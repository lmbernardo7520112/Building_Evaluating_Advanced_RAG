"""Sentence-Window Retrieval Adapter for RAGLab v7.

Implements sentence-level indexing with window expansion,
Portuguese abbreviation handling, and deduplication with page provenance.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from raglab.application.ports.retrieval import RetrievalPort
from raglab.domain.entities import Chunk, RetrievedEvidence
from raglab.domain.value_objects import ChunkId, DocumentPage
from raglab.infrastructure.embeddings.fastembed_adapter import FastEmbedEmbeddingAdapter

# Portuguese abbreviation pattern to avoid false sentence splits
_ABBREVIATIONS = re.compile(
    r"\b(e\.g|i\.e|ex|pág|pag|fig|cap|tab|vs|art|sec|seção|vol|dr|prof|sr|sra|pp|nº)\.\s*$",
    re.IGNORECASE,
)


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences while respecting Portuguese abbreviations."""
    if not text or not text.strip():
        return []

    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences: list[str] = []
    buffer = ""

    for s in raw_sentences:
        if not s.strip():
            continue
        combined = f"{buffer} {s}" if buffer else s

        # Check if sentence ends with a known abbreviation
        if _ABBREVIATIONS.search(combined):
            buffer = combined
        else:
            sentences.append(combined)
            buffer = ""

    if buffer:
        sentences.append(buffer)

    return [s.strip() for s in sentences if s.strip()]


class SentenceWindowAdapter(RetrievalPort):
    """Sentence-Window retrieval adapter with configurable window expansion."""

    def __init__(
        self,
        embedding_adapter: FastEmbedEmbeddingAdapter | None = None,
        window_size: int = 2,
    ) -> None:
        self.embedding_adapter = embedding_adapter or FastEmbedEmbeddingAdapter()
        self.window_size = window_size
        self._sentence_nodes: list[dict[str, Any]] = []
        self._sentence_embeddings: list[list[float]] = []

    def index_pages(self, pages: Sequence[DocumentPage]) -> int:
        """Index source pages by sentence units and attach window context."""
        self._sentence_nodes.clear()
        self._sentence_embeddings.clear()

        all_sentence_texts: list[str] = []

        for page in pages:
            doc_id = page.document_id
            page_num = page.page_number
            sentences = split_into_sentences(page.text)

            for idx, sentence in enumerate(sentences):
                # Calculate window bounds within the same page
                start_w = max(0, idx - self.window_size)
                end_w = min(len(sentences), idx + self.window_size + 1)
                window_text = " ".join(sentences[start_w:end_w])

                chunk_id_val = f"{doc_id}_p{page_num}_s{idx}"

                node = {
                    "chunk_id": chunk_id_val,
                    "document_id": doc_id,
                    "page_number": page_num,
                    "sentence_index": idx,
                    "anchor_text": sentence,
                    "window_text": window_text,
                    "window_sentence_indices": list(range(start_w, end_w)),
                }
                self._sentence_nodes.append(node)
                all_sentence_texts.append(sentence)

        if all_sentence_texts:
            embeddings = self.embedding_adapter.embed_texts(all_sentence_texts)
            self._sentence_embeddings = [list(vec) for vec in embeddings]

        return len(self._sentence_nodes)

    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Fallback chunk indexing compatibility."""
        pages: list[DocumentPage] = []
        for c in chunks:
            pages.append(
                DocumentPage(
                    document_id=c.document_id,
                    page_number=c.start_page,
                    text=c.text,
                )
            )
        self.index_pages(pages)

    def clear(self) -> None:
        self._sentence_nodes.clear()
        self._sentence_embeddings.clear()

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedEvidence]:
        """Retrieve top_k sentence anchors and expand into deduplicated windows."""
        if not query or not query.strip() or top_k <= 0 or not self._sentence_nodes:
            return []

        query_emb = self.embedding_adapter._get_query_embedding(query)

        # Compute cosine similarities
        scores_with_nodes: list[tuple[float, dict[str, Any]]] = []
        for vec, node in zip(
            self._sentence_embeddings, self._sentence_nodes, strict=False
        ):
            dot = sum(q * v for q, v in zip(query_emb, vec, strict=False))
            q_norm = sum(q * q for q in query_emb) ** 0.5
            v_norm = sum(v * v for v in vec) ** 0.5
            sim = (dot / (q_norm * v_norm)) if (q_norm > 0 and v_norm > 0) else 0.0
            scores_with_nodes.append((sim, node))

        # Sort by similarity descending, then chunk_id ascending for stability
        scores_with_nodes.sort(key=lambda x: (-x[0], x[1]["chunk_id"]))

        # Retrieve top candidates for window expansion & deduplication
        candidate_count = min(len(scores_with_nodes), top_k * 3)
        top_candidates = scores_with_nodes[:candidate_count]

        # Window deduplication: merge overlapping windows on the same page
        seen_sentence_spans: set[tuple[str, int, int]] = set()
        retrieved_evidence: list[RetrievedEvidence] = []
        rank = 1

        for raw_score, node in top_candidates:
            doc_id = node["document_id"]
            page_num = node["page_number"]
            span_key = (doc_id, page_num, node["sentence_index"])

            # Deduplicate windows covering identical sentence spans
            if span_key in seen_sentence_spans:
                continue

            # Mark all sentences in this window as covered
            for s_idx in node["window_sentence_indices"]:
                seen_sentence_spans.add((doc_id, page_num, s_idx))

            # Clamp score to [0,1] range safely
            clamped_score = max(0.0, min(1.0, (raw_score + 1.0) / 2.0))

            evidence = RetrievedEvidence(
                chunk_id=ChunkId(node["chunk_id"]),
                document_id=f"{doc_id}_p{page_num}",
                text=node["window_text"],
                rank=rank,
                score=round(clamped_score, 4),
            )
            retrieved_evidence.append(evidence)
            rank += 1

            if len(retrieved_evidence) >= top_k:
                break

        return retrieved_evidence
