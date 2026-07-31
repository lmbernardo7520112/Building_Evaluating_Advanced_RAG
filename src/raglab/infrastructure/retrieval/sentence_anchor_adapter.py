"""Sentence-Anchor Retrieval Adapter (S0) for RAGLab v7 Slice 3.

S0 is the causal control that isolates ONLY the effect of granularity
(sentence vs. fixed chunk) WITHOUT any window expansion.

Causal role:
  F0 vs S0  →  effect of indexing granularity (fixed chunk vs sentence)
  S0 vs W0  →  effect of window expansion (same anchors, same scores)

Implementation contract:
  - Index and retrieve individual SENTENCES (anchor = retrieval unit)
  - Return ONLY the anchor sentence — NO window expansion
  - Embeddings are sentence-level (same model as W0 for fair comparison)
  - Deduplication: a sentence can only appear once in results
  - top_k is applied to final results (not candidates)
  - Page provenance is preserved on every RetrievedEvidence

The returned text MUST be the anchor sentence only (not the window),
so that F0 × S0 isolates granularity and S0 × W0 isolates expansion.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from raglab.application.ports.retrieval import RetrievalPort
from raglab.domain.entities import Chunk, RetrievedEvidence
from raglab.domain.value_objects import ChunkId, DocumentPage
from raglab.infrastructure.embeddings.fastembed_adapter import FastEmbedEmbeddingAdapter
from raglab.infrastructure.retrieval.sentence_window_adapter import split_into_sentences


class SentenceAnchorAdapter(RetrievalPort):
    """S0: Sentence-level indexing and retrieval WITHOUT window expansion.

    Each sentence is indexed independently.  On retrieval the anchor
    sentence itself is returned — no surrounding context is added.
    This is the causal control needed to separate:
      - granularity effect  (F0 × S0)
      - expansion effect    (S0 × W0)
    """

    def __init__(
        self,
        embedding_adapter: FastEmbedEmbeddingAdapter | None = None,
    ) -> None:
        self.embedding_adapter = embedding_adapter or FastEmbedEmbeddingAdapter()
        self._sentence_nodes: list[dict[str, Any]] = []
        self._sentence_embeddings: list[list[float]] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_pages(self, pages: Sequence[DocumentPage]) -> int:
        """Index source pages by sentence, storing anchor text only."""
        self._sentence_nodes.clear()
        self._sentence_embeddings.clear()

        all_sentence_texts: list[str] = []

        for page in pages:
            doc_id = page.document_id
            page_num = page.page_number
            sentences = split_into_sentences(page.text)

            for idx, sentence in enumerate(sentences):
                chunk_id_val = f"{doc_id}_p{page_num}_s{idx}_anchor"
                node: dict[str, Any] = {
                    "chunk_id": chunk_id_val,
                    "document_id": doc_id,
                    "page_number": page_num,
                    "sentence_index": idx,
                    "anchor_text": sentence,   # returned text = anchor only
                }
                self._sentence_nodes.append(node)
                all_sentence_texts.append(sentence)

        if all_sentence_texts:
            embeddings = self.embedding_adapter.embed_texts(all_sentence_texts)
            self._sentence_embeddings = [list(vec) for vec in embeddings]

        return len(self._sentence_nodes)

    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Fallback chunk indexing — converts chunks to pages."""
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

    # ------------------------------------------------------------------
    # Retrieval — anchor-only, no expansion
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedEvidence]:
        """Retrieve top_k sentence anchors.  Returns anchor text only.

        Key invariant: the returned text is the ANCHOR sentence,
        never a window.  This distinguishes S0 from W0.
        """
        if not query or not query.strip() or top_k <= 0 or not self._sentence_nodes:
            return []

        query_emb = self.embedding_adapter._get_query_embedding(query)

        scores_with_nodes: list[tuple[float, dict[str, Any]]] = []
        for vec, node in zip(
            self._sentence_embeddings, self._sentence_nodes, strict=False
        ):
            dot = sum(q * v for q, v in zip(query_emb, vec, strict=False))
            q_norm = sum(q * q for q in query_emb) ** 0.5
            v_norm = sum(v * v for v in vec) ** 0.5
            sim = (dot / (q_norm * v_norm)) if (q_norm > 0 and v_norm > 0) else 0.0
            scores_with_nodes.append((sim, node))

        scores_with_nodes.sort(key=lambda x: (-x[0], x[1]["chunk_id"]))

        retrieved_evidence: list[RetrievedEvidence] = []
        seen_chunk_ids: set[str] = set()
        rank = 1

        for raw_score, node in scores_with_nodes:
            cid = node["chunk_id"]
            if cid in seen_chunk_ids:
                continue
            seen_chunk_ids.add(cid)

            clamped_score = max(0.0, min(1.0, (raw_score + 1.0) / 2.0))

            evidence = RetrievedEvidence(
                chunk_id=ChunkId(cid),
                document_id=f"{node['document_id']}_p{node['page_number']}",
                text=node["anchor_text"],   # anchor only — never window text
                rank=rank,
                score=round(clamped_score, 4),
            )
            retrieved_evidence.append(evidence)
            rank += 1

            if len(retrieved_evidence) >= top_k:
                break

        return retrieved_evidence
