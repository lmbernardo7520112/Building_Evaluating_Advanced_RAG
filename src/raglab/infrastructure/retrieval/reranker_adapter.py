"""Local Reranker Adapter and Damage Assessment for RAGLab v7.

Provides second-stage candidate reranking, passage elimination tracking,
and damage metrics calculation (recall_pre vs recall_post).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from raglab.domain.entities import RetrievedEvidence
from raglab.infrastructure.embeddings.fastembed_adapter import FastEmbedEmbeddingAdapter


@dataclass(frozen=True, slots=True)
class RerankerDamageMetrics:
    """Metrics assessing potential evidence loss caused by second-stage reranking."""

    candidate_k: int
    top_n: int
    recall_pre: float
    recall_post: float
    delta_recall: float
    passages_dropped_count: int
    relevant_passages_dropped_count: int
    relevant_passage_dropped_rate: float


class LocalRerankerAdapter:
    """Local CPU reranker evaluating second-stage query-candidate similarity."""

    def __init__(
        self, embedding_adapter: FastEmbedEmbeddingAdapter | None = None
    ) -> None:
        self.embedding_adapter = embedding_adapter or FastEmbedEmbeddingAdapter()

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedEvidence],
        top_n: int = 3,
    ) -> tuple[list[RetrievedEvidence], list[RetrievedEvidence]]:
        """Rerank candidate evidence and return (reranked_top_n, dropped_candidates)."""
        if not candidates or top_n <= 0 or not query.strip():
            return list(candidates[:top_n]), list(candidates[top_n:])

        query_emb = self.embedding_adapter._get_query_embedding(query)
        scored_candidates: list[tuple[float, RetrievedEvidence]] = []

        for candidate in candidates:
            cand_emb = self.embedding_adapter._get_text_embedding(candidate.text)
            dot = sum(q * c for q, c in zip(query_emb, cand_emb, strict=False))
            q_norm = sum(q * q for q in query_emb) ** 0.5
            c_norm = sum(c * c for c in cand_emb) ** 0.5
            sim = (dot / (q_norm * c_norm)) if (q_norm > 0 and c_norm > 0) else 0.0
            scored_candidates.append((sim, candidate))

        # Sort descending by reranker score, maintaining stability by original rank
        scored_candidates.sort(key=lambda x: (-x[0], x[1].rank))

        reranked: list[RetrievedEvidence] = []
        rank = 1
        for sim, orig in scored_candidates[:top_n]:
            clamped_score = max(0.0, min(1.0, (sim + 1.0) / 2.0))
            reranked.append(
                RetrievedEvidence(
                    chunk_id=orig.chunk_id,
                    document_id=orig.document_id,
                    text=orig.text,
                    rank=rank,
                    score=round(clamped_score, 4),
                )
            )
            rank += 1

        top_n_ids = {r.chunk_id.value for r in reranked}
        dropped = [c for c in candidates if c.chunk_id.value not in top_n_ids]

        return reranked, dropped

    @staticmethod
    def calculate_damage_metrics(
        candidates_pre: Sequence[RetrievedEvidence],
        candidates_post: Sequence[RetrievedEvidence],
        relevant_chunk_ids: set[str],
        candidate_k: int,
        top_n: int,
    ) -> RerankerDamageMetrics:
        """Calculate damage metrics assessing if reranking dropped relevant passages."""
        pre_ids = [c.chunk_id.value for c in candidates_pre[:top_n]]
        post_ids = [c.chunk_id.value for c in candidates_post[:top_n]]

        if not relevant_chunk_ids:
            dropped_cnt = max(0, len(candidates_pre) - len(candidates_post))
            return RerankerDamageMetrics(
                candidate_k=candidate_k,
                top_n=top_n,
                recall_pre=0.0,
                recall_post=0.0,
                delta_recall=0.0,
                passages_dropped_count=dropped_cnt,
                relevant_passages_dropped_count=0,
                relevant_passage_dropped_rate=0.0,
            )

        relevant_in_pre = sum(1 for cid in pre_ids if cid in relevant_chunk_ids)
        relevant_in_post = sum(1 for cid in post_ids if cid in relevant_chunk_ids)

        recall_pre = relevant_in_pre / len(relevant_chunk_ids)
        recall_post = relevant_in_post / len(relevant_chunk_ids)
        delta_recall = recall_post - recall_pre

        all_candidate_ids = [c.chunk_id.value for c in candidates_pre]
        dropped_ids = [cid for cid in all_candidate_ids if cid not in post_ids]
        rel_dropped = sum(1 for cid in dropped_ids if cid in relevant_chunk_ids)
        tot_rel = sum(1 for cid in all_candidate_ids if cid in relevant_chunk_ids)

        dropped_rate = (rel_dropped / tot_rel) if tot_rel > 0 else 0.0

        return RerankerDamageMetrics(
            candidate_k=candidate_k,
            top_n=top_n,
            recall_pre=recall_pre,
            recall_post=recall_post,
            delta_recall=delta_recall,
            passages_dropped_count=len(dropped_ids),
            relevant_passages_dropped_count=rel_dropped,
            relevant_passage_dropped_rate=round(dropped_rate, 4),
        )
