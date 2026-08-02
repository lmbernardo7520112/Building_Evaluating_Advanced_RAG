"""Deterministic retrieval metrics — Recall@k and MRR.

Framework-independent, domain-pure implementation.
These metrics evaluate retrieval quality against ground truth
relevance judgments. They do NOT depend on LLM evaluation
(RAG Triad) and are computed entirely offline.

Recall@k = |relevant ∩ retrieved_up_to_k| / |relevant|
MRR = (1/N) * Σ(1/rank_i) for first relevant result per query

Policies:
- Empty ground truth: explicit policy (skip or 0), not silent.
- Duplicates: deduplicated before scoring; no inflation.
- Unknown IDs: treated as non-relevant.
- Output always in [0, 1].
- Absent result ≠ zero; use None for not-computed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RecallAtK:
    """Recall@k for a single query.

    Attributes:
        value: The recall score in [0, 1], or None if not computable.
        k: The cutoff.
        retrieved_relevant: Number of relevant items found up to k.
        total_relevant: Total number of relevant items in ground truth.
        skipped: True if ground truth was empty (policy: skip).
    """

    value: float | None
    k: int
    retrieved_relevant: int
    total_relevant: int
    skipped: bool = False


@dataclass(frozen=True, slots=True)
class MRRResult:
    """Mean Reciprocal Rank across queries.

    Attributes:
        value: The MRR score in [0, 1], or None if no queries.
        num_queries: Number of queries evaluated.
        reciprocal_ranks: Per-query reciprocal ranks.
    """

    value: float | None
    num_queries: int
    reciprocal_ranks: tuple[float, ...]


def compute_recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
    *,
    empty_gt_policy: str = "skip",
) -> RecallAtK:
    """Compute Recall@k for a single query.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs (rank order).
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k: Cutoff position.
        empty_gt_policy: How to handle empty ground truth.
            "skip" → RecallAtK with value=None, skipped=True
            "zero" → RecallAtK with value=0.0

    Returns:
        RecallAtK result.

    Raises:
        ValueError: If k < 1 or empty_gt_policy is unknown.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if empty_gt_policy not in ("skip", "zero"):
        raise ValueError(f"Unknown empty_gt_policy: {empty_gt_policy}")

    if not relevant_ids:
        if empty_gt_policy == "skip":
            return RecallAtK(
                value=None, k=k, retrieved_relevant=0,
                total_relevant=0, skipped=True,
            )
        return RecallAtK(
            value=0.0, k=k, retrieved_relevant=0,
            total_relevant=0, skipped=False,
        )

    # Deduplicate retrieved IDs preserving order (first occurrence)
    seen: set[str] = set()
    deduped: list[str] = []
    for rid in retrieved_ids:
        if rid not in seen:
            seen.add(rid)
            deduped.append(rid)

    top_k = deduped[:k]
    retrieved_relevant = len(set(top_k) & relevant_ids)
    total_relevant = len(relevant_ids)

    value = retrieved_relevant / total_relevant
    # Clamp to [0, 1] for safety (should already be in range)
    value = max(0.0, min(1.0, value))

    return RecallAtK(
        value=value,
        k=k,
        retrieved_relevant=retrieved_relevant,
        total_relevant=total_relevant,
    )


def compute_reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Compute reciprocal rank for a single query.

    Returns 1/rank of the first relevant result, or 0.0 if none found.
    Deduplicates retrieved_ids before ranking.
    """
    seen: set[str] = set()
    rank = 0
    for rid in retrieved_ids:
        if rid in seen:
            continue
        seen.add(rid)
        rank += 1
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_mrr(
    queries: list[tuple[list[str], set[str]]],
    *,
    empty_gt_policy: str = "skip",
) -> MRRResult:
    """Compute Mean Reciprocal Rank across queries.

    Args:
        queries: List of (retrieved_ids, relevant_ids) tuples.
        empty_gt_policy: "skip" excludes queries with empty GT;
                         "zero" treats them as RR=0.

    Returns:
        MRRResult.

    Raises:
        ValueError: If empty_gt_policy is unknown.
    """
    if empty_gt_policy not in ("skip", "zero"):
        raise ValueError(f"Unknown empty_gt_policy: {empty_gt_policy}")

    reciprocal_ranks: list[float] = []

    for retrieved_ids, relevant_ids in queries:
        if not relevant_ids:
            if empty_gt_policy == "skip":
                continue
            reciprocal_ranks.append(0.0)
            continue
        rr = compute_reciprocal_rank(retrieved_ids, relevant_ids)
        reciprocal_ranks.append(rr)

    if not reciprocal_ranks:
        return MRRResult(
            value=None,
            num_queries=0,
            reciprocal_ranks=(),
        )

    mrr_value = sum(reciprocal_ranks) / len(reciprocal_ranks)
    # Clamp to [0, 1]
    mrr_value = max(0.0, min(1.0, mrr_value))

    return MRRResult(
        value=mrr_value,
        num_queries=len(reciprocal_ranks),
        reciprocal_ranks=tuple(reciprocal_ranks),
    )
