"""Deterministic Metrics Computation Engine (Ground Truth v2).

All metrics here are pure deterministic math functions.
No LLM calls, no API calls, no non-deterministic side effects.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from raglab.evaluation.contracts.ground_truth_v2 import CanonicalEvidence


def compute_passage_recall_at_k(
    retrieved_passage_ids: Sequence[str],
    gold_evidences: Sequence[CanonicalEvidence],
    k: int = 3,
) -> float:
    """Compute Passage Recall@k."""
    if not gold_evidences:
        return 1.0  # Empty gold set: default to 1.0

    gold_ids = {ev.passage_id for ev in gold_evidences}
    retrieved_k = set(retrieved_passage_ids[:k])

    hits = len(retrieved_k.intersection(gold_ids))
    return float(hits / len(gold_ids))


def compute_mrr(
    retrieved_passage_ids: Sequence[str],
    gold_evidences: Sequence[CanonicalEvidence],
) -> float:
    """Compute Mean Reciprocal Rank (MRR)."""
    if not gold_evidences:
        return 1.0

    gold_ids = {ev.passage_id for ev in gold_evidences}
    for rank, pid in enumerate(retrieved_passage_ids, start=1):
        if pid in gold_ids:
            return float(1.0 / rank)
    return 0.0


def compute_ndcg_at_k(
    retrieved_passage_ids: Sequence[str],
    gold_evidences: Sequence[CanonicalEvidence],
    k: int = 3,
) -> str | float:
    """Compute nDCG@k.

    EXECUTION GUARD 2:
    If relevance_grade is None (unannotated/legacy binary qrels), returns
    'NOT_COMPUTABLE_MISSING_GRADED_QRELS' rather than fabricating artificial grades.
    """
    if not gold_evidences:
        return 1.0

    for ev in gold_evidences:
        if ev.relevance_grade is None:
            return "NOT_COMPUTABLE_MISSING_GRADED_QRELS"

    import math

    grade_map: dict[str, int] = {
        ev.passage_id: ev.relevance_grade
        for ev in gold_evidences
        if ev.relevance_grade is not None
    }

    # DCG@k
    dcg = 0.0
    for i, pid in enumerate(retrieved_passage_ids[:k], start=1):
        rel = grade_map.get(pid, 0)
        dcg += float((2**rel - 1) / math.log2(i + 1))

    # IDCG@k
    ideal_grades = sorted(grade_map.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_grades, start=1):
        idcg += float((2**rel - 1) / math.log2(i + 1))

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)


def compute_citation_precision_recall(
    cited_passage_ids: Sequence[str],
    retrieved_passage_ids: Sequence[str],
    gold_evidences: Sequence[CanonicalEvidence],
) -> dict[str, float]:
    """Compute Citation Precision and Citation Recall against gold evidence."""
    if not cited_passage_ids:
        return {"citation_precision": 0.0, "citation_recall": 0.0}

    gold_ids = {ev.passage_id for ev in gold_evidences}
    retrieved_set = set(retrieved_passage_ids)

    valid_citations = [pid for pid in cited_passage_ids if pid in retrieved_set]
    gold_hits = [pid for pid in valid_citations if pid in gold_ids]

    precision = len(gold_hits) / len(cited_passage_ids) if cited_passage_ids else 0.0
    recall = len(gold_hits) / len(gold_ids) if gold_ids else 1.0

    return {
        "citation_precision": float(precision),
        "citation_recall": float(recall),
    }


def compute_abstention_confusion_matrix(
    is_abstained: bool,
    is_unanswerable: bool,
) -> dict[str, Any]:
    """Compute 2x2 confusion matrix entry for abstention correctness."""
    if is_unanswerable and is_abstained:
        matrix_category = "TRUE_POSITIVE_ABSTENTION"
    elif is_unanswerable and not is_abstained:
        matrix_category = "FALSE_NEGATIVE_HALLUCINATED_ANSWER"
    elif not is_unanswerable and is_abstained:
        matrix_category = "FALSE_POSITIVE_UNNECESSARY_ABSTENTION"
    else:
        matrix_category = "TRUE_NEGATIVE_CORRECT_ANSWER"

    correct = is_abstained == is_unanswerable
    return {
        "is_abstained": is_abstained,
        "is_unanswerable": is_unanswerable,
        "matrix_category": matrix_category,
        "abstention_correct": correct,
    }


def compute_nugget_and_contradiction_metrics(
    answer_text: str,
    gold_nuggets: Sequence[str] | None,
) -> str:
    """Nugget recall / precision / contradiction count.

    Returns 'NOT_COMPUTABLE_MATCHER_NOT_CONFIGURED' when matcher configuration is
    absent.
    """
    return "NOT_COMPUTABLE_MATCHER_NOT_CONFIGURED"
