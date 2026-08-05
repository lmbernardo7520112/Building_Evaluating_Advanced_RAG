# ruff: noqa: E501
"""Human Qrels Metrics Module — Deterministic Retrieval & Damage Evaluation.

Computes:
- nDCG@k with 0-3 graded relevance
- Recall@k (binary threshold grade >= 1)
- MRR@k for first relevant item
- Judged coverage@k, unjudged count@k, unresolved mapping count@k
- Reranker damage metrics using human qrels
- Abstention / negative control retrieval metrics for unanswerable queries (e.g., q_test_04)
"""

from __future__ import annotations

import math
from typing import Any

from raglab.evaluation.contracts.human_qrels_v2 import HumanQrelsSet


def _compute_dcg(grades: list[int]) -> float:
    dcg = 0.0
    for idx, g in enumerate(grades, 1):
        if g > 0:
            dcg += (2**g - 1) / math.log2(idx + 1)
    return dcg


def compute_human_qrels_metrics_for_question(
    qrels_set: HumanQrelsSet,
    question_id: str,
    retrieved_passage_ids: list[str | None],
    k: int = 3,
    candidate_passage_ids_pre_rerank: list[str | None] | None = None,
) -> dict[str, Any]:
    """Compute deterministic retrieval metrics for a single question using HumanQrelsSet."""
    question_qrels = qrels_set.get_qrels_for_question(question_id)
    is_abstention = qrels_set.is_abstention_question(question_id)

    # Top-k retrieved items
    top_k_retrieved = retrieved_passage_ids[:k]

    # Resolve passage IDs
    valid_top_k: list[str] = []
    unresolved_count = 0
    for pid in top_k_retrieved:
        if pid and isinstance(pid, str) and pid.startswith("ps_"):
            valid_top_k.append(pid)
        else:
            unresolved_count += 1

    # Judged vs unjudged status
    judged_count = 0
    unjudged_count = 0
    retrieved_grades: list[int] = []

    for pid in valid_top_k:
        qrel = qrels_set.get_qrel(question_id, pid)
        if qrel is not None:
            judged_count += 1
            retrieved_grades.append(qrel.relevance_grade)
        else:
            unjudged_count += 1
            retrieved_grades.append(0)  # Policy: unjudged treated as 0 for DCG, but counted in unjudged_count

    total_retrieved_items = len(top_k_retrieved)
    judged_coverage = (
        float(judged_count) / float(total_retrieved_items)
        if total_retrieved_items > 0
        else 0.0
    )

    # All relevant passages for this question (grade >= 1)
    relevant_qrels = [q for q in question_qrels if q.relevance_grade >= 1]
    total_relevant_count = len(relevant_qrels)

    # Calculate Ideal DCG (IDCG@k)
    all_question_grades = sorted(
        [q.relevance_grade for q in question_qrels], reverse=True
    )
    idcg = _compute_dcg(all_question_grades[:k])

    # 1. nDCG@k
    if idcg == 0.0:
        ndcg_status = "NOT_APPLICABLE"
        ndcg_score = None
    else:
        dcg = _compute_dcg(retrieved_grades)
        ndcg_status = "COMPUTED"
        ndcg_score = round(dcg / idcg, 4)

    # 2. Recall@k & MRR@k
    if total_relevant_count == 0:
        recall_status = "NOT_APPLICABLE"
        recall_score = None
        mrr_status = "NOT_APPLICABLE"
        mrr_score = None
        relevant_retrieved_count = 0
    else:
        # Count relevant items retrieved in top-k
        rel_retrieved = sum(1 for g in retrieved_grades if g >= 1)
        relevant_retrieved_count = rel_retrieved

        recall_status = "COMPUTED"
        recall_score = round(float(rel_retrieved) / float(total_relevant_count), 4)

        # MRR@k
        first_rel_rank = None
        for idx, g in enumerate(retrieved_grades, 1):
            if g >= 1:
                first_rel_rank = idx
                break

        mrr_status = "COMPUTED"
        if first_rel_rank is not None:
            mrr_score = round(1.0 / float(first_rel_rank), 4)
        else:
            mrr_score = 0.0

    # 3. Abstention & False Positive retrieval metrics (e.g. q_test_04)
    false_positive_negative_control_retrieved = sum(
        1
        for pid in valid_top_k
        if (q := qrels_set.get_qrel(question_id, pid)) is not None
        and q.evidence_role == "NEGATIVE_CONTROL"
    )

    # 4. Reranker Damage (if pre-rerank candidates provided)
    reranker_damage: dict[str, Any] | None = None
    if candidate_passage_ids_pre_rerank is not None:
        pre_valid = [
            pid for pid in candidate_passage_ids_pre_rerank
            if pid and isinstance(pid, str) and pid.startswith("ps_")
        ]
        pre_rel = [
            pid for pid in pre_valid
            if (q := qrels_set.get_qrel(question_id, pid)) is not None and q.relevance_grade >= 1
        ]
        post_rel = [
            pid for pid in valid_top_k
            if (q := qrels_set.get_qrel(question_id, pid)) is not None and q.relevance_grade >= 1
        ]

        dropped = set(pre_rel) - set(post_rel)
        dropped_count = len(dropped)
        dropped_rate = (
            float(dropped_count) / float(len(pre_rel))
            if pre_rel
            else 0.0
        )

        recall_pre = (
            float(len(pre_rel)) / float(total_relevant_count)
            if total_relevant_count > 0
            else None
        )
        recall_post = recall_score  # None for unanswerable

        delta_recall = (
            round(recall_post - recall_pre, 4)
            if (recall_post is not None and recall_pre is not None)
            else None
        )

        reranker_damage = {
            "pre_rerank_candidate_count": len(candidate_passage_ids_pre_rerank),
            "pre_rerank_relevant_count": len(pre_rel),
            "post_rerank_relevant_count": len(post_rel),
            "dropped_relevant_count": dropped_count,
            "dropped_relevant_passage_ids": sorted(dropped),
            "dropped_relevant_rate": round(dropped_rate, 4),
            "recall_pre": round(recall_pre, 4) if recall_pre is not None else None,
            "recall_post": recall_post,
            "delta_recall": delta_recall,
        }

    return {
        "question_id": question_id,
        "is_abstention_question": is_abstention,
        "k": k,
        "metrics": {
            "ndcg_at_k": {
                "status": ndcg_status,
                "score": ndcg_score,
                "k": k,
            },
            "recall_at_k": {
                "status": recall_status,
                "score": recall_score,
                "k": k,
            },
            "mrr_at_k": {
                "status": mrr_status,
                "score": mrr_score,
                "k": k,
            },
        },
        "retrieval_accounting": {
            "total_relevant_in_qrels": total_relevant_count,
            "relevant_retrieved_count": relevant_retrieved_count,
            "judged_count": judged_count,
            "unjudged_count": unjudged_count,
            "unresolved_mapping_count": unresolved_count,
            "judged_coverage_rate": round(judged_coverage, 4),
            "false_positive_negative_control_count": false_positive_negative_control_retrieved,
        },
        "reranker_damage": reranker_damage,
    }
