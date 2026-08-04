"""Inter-Annotator Agreement Tool (Gate B1 - Etapa 6).

Calculates inter-annotator agreement between Annotator A and Annotator B offline:
- Percentage agreement for answerability & relevance
- Cohen's Kappa for answerability
- Linear Weighted Cohen's Kappa for relevance grades (0-3)
- Explicit divergence reporting for question_ids and passage_ids
- Non-computable metric status reporting (NOT_COMPUTABLE_*)

Generates:
- benchmarks/ground_truth/v2/agreement_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# Ensure src is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def compute_cohens_kappa(r1: list[Any], r2: list[Any]) -> str | float:
    """Compute Cohen's Kappa for categorical ratings."""
    if not r1 or not r2 or len(r1) != len(r2):
        return "NOT_COMPUTABLE_EMPTY_SAMPLE"

    total = len(r1)
    if total == 0:
        return "NOT_COMPUTABLE_EMPTY_SAMPLE"

    # Count observed agreement
    agree_count = sum(1 for a, b in zip(r1, r2, strict=False) if a == b)
    p_o = agree_count / total

    # Categories
    categories = sorted(set(r1) | set(r2))
    if len(categories) <= 1:
        return "NOT_COMPUTABLE_SINGLE_CLASS"

    # Expected agreement
    freq1 = {cat: sum(1 for x in r1 if x == cat) for cat in categories}
    freq2 = {cat: sum(1 for x in r2 if x == cat) for cat in categories}

    p_e = sum((freq1[cat] / total) * (freq2[cat] / total) for cat in categories)

    if math.isclose(p_e, 1.0):
        return "NOT_COMPUTABLE_SINGLE_CLASS"

    kappa = (p_o - p_e) / (1.0 - p_e)
    return float(round(kappa, 4))


def compute_weighted_kappa(
    r1: list[int], r2: list[int], num_categories: int = 4
) -> str | float:
    """Compute Linear Weighted Cohen's Kappa for ordinal ratings (0-3)."""
    if not r1 or not r2 or len(r1) != len(r2):
        return "NOT_COMPUTABLE_EMPTY_SAMPLE"

    total = len(r1)
    if total == 0:
        return "NOT_COMPUTABLE_EMPTY_SAMPLE"

    unique_vals = set(r1) | set(r2)
    if len(unique_vals) <= 1:
        return "NOT_COMPUTABLE_SINGLE_CLASS"

    max_diff = num_categories - 1

    # Observed weight matrix & agreement
    weights = [
        [1.0 - (abs(i - j) / max_diff) for j in range(num_categories)]
        for i in range(num_categories)
    ]

    # Matrix counts
    obs = [[0 for _ in range(num_categories)] for _ in range(num_categories)]
    for a, b in zip(r1, r2, strict=False):
        if 0 <= a < num_categories and 0 <= b < num_categories:
            obs[a][b] += 1

    p_o = (
        sum(
            obs[i][j] * weights[i][j]
            for i in range(num_categories)
            for j in range(num_categories)
        )
        / total
    )

    # Marginals
    row_sums = [
        sum(obs[i][j] for j in range(num_categories)) for i in range(num_categories)
    ]
    col_sums = [
        sum(obs[i][j] for i in range(num_categories)) for j in range(num_categories)
    ]

    p_e = sum(
        row_sums[i] * col_sums[j] * weights[i][j]
        for i in range(num_categories)
        for j in range(num_categories)
    ) / (total * total)

    if math.isclose(p_e, 1.0):
        return "NOT_COMPUTABLE_SINGLE_CLASS"

    w_kappa = (p_o - p_e) / (1.0 - p_e)
    return float(round(w_kappa, 4))


def load_annotation_records_by_qid(annotator_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all records from annotator JSONL files keyed by question_id."""
    records: dict[str, dict[str, Any]] = {}
    if not annotator_dir.exists():
        return records

    for jsonl_file in annotator_dir.glob("*.jsonl"):
        with jsonl_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    qid = item["question_id"]
                    records[qid] = item
    return records


def compute_agreement(root_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Compute agreement report for Annotator A and Annotator B."""
    packages_dir = root_dir / "annotation_packages"
    ann_a_dir = packages_dir / "annotator_a"
    ann_b_dir = packages_dir / "annotator_b"

    recs_a = load_annotation_records_by_qid(ann_a_dir)
    recs_b = load_annotation_records_by_qid(ann_b_dir)

    common_qids = sorted(set(recs_a.keys()) & set(recs_b.keys()))

    # Answerability agreement vectors
    ans_a: list[bool] = []
    ans_b: list[bool] = []

    # Relevance grade vectors
    rel_a: list[int] = []
    rel_b: list[int] = []

    divergent_questions: list[str] = []
    divergent_passages: list[dict[str, str]] = []
    pending_items_count = 0

    for qid in common_qids:
        item_a = recs_a[qid]
        item_b = recs_b[qid]

        a_ans = item_a.get("answerability")
        b_ans = item_b.get("answerability")

        if a_ans is None or b_ans is None:
            pending_items_count += 1
        else:
            ans_a.append(bool(a_ans))
            ans_b.append(bool(b_ans))
            if a_ans != b_ans:
                divergent_questions.append(qid)

        # Candidate relevance grades
        cands_a = {c["passage_id"]: c for c in item_a.get("candidate_passages", [])}
        cands_b = {c["passage_id"]: c for c in item_b.get("candidate_passages", [])}

        common_pids = sorted(set(cands_a.keys()) & set(cands_b.keys()))
        for pid in common_pids:
            g_a = cands_a[pid].get("relevance_grade")
            g_b = cands_b[pid].get("relevance_grade")

            if g_a is None or g_b is None:
                pending_items_count += 1
            else:
                rel_a.append(g_a)
                rel_b.append(g_b)
                if g_a != g_b:
                    divergent_passages.append(
                        {
                            "question_id": qid,
                            "passage_id": pid,
                            "grade_a": g_a,
                            "grade_b": g_b,
                        }
                    )

    # Calculations
    if pending_items_count > 0 and len(ans_a) == 0:
        ans_agreement_pct = "NOT_COMPUTABLE_INCOMPLETE_ANNOTATIONS"
        ans_kappa = "NOT_COMPUTABLE_INCOMPLETE_ANNOTATIONS"
        rel_agreement_pct = "NOT_COMPUTABLE_INCOMPLETE_ANNOTATIONS"
        rel_weighted_kappa = "NOT_COMPUTABLE_INCOMPLETE_ANNOTATIONS"
    else:
        ans_agree = sum(1 for x, y in zip(ans_a, ans_b, strict=True) if x == y)
        ans_agreement_pct = (
            round(ans_agree / len(ans_a), 4) if ans_a else "NOT_COMPUTABLE_EMPTY_SAMPLE"
        )
        ans_kappa = compute_cohens_kappa(ans_a, ans_b)

        rel_agree = sum(1 for x, y in zip(rel_a, rel_b, strict=True) if x == y)
        rel_agreement_pct = (
            round(rel_agree / len(rel_a), 4) if rel_a else "NOT_COMPUTABLE_EMPTY_SAMPLE"
        )
        rel_weighted_kappa = compute_weighted_kappa(rel_a, rel_b)

    report_data = {
        "report_version": "2.0.0",
        "question_count_evaluated": len(common_qids),
        "pending_items_count": pending_items_count,
        "answerability_metrics": {
            "percent_agreement": ans_agreement_pct,
            "cohens_kappa": ans_kappa,
            "divergent_questions_count": len(divergent_questions),
            "divergent_question_ids": divergent_questions,
        },
        "relevance_metrics": {
            "percent_agreement": rel_agreement_pct,
            "linear_weighted_kappa": rel_weighted_kappa,
            "divergent_passages_count": len(divergent_passages),
            "divergent_passages": divergent_passages,
        },
        "items_requiring_adjudication": (
            len(divergent_questions) + len(divergent_passages)
        ),
        "status": (
            "COMPLETED"
            if pending_items_count == 0
            else "INCOMPLETE_ANNOTATIONS_PENDING"
        ),
    }

    output_file = root_dir / "agreement_report.json"
    output_file.write_text(
        json.dumps(report_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return output_file, report_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute inter-annotator agreement for Gate B1"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_REPO_ROOT / "benchmarks" / "ground_truth" / "v2",
        help="Root ground_truth/v2 directory",
    )
    args = parser.parse_args()

    out_file, report = compute_agreement(args.root)
    print("INTER-ANNOTATOR AGREEMENT REPORT COMPUTED")
    print(f"Report Output: {out_file}")
    print(f"Status: {report['status']}")
    print(f"Items Requiring Adjudication: {report['items_requiring_adjudication']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
