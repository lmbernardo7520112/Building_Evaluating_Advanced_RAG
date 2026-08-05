# ruff: noqa: E501, B905
"""Compute Human Annotation Inter-Annotator Agreement Report (Gate B).

CLI:
  python scripts/compute_human_agreement.py \\
    --annotator-a PATH \\
    --annotator-b PATH \\
    --output-report PATH \\
    --output-disagreements PATH

Calculates exact agreement, unweighted Cohen's Kappa, quadratic weighted Cohen's Kappa,
binary relevant agreement, 4x4 and 2x2 confusion matrices, grade distance distributions,
and outputs an auditable JSON report.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

PROTOCOL_VERSION: Final[str] = "raglab_v7_slice4_v3"
SCHEMA_VERSION: Final[str] = "3.0.0"
HOLDOUT_QIDS: Final[frozenset[str]] = frozenset({"q_holdout_01", "q_holdout_02"})

BLINDING_FORBIDDEN_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "confidence",
        "reasoning",
        "supporting_span",
        "judge_model",
        "judge_provider",
        "judge_id",
        "label_source",
        "strategy",
        "retrieval_rank",
        "retrieval_score",
        "pre_rerank_rank",
        "post_rerank_rank",
        "selected_by_reranker",
        "dropped_by_reranker",
        "relevant_pages",
        "gold_answer",
    }
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_json(target_path: Path, data: Any) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=target_path.parent, prefix=f".tmp_{target_path.name}_"
    )
    tmp_path = Path(tmp_path_str)

    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, target_path)

        with contextlib.suppress(OSError):
            parent_fd = os.open(str(target_path.parent), os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
    except Exception:
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()
        raise


def load_and_validate_export(
    file_path: Path, expected_annotator_id: str
) -> dict[tuple[str, str], dict[str, Any]]:
    if not file_path.exists():
        raise FileNotFoundError(f"Export file not found: {file_path}")

    lines = [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError(f"Export file is empty: {file_path}")

    records: dict[tuple[str, str], dict[str, Any]] = {}

    for line_num, line in enumerate(lines, 1):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on line {line_num} of {file_path}: {exc}"
            ) from exc

        ann_id = rec.get("annotator_id")
        if ann_id != expected_annotator_id:
            raise ValueError(
                f"Identity mismatch in {file_path} line {line_num}: expected '{expected_annotator_id}', got '{ann_id}'"
            )

        qid = str(rec.get("question_id", "")).strip()
        ps_id = str(rec.get("passage_id", "")).strip()

        if not qid or not ps_id:
            raise ValueError(
                f"Missing question_id or passage_id on line {line_num} of {file_path}"
            )

        if qid in HOLDOUT_QIDS or "holdout" in qid.lower():
            raise ValueError(f"HOLDOUT VIOLATION: item '{qid}' found in {file_path}")

        pair = (qid, ps_id)
        if pair in records:
            raise ValueError(f"Duplicate pair {pair} in {file_path}")

        # Check for forbidden silver fields
        for forbidden in BLINDING_FORBIDDEN_FIELDS:
            if forbidden in rec:
                raise ValueError(
                    f"BLINDING VIOLATION: forbidden field '{forbidden}' in {file_path}"
                )

        grade = rec.get("relevance_grade")
        if not isinstance(grade, int) or grade not in (0, 1, 2, 3):
            raise ValueError(
                f"Invalid relevance_grade '{grade}' in {file_path} for pair {pair}"
            )

        records[pair] = rec

    return records


def compute_cohens_kappa_unweighted(
    r1: list[int], r2: list[int], num_classes: int = 4
) -> float:
    n = len(r1)
    if n == 0:
        return 0.0

    cm = [[0] * num_classes for _ in range(num_classes)]
    for a, b in zip(r1, r2):
        cm[a][b] += 1

    po = sum(cm[i][i] for i in range(num_classes)) / n

    row_sums = [sum(cm[i][j] for j in range(num_classes)) for i in range(num_classes)]
    col_sums = [sum(cm[j][i] for j in range(num_classes)) for i in range(num_classes)]

    pe = sum(row_sums[i] * col_sums[i] for i in range(num_classes)) / (n * n)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def compute_cohens_kappa_quadratic(
    r1: list[int], r2: list[int], num_classes: int = 4
) -> float:
    n = len(r1)
    if n == 0:
        return 0.0

    cm = [[0] * num_classes for _ in range(num_classes)]
    for a, b in zip(r1, r2):
        cm[a][b] += 1

    weights = [
        [((i - j) ** 2) / ((num_classes - 1) ** 2) for j in range(num_classes)]
        for i in range(num_classes)
    ]

    row_sums = [sum(cm[i][j] for j in range(num_classes)) for i in range(num_classes)]
    col_sums = [sum(cm[j][i] for j in range(num_classes)) for i in range(num_classes)]

    expected = [
        [row_sums[i] * col_sums[j] / n for j in range(num_classes)]
        for i in range(num_classes)
    ]

    num = sum(
        weights[i][j] * cm[i][j] for i in range(num_classes) for j in range(num_classes)
    )
    den = sum(
        weights[i][j] * expected[i][j]
        for i in range(num_classes)
        for j in range(num_classes)
    )

    if den == 0:
        return 1.0
    return 1.0 - (num / den)


def compute_agreement(
    annotator_a_path: Path,
    annotator_b_path: Path,
    output_report_path: Path,
    output_disagreements_path: Path,
) -> tuple[dict[str, Any], Path]:
    """Compute complete inter-annotator agreement metrics and report."""

    map_a = load_and_validate_export(
        annotator_a_path, expected_annotator_id="annotator_a"
    )
    map_b = load_and_validate_export(
        annotator_b_path, expected_annotator_id="annotator_b"
    )

    hash_a = sha256_file(annotator_a_path)
    hash_b = sha256_file(annotator_b_path)

    pairs_a = set(map_a.keys())
    pairs_b = set(map_b.keys())

    if pairs_a != pairs_b:
        missing_in_b = pairs_a - pairs_b
        missing_in_a = pairs_b - pairs_a
        raise ValueError(
            f"Universe mismatch between Annotator A and B exports: "
            f"in_A_not_B={len(missing_in_b)}, in_B_not_A={len(missing_in_a)}"
        )

    common_pairs = sorted(pairs_a)
    n = len(common_pairs)

    grades_a = [map_a[p]["relevance_grade"] for p in common_pairs]
    grades_b = [map_b[p]["relevance_grade"] for p in common_pairs]

    exact_agreements = sum(1 for a, b in zip(grades_a, grades_b) if a == b)
    disagreements_count = n - exact_agreements

    adjacent_count = sum(1 for a, b in zip(grades_a, grades_b) if abs(a - b) == 1)
    severe_count = sum(1 for a, b in zip(grades_a, grades_b) if abs(a - b) >= 2)

    # Binary relevance (grade >= 1 is relevant, grade 0 is non-relevant)
    binary_a = [1 if g >= 1 else 0 for g in grades_a]
    binary_b = [1 if g >= 1 else 0 for g in grades_b]
    binary_agreements = sum(1 for ba, bb in zip(binary_a, binary_b) if ba == bb)

    # 4x4 Confusion Matrix
    cm_4x4 = [[0] * 4 for _ in range(4)]
    for a, b in zip(grades_a, grades_b):
        cm_4x4[a][b] += 1

    # 2x2 Confusion Matrix
    cm_2x2 = [[0] * 2 for _ in range(2)]
    for ba, bb in zip(binary_a, binary_b):
        cm_2x2[ba][bb] += 1

    kappa_unweighted = compute_cohens_kappa_unweighted(grades_a, grades_b)
    kappa_quadratic = compute_cohens_kappa_quadratic(grades_a, grades_b)

    dist_abs = {
        "dist_0_exact": exact_agreements,
        "dist_1_adjacent": adjacent_count,
        "dist_2": sum(1 for a, b in zip(grades_a, grades_b) if abs(a - b) == 2),
        "dist_3": sum(1 for a, b in zip(grades_a, grades_b) if abs(a - b) == 3),
    }

    # Per-question disagreement counts
    per_question_disagreements: dict[str, int] = {}
    disagreements_list: list[dict[str, Any]] = []

    for qid, ps_id in common_pairs:
        ga = map_a[(qid, ps_id)]["relevance_grade"]
        gb = map_b[(qid, ps_id)]["relevance_grade"]
        ra = map_a[(qid, ps_id)]["evidence_role"]
        rb = map_b[(qid, ps_id)]["evidence_role"]

        if ga != gb:
            per_question_disagreements[qid] = per_question_disagreements.get(qid, 0) + 1
            disagreements_list.append(
                {
                    "question_id": qid,
                    "passage_id": ps_id,
                    "page_number": map_a[(qid, ps_id)].get("page_number", 0),
                    "annotator_a_grade": ga,
                    "annotator_b_grade": gb,
                    "annotator_a_role": ra,
                    "annotator_b_role": rb,
                    "grade_difference_abs": abs(ga - gb),
                    "disagreement_severity": "ADJACENT"
                    if abs(ga - gb) == 1
                    else "SEVERE",
                }
            )

    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "calculated_at_utc": datetime.now(UTC).isoformat(),
        "annotator_a_file_sha256": hash_a,
        "annotator_b_file_sha256": hash_b,
        "total_pairs": n,
        "exact_agreement_count": exact_agreements,
        "exact_agreement_rate": round(exact_agreements / n, 4) if n > 0 else 1.0,
        "disagreements": disagreements_count,
        "adjacent_disagreements": adjacent_count,
        "severe_disagreements": severe_count,
        "cohen_kappa_unweighted": round(kappa_unweighted, 4),
        "cohen_kappa_quadratic": round(kappa_quadratic, 4),
        "binary_relevant_agreement": round(binary_agreements / n, 4) if n > 0 else 1.0,
        "confusion_matrix_4x4": {
            "rows_annotator_a": ["grade_0", "grade_1", "grade_2", "grade_3"],
            "cols_annotator_b": ["grade_0", "grade_1", "grade_2", "grade_3"],
            "matrix": cm_4x4,
        },
        "confusion_matrix_2x2_binary": {
            "rows_annotator_a": ["non_relevant_0", "relevant_1_2_3"],
            "cols_annotator_b": ["non_relevant_0", "relevant_1_2_3"],
            "matrix": cm_2x2,
        },
        "grade_distribution": {
            "annotator_a": {g: grades_a.count(g) for g in (0, 1, 2, 3)},
            "annotator_b": {g: grades_b.count(g) for g in (0, 1, 2, 3)},
        },
        "absolute_grade_distance_distribution": dist_abs,
        "disagreements_per_question": per_question_disagreements,
    }

    atomic_write_json(output_report_path, report)
    atomic_write_json(output_disagreements_path, disagreements_list)

    return report, output_report_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute Inter-Annotator Agreement Report (Gate B)"
    )
    parser.add_argument(
        "--annotator-a",
        type=Path,
        required=True,
        help="Path to Annotator A final export",
    )
    parser.add_argument(
        "--annotator-b",
        type=Path,
        required=True,
        help="Path to Annotator B combined final export",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        required=True,
        help="Path to output agreement report JSON",
    )
    parser.add_argument(
        "--output-disagreements",
        type=Path,
        required=True,
        help="Path to output disagreements JSON",
    )
    args = parser.parse_args()

    try:
        report, rep_path = compute_agreement(
            annotator_a_path=args.annotator_a,
            annotator_b_path=args.annotator_b,
            output_report_path=args.output_report,
            output_disagreements_path=args.output_disagreements,
        )
        print("INTER-ANNOTATOR AGREEMENT REPORT COMPUTED SUCCESSFULLY")
        print(f"Report:                 {rep_path}")
        print(f"Total Pairs:            {report['total_pairs']}")
        print(
            f"Exact Agreement Rate:   {report['exact_agreement_rate']} ({report['exact_agreement_count']}/{report['total_pairs']})"
        )
        print(f"Cohen's Kappa (Unw):    {report['cohen_kappa_unweighted']}")
        print(f"Cohen's Kappa (Quad):   {report['cohen_kappa_quadratic']}")
        print(f"Binary Agreement Rate:  {report['binary_relevant_agreement']}")
        print(
            f"Disagreements Total:    {report['disagreements']} (Adjacent: {report['adjacent_disagreements']}, Severe: {report['severe_disagreements']})"
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
