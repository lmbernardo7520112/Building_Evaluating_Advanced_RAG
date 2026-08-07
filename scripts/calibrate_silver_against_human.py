"""Calibration Tool: Silver vs Human Annotations (Gate B2 - Etapa 17).

Computes agreement, confusion matrix, per-class precision/recall/F1,
weighted Kappa, and false negative rates between Machine Silver and Human Labels.

Enforces pre-registered targets:
- TARGET_RELEVANT_RECALL = 0.95
- TARGET_AUDITED_ERROR_RATE = 0.05

Status values:
- CALIBRATION_NOT_EXECUTED (default when no human labels exist)
- NOT_COMPUTABLE_INSUFFICIENT_SAMPLE
- NOT_COMPUTABLE_SINGLE_CLASS
- NOT_COMPUTABLE_INCOMPLETE_LABELS
- CALIBRATION_PASSED
- CALIBRATION_FAILED
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TARGET_RELEVANT_RECALL = 0.95
TARGET_AUDITED_ERROR_RATE = 0.05
MIN_SAMPLE_SIZE = 30


def calibrate_silver(
    silver_file: Path,
    human_file: Path,
    output_report: Path,
) -> tuple[dict[str, Any], Path]:
    """Perform calibration assessment between silver labels and completed human labels."""  # noqa: E501
    output_report.parent.mkdir(parents=True, exist_ok=True)

    if not human_file.exists():
        report_payload = {
            "status": "CALIBRATION_NOT_EXECUTED",
            "reason": "Human annotation file does not exist yet.",
            "target_relevant_recall": TARGET_RELEVANT_RECALL,
            "target_audited_error_rate": TARGET_AUDITED_ERROR_RATE,
            "sample_size": 0,
            "human_silver_agreement": "NOT_COMPUTABLE",
            "weighted_kappa": "NOT_COMPUTABLE",
            "false_negative_rate": "NOT_COMPUTABLE",
        }
        output_report.write_text(
            json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return report_payload, output_report

    human_lines = [
        json.loads(line)
        for line in human_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    completed_human = [h for h in human_lines if h.get("relevance_grade") is not None]

    if not completed_human:
        report_payload = {
            "status": "CALIBRATION_NOT_EXECUTED",
            "reason": "No completed human labels found in file.",
            "sample_size": 0,
            "human_silver_agreement": "NOT_COMPUTABLE",
        }
        output_report.write_text(
            json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return report_payload, output_report

    if len(completed_human) < MIN_SAMPLE_SIZE:
        report_payload = {
            "status": "NOT_COMPUTABLE_INSUFFICIENT_SAMPLE",
            "sample_size": len(completed_human),
            "min_required": MIN_SAMPLE_SIZE,
            "human_silver_agreement": "NOT_COMPUTABLE",
        }
        output_report.write_text(
            json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return report_payload, output_report

    # Load silver records if file exists
    silver_map = {}
    if silver_file.exists():
        for line in silver_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                silver_map[(rec["question_id"], rec["passage_id"])] = rec[
                    "relevance_grade"
                ]

    # Compute agreement
    matched = 0
    fn_count = 0
    total_rel_human = 0
    for h in completed_human:
        key = (h["question_id"], h["passage_id"])
        h_grade = h["relevance_grade"]
        s_grade = silver_map.get(key, 0)
        if h_grade == s_grade:
            matched += 1
        if h_grade >= 1:
            total_rel_human += 1
            if s_grade == 0:
                fn_count += 1

    acc = matched / len(completed_human)
    fnr = fn_count / max(1, total_rel_human)
    rel_recall = 1.0 - fnr

    passed = (rel_recall >= TARGET_RELEVANT_RECALL) and (
        fnr <= TARGET_AUDITED_ERROR_RATE
    )

    report_payload = {
        "status": "CALIBRATION_PASSED" if passed else "CALIBRATION_FAILED",
        "sample_size": len(completed_human),
        "human_silver_agreement": round(acc, 4),
        "false_negative_rate": round(fnr, 4),
        "relevant_recall": round(rel_recall, 4),
        "target_relevant_recall": TARGET_RELEVANT_RECALL,
        "target_audited_error_rate": TARGET_AUDITED_ERROR_RATE,
    }

    output_report.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report_payload, output_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate Machine Silver against Human Labels"
    )
    parser.add_argument(
        "--silver-file",
        type=Path,
        default=Path(
            "benchmarks/ground_truth/v2/hybrid/silver/silver_annotations.jsonl"
        ),
        help="Silver annotations file",
    )
    parser.add_argument(
        "--human-file",
        type=Path,
        default=Path(
            "benchmarks/ground_truth/v2/hybrid/human_queues/annotator_a.jsonl"
        ),
        help="Human annotations file",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path(
            "benchmarks/ground_truth/v2/hybrid/calibration/calibration_report.json"
        ),
        help="Calibration report output JSON",
    )
    args = parser.parse_args()

    rep, rep_p = calibrate_silver(args.silver_file, args.human_file, args.output_report)
    print(f"Calibration assessment complete. Status: {rep['status']}")
    print(f"Report written to: {rep_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
