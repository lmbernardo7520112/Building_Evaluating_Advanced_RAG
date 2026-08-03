"""Offline Artifact Auditor for Slice4 v4 Ground Truth v2 Benchmark Results.

Validates positive smoke JSON and abstention smoke JSON against contract requirements
without network, APIs, or credential access.
Exits 0 if valid, non-zero if any contract invariant is violated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def validate_positive_json(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    # Protocol and Schema Versions
    if data.get("protocol_version") != "raglab_v7_slice4_v3":
        pv = data.get("protocol_version")
        errors.append(f"Invalid protocol_version: {pv} != raglab_v7_slice4_v3")
    if data.get("artifact_schema_version") != "slice4_v4":
        asv = data.get("artifact_schema_version")
        errors.append(f"Invalid artifact_schema_version: {asv} != slice4_v4")

    # Find the result entry
    results = data.get("results", {})
    entry = None
    for strat_entries in results.values():
        if isinstance(strat_entries, list) and len(strat_entries) > 0:
            entry = strat_entries[0]
            break

    if not entry:
        errors.append("No result entries found in positive JSON")
        return errors

    # Check Sealed Holdout
    if "holdout" in str(entry.get("qid", "")).lower():
        errors.append(f"SEALED HOLDOUT VIOLATION: qid={entry.get('qid')}")

    # Ground Truth Subtree
    gt = entry.get("ground_truth", {})
    if gt.get("contract_version") != "v2":
        errors.append(f"Invalid contract_version: {gt.get('contract_version')}")
    if gt.get("source_schema") != "legacy_active_questions":
        errors.append(f"Invalid source_schema: {gt.get('source_schema')}")
    if gt.get("provenance_status") != "LEGACY_METADATA_UNAVAILABLE":
        errors.append(f"Invalid provenance_status: {gt.get('provenance_status')}")
    if gt.get("legacy_relevant_pages") != [92]:
        pages = gt.get("legacy_relevant_pages")
        errors.append(f"Invalid legacy_relevant_pages: {pages} != [92]")
    if gt.get("passage_qrels_status") != "NOT_ANNOTATED":
        errors.append(f"Invalid passage_qrels_status: {gt.get('passage_qrels_status')}")
    if gt.get("graded_qrels_status") != "NOT_ANNOTATED":
        errors.append(f"Invalid graded_qrels_status: {gt.get('graded_qrels_status')}")
    if gt.get("gold_answer_status") != "NOT_ANNOTATED":
        errors.append(f"Invalid gold_answer_status: {gt.get('gold_answer_status')}")

    # Evaluation Subtree
    ev = entry.get("evaluation", {})
    if ev.get("protocol_version") != "raglab_v7_slice4_v3":
        errors.append(f"Invalid eval protocol_version: {ev.get('protocol_version')}")
    if ev.get("artifact_schema_version") != "slice4_v4":
        asv = ev.get("artifact_schema_version")
        errors.append(f"Invalid eval schema_version: {asv}")

    legacy_page_metrics = ev.get("legacy_page_metrics", {})
    if not legacy_page_metrics:
        errors.append("Missing evaluation.legacy_page_metrics")
    else:
        prec = legacy_page_metrics.get("citation_page_precision")
        rec = legacy_page_metrics.get("citation_page_recall")
        if prec is not None and abs(prec - (1.0 / 3.0)) > 1e-3:
            errors.append(f"Unexpected citation_page_precision: {prec} != 1/3")
        if rec is not None and abs(rec - 1.0) > 1e-3:
            errors.append(f"Unexpected citation_page_recall: {rec} != 1.0")

    det_v2 = ev.get("deterministic_v2_metrics", {})
    if not det_v2:
        errors.append("Missing evaluation.deterministic_v2_metrics")
    else:
        for metric_name, val in det_v2.items():
            if not str(val).startswith("NOT_COMPUTABLE_"):
                errors.append(
                    f"Deterministic v2 metric {metric_name} invalid: {val}"
                )

    return errors


def validate_abstention_json(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    # Protocol and Schema Versions
    if data.get("protocol_version") != "raglab_v7_slice4_v3":
        pv = data.get("protocol_version")
        errors.append(f"Invalid protocol_version: {pv} != raglab_v7_slice4_v3")
    if data.get("artifact_schema_version") != "slice4_v4":
        asv = data.get("artifact_schema_version")
        errors.append(f"Invalid artifact_schema_version: {asv} != slice4_v4")

    # Find the result entry
    results = data.get("results", {})
    entry = None
    for strat_entries in results.values():
        if isinstance(strat_entries, list) and len(strat_entries) > 0:
            entry = strat_entries[0]
            break

    if not entry:
        errors.append("No result entries found in abstention JSON")
        return errors

    # Check Sealed Holdout
    if "holdout" in str(entry.get("qid", "")).lower():
        errors.append(f"SEALED HOLDOUT VIOLATION: qid={entry.get('qid')}")

    # Abstention fields
    if entry.get("abstained") is not True:
        errors.append(f"Abstention entry abstained not True: {entry.get('abstained')}")

    gt = entry.get("ground_truth", {})
    if gt.get("answerable") is not False:
        errors.append(f"Abstention answerable is not False: {gt.get('answerable')}")

    c_pages = entry.get("citation_pages")
    if c_pages and len(c_pages) > 0:
        errors.append(f"Abstention entry contains unexpected citations: {c_pages}")

    # Evaluation Subtree
    ev = entry.get("evaluation", {})
    det_v2 = ev.get("deterministic_v2_metrics", {})
    if not det_v2:
        errors.append("Missing deterministic_v2_metrics in abstention result")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline Slice4 v4 Artifact Auditor")
    parser.add_argument(
        "--positive-json", type=Path, required=True, help="Positive smoke JSON"
    )
    parser.add_argument(
        "--abstention-json", type=Path, required=True, help="Abstention smoke JSON"
    )
    args = parser.parse_args()

    pos_path = args.positive_json
    abs_path = args.abstention_json

    if not pos_path.exists():
        print(f"ERROR: Positive smoke JSON not found: {pos_path}", file=sys.stderr)
        return 1

    if not abs_path.exists():
        print(f"ERROR: Abstention smoke JSON not found: {abs_path}", file=sys.stderr)
        return 1

    try:
        pos_data = json.loads(pos_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR: Positive smoke JSON invalid JSON format: {e}", file=sys.stderr)
        return 1

    try:
        abs_data = json.loads(abs_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR: Abstention smoke JSON invalid JSON format: {e}", file=sys.stderr)
        return 1

    pos_errors = validate_positive_json(pos_data)
    abs_errors = validate_abstention_json(abs_data)

    all_errors = [f"[POSITIVE] {e}" for e in pos_errors] + [
        f"[ABSTENTION] {e}" for e in abs_errors
    ]

    if all_errors:
        print("ARTIFACT AUDIT FAILED:", file=sys.stderr)
        for err in all_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("ARTIFACT AUDIT PASSED: All v4 contract invariants verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
