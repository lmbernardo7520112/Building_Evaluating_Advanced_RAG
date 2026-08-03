"""Offline Artifact Auditor for Slice4 v4 Ground Truth v2 Benchmark Results.

Validates positive smoke JSON and abstention smoke JSON against contract requirements
data-driven without network, APIs, or credential access.
Exits 0 if valid, non-zero if any contract invariant is violated.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from raglab.evaluation.metrics.deterministic_v2 import (
    compute_legacy_page_metrics,
)

# Ensure src is on sys.path if invoked as standalone script
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


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
    qid = str(entry.get("qid", ""))
    if "holdout" in qid.lower():
        errors.append(f"SEALED HOLDOUT VIOLATION: qid={qid}")

    # Ground Truth Subtree
    gt = entry.get("ground_truth", {})
    if gt.get("contract_version") != "v2":
        errors.append(f"Invalid contract_version: {gt.get('contract_version')}")
    if gt.get("source_schema") != "legacy_active_questions":
        errors.append(f"Invalid source_schema: {gt.get('source_schema')}")
    if gt.get("provenance_status") != "LEGACY_METADATA_UNAVAILABLE":
        errors.append(f"Invalid provenance_status: {gt.get('provenance_status')}")
    if gt.get("passage_qrels_status") != "NOT_ANNOTATED":
        errors.append(f"Invalid passage_qrels_status: {gt.get('passage_qrels_status')}")
    if gt.get("graded_qrels_status") != "NOT_ANNOTATED":
        errors.append(f"Invalid graded_qrels_status: {gt.get('graded_qrels_status')}")
    if gt.get("gold_answer_status") != "NOT_ANNOTATED":
        errors.append(f"Invalid gold_answer_status: {gt.get('gold_answer_status')}")

    # Check synthetic passage_id manufactured from page numbers
    relevant_evidences = gt.get("relevant_evidences", [])
    if relevant_evidences:
        for ev_item in relevant_evidences:
            pid = str(
                ev_item.get("passage_id", "")
                if isinstance(ev_item, dict)
                else getattr(ev_item, "passage_id", "")
            )
            if pid.startswith("p") and pid[1:].isdigit():
                errors.append(f"SYNTHETIC PASSAGE_ID DETECTED: {pid}")

    # Extract pages for data-driven recalculation
    legacy_relevant_pages = gt.get("legacy_relevant_pages", [])
    if not isinstance(legacy_relevant_pages, (list, tuple)):
        errors.append("legacy_relevant_pages is not a list/tuple")
        legacy_relevant_pages = []

    citation_pages = entry.get("citation_pages")
    if citation_pages is None:
        cit_map = entry.get("citation_map", [])
        citation_pages = [
            c["page_number"]
            for c in cit_map
            if isinstance(c, dict) and "page_number" in c
        ]

    ret_ev = entry.get("retrieval_evidence", {})
    cands = ret_ev.get("candidates", [])
    retrieved_pages = [
        c.get("page_number")
        for c in cands
        if isinstance(c, dict) and c.get("page_number") is not None
    ]

    # Recalculate metrics dynamically
    recalculated = compute_legacy_page_metrics(
        retrieved_pages=retrieved_pages,
        relevant_pages=legacy_relevant_pages,
        cited_pages=citation_pages,
    )

    ev = entry.get("evaluation", {})
    if ev.get("protocol_version") != "raglab_v7_slice4_v3":
        errors.append(f"Invalid eval protocol_version: {ev.get('protocol_version')}")
    if ev.get("artifact_schema_version") != "slice4_v4":
        asv = ev.get("artifact_schema_version")
        errors.append(f"Invalid eval schema_version: {asv}")

    serialized_page_metrics = ev.get("legacy_page_metrics", {})
    if not serialized_page_metrics:
        errors.append("Missing evaluation.legacy_page_metrics")
    else:
        for key in (
            "page_hit_at_k",
            "page_mrr",
            "citation_page_precision",
            "citation_page_recall",
        ):
            ser_val = serialized_page_metrics.get(key)
            recalc_val = recalculated.get(key)
            if ser_val is None:
                errors.append(f"Missing serialized metric {key}")
            elif (
                recalc_val is not None
                and not math.isclose(float(ser_val), float(recalc_val), abs_tol=1e-3)
            ):
                errors.append(
                    f"Mismatch in {key}: {ser_val} != {recalc_val}"
                )

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
    qid = str(entry.get("qid", ""))
    if "holdout" in qid.lower():
        errors.append(f"SEALED HOLDOUT VIOLATION: qid={qid}")

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
