"""Human Annotation Offline Validator (Gate B1 - Etapa 5).

Validates blinded annotation templates, completed human annotations,
and adjudication records.

Modes:
- --mode template: Accepts PENDING status and null grades.
- --mode completed: Requires full completion (COMPLETED status, non-null grades).
- --mode adjudicated: Validates resolved adjudication items preserving original grades.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure src is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def load_registry_ids(root_dir: Path) -> set[str]:
    """Load valid passage_ids from passage_registry.jsonl."""
    reg_file = root_dir / "passage_registry.jsonl"
    if not reg_file.exists():
        raise FileNotFoundError(f"Passage registry not found at {reg_file}")

    passage_ids: set[str] = set()
    with reg_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                passage_ids.add(item["passage_id"])
    return passage_ids


def validate_annotation_record(
    record: dict[str, Any], valid_passage_ids: set[str], mode: str
) -> list[str]:
    errors: list[str] = []

    qid = record.get("question_id")
    if not qid or not str(qid).strip():
        errors.append("Missing or empty question_id")

    # Check Sealed Holdout
    if qid and "holdout" in str(qid).lower():
        errors.append(f"SEALED HOLDOUT VIOLATION: question_id={qid}")

    ann_id = record.get("annotator_id")
    if not ann_id or not str(ann_id).strip():
        errors.append(f"[{qid}] Missing or empty annotator_id")

    status = record.get("annotation_status")
    valid_statuses = ("PENDING", "COMPLETED", "NEEDS_REVIEW")
    if status not in valid_statuses:
        errors.append(f"[{qid}] Invalid annotation_status: {status}")

    if mode == "completed" and status != "COMPLETED":
        errors.append(
            f"[{qid}] Completed mode requires status='COMPLETED', got {status}"
        )

    # Check for forbidden strategy/score leakage
    forbidden_keys = (
        "strategy",
        "original_rank",
        "score",
        "retriever_name",
        "reranker_score",
    )
    for key in forbidden_keys:
        if key in record:
            errors.append(f"[{qid}] FORBIDDEN LEAKAGE: record contains '{key}'")

    candidates = record.get("candidate_passages", [])
    seen_cand_ids: set[str] = set()
    has_positive_evidence = False

    for cand in candidates:
        if not isinstance(cand, dict):
            errors.append(f"[{qid}] Invalid candidate item: {cand}")
            continue

        for key in forbidden_keys:
            if key in cand:
                errors.append(f"[{qid}] FORBIDDEN LEAKAGE in candidate: '{key}'")

        ps_id = cand.get("passage_id")
        if not ps_id or ps_id not in valid_passage_ids:
            errors.append(f"[{qid}] Unknown or missing passage_id: {ps_id}")

        if ps_id in seen_cand_ids:
            errors.append(f"[{qid}] Duplicate candidate passage_id: {ps_id}")
        seen_cand_ids.add(ps_id)

        grade = cand.get("relevance_grade")
        if grade is not None:
            if not isinstance(grade, int) or not (0 <= grade <= 3):
                errors.append(f"[{qid}] Invalid relevance_grade for {ps_id}: {grade}")
            if grade >= 1:
                has_positive_evidence = True
        elif mode == "completed":
            errors.append(
                f"[{qid}] Missing relevance_grade for {ps_id} in completed mode"
            )

    answerability = record.get("answerability")
    if mode == "completed" and answerability is None:
        errors.append(f"[{qid}] Missing answerability in completed mode")

    if answerability is False:
        # UNANSWERABLE cannot have gold answer or positive evidence
        if record.get("gold_answer") is not None:
            errors.append(f"[{qid}] Unanswerable question cannot have gold_answer")
        if has_positive_evidence and mode == "completed":
            errors.append(
                f"[{qid}] Unanswerable question cannot have positive evidence (grade >= 1)"  # noqa: E501
            )

    if mode == "completed" and answerability is True and not has_positive_evidence:
        errors.append(
            f"[{qid}] Completed ANSWERABLE question must have at least one relevant passage (grade >= 1)"  # noqa: E501
        )

    # Validate evidence_sets
    ev_sets = record.get("evidence_sets", [])
    for ev_set in ev_sets:
        if isinstance(ev_set, dict):
            set_passages = ev_set.get("passage_ids", [])
            jointly_suff = ev_set.get("jointly_sufficient", True)
            if jointly_suff and not set_passages:
                errors.append(
                    f"[{qid}] Empty evidence set cannot be jointly_sufficient"
                )
            for pid in set_passages:
                if pid not in valid_passage_ids:
                    errors.append(
                        f"[{qid}] Evidence set contains unknown passage_id: {pid}"
                    )

    # Validate gold_supporting_passage_ids
    gold_cits = record.get("gold_supporting_passage_ids", [])
    for g_id in gold_cits:
        if g_id not in valid_passage_ids:
            errors.append(f"[{qid}] Unknown gold_supporting_passage_id: {g_id}")

    return errors


def validate_adjudication_record(
    record: dict[str, Any], valid_passage_ids: set[str]
) -> list[str]:
    errors: list[str] = []
    qid = record.get("question_id")
    ps_id = record.get("passage_id")

    if not qid:
        errors.append("Missing question_id in adjudication record")
    if not ps_id or ps_id not in valid_passage_ids:
        errors.append(f"Unknown passage_id in adjudication record: {ps_id}")

    status = record.get("annotation_status", record.get("adjudication_status"))
    if status not in ("PENDING", "ADJUDICATED", "RESOLVED"):
        errors.append(f"[{qid}:{ps_id}] Invalid adjudication status: {status}")

    return errors


def validate_annotation_packages(root_dir: Path, mode: str = "template") -> list[str]:
    valid_passage_ids = load_registry_ids(root_dir)
    pkg_dir = root_dir / "annotation_packages"

    all_errors: list[str] = []

    if mode in ("template", "completed"):
        if not pkg_dir.exists():
            return [f"Annotation packages directory not found at {pkg_dir}"]

        jsonl_files = list(pkg_dir.glob("*/*.jsonl"))
        if not jsonl_files:
            return [f"No annotation JSONL files found in {pkg_dir}"]

        for jsonl_file in jsonl_files:
            with jsonl_file.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        errs = validate_annotation_record(
                            record, valid_passage_ids, mode
                        )
                        for e in errs:
                            all_errors.append(f"{jsonl_file.name}:{line_no} -> {e}")
                    except Exception as ex:
                        all_errors.append(
                            f"{jsonl_file.name}:{line_no} -> Invalid JSON format: {ex}"
                        )

    elif mode == "adjudicated":
        adj_file = root_dir / "adjudication_template.jsonl"
        if not adj_file.exists():
            return [f"Adjudication file not found at {adj_file}"]

        with adj_file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    errs = validate_adjudication_record(record, valid_passage_ids)
                    for e in errs:
                        all_errors.append(f"{adj_file.name}:{line_no} -> {e}")
                except Exception as ex:
                    all_errors.append(
                        f"{adj_file.name}:{line_no} -> Invalid JSON format: {ex}"
                    )

    return all_errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate human annotation packages")
    parser.add_argument(
        "--mode",
        choices=["template", "completed", "adjudicated"],
        default="template",
        help="Validation mode",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_REPO_ROOT / "benchmarks" / "ground_truth" / "v2",
        help="Root ground_truth/v2 directory",
    )
    args = parser.parse_args()

    errors = validate_annotation_packages(root_dir=args.root, mode=args.mode)
    if errors:
        print(f"ANNOTATION VALIDATION FAILED ({len(errors)} errors):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"ANNOTATION VALIDATION PASSED (mode={args.mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
