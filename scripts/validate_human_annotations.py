"""Offline Human Annotations Validator and Sanitized Exporter (Gate B).

CLI:
  python scripts/validate_human_annotations.py \\
    --annotator-id annotator_a|annotator_b \\
    --queue-file PATH \\
    --questions-file PATH \\
    --work-file PATH \\
    --export-file PATH

Validates 100% coverage, atomic integrity, grade/role consistency, literal spans,
blinding, and exports sanitized human qrels.
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

VALID_GRADES: Final[frozenset[int]] = frozenset({0, 1, 2, 3})
VALID_ROLES: Final[frozenset[str]] = frozenset(
    {"NEGATIVE_CONTROL", "CONTEXTUAL", "SUPPORTING", "PRIMARY"}
)

DEFAULT_ROLE_FOR_GRADE: Final[dict[int, str]] = {
    0: "NEGATIVE_CONTROL",
    1: "CONTEXTUAL",
    2: "SUPPORTING",
    3: "PRIMARY",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_jsonl(target_path: Path, records: list[dict[str, Any]]) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=target_path.parent, prefix=f".tmp_{target_path.name}_"
    )
    tmp_path = Path(tmp_path_str)

    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, target_path)

        try:
            parent_fd = os.open(str(target_path.parent), os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except OSError:
            pass
    except Exception:
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()
        raise


def validate_and_export_human_annotations(
    annotator_id: str,
    queue_file: Path,
    questions_file: Path,
    work_file: Path,
    export_file: Path,
) -> Path:
    """Validate human work annotations fail-closed and export sanitized qrels."""

    if annotator_id not in ("annotator_a", "annotator_b"):
        msg = f"Invalid annotator_id '{annotator_id}'."
        raise ValueError(msg)

    if not queue_file.exists():
        raise FileNotFoundError(f"Queue file not found: {queue_file}")
    if not questions_file.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")
    if not work_file.exists():
        raise FileNotFoundError(f"Work file not found: {work_file}")

    queue_sha = sha256_file(queue_file)
    questions_sha = sha256_file(questions_file)

    # 1. Load queue items
    queue_lines = [
        json.loads(line)
        for line in queue_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    queue_pairs = {(it["question_id"], it["passage_id"]): it for it in queue_lines}

    # Check for holdout in queue
    for qid, _ps_id in queue_pairs:
        if qid in HOLDOUT_QIDS or "holdout" in qid.lower():
            raise ValueError(f"HOLDOUT VIOLATION: item {qid} in queue file")

    # 2. Load work file records
    work_bytes = work_file.read_bytes()
    if not work_bytes.strip():
        raise ValueError(f"Work file is empty: {work_file}")

    work_records: list[dict[str, Any]] = []
    for line_num, line in enumerate(work_bytes.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            work_records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in work file at line {line_num}: {exc}"
            ) from exc

    # 3. Fail-Closed Inspections
    seen_work_pairs: set[tuple[str, str]] = set()
    sanitized_export_records: list[dict[str, Any]] = []

    for r_idx, r in enumerate(work_records, 1):
        # Identity check
        rec_ann = r.get("annotator_id")
        if rec_ann != annotator_id:
            raise ValueError(
                f"Record {r_idx} annotator_id mismatch: '{rec_ann}' != '{annotator_id}'"
            )

        # Hashes check
        if r.get("queue_file_sha256") != queue_sha:
            raise ValueError(f"Record {r_idx} queue_file_sha256 mismatch")
        if r.get("questions_file_sha256") != questions_sha:
            raise ValueError(f"Record {r_idx} questions_file_sha256 mismatch")

        # Status check
        if r.get("status") != "COMPLETED":
            raise ValueError(
                f"Record {r_idx} ({r.get('question_id')}) is not COMPLETED"
            )

        qid = str(r.get("question_id", ""))
        ps_id = str(r.get("passage_id", ""))

        if qid in HOLDOUT_QIDS or "holdout" in qid.lower():
            raise ValueError(
                f"HOLDOUT VIOLATION: record {r_idx} contains holdout '{qid}'"
            )

        pair = (qid, ps_id)
        if pair in seen_work_pairs:
            raise ValueError(f"Duplicate record for pair {pair} in work file")
        seen_work_pairs.add(pair)

        if pair not in queue_pairs:
            raise ValueError(
                f"Unexpected pair {pair} in work file (not present in input queue file)"
            )

        # Blinding checks
        for forbidden in BLINDING_FORBIDDEN_FIELDS:
            if forbidden in r:
                msg = f"BLINDING VIOLATION: '{forbidden}' in record {pair}"
                raise ValueError(msg)

        # Grade and Role validation
        grade = r.get("relevance_grade")
        if not isinstance(grade, int) or grade not in VALID_GRADES:
            raise ValueError(f"Invalid relevance_grade '{grade}' in record {pair}")

        role = str(r.get("evidence_role") or "").strip().upper()
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid evidence_role '{role}' in record {pair}")

        exp_role = DEFAULT_ROLE_FOR_GRADE[grade]
        notes = str(r.get("annotation_notes") or "").strip()
        if role != exp_role and not notes:
            raise ValueError(
                f"Divergent evidence_role '{role}' for grade {grade} in record {pair} "
                f"requires non-empty annotation_notes"
            )

        # Literal span validation
        span = str(r.get("supporting_span_human") or "").strip()
        queue_item = queue_pairs[pair]
        passage_text = queue_item["text"]

        if span and span not in passage_text:
            raise ValueError(
                f"Literal span violation in record {pair}: supporting_span_human "
                f"is not a literal substring of passage_text"
            )

        # Construct sanitized export record
        export_rec = {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "annotator_id": annotator_id,
            "question_id": qid,
            "passage_id": ps_id,
            "page_number": queue_item.get("page_number", 0),
            "relevance_grade": grade,
            "evidence_role": role,
            "supporting_span_human": span,
            "annotation_notes": notes,
            "annotated_at_utc": r.get(
                "annotated_at_utc", datetime.now(UTC).isoformat()
            ),
            "queue_file_sha256": queue_sha,
            "questions_file_sha256": questions_sha,
            "export_status": "VALIDATED_HUMAN_QRELS",
        }
        sanitized_export_records.append(export_rec)

    # Coverage check: work file must contain ALL queue pairs
    if set(queue_pairs.keys()) != seen_work_pairs:
        missing = set(queue_pairs.keys()) - seen_work_pairs
        raise ValueError(
            f"Incomplete annotation coverage: missing {len(missing)} items"
        )

    # Atomic write to export_file
    atomic_write_jsonl(export_file, sanitized_export_records)

    return export_file


# ─────────────────────────────────────────────────────────────────
# Legacy Gate B1 Backward Compatibility Helpers
# ─────────────────────────────────────────────────────────────────


def load_registry_ids(root_dir: Path) -> set[str]:
    reg_file = root_dir / "passage_registry.jsonl"
    passage_ids: set[str] = set()
    if reg_file.exists():
        for line in reg_file.read_text(encoding="utf-8").splitlines():
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

        ps_id = str(cand.get("passage_id") or "")
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
        if record.get("gold_answer") is not None:
            errors.append(f"[{qid}] Unanswerable question cannot have gold_answer")
        if has_positive_evidence and mode == "completed":
            errors.append(
                f"[{qid}] Unanswerable question cannot have evidence grade >= 1"
            )

    if mode == "completed" and answerability is True and not has_positive_evidence:
        errors.append(
            f"[{qid}] Completed ANSWERABLE question must have at least one relevant passage (grade >= 1)"  # noqa: E501
        )

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
    parser = argparse.ArgumentParser(
        description="Validate and Export Human Annotations (Gate B)"
    )
    parser.add_argument(
        "--annotator-id",
        required=True,
        choices=["annotator_a", "annotator_b"],
        help="Annotator identity",
    )
    parser.add_argument(
        "--queue-file",
        type=Path,
        required=True,
        help="Path to input blinded queue file",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        required=True,
        help="Path to input questions file",
    )
    parser.add_argument(
        "--work-file",
        type=Path,
        required=True,
        help="Path to work annotations file",
    )
    parser.add_argument(
        "--export-file",
        type=Path,
        required=True,
        help="Path to output sanitized export file",
    )
    args = parser.parse_args()

    try:
        exported = validate_and_export_human_annotations(
            annotator_id=args.annotator_id,
            queue_file=args.queue_file,
            questions_file=args.questions_file,
            work_file=args.work_file,
            export_file=args.export_file,
        )
        print("HUMAN ANNOTATIONS VALIDATED AND EXPORTED SUCCESSFULLY")
        count = sum(1 for _ in exported.read_text("utf-8").splitlines() if _.strip())
        print(f"Annotator: {args.annotator_id}")
        print(f"Export File: {exported} ({count} items)")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
