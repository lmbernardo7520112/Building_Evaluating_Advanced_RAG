# ruff: noqa: E501
"""Validate and merge Annotator B original export (53 items) and supplemental export (16 items) (Gate B).

CLI:
  python scripts/merge_human_annotations_b.py \\
    --original-export-b PATH \\
    --supplemental-export-b PATH \\
    --full-queue-a PATH \\
    --output-combined PATH \\
    --output-manifest PATH

Enforces zero overlap between exports, exact 69 item coverage matching Queue A,
holdout rejection, provenance tagging, and atomic persistence.
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_json(target_path: Path, data: Any, is_jsonl: bool = False) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=target_path.parent, prefix=f".tmp_{target_path.name}_"
    )
    tmp_path = Path(tmp_path_str)

    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            if is_jsonl:
                for rec in data:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
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
    export_path: Path, expected_batch_name: str
) -> list[dict[str, Any]]:
    if not export_path.exists():
        raise FileNotFoundError(f"Export file not found: {export_path}")

    lines = export_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Export file is empty: {export_path}")

    records: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for line_num, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on line {line_num} of {export_path}: {exc}"
            ) from exc

        ann_id = rec.get("annotator_id")
        if ann_id != "annotator_b":
            raise ValueError(
                f"Identity mismatch in {export_path} at line {line_num}: expected 'annotator_b', got '{ann_id}'"
            )

        qid = str(rec.get("question_id", "")).strip()
        ps_id = str(rec.get("passage_id", "")).strip()

        if not qid or not ps_id:
            raise ValueError(
                f"Line {line_num} in {export_path} missing question_id or passage_id"
            )

        if qid in HOLDOUT_QIDS or "holdout" in qid.lower():
            raise ValueError(
                f"HOLDOUT VIOLATION: item '{qid}' in export file {export_path}"
            )

        pair = (qid, ps_id)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate pair {pair} inside export file {export_path}")
        seen_pairs.add(pair)

        # Check for forbidden silver fields
        for forbidden in BLINDING_FORBIDDEN_FIELDS:
            if forbidden in rec:
                raise ValueError(
                    f"BLINDING VIOLATION: Forbidden field '{forbidden}' present in export item {pair}"
                )

        # Check grade and role validity
        grade = rec.get("relevance_grade")
        if not isinstance(grade, int) or grade not in VALID_GRADES:
            raise ValueError(
                f"Invalid relevance_grade '{grade}' in record {pair} of {export_path}"
            )

        role = str(rec.get("evidence_role") or "").strip().upper()
        if role not in VALID_ROLES:
            raise ValueError(
                f"Invalid evidence_role '{role}' in record {pair} of {export_path}"
            )

        rec_copy = dict(rec)
        rec_copy["provenance_batch"] = expected_batch_name
        records.append(rec_copy)

    return records


def merge_annotator_b_exports(
    original_export_b_path: Path,
    supplemental_export_b_path: Path,
    full_queue_a_path: Path,
    output_combined_path: Path,
    output_manifest_path: Path,
) -> tuple[Path, Path]:
    """Validate and combine Annotator B original and supplemental exports into a single 69-item combined export."""

    if not full_queue_a_path.exists():
        raise FileNotFoundError(f"Queue A file not found: {full_queue_a_path}")

    # Load Queue A pairs for coverage validation
    queue_a_lines = [
        json.loads(line)
        for line in full_queue_a_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    queue_a_pairs = {(it["question_id"], it["passage_id"]) for it in queue_a_lines}

    # Load and validate original (53) and supplemental (16) exports
    orig_records = load_and_validate_export(
        original_export_b_path, expected_batch_name="original_queue"
    )
    supp_records = load_and_validate_export(
        supplemental_export_b_path, expected_batch_name="supplemental_coverage_review"
    )

    orig_pairs = {(r["question_id"], r["passage_id"]): r for r in orig_records}
    supp_pairs = {(r["question_id"], r["passage_id"]): r for r in supp_records}

    # 1. Overlap Check: original and supplemental MUST be disjoint
    overlap = set(orig_pairs.keys()) & set(supp_pairs.keys())
    if overlap:
        raise ValueError(
            f"Overlap detected between original and supplemental exports ({len(overlap)} items overlap): {overlap}"
        )

    # 2. Combined count check
    combined_records = orig_records + supp_records
    combined_pairs = set(orig_pairs.keys()) | set(supp_pairs.keys())

    if len(combined_records) != len(queue_a_pairs):
        raise ValueError(
            f"Combined count mismatch: expected {len(queue_a_pairs)} items matching Queue A, "
            f"got {len(combined_records)} ({len(orig_records)} original + {len(supp_records)} supplemental)"
        )

    # 3. 100% Set Equality check with Queue A
    if combined_pairs != queue_a_pairs:
        missing = queue_a_pairs - combined_pairs
        unexpected = combined_pairs - queue_a_pairs
        raise ValueError(
            f"Combined export coverage mismatch with Queue A: missing={missing}, unexpected={unexpected}"
        )

    # Sort combined records deterministically by (question_id, passage_id)
    combined_records.sort(key=lambda x: (x["question_id"], x["passage_id"]))

    # 4. Atomic write of combined export
    atomic_write_json(output_combined_path, combined_records, is_jsonl=True)

    # 5. Build and atomic write manifest
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "annotator_id": "annotator_b",
        "original_b_export_sha256": sha256_file(original_export_b_path),
        "supplemental_b_export_sha256": sha256_file(supplemental_export_b_path),
        "queue_a_file_sha256": sha256_file(full_queue_a_path),
        "combined_export_sha256": sha256_file(output_combined_path),
        "original_count": len(orig_records),
        "supplemental_count": len(supp_records),
        "total_combined_count": len(combined_records),
        "matches_queue_a_coverage": True,
        "holdout_sealed": True,
        "export_status": "VALIDATED_COMBINED_HUMAN_QRELS",
    }

    atomic_write_json(output_manifest_path, manifest, is_jsonl=False)

    return output_combined_path, output_manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge Annotator B Original and Supplemental Exports (Gate B)"
    )
    parser.add_argument(
        "--original-export-b",
        type=Path,
        required=True,
        help="Path to original export B (53 items)",
    )
    parser.add_argument(
        "--supplemental-export-b",
        type=Path,
        required=True,
        help="Path to supplemental export B (16 items)",
    )
    parser.add_argument(
        "--full-queue-a",
        type=Path,
        required=True,
        help="Path to full Queue A (annotator_a.jsonl)",
    )
    parser.add_argument(
        "--output-combined",
        type=Path,
        required=True,
        help="Path to output combined export file",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
        help="Path to output combined manifest file",
    )
    args = parser.parse_args()

    try:
        c_path, m_path = merge_annotator_b_exports(
            original_export_b_path=args.original_export_b,
            supplemental_export_b_path=args.supplemental_export_b,
            full_queue_a_path=args.full_queue_a,
            output_combined_path=args.output_combined,
            output_manifest_path=args.output_manifest,
        )
        print("ANNOTATOR B EXPORTS COMBINED SUCCESSFULLY")
        print(f"Combined Export: {c_path} ({sha256_file(c_path)})")
        print(f"Manifest:        {m_path}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
