# ruff: noqa: E501
"""Build blinded supplemental coverage review queue for Annotator B (Gate B).

CLI:
  python scripts/build_supplemental_b_queue.py \\
    --full-queue-a PATH \\
    --original-queue-b PATH \\
    --output-queue PATH \\
    --output-manifest PATH

Calculates strict set difference (Queue A - Queue B) using ONLY blinded input queue files.
Guarantees zero reading of human exports or silver predictions, blinded fields for Annotator B,
holdout rejection, and atomic persistence.
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
    """Compute SHA-256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_json(target_path: Path, data: Any, is_jsonl: bool = False) -> None:
    """Write data to target_path atomically using tmp file + fsync + replace."""
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


def load_blinded_queue_items(
    queue_path: Path, expected_annotator_id: str | None = None
) -> list[dict[str, Any]]:
    """Load and validate a blinded input queue file fail-closed."""
    if not queue_path.exists():
        raise FileNotFoundError(f"Input queue file not found: {queue_path}")

    # Guard: Fail-closed if someone passes a human export file or silver annotation file
    name_lower = queue_path.name.lower()
    if (
        "final" in name_lower
        or "export" in name_lower
        or "silver" in name_lower
        or "work" in name_lower
    ):
        raise ValueError(
            f"SECURITY VIOLATION: Input file '{queue_path}' appears to be a human export or silver file. "
            "Must use strictly original blinded input queue files."
        )

    lines = queue_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Input queue file is empty: {queue_path}")

    items: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for line_num, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on line {line_num} of {queue_path}: {exc}"
            ) from exc

        qid = str(rec.get("question_id", "")).strip()
        ps_id = str(rec.get("passage_id", "")).strip()

        if not qid or not ps_id:
            raise ValueError(
                f"Line {line_num} in {queue_path} missing question_id or passage_id"
            )

        if qid in HOLDOUT_QIDS or "holdout" in qid.lower():
            raise ValueError(
                f"HOLDOUT VIOLATION: item '{qid}' found in input queue {queue_path}"
            )

        pair = (qid, ps_id)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate pair {pair} in input queue {queue_path}")
        seen_pairs.add(pair)

        # Check for forbidden silver or human judgment fields in input
        for forbidden in BLINDING_FORBIDDEN_FIELDS:
            if forbidden in rec:
                raise ValueError(
                    f"BLINDING VIOLATION: Forbidden field '{forbidden}' present in input item {pair}"
                )

        if (
            expected_annotator_id
            and rec.get("annotator_id")
            and rec["annotator_id"] != expected_annotator_id
        ):
            raise ValueError(
                f"Identity mismatch in {queue_path}: expected '{expected_annotator_id}', got '{rec['annotator_id']}'"
            )

        items.append(rec)

    return items


def build_supplemental_b_queue(
    full_queue_a_path: Path,
    original_queue_b_path: Path,
    output_queue_path: Path,
    output_manifest_path: Path,
    expected_supplemental_count: int | None = None,
) -> tuple[Path, Path]:
    """Calculate strict set difference A - B from blinded queues and construct supplemental queue for B."""

    items_a = load_blinded_queue_items(
        full_queue_a_path, expected_annotator_id="annotator_a"
    )
    items_b = load_blinded_queue_items(
        original_queue_b_path, expected_annotator_id="annotator_b"
    )

    hash_a = sha256_file(full_queue_a_path)
    hash_b = sha256_file(original_queue_b_path)

    map_a = {(it["question_id"], it["passage_id"]): it for it in items_a}
    pairs_b = {(it["question_id"], it["passage_id"]) for it in items_b}

    diff_pairs = sorted(set(map_a.keys()) - pairs_b)

    if (
        expected_supplemental_count is not None
        and len(diff_pairs) != expected_supplemental_count
    ):
        raise ValueError(
            f"Supplemental count mismatch: expected {expected_supplemental_count} items in (A - B), "
            f"found {len(diff_pairs)}"
        )

    supplemental_items: list[dict[str, Any]] = []

    for qid, ps_id in diff_pairs:
        raw_a_item = map_a[(qid, ps_id)]

        # Construct sanitized supplemental item for Annotator B
        b_item = {
            "question_id": qid,
            "passage_id": ps_id,
            "page_number": raw_a_item.get("page_number", 0),
            "text": raw_a_item["text"],
            "annotator_id": "annotator_b",
            "priority_rank": raw_a_item.get("priority_rank", 1),
            "routing_reasons": ["supplemental_coverage_review"],
            "is_overlap_sample": False,
            "relevance_grade": None,
            "evidence_role": None,
            "supporting_span_human": "",
            "annotation_notes": "",
            "status": "PENDING",
        }

        # Verify no leaking fields remain
        for forbidden in BLINDING_FORBIDDEN_FIELDS:
            if forbidden in b_item:
                b_item.pop(forbidden)

        supplemental_items.append(b_item)

    # Sort deterministically by question_id then passage_id
    supplemental_items.sort(key=lambda x: (x["question_id"], x["passage_id"]))

    # Write output queue file atomically
    atomic_write_json(output_queue_path, supplemental_items, is_jsonl=True)
    supp_queue_hash = sha256_file(output_queue_path)

    # Construct manifest
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "queue_a_file_sha256": hash_a,
        "queue_b_file_sha256": hash_b,
        "total_queue_a": len(items_a),
        "total_queue_b_original": len(items_b),
        "supplemental_count": len(supplemental_items),
        "blinding_verified": True,
        "human_labels_consumed": False,
        "holdout_sealed": True,
        "supplemental_queue_sha256": supp_queue_hash,
    }

    # Write output manifest file atomically
    atomic_write_json(output_manifest_path, manifest, is_jsonl=False)

    return output_queue_path, output_manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Supplemental Coverage Review Queue for Annotator B (Gate B)"
    )
    parser.add_argument(
        "--full-queue-a",
        type=Path,
        required=True,
        help="Path to full Queue A (annotator_a.jsonl)",
    )
    parser.add_argument(
        "--original-queue-b",
        type=Path,
        required=True,
        help="Path to original Queue B (annotator_b.jsonl)",
    )
    parser.add_argument(
        "--output-queue",
        type=Path,
        required=True,
        help="Path to output supplemental queue file (annotator_b_supplemental.jsonl)",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
        help="Path to output supplemental manifest file",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=16,
        help="Expected supplemental count (default 16)",
    )
    args = parser.parse_args()

    try:
        q_path, m_path = build_supplemental_b_queue(
            full_queue_a_path=args.full_queue_a,
            original_queue_b_path=args.original_queue_b,
            output_queue_path=args.output_queue,
            output_manifest_path=args.output_manifest,
            expected_supplemental_count=args.expected_count,
        )
        print("BLINDED SUPPLEMENTAL B QUEUE BUILT SUCCESSFULLY")
        print(f"Output Queue:    {q_path} ({sha256_file(q_path)})")
        print(f"Output Manifest: {m_path}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
