# ruff: noqa: E501, UP012
"""Build Blinded Human Adjudication Queue (Gate B).

CLI:
  python scripts/build_human_adjudication_queue.py \\
    --annotator-a PATH \\
    --annotator-b-combined PATH \\
    --questions-file PATH \\
    --output-queue PATH \\
    --output-manifest PATH

Calculates the deduplicated union of grade disagreements (21) and structural abstention
audit questions (10), anonymizes A & B into Reviewer 1 & 2 via deterministic pair hashing,
and outputs a blinded adjudication queue and manifest.
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
        "expected_answer_summary",
    }
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

        for forbidden in BLINDING_FORBIDDEN_FIELDS:
            if forbidden in rec:
                raise ValueError(
                    f"BLINDING VIOLATION: forbidden field '{forbidden}' in {file_path}"
                )

        records[pair] = rec

    return records


def build_adjudication_queue(
    annotator_a_path: Path,
    annotator_b_path: Path,
    questions_file_path: Path,
    output_queue_path: Path,
    output_manifest_path: Path,
    full_queue_a_path: Path | None = None,
) -> tuple[Path, Path]:
    """Build blinded human adjudication queue from A & B export files and questions file."""

    map_a = load_and_validate_export(
        annotator_a_path, expected_annotator_id="annotator_a"
    )
    map_b = load_and_validate_export(
        annotator_b_path, expected_annotator_id="annotator_b"
    )

    # Load passage text map from full Queue A file
    if full_queue_a_path is None:
        full_queue_a_path = Path(
            "benchmarks/ground_truth/v2/hybrid/human_queues/annotator_a.jsonl"
        )

    text_map: dict[tuple[str, str], str] = {}
    if full_queue_a_path.exists():
        for line in full_queue_a_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                text_map[(rec["question_id"], rec["passage_id"])] = rec.get("text", "")

    hash_a = sha256_file(annotator_a_path)
    hash_b = sha256_file(annotator_b_path)
    hash_q = sha256_file(questions_file_path)

    if set(map_a.keys()) != set(map_b.keys()):
        raise ValueError("Universe mismatch between Annotator A and B exports")

    # Load questions data and discover structural abstention questions
    questions_content = json.loads(questions_file_path.read_text(encoding="utf-8"))
    q_dict: dict[str, dict[str, Any]] = {}

    for q in questions_content.get("questions", []):
        qid = q.get("qid") or q.get("question_id")
        if qid:
            q_dict[qid] = q

    abstention_qids: set[str] = set()
    for qid, q_obj in q_dict.items():
        if (
            q_obj.get("is_abstention") is True
            or q_obj.get("question_type") == "abstention"
        ):
            abstention_qids.add(qid)

    # Calculate union of disagreements and structural abstention pairs
    disagreement_pairs: set[tuple[str, str]] = set()
    abstention_pairs: set[tuple[str, str]] = set()

    for pair, rec_a in map_a.items():
        rec_b = map_b[pair]
        qid, ps_id = pair

        if rec_a["relevance_grade"] != rec_b["relevance_grade"]:
            disagreement_pairs.add(pair)

        if qid in abstention_qids:
            abstention_pairs.add(pair)

    union_pairs = sorted(disagreement_pairs | abstention_pairs)

    adjudication_items: list[dict[str, Any]] = []
    anonymization_mapping: list[dict[str, Any]] = []

    for qid, ps_id in union_pairs:
        rec_a = map_a[(qid, ps_id)]
        rec_b = map_b[(qid, ps_id)]

        reasons: list[str] = []
        if (qid, ps_id) in disagreement_pairs:
            reasons.append("disagreement")
        if (qid, ps_id) in abstention_pairs:
            reasons.append("structural_abstention_audit")

        # Position bias prevention: deterministic hash modulo 2 for reviewer order
        pair_hash = hashlib.sha256(f"{qid}:{ps_id}".encode("utf-8")).hexdigest()
        is_even = int(pair_hash, 16) % 2 == 0

        if is_even:
            rev_1_grade, rev_2_grade = (
                rec_a["relevance_grade"],
                rec_b["relevance_grade"],
            )
            rev_1_role, rev_2_role = rec_a["evidence_role"], rec_b["evidence_role"]
            mapping_str = "reviewer_1=A, reviewer_2=B"
        else:
            rev_1_grade, rev_2_grade = (
                rec_b["relevance_grade"],
                rec_a["relevance_grade"],
            )
            rev_1_role, rev_2_role = rec_b["evidence_role"], rec_a["evidence_role"]
            mapping_str = "reviewer_1=B, reviewer_2=A"

        anonymization_mapping.append(
            {"question_id": qid, "passage_id": ps_id, "mapping": mapping_str}
        )

        q_text = q_dict.get(qid, {}).get("question") or q_dict.get(qid, {}).get(
            "text", ""
        )

        adj_item = {
            "question_id": qid,
            "question_text": q_text,
            "passage_id": ps_id,
            "page_number": rec_a.get("page_number", 0),
            "passage_text": text_map.get((qid, ps_id)) or rec_a.get("text", ""),
            "adjudication_reasons": reasons,
            "reviewer_1_grade": rev_1_grade,
            "reviewer_2_grade": rev_2_grade,
            "reviewer_1_role": rev_1_role,
            "reviewer_2_role": rev_2_role,
            "adjudicated_grade": None,
            "adjudicated_role": None,
            "adjudication_reasoning": "",
            "supporting_span_human": "",
            "status": "PENDING",
        }

        # Double check no forbidden fields remain
        for forbidden in BLINDING_FORBIDDEN_FIELDS:
            if forbidden in adj_item:
                adj_item.pop(forbidden)

        adjudication_items.append(adj_item)

    # Sort items deterministically by (question_id, passage_id)
    adjudication_items.sort(key=lambda x: (x["question_id"], x["passage_id"]))

    # Atomic write of adjudication queue
    atomic_write_json(output_queue_path, adjudication_items, is_jsonl=True)

    # Build manifest
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "annotator_a_file_sha256": hash_a,
        "annotator_b_file_sha256": hash_b,
        "questions_file_sha256": hash_q,
        "total_disagreement_pairs": len(disagreement_pairs),
        "total_abstention_pairs": len(abstention_pairs),
        "overlap_disagreement_abstention_count": len(
            disagreement_pairs & abstention_pairs
        ),
        "total_adjudication_queue": len(adjudication_items),
        "abstention_detection_mechanism": "question_attribute_is_abstention_or_type_abstention",
        "anonymization_verified": True,
        "holdout_sealed": True,
        "adjudication_queue_sha256": sha256_file(output_queue_path),
        "anonymization_provenance": anonymization_mapping,
    }

    # Atomic write of manifest
    atomic_write_json(output_manifest_path, manifest, is_jsonl=False)

    return output_queue_path, output_manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Blinded Human Adjudication Queue (Gate B)"
    )
    parser.add_argument(
        "--annotator-a",
        type=Path,
        required=True,
        help="Path to Annotator A final export",
    )
    parser.add_argument(
        "--annotator-b-combined",
        type=Path,
        required=True,
        help="Path to Annotator B combined export",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        required=True,
        help="Path to controlled questions JSON",
    )
    parser.add_argument(
        "--output-queue",
        type=Path,
        required=True,
        help="Path to output adjudication queue file",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
        help="Path to output manifest file",
    )
    args = parser.parse_args()

    try:
        q_path, m_path = build_adjudication_queue(
            annotator_a_path=args.annotator_a,
            annotator_b_path=args.annotator_b_combined,
            questions_file_path=args.questions_file,
            output_queue_path=args.output_queue,
            output_manifest_path=args.output_manifest,
        )
        print("BLINDED HUMAN ADJUDICATION QUEUE BUILT SUCCESSFULLY")
        print(f"Output Queue:    {q_path} ({sha256_file(q_path)})")
        print(f"Output Manifest: {m_path}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
