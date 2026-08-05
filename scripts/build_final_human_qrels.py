# ruff: noqa: E501
"""Build Final Consolidated Human Graded Qrels (Gate B).

CLI:
  python scripts/build_final_human_qrels.py \\
    --annotator-a PATH \\
    --annotator-b-combined PATH \\
    --questions-file PATH \\
    --adjudication-file PATH \\
    --output-qrels PATH \\
    --output-manifest PATH

Consolidates exact 69 human graded qrels:
- Non-adjudicated consensus pairs -> HUMAN_EXACT_CONSENSUS
- Adjudicated disagreement/abstention pairs -> HUMAN_ADJUDICATED (from human adjudication work)
Enforces zero silver usage, zero automatic grade averaging, holdout sealing, and atomic persistence.
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


def load_and_validate_adjudication(
    file_path: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    if not file_path.exists():
        raise FileNotFoundError(f"Adjudication file not found: {file_path}")

    lines = [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError(f"Adjudication file is empty: {file_path}")

    records: dict[tuple[str, str], dict[str, Any]] = {}

    for line_num, line in enumerate(lines, 1):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on line {line_num} of {file_path}: {exc}"
            ) from exc

        qid = str(rec.get("question_id", "")).strip()
        ps_id = str(rec.get("passage_id", "")).strip()

        if not qid or not ps_id:
            raise ValueError(
                f"Missing question_id or passage_id in adjudication file line {line_num}"
            )

        if qid in HOLDOUT_QIDS or "holdout" in qid.lower():
            raise ValueError(f"HOLDOUT VIOLATION: item '{qid}' in adjudication file")

        pair = (qid, ps_id)
        if pair in records:
            raise ValueError(f"Duplicate pair {pair} in adjudication file")

        status = rec.get("status")
        if status not in ("COMPLETED", "VALIDATED_HUMAN_QRELS"):
            raise ValueError(
                f"Adjudication record for pair {pair} is not completed (status='{status}')"
            )

        adj_grade = rec.get("adjudicated_grade")
        if not isinstance(adj_grade, int) or adj_grade not in (0, 1, 2, 3):
            raise ValueError(f"Invalid adjudicated_grade '{adj_grade}' for pair {pair}")

        reasoning = str(rec.get("adjudication_reasoning", "")).strip()
        if not reasoning:
            raise ValueError(f"Missing adjudication_reasoning for pair {pair}")

        records[pair] = rec

    return records


def build_final_human_qrels(
    annotator_a_path: Path,
    annotator_b_path: Path,
    questions_file_path: Path,
    adjudication_file_path: Path,
    output_qrels_path: Path,
    output_manifest_path: Path,
    full_queue_a_path: Path | None = None,
) -> tuple[Path, Path]:
    """Consolidate final 69 human graded qrels from A, B, and Adjudication files."""

    map_a = load_and_validate_export(
        annotator_a_path, expected_annotator_id="annotator_a"
    )
    map_b = load_and_validate_export(
        annotator_b_path, expected_annotator_id="annotator_b"
    )
    map_adj = load_and_validate_adjudication(adjudication_file_path)

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
    hash_adj = sha256_file(adjudication_file_path)

    if set(map_a.keys()) != set(map_b.keys()):
        raise ValueError("Universe mismatch between Annotator A and B exports")

    # Discover structural abstention questions
    questions_content = json.loads(questions_file_path.read_text(encoding="utf-8"))
    abstention_qids: set[str] = set()

    for q in questions_content.get("questions", []):
        qid = q.get("qid") or q.get("question_id")
        if qid and (
            q.get("is_abstention") is True or q.get("question_type") == "abstention"
        ):
            abstention_qids.add(qid)

    common_pairs = sorted(map_a.keys())
    final_qrels: list[dict[str, Any]] = []

    consensus_count = 0
    adjudicated_count = 0

    for qid, ps_id in common_pairs:
        rec_a = map_a[(qid, ps_id)]
        rec_b = map_b[(qid, ps_id)]

        is_disagreement = rec_a["relevance_grade"] != rec_b["relevance_grade"]
        is_abstention_audit = qid in abstention_qids

        needs_adjudication = is_disagreement or is_abstention_audit
        pair = (qid, ps_id)

        if needs_adjudication:
            if pair not in map_adj:
                raise ValueError(
                    f"Pair {pair} requires adjudication but is missing in adjudication file"
                )

            adj_rec = map_adj[pair]
            final_grade = adj_rec["adjudicated_grade"]
            final_role = adj_rec["adjudicated_role"]
            span_human = adj_rec.get("supporting_span_human", "")
            reasoning = adj_rec["adjudication_reasoning"]
            provenance = "HUMAN_ADJUDICATED"
            adjudicated_count += 1
        else:
            final_grade = rec_a["relevance_grade"]
            final_role = rec_a["evidence_role"]
            span_human = rec_a.get("supporting_span_human", "")
            reasoning = "Consensus agreement between Annotator A and Annotator B"
            provenance = "HUMAN_EXACT_CONSENSUS"
            consensus_count += 1

        qrel_item = {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "question_id": qid,
            "passage_id": ps_id,
            "page_number": rec_a.get("page_number", 0),
            "text": text_map.get((qid, ps_id)) or rec_a.get("text", ""),
            "relevance_grade": final_grade,
            "evidence_role": final_role,
            "supporting_span_human": span_human,
            "provenance": provenance,
            "consolidation_reasoning": reasoning,
            "authoritative_for_evaluation": True,
        }

        final_qrels.append(qrel_item)

    # Sort final qrels deterministically by (question_id, passage_id)
    final_qrels.sort(key=lambda x: (x["question_id"], x["passage_id"]))

    # Atomic write of final qrels
    atomic_write_json(output_qrels_path, final_qrels, is_jsonl=True)

    # Build manifest
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "authoritative_for_evaluation": True,
        "silver_used_as_ground_truth": False,
        "holdout_sealed": True,
        "annotator_a_file_sha256": hash_a,
        "annotator_b_file_sha256": hash_b,
        "questions_file_sha256": hash_q,
        "adjudication_file_sha256": hash_adj,
        "final_qrels_file_sha256": sha256_file(output_qrels_path),
        "total_pairs": len(final_qrels),
        "consensus_pairs_count": consensus_count,
        "adjudicated_pairs_count": adjudicated_count,
    }

    # Atomic write of manifest
    atomic_write_json(output_manifest_path, manifest, is_jsonl=False)

    return output_qrels_path, output_manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Final Consolidated Human Graded Qrels (Gate B)"
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
        "--adjudication-file",
        type=Path,
        required=True,
        help="Path to completed adjudication export",
    )
    parser.add_argument(
        "--output-qrels",
        type=Path,
        required=True,
        help="Path to output final human qrels JSONL",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
        help="Path to output manifest JSON",
    )
    args = parser.parse_args()

    try:
        q_path, m_path = build_final_human_qrels(
            annotator_a_path=args.annotator_a,
            annotator_b_path=args.annotator_b_combined,
            questions_file_path=args.questions_file,
            adjudication_file_path=args.adjudication_file,
            output_qrels_path=args.output_qrels,
            output_manifest_path=args.output_manifest,
        )
        print("FINAL HUMAN GRADED QRELS CONSOLIDATED SUCCESSFULLY")
        print(f"Final Qrels:     {q_path} ({sha256_file(q_path)})")
        print(f"Final Manifest:  {m_path}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
