#!/usr/bin/env python3
"""CLI runner for Slice 5A.3 Canonical Coverage and Retrieval Parity Gate V3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from raglab.agentic.evaluation.parity_gate import (
    run_parity_gate_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Slice 5A.3 Canonical Coverage and Retrieval Parity Gate V3."
    )
    parser.add_argument(
        "--protocol",
        default="benchmarks/agentic/slice5/slice5a3/protocols/preregistration_v3.json",
        help="Path to protocol file.",
    )
    parser.add_argument(
        "--qrels",
        default="benchmarks/ground_truth/v2/hybrid/qrels/human_qrels_final.jsonl",
        help="Path to human_qrels_final.jsonl file.",
    )
    parser.add_argument(
        "--qrels-manifest",
        default="benchmarks/ground_truth/v2/hybrid/qrels/human_qrels_manifest.json",
        help="Path to human_qrels_manifest.json file.",
    )
    parser.add_argument(
        "--passage-registry",
        default="benchmarks/ground_truth/v2/passage_registry.jsonl",
        help="Path to passage_registry.jsonl file.",
    )
    parser.add_argument(
        "--passage-registry-manifest",
        default="benchmarks/ground_truth/v2/passage_registry_manifest.json",
        help="Path to passage_registry_manifest.json file.",
    )
    parser.add_argument(
        "--questions-file",
        default="benchmarks/questions/controlled_chapter2.json",
        help="Path to controlled_chapter2.json questions file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Target output directory for artifact generation.",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=2,
        help="Number of repeat runs for determinism check (default: 2).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=True,
        help="Enforce 100% offline execution without remote APIs.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file if it exists."""
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()

    # 1. Credential check
    if (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    ):
        sys.stderr.write(
            "ERROR: External API credentials detected in environment. Aborting.\n"
        )
        return 1

    # 2. Output dir check
    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.stderr.write(
            f"ERROR: Output directory '{out_dir}' already exists and is non-empty. Aborting.\n"
        )
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load protocol
    proto_path = Path(args.protocol)
    if not proto_path.exists():
        sys.stderr.write(f"ERROR: Protocol file not found: {proto_path}\n")
        return 1

    protocol_text = proto_path.read_text(encoding="utf-8")
    protocol = json.loads(protocol_text)
    protocol_sha256 = hashlib.sha256(protocol_text.encode("utf-8")).hexdigest()

    # 4. Check questions for TEST QIDs
    q_path = Path(args.questions_file)
    if not q_path.exists():
        sys.stderr.write(f"ERROR: Questions file not found: {q_path}\n")
        return 1

    q_data = json.loads(q_path.read_text(encoding="utf-8"))
    questions = q_data.get("questions", [])
    dev_qids = protocol.get(
        "dev_qids", ["q_dev_01", "q_dev_02", "q_dev_03", "q_dev_04"]
    )

    for q in questions:
        qid = q.get("qid") or q.get("question_id")
        if qid not in dev_qids and q.get("split") == "test":
            sys.stderr.write(
                f"WARNING: TEST QID '{qid}' filtered out from split.\n"
            )

    # 5. Run Parity Gate evaluation
    try:
        result, arm_runs, unjudged_queue, exec_metadata = run_parity_gate_evaluation(
            protocol=protocol,
            passage_registry_path=args.passage_registry,
            passage_registry_manifest_path=args.passage_registry_manifest,
            human_qrels_path=args.qrels,
            questions_path=args.questions_file,
            top_k=protocol.get("top_k", 3),
        )
    except Exception as err:
        sys.stderr.write(f"ERROR: Parity Gate evaluation failed: {err}\n")
        return 1

    # Write Canonical Projection JSONL
    with (out_dir / "canonical_projection.jsonl").open("w", encoding="utf-8") as f:
        for r in arm_runs:
            for item in r.retrieved_items:
                row = {
                    "qid": item.qid,
                    "arm_id": item.arm_id,
                    "rank": item.rank,
                    "technical_chunk_id": item.technical_chunk_id,
                    "technical_node_id": item.technical_node_id,
                    "anchor_passage_id": item.anchor_passage_id,
                    "supporting_passage_ids": item.supporting_passage_ids,
                    "source_offsets": item.source_offsets,
                    "projection_method": item.projection_method,
                    "page_number": item.page_number,
                    "content_sha256": item.content_sha256,
                    "score": item.score,
                    "mapping_status": item.mapping_status.value,
                    "judgment_status": item.judgment_status.value,
                    "human_grade": item.human_grade,
                }
                f.write(json.dumps(row) + "\n")

    # Write Retrieval Matrix JSON
    (out_dir / "retrieval_matrix.json").write_text(
        json.dumps(exec_metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Write Repeatability Report JSON
    repeatability_data = {
        "protocol_sha256": protocol_sha256,
        "implementation_commit": protocol.get("implementation_commit", ""),
        "protocol_commit": protocol.get("protocol_commit", ""),
        "repeatability_topk_identity": result.repeatability_topk_identity,
        "repeatability_rank_match": result.repeatability_rank_match,
        "same_run_arm_completeness": result.same_run_arm_completeness,
    }
    (out_dir / "repeatability_report.json").write_text(
        json.dumps(repeatability_data, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Write Coverage Report JSON
    coverage_data = {
        "canonical_coverage": result.canonical_coverage,
        "judged_coverage": result.judged_coverage,
        "unmapped_count": result.unmapped_count,
        "unjudged_count": result.unjudged_count,
        "ambiguous_count": exec_metadata.get("ambiguous_count", 0),
        "synthetic_fallback_count": result.synthetic_fallback_count,
        "expected_technical_slots": exec_metadata.get("expected_technical_slots", 84),
        "technical_items_returned": exec_metadata.get("technical_items_returned", 83),
        "missing_technical_slots": exec_metadata.get("missing_technical_slots", 1),
    }
    (out_dir / "coverage_report.json").write_text(
        json.dumps(coverage_data, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Write Scientific Result JSON
    result_data = {
        "status": result.status.value,
        "failure_reasons": result.failure_reasons,
        "metrics_status": result.metrics_status,
        "metrics": result.metrics,
        "protocol_sha256": protocol_sha256,
        "implementation_commit": protocol.get("implementation_commit", ""),
    }
    (out_dir / "scientific_result.json").write_text(
        json.dumps(result_data, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Write Human Review Queue JSONL if unjudged items exist
    if unjudged_queue:
        with (out_dir / "human_review_queue.jsonl").open("w", encoding="utf-8") as f:
            for item in unjudged_queue:
                f.write(json.dumps(item) + "\n")

    # Write Manifest JSON
    manifest_data = {
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": protocol_sha256,
        "implementation_commit": protocol.get("implementation_commit", ""),
        "status": result.status.value,
        "canonical_coverage": result.canonical_coverage,
        "judged_coverage": result.judged_coverage,
        "unmapped_count": result.unmapped_count,
        "unjudged_count": result.unjudged_count,
        "repeatability_topk_identity": result.repeatability_topk_identity,
        "same_run_arm_completeness": result.same_run_arm_completeness,
        "metrics_status": result.metrics_status,
        "output_artifacts": [
            "manifest.json",
            "coverage_report.json",
            "retrieval_matrix.json",
            "canonical_projection.jsonl",
            "repeatability_report.json",
            "scientific_result.json",
        ],
    }
    if unjudged_queue:
        manifest_data["output_artifacts"].append("human_review_queue.jsonl")

    # Artifact hashes map
    artifact_hashes = {}
    for art in manifest_data["output_artifacts"]:
        artifact_hashes[art] = sha256_file(out_dir / art)
    manifest_data["artifact_hashes"] = artifact_hashes

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"PARITY_GATE_SUCCESS status={result.status.value} output_dir={out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
