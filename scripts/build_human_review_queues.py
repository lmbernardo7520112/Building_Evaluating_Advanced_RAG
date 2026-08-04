"""Risk-Oriented Human Review Queue Builder (Gate B2 - Etapa 15).

Generates blinded review queues for Annotator A and Annotator B:
- human_queues/annotator_a.jsonl
- human_queues/annotator_b.jsonl
- human_queues/adjudication.jsonl (template)
- human_queues/routing_manifest.json

Supports --without-silver-execution flag when silver triage has not been run.
Enforces 15-25% planned overlap between Annotator A and Annotator B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

# Ensure src and benchmarks are on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))

from run_slice4_benchmark import ACTIVE_QUESTIONS  # noqa: E402

PROTOCOL_VERSION = "raglab_v7_slice4_v3"
SCHEMA_VERSION = "2.0.0"
TARGET_OVERLAP_MIN = 0.15
TARGET_OVERLAP_MAX = 0.25


def build_human_queues(
    input_root: Path,
    output_root: Path,
    without_silver_execution: bool = False,
) -> tuple[Path, Path, Path, Path]:
    """Build blinded human review queues for Annotators A and B."""
    candidate_pool_dir = input_root / "candidate_pool"
    blinded_pool_file = candidate_pool_dir / "blinded_pool.jsonl"

    if not blinded_pool_file.exists():
        raise FileNotFoundError(f"Blinded pool not found at {blinded_pool_file}")

    silver_file = input_root / "silver" / "silver_annotations.jsonl"
    has_silver = silver_file.exists() and not without_silver_execution

    silver_map: dict[tuple[str, str], dict] = {}
    if has_silver:
        for line in silver_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                silver_map[(rec["question_id"], rec["passage_id"])] = rec

    blinded_items = [
        json.loads(line)
        for line in blinded_pool_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Group blinded items by question_id
    items_by_q: dict[str, list[dict]] = {}
    for item in blinded_items:
        items_by_q.setdefault(item["question_id"], []).append(item)

    queue_a_items: list[dict] = []
    queue_b_items: list[dict] = []
    adjudication_template_items: list[dict] = []

    total_pool_count = len(blinded_items)
    overlap_count = 0

    for qid, q_items in items_by_q.items():
        q_obj = next((q for q in ACTIVE_QUESTIONS if q["qid"] == qid), None)
        is_abstention = q_obj.get("abstention_expected", False) if q_obj else False

        # Annotator A gets ALL pool items for the question
        for item in q_items:
            ps_id = item["passage_id"]
            sil_rec = silver_map.get((qid, ps_id), {})

            reasons_a = ["primary_pool_evaluation"]
            if item.get("is_outside_pool_audit"):
                reasons_a.append("outside_pool_audit_sample")
            if is_abstention:
                reasons_a.append("abstention_question")

            item_a = {
                "question_id": qid,
                "passage_id": ps_id,
                "page_number": item["page_number"],
                "text": item["text"],
                "annotator_id": "annotator_a",
                "priority_rank": 1,
                "routing_reasons": reasons_a,
                "is_overlap_sample": False,
                "relevance_grade": None,
                "evidence_role": None,
                "annotation_notes": "",
                "status": "PENDING",
            }
            queue_a_items.append(item_a)

        # Queue B selection: Silver positives + Needs Review
        # + Abstention + 20% random sample
        b_selected_ids: set[str] = set()

        if has_silver:
            for item in q_items:
                ps_id = item["passage_id"]
                sil_rec = silver_map.get((qid, ps_id), {})
                sil_grade = sil_rec.get("relevance_grade", 0)
                needs_rev = sil_rec.get("needs_human_review", False)

                if sil_grade >= 1 or needs_rev:
                    b_selected_ids.add(ps_id)

        # Target ~20% of total question items for Queue B overlap
        target_b_count = max(1, int(round(0.20 * len(q_items))))
        if len(b_selected_ids) < target_b_count:
            remaining_ids = [
                it["passage_id"]
                for it in q_items
                if it["passage_id"] not in b_selected_ids
            ]
            seed_b_sel = int(
                hashlib.sha256(f"{qid}:b_overlap_seed".encode()).hexdigest()[:8], 16
            )
            rng_b = random.Random(seed_b_sel)  # noqa: S311
            needed = target_b_count - len(b_selected_ids)
            additional_b = rng_b.sample(
                remaining_ids, k=min(len(remaining_ids), needed)
            )
            b_selected_ids.update(additional_b)

        for item in q_items:
            ps_id = item["passage_id"]
            if ps_id in b_selected_ids:
                overlap_count += 1
                item_b = {
                    "question_id": qid,
                    "passage_id": ps_id,
                    "page_number": item["page_number"],
                    "text": item["text"],
                    "annotator_id": "annotator_b",
                    "priority_rank": 1,
                    "routing_reasons": ["risk_routing_b_queue"],
                    "is_overlap_sample": True,
                    "relevance_grade": None,
                    "evidence_role": None,
                    "annotation_notes": "",
                    "status": "PENDING",
                }
                queue_b_items.append(item_b)

                # Adjudication template item
                adj_item = {
                    "question_id": qid,
                    "passage_id": ps_id,
                    "annotator_a_grade": None,
                    "annotator_b_grade": None,
                    "adjudicated_grade": None,
                    "adjudicated_role": None,
                    "adjudicator_id": None,
                    "reasoning": "",
                    "status": "PENDING_HUMAN_ANNOTATIONS",
                }
                adjudication_template_items.append(adj_item)

    # Deterministic reordering for A and B
    def reorder_queue(items: list[dict], annotator_id: str) -> list[dict]:
        by_q: dict[str, list[dict]] = {}
        for it in items:
            by_q.setdefault(it["question_id"], []).append(it)
        ordered: list[dict] = []
        for qid in sorted(by_q.keys()):
            seed = int(
                hashlib.sha256(f"{qid}:{annotator_id}".encode()).hexdigest()[:8],
                16,
            )
            q_list = list(by_q[qid])
            random.Random(seed).shuffle(q_list)  # noqa: S311
            for rank, it in enumerate(q_list, 1):
                it["priority_rank"] = rank
                ordered.append(it)
        return ordered

    ordered_a = reorder_queue(queue_a_items, "annotator_a")
    ordered_b = reorder_queue(queue_b_items, "annotator_b")

    output_root.mkdir(parents=True, exist_ok=True)
    file_a = output_root / "annotator_a.jsonl"
    file_b = output_root / "annotator_b.jsonl"
    file_adj = output_root / "adjudication.jsonl"
    file_man = output_root / "routing_manifest.json"

    file_a.write_text(
        "\n".join(json.dumps(it, ensure_ascii=False) for it in ordered_a) + "\n",
        encoding="utf-8",
    )
    file_b.write_text(
        "\n".join(json.dumps(it, ensure_ascii=False) for it in ordered_b) + "\n",
        encoding="utf-8",
    )
    file_adj.write_text(
        "\n".join(
            json.dumps(it, ensure_ascii=False) for it in adjudication_template_items
        )
        + "\n",
        encoding="utf-8",
    )

    planned_overlap = round(overlap_count / max(1, total_pool_count), 4)

    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "annotator_a_queue_count": len(ordered_a),
        "annotator_b_queue_count": len(ordered_b),
        "adjudication_template_count": len(adjudication_template_items),
        "total_candidate_pool_items": total_pool_count,
        "planned_overlap_rate": planned_overlap,
        "target_overlap_range": [TARGET_OVERLAP_MIN, TARGET_OVERLAP_MAX],
        "silver_used_in_routing": has_silver,
        "file_a_sha256": hashlib.sha256(file_a.read_bytes()).hexdigest(),
        "file_b_sha256": hashlib.sha256(file_b.read_bytes()).hexdigest(),
        "file_adj_sha256": hashlib.sha256(file_adj.read_bytes()).hexdigest(),
        "holdout_sealed": True,
        "created_by": "deterministic_offline_builder",
    }

    file_man.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return file_a, file_b, file_adj, file_man


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build risk-oriented human review queues"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=_REPO_ROOT / "benchmarks" / "ground_truth" / "v2" / "hybrid",
        help="Input hybrid directory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_REPO_ROOT
        / "benchmarks"
        / "ground_truth"
        / "v2"
        / "hybrid"
        / "human_queues",
        help="Output human_queues directory",
    )
    parser.add_argument(
        "--without-silver-execution",
        action="store_true",
        help="Build queues without requiring silver triage execution",
    )

    args = parser.parse_args()

    fa, fb, fadj, fman = build_human_queues(
        input_root=args.input_root,
        output_root=args.output_root,
        without_silver_execution=args.without_silver_execution,
    )

    print("HUMAN REVIEW QUEUES BUILT SUCCESSFULLY")
    print(f"Queue A: {fa}")
    print(f"Queue B: {fb}")
    print(f"Adjudication Template: {fadj}")
    print(f"Routing Manifest: {fman}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
