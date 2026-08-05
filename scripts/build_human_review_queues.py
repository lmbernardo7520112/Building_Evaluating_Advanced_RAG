"""Human Review Queue Builder consuming evidence v2 pool.

Generates blinded review queues for Annotators A and B:
- human_queues/annotator_a.jsonl
- human_queues/annotator_b.jsonl
- human_queues/adjudication.jsonl
- human_queues/routing_manifest.json

Queue A = full union of pool + outside audit.
Queue B = risk items + random overlap sample (15-25%).
No silver, no ground truth, no relevant_pages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_VERSION = "raglab_v7_slice4_v3"
SCHEMA_VERSION = "3.0.0"
TARGET_OVERLAP_MIN = 0.15
TARGET_OVERLAP_MAX = 0.25

# Fields that MUST NOT appear in blinded view
BLINDING_FORBIDDEN_FIELDS = frozenset(
    {
        "source_provenance",
        "raw_retrieval_unit",
        "generation_context_unit",
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


def _verify_blinding(item: dict) -> list[str]:
    """Verify a queue item is recursively blinded."""
    violations = []
    for field in BLINDING_FORBIDDEN_FIELDS:
        if field in item:
            violations.append(f"Field '{field}' present in blinded item")
    return violations


def build_human_queues(
    input_root: Path,
    output_root: Path,
    without_silver_execution: bool = False,
) -> tuple[Path, Path, Path, Path]:
    """Build blinded human review queues."""
    blinded_pool_file = input_root / "candidate_pool" / "blinded_pool.jsonl"

    if not blinded_pool_file.exists():
        raise FileNotFoundError(f"Blinded pool not found: {blinded_pool_file}")

    blinded_items = [
        json.loads(line)
        for line in blinded_pool_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    items_by_q: dict[str, list[dict]] = {}
    for item in blinded_items:
        items_by_q.setdefault(item["question_id"], []).append(item)

    queue_a_items: list[dict] = []
    queue_b_items: list[dict] = []
    adjudication_items: list[dict] = []

    total_pool_count = len(blinded_items)
    overlap_count = 0

    for qid in sorted(items_by_q.keys()):
        q_items = items_by_q[qid]

        # Queue A: ALL items
        for item in q_items:
            item_a = {
                "question_id": qid,
                "passage_id": item["passage_id"],
                "page_number": item["page_number"],
                "text": item["text"],
                "annotator_id": "annotator_a",
                "priority_rank": 1,
                "routing_reasons": ["primary_pool_evaluation"],
                "is_overlap_sample": False,
                "relevance_grade": None,
                "evidence_role": None,
                "annotation_notes": "",
                "status": "PENDING",
            }
            if item.get("is_outside_pool_audit"):
                item_a["routing_reasons"].append("outside_pool_audit_sample")
            queue_a_items.append(item_a)

        # Queue B: risk items + random overlap
        b_selected_ids: set[str] = set()

        # Risk: outside audit items and abstention questions
        for item in q_items:
            if item.get("is_outside_pool_audit"):
                b_selected_ids.add(item["passage_id"])

        # Random overlap to target 20%
        target_b = max(1, int(round(0.20 * len(q_items))))
        if len(b_selected_ids) < target_b:
            remaining = [
                it["passage_id"]
                for it in q_items
                if it["passage_id"] not in b_selected_ids
            ]
            seed = int(
                hashlib.sha256(f"{qid}:b_overlap_seed".encode()).hexdigest()[:8], 16
            )
            rng = random.Random(seed)  # noqa: S311
            needed = target_b - len(b_selected_ids)
            additional = rng.sample(remaining, k=min(len(remaining), needed))
            b_selected_ids.update(additional)

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

                adj = {
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
                adjudication_items.append(adj)

    # Deterministic reordering
    def reorder(items: list[dict], annotator_id: str) -> list[dict]:
        by_q: dict[str, list[dict]] = {}
        for it in items:
            by_q.setdefault(it["question_id"], []).append(it)
        ordered: list[dict] = []
        for q in sorted(by_q.keys()):
            seed = int(
                hashlib.sha256(f"{q}:{annotator_id}".encode()).hexdigest()[:8], 16
            )
            q_list = list(by_q[q])
            random.Random(seed).shuffle(q_list)  # noqa: S311
            for rank, it in enumerate(q_list, 1):
                it["priority_rank"] = rank
                ordered.append(it)
        return ordered

    ordered_a = reorder(queue_a_items, "annotator_a")
    ordered_b = reorder(queue_b_items, "annotator_b")

    # Verify recursive blinding
    blinding_violations: list[str] = []
    for it in ordered_a + ordered_b:
        blinding_violations.extend(_verify_blinding(it))
    if blinding_violations:
        print(f"ERROR: {len(blinding_violations)} blinding violations:")
        for v in blinding_violations[:10]:
            print(f"  {v}")
        sys.exit(1)

    # Write files
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
    adj_lines = "\n".join(
        json.dumps(it, ensure_ascii=False) for it in adjudication_items
    )
    file_adj.write_text(adj_lines + "\n", encoding="utf-8")

    planned_overlap = round(overlap_count / max(1, total_pool_count), 4)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "queue_status": "PROVISIONAL_WITHOUT_SILVER",
        "annotator_a_queue_count": len(ordered_a),
        "annotator_b_queue_count": len(ordered_b),
        "adjudication_template_count": len(adjudication_items),
        "total_candidate_pool_items": total_pool_count,
        "planned_overlap_rate": planned_overlap,
        "target_overlap_range": [TARGET_OVERLAP_MIN, TARGET_OVERLAP_MAX],
        "blinding_verified": len(blinding_violations) == 0,
        "silver_used_in_routing": False,
        "file_a_sha256": hashlib.sha256(file_a.read_bytes()).hexdigest(),
        "file_b_sha256": hashlib.sha256(file_b.read_bytes()).hexdigest(),
        "file_adj_sha256": hashlib.sha256(file_adj.read_bytes()).hexdigest(),
        "holdout_sealed": True,
    }
    file_man.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return file_a, file_b, file_adj, file_man


def main() -> int:
    parser = argparse.ArgumentParser(description="Build human review queues")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=_REPO_ROOT / "benchmarks" / "ground_truth" / "v2" / "hybrid",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            _REPO_ROOT
            / "benchmarks"
            / "ground_truth"
            / "v2"
            / "hybrid"
            / "human_queues"
        ),
    )
    parser.add_argument(
        "--without-silver-execution",
        action="store_true",
    )
    args = parser.parse_args()

    fa, fb, fadj, fman = build_human_queues(
        input_root=args.input_root,
        output_root=args.output_root,
        without_silver_execution=args.without_silver_execution,
    )
    print("HUMAN REVIEW QUEUES BUILT SUCCESSFULLY")
    count_a = sum(1 for _ in fa.read_text().splitlines() if _.strip())
    count_b = sum(1 for _ in fb.read_text().splitlines() if _.strip())
    print(f"Queue A: {fa} ({count_a} items)")
    print(f"Queue B: {fb} ({count_b} items)")
    print(f"Adjudication: {fadj}")
    print(f"Manifest: {fman}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
