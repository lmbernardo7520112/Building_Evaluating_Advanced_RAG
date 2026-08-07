"""Human Review Queue Builder consuming evidence v2 pool and Machine Silver triage.

Generates blinded review queues for Annotators A and B:
- human_queues/annotator_a.jsonl
- human_queues/annotator_b.jsonl
- human_queues/adjudication.jsonl
- human_queues/routing_manifest.json

Queue A = full union of pool + outside audit (69 items).
Queue B = risk items (silver grade > 0, needs_human_review=True, outside audit)
          + random negative control overlap sample (15-25%).
Strictly blinded: no silver grades, roles, confidence, reasoning, model, or ranks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_VERSION = "raglab_v7_slice4_v3"
SCHEMA_VERSION = "3.0.0"
TARGET_OVERLAP_MIN = 0.15
TARGET_OVERLAP_MAX = 0.25
HOLDOUT_QIDS = frozenset({"q_holdout_01", "q_holdout_02"})

# Fields that MUST NOT appear in blinded queue items
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
        "confidence",
        "reasoning",
        "supporting_span",
        "judge_model",
        "judge_provider",
        "judge_id",
        "label_source",
    }
)

BLINDING_FORBIDDEN_KEYWORDS = frozenset(
    {
        "silver",
        "judge",
        "predicted",
        "low_confidence",
        "positive",
    }
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_blinding(item: dict[str, Any]) -> list[str]:
    """Verify a queue item is recursively blinded."""
    violations: list[str] = []

    def _scan(val: Any, key_context: str = "") -> None:
        if isinstance(val, dict):
            for k, v in val.items():
                if k in BLINDING_FORBIDDEN_FIELDS:
                    violations.append(f"Forbidden field '{k}' present in item")
                _scan(v, k)
        elif isinstance(val, list):
            for elem in val:
                _scan(elem, key_context)
        elif isinstance(val, str) and key_context == "routing_reasons":
            lowered = val.lower()
            for kw in BLINDING_FORBIDDEN_KEYWORDS:
                if kw in lowered:
                    violations.append(
                        f"Revealing keyword '{kw}' found in reasons: '{val}'"
                    )

    _scan(item)
    return violations


def validate_silver_execution(
    silver_file: Path, pool_items: list[dict[str, Any]]
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Validate silver execution file, manifest, and checkpoint."""
    if not silver_file.exists():
        raise FileNotFoundError(f"Silver file not found at {silver_file}")

    silver_bytes = silver_file.read_bytes()
    silver_sha = hashlib.sha256(silver_bytes).hexdigest()

    # 1. Parse JSONL & check validity
    silver_records_list: list[dict[str, Any]] = []
    lines = silver_bytes.decode("utf-8").splitlines()
    for line_idx, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            silver_records_list.append(r)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in silver file at line {line_idx}: {exc}"
            ) from exc

    if not silver_records_list:
        raise ValueError("Silver file is empty or contains no records")

    # 2. Check for duplicate pairs & holdout & label_source
    seen_pairs: set[tuple[str, str]] = set()
    silver_records: dict[tuple[str, str], dict[str, Any]] = {}

    for r in silver_records_list:
        qid = r.get("question_id", "")
        ps_id = r.get("passage_id", "")
        pair = (qid, ps_id)

        if pair in seen_pairs:
            raise ValueError(f"Duplicate pair {pair} in silver records")
        seen_pairs.add(pair)

        if qid in HOLDOUT_QIDS or "holdout" in qid.lower():
            raise ValueError(f"HOLDOUT VIOLATION: item {qid} found in silver file")

        if r.get("label_source") != "MACHINE_SILVER":
            src = r.get("label_source")
            raise ValueError(
                f"Invalid label_source '{src}' in silver record for {pair}"
            )

        silver_records[pair] = r

    # 3. Check exact pool pairs matching
    pool_pairs = {(it["question_id"], it["passage_id"]) for it in pool_items}
    if pool_pairs != set(silver_records.keys()):
        missing = pool_pairs - set(silver_records.keys())
        extra = set(silver_records.keys()) - pool_pairs
        raise ValueError(
            f"Silver records do not match candidate pool pairs. "
            f"Missing from silver: {len(missing)}, Extra in silver: {len(extra)}"
        )

    # 4. Manifest Validation
    run_dir = silver_file.parent
    manifest_file = run_dir / "silver_manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Silver manifest not found at {manifest_file}")

    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))

    if manifest_data.get("mode") != "full":
        raise ValueError(
            f"Manifest mode must be 'full', got '{manifest_data.get('mode')}'"
        )
    exec_mode = manifest_data.get("execution_mode")
    if exec_mode != "FULL_REAL":
        raise ValueError(
            f"Manifest execution_mode must be 'FULL_REAL', got '{exec_mode}'"
        )

    exec_auth = manifest_data.get("execution_authenticity")
    if exec_auth != "REAL_MODEL_CALL":
        msg = (
            "Manifest execution_authenticity must be 'REAL_MODEL_CALL', "
            f"got '{exec_auth}'"
        )
        raise ValueError(msg)
    if manifest_data.get("status") != "COMPLETED":
        raise ValueError(
            f"Manifest status must be 'COMPLETED', got '{manifest_data.get('status')}'"
        )
    p_cnt = manifest_data.get("pending_count")
    if p_cnt != 0:
        raise ValueError(f"Manifest pending_count must be 0, got {p_cnt}")
    if manifest_data.get("authoritative_for_human_qrels") is not False:
        raise ValueError("Manifest authoritative_for_human_qrels must be False")
    if manifest_data.get("holdout_sealed") is not True:
        raise ValueError("Manifest holdout_sealed must be True")

    # 5. Checkpoint Validation
    checkpoint_file = run_dir / "checkpoint.json"
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Silver checkpoint not found at {checkpoint_file}")

    checkpoint_data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    records_sha_in_cp = checkpoint_data.get("records_sha256", "")
    if records_sha_in_cp != silver_sha:
        raise ValueError(
            f"SHA-256 mismatch: silver_annotations.jsonl ({silver_sha[:16]}...) "
            f"does not match checkpoint records_sha256 ({records_sha_in_cp[:16]}...)"
        )

    return (
        silver_records,
        manifest_data,
        checkpoint_data,
        {
            "silver_file_sha256": silver_sha,
            "silver_manifest_sha256": sha256_file(manifest_file),
            "silver_checkpoint_sha256": sha256_file(checkpoint_file),
        },
    )


def build_human_queues(
    input_root: Path,
    output_root: Path,
    silver_file: Path | None = None,
    without_silver_execution: bool = False,
) -> tuple[Path, Path, Path, Path]:
    """Build blinded human review queues for Annotators A and B."""

    # ── Enforce CLI Option Rules Fail-Closed ──────────────────────
    if not silver_file and not without_silver_execution:
        raise ValueError(
            "Must provide either --silver-file PATH for definitive review "
            "or --without-silver-execution for provisional review."
        )

    if silver_file and without_silver_execution:
        raise ValueError(
            "Cannot specify both --silver-file and --without-silver-execution."
        )

    blinded_pool_file = input_root / "candidate_pool" / "blinded_pool.jsonl"
    if not blinded_pool_file.exists():
        raise FileNotFoundError(f"Blinded pool not found at {blinded_pool_file}")

    blinded_items = [
        json.loads(line)
        for line in blinded_pool_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for item in blinded_items:
        qid = item["question_id"]
        if qid in HOLDOUT_QIDS or "holdout" in qid.lower():
            raise ValueError(f"HOLDOUT VIOLATION: item {qid} in candidate pool")

    items_by_q: dict[str, list[dict[str, Any]]] = {}
    for item in blinded_items:
        items_by_q.setdefault(item["question_id"], []).append(item)

    total_pool_count = len(blinded_items)

    # ── Load and Validate Silver if Provided ──────────────────────
    silver_records: dict[tuple[str, str], dict[str, Any]] = {}
    silver_manifest_data: dict[str, Any] = {}
    silver_hashes: dict[str, str] = {}

    if silver_file:
        silver_records, silver_manifest_data, _, silver_hashes = (
            validate_silver_execution(silver_file, blinded_items)
        )

    queue_a_items: list[dict[str, Any]] = []
    queue_b_items: list[dict[str, Any]] = []
    adjudication_items: list[dict[str, Any]] = []

    risk_routed_count = 0
    random_negative_overlap_count = 0
    silver_positive_count = 0
    silver_needs_review_count = 0

    # ── Process Questions for Queue A and Queue B ──────────────────
    for qid in sorted(items_by_q.keys()):
        q_items = items_by_q[qid]

        # Queue A: All items from candidate pool
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

        # Queue B: Risk items + Random negative control sample
        risk_b_keys: set[str] = set()

        if silver_file:
            for item in q_items:
                ps_id = item["passage_id"]
                sil = silver_records.get((qid, ps_id), {})
                grade = sil.get("relevance_grade", 0)
                needs_review = sil.get("needs_human_review", False)
                is_audit = item.get("is_outside_pool_audit", False)

                if grade > 0:
                    silver_positive_count += 1
                if needs_review:
                    silver_needs_review_count += 1

                if grade > 0 or needs_review or is_audit:
                    risk_b_keys.add(ps_id)
        else:
            # Provisional mode: outside audit items are risk
            for item in q_items:
                if item.get("is_outside_pool_audit"):
                    risk_b_keys.add(item["passage_id"])

        risk_routed_count += len(risk_b_keys)

        # Random control negative sample (20% per-question of remaining negative items)
        remaining_negatives = [
            it["passage_id"] for it in q_items if it["passage_id"] not in risk_b_keys
        ]

        target_neg_sample = max(1, int(round(0.20 * len(q_items))))
        seed = int(
            hashlib.sha256(f"{qid}:random_negative_control_seed".encode()).hexdigest()[
                :8
            ],
            16,
        )
        rng = random.Random(seed)  # noqa: S311
        sampled_negatives = set(
            rng.sample(
                remaining_negatives, k=min(len(remaining_negatives), target_neg_sample)
            )
        )
        random_negative_overlap_count += len(sampled_negatives)

        b_selected_ids = risk_b_keys | sampled_negatives

        for item in q_items:
            ps_id = item["passage_id"]
            if ps_id in b_selected_ids:
                is_risk = ps_id in risk_b_keys
                reasons = ["secondary_review_selection"]
                if item.get("is_outside_pool_audit"):
                    reasons.append("outside_pool_audit_sample")

                item_b = {
                    "question_id": qid,
                    "passage_id": ps_id,
                    "page_number": item["page_number"],
                    "text": item["text"],
                    "annotator_id": "annotator_b",
                    "priority_rank": 1,
                    "routing_reasons": reasons,
                    "is_overlap_sample": not is_risk,
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

    # ── Deterministic Reordering per Question ────────────────────
    def reorder(items: list[dict[str, Any]], annotator_id: str) -> list[dict[str, Any]]:
        by_q: dict[str, list[dict[str, Any]]] = {}
        for it in items:
            by_q.setdefault(it["question_id"], []).append(it)
        ordered: list[dict[str, Any]] = []
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

    # ── Verify Blinding ──────────────────────────────────────────
    blinding_violations: list[str] = []
    for it in ordered_a + ordered_b:
        blinding_violations.extend(_verify_blinding(it))

    if blinding_violations:
        raise ValueError(
            f"Blinding verification failed with {len(blinding_violations)} violations: "
            f"{blinding_violations[:5]}"
        )

    # ── Write Output Files ────────────────────────────────────────
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
        "\n".join(json.dumps(it, ensure_ascii=False) for it in adjudication_items)
        + "\n",
        encoding="utf-8",
    )

    random_overlap_rate = round(
        random_negative_overlap_count / max(1, total_pool_count), 4
    )

    manifest_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "queue_status": (
            "DEFINITIVE_HUMAN_REVIEW" if silver_file else "PROVISIONAL_WITHOUT_SILVER"
        ),
        "annotator_a_queue_count": len(ordered_a),
        "annotator_b_queue_count": len(ordered_b),
        "adjudication_template_count": len(adjudication_items),
        "total_candidate_pool_items": total_pool_count,
        "risk_routed_count": risk_routed_count,
        "random_negative_overlap_count": random_negative_overlap_count,
        "queue_b_total_count": len(ordered_b),
        "random_overlap_rate": random_overlap_rate,
        "planned_overlap_rate": random_overlap_rate,
        "target_overlap_range": [TARGET_OVERLAP_MIN, TARGET_OVERLAP_MAX],
        "blinding_verified": True,
        "silver_used_in_routing": bool(silver_file),
    }

    if silver_file:
        manifest_payload.update(
            {
                "silver_records_count": len(silver_records),
                "silver_needs_human_review_count": silver_needs_review_count,
                "silver_positive_count": silver_positive_count,
                "silver_file_sha256": silver_hashes["silver_file_sha256"],
                "silver_manifest_sha256": silver_hashes["silver_manifest_sha256"],
                "silver_checkpoint_sha256": silver_hashes["silver_checkpoint_sha256"],
                "silver_run_id": silver_manifest_data.get("run_id", ""),
            }
        )

    manifest_payload.update(
        {
            "authoritative_for_human_qrels": False,
            "holdout_sealed": True,
            "file_a_sha256": sha256_file(file_a),
            "file_b_sha256": sha256_file(file_b),
            "file_adj_sha256": sha256_file(file_adj),
        }
    )

    file_man.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return file_a, file_b, file_adj, file_man


def main() -> int:
    parser = argparse.ArgumentParser(description="Build human review queues")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=_REPO_ROOT / "benchmarks" / "ground_truth" / "v2" / "hybrid",
        help="Input directory containing candidate_pool/",
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
        help="Output directory for human queues",
    )
    parser.add_argument(
        "--silver-file",
        type=Path,
        default=None,
        help="Path to Machine Silver annotations.jsonl file for definitive review",
    )
    parser.add_argument(
        "--without-silver-execution",
        action="store_true",
        help="Flag for provisional review run without silver execution",
    )
    args = parser.parse_args()

    try:
        fa, fb, fadj, fman = build_human_queues(
            input_root=args.input_root,
            output_root=args.output_root,
            silver_file=args.silver_file,
            without_silver_execution=args.without_silver_execution,
        )
        print("HUMAN REVIEW QUEUES BUILT SUCCESSFULLY")
        count_a = sum(1 for _ in fa.read_text("utf-8").splitlines() if _.strip())
        count_b = sum(1 for _ in fb.read_text("utf-8").splitlines() if _.strip())
        print(f"Queue A: {fa} ({count_a} items)")
        print(f"Queue B: {fb} ({count_b} items)")
        print(f"Adjudication: {fadj}")
        print(f"Manifest: {fman}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
