"""Reproducibility and byte-by-byte scientific comparison for Slice 5A.3."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from raglab.agentic.evaluation.parity_contracts import ArmRun, ParityComparison


def compute_scientific_payload_hash(data: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash excluding temporal or ephemeral fields.

    Fields excluded from scientific hash:
    - timestamp, created_at_utc, updated_at, latency, duration, run_id, output_dir.
    """
    cleaned = _strip_ephemeral_fields(data)
    canonical_json = json.dumps(cleaned, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _strip_ephemeral_fields(obj: Any) -> Any:
    """Recursively strip ephemeral/timestamp fields for deterministic hashing."""
    ephemeral_keys = {
        "timestamp",
        "created_at_utc",
        "updated_at",
        "latency",
        "duration",
        "run_id",
        "output_dir",
        "latency_seconds",
    }
    if isinstance(obj, dict):
        return {
            k: _strip_ephemeral_fields(v)
            for k, v in obj.items()
            if k not in ephemeral_keys
        }
    if isinstance(obj, list):
        return [_strip_ephemeral_fields(item) for item in obj]
    return obj


def compare_arm_runs(
    run_1: list[ArmRun],
    run_2: list[ArmRun],
) -> tuple[list[ParityComparison], float, bool]:
    """Compare two sets of arm runs byte-by-byte for repeatability.

    Args:
        run_1: ArmRun list from first execution.
        run_2: ArmRun list from second execution.

    Returns:
        Tuple of (list of ParityComparison, repeatability_topk_identity,
        repeatability_rank_match).
    """
    map_1 = {(r.qid, r.arm_id): r for r in run_1}
    map_2 = {(r.qid, r.arm_id): r for r in run_2}

    all_keys = sorted(set(map_1.keys()) | set(map_2.keys()))
    comparisons: list[ParityComparison] = []

    exact_matches = 0
    all_ranks_matched = True

    for key in all_keys:
        qid, arm_id = key
        r1 = map_1.get(key)
        r2 = map_2.get(key)

        if not r1 or not r2:
            f_ids = [i.passage_id or "" for i in r1.retrieved_items] if r1 else []
            s_ids = [i.passage_id or "" for i in r2.retrieved_items] if r2 else []
            comparisons.append(
                ParityComparison(
                    qid=qid,
                    arm_id=arm_id,
                    first_run_ids=f_ids,
                    second_run_ids=s_ids,
                    exact_topk_match=False,
                    rank_match=False,
                    configuration_match=False,
                    corpus_match=False,
                )
            )
            all_ranks_matched = False
            continue

        ids_1 = [i.passage_id or "" for i in r1.retrieved_items]
        ids_2 = [i.passage_id or "" for i in r2.retrieved_items]

        topk_match = (ids_1 == ids_2)
        if topk_match:
            exact_matches += 1

        ranks_1 = [(i.rank, i.passage_id) for i in r1.retrieved_items]
        ranks_2 = [(i.rank, i.passage_id) for i in r2.retrieved_items]
        rank_match = (ranks_1 == ranks_2)
        if not rank_match:
            all_ranks_matched = False

        config_match = (r1.configuration_hashes == r2.configuration_hashes)
        corpus_match = (r1.corpus_hash == r2.corpus_hash)

        comparisons.append(
            ParityComparison(
                qid=qid,
                arm_id=arm_id,
                first_run_ids=ids_1,
                second_run_ids=ids_2,
                exact_topk_match=topk_match,
                rank_match=rank_match,
                configuration_match=config_match,
                corpus_match=corpus_match,
            )
        )

    total = len(all_keys)
    identity = (exact_matches / total) if total > 0 else 0.0

    return comparisons, identity, all_ranks_matched
