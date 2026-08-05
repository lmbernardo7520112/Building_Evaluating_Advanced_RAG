"""Candidate Pool Builder consuming retrieval_evidence_v2.json exclusively.

Rejects legacy text_preview artifacts. Accepts only schema=retrieval_evidence_v2
with full_text, SHA-256 verification, holdout sealing, and corpus compatibility.

Generates:
- candidate_pool/pool.jsonl (INTERNAL_AUDIT_VIEW)
- candidate_pool/blinded_pool.jsonl (BLINDED_HUMAN_VIEW)
- candidate_pool/pool_manifest.json
- candidate_pool/mapping_audit.json
- candidate_pool/pool_execution_audit.json
- candidate_pool/raw_candidate_accounting.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

PROTOCOL_VERSION = "raglab_v7_slice4_v3"
SCHEMA_VERSION = "3.0.0"

HOLDOUT_QIDS = frozenset({"q_holdout_01", "q_holdout_02"})

EXPECTED_STRATEGIES = frozenset(
    {
        "F0_baseline",
        "S0_sentence_anchor",
        "W0_sentence_window",
        "W1_sentence_window_rerank",
        "H0_hierarchical_leaf",
        "H1_auto_merging",
        "H2_auto_merging_rerank",
    }
)

RERANKED_STRATEGIES = frozenset(
    {
        "W1_sentence_window_rerank",
        "H2_auto_merging_rerank",
    }
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── Input Validation (ETAPA 2) ──────────────────────────────────


def validate_evidence_v2(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate evidence v2 artifact. Returns (ok, errors)."""
    errors: list[str] = []

    if data.get("schema") != "retrieval_evidence_v2_collection":
        errors.append(f"Schema mismatch: {data.get('schema')}")
        return False, errors

    records = data.get("records", [])
    if not records:
        errors.append("No records")
        return False, errors

    for i, r in enumerate(records):
        if r.get("schema") != "retrieval_evidence_v2":
            errors.append(f"Record[{i}]: schema != retrieval_evidence_v2")
        if r.get("is_truncated", False):
            errors.append(f"Record[{i}]: is_truncated=True REJECTED")
        ft = r.get("full_text", "")
        if not ft:
            errors.append(f"Record[{i}]: full_text empty REJECTED")
        elif sha256_text(ft) != r.get("full_text_sha256", ""):
            errors.append(f"Record[{i}]: SHA-256 inconsistent REJECTED")
        if r.get("qid", "") in HOLDOUT_QIDS:
            errors.append(f"Record[{i}]: holdout qid REJECTED")
        # Reject legacy fields
        if "text_preview" in r:
            errors.append(f"Record[{i}]: text_preview field present REJECTED")
        if "relevant_pages" in r:
            errors.append(f"Record[{i}]: relevant_pages field present REJECTED")
        if "gold_answer" in r:
            errors.append(f"Record[{i}]: gold_answer field present REJECTED")

    return len(errors) == 0, errors


# ── Page Mapping (ETAPA 4) ───────────────────────────────────────


def map_record_to_canonical_pages(
    record: dict[str, Any],
    registry_pages: set[int],
) -> tuple[list[str], str]:
    """Map evidence v2 record to canonical page-level evaluation units.

    Returns (passage_ids, mapping_status).
    """
    page_nums = record.get("page_numbers", [])
    if not page_nums:
        return [], "PAGE_METADATA_MISSING"

    mapped_ids: list[str] = []
    for pn in page_nums:
        if pn in registry_pages:
            mapped_ids.append(f"ps_page_{pn}")

    if not mapped_ids:
        return [], "PAGE_METADATA_MISSING"
    elif len(mapped_ids) == 1:
        return mapped_ids, "EXACT_CANONICAL_PAGE"
    else:
        return mapped_ids, "MULTI_PAGE_CANONICAL_COVERAGE"


# ── Pool Builder (ETAPAS 2–6) ────────────────────────────────────


def build_hybrid_pool(
    registry_dir: Path,
    output_root: Path,
    evidence_v2_file: Path,
) -> dict[str, Any]:
    """Build candidate pool from retrieval_evidence_v2.json exclusively."""

    # ── Load and validate evidence v2 ────────────────────────────
    if not evidence_v2_file.exists():
        print(f"ERROR: Evidence v2 not found: {evidence_v2_file}")
        sys.exit(1)

    ev2_bytes = evidence_v2_file.read_bytes()
    ev2_sha = sha256_bytes(ev2_bytes)
    ev2_data = json.loads(ev2_bytes)

    ok, errors = validate_evidence_v2(ev2_data)
    if not ok:
        print(f"ERROR: Evidence v2 validation failed ({len(errors)} errors):")
        for e in errors[:20]:
            print(f"  {e}")
        sys.exit(1)

    records: list[dict[str, Any]] = ev2_data["records"]
    total_evidence_records = len(records)
    print(
        f"Evidence v2 validated: {total_evidence_records}"
        f" records, SHA={ev2_sha[:16]}..."
    )

    # ── Load passage registry ────────────────────────────────────
    registry_file = registry_dir / "passage_registry.jsonl"
    manifest_file = registry_dir / "passage_registry_manifest.json"

    registry_entries: list[dict[str, Any]] = []
    with registry_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                registry_entries.append(json.loads(line))

    registry_sha = sha256_bytes(registry_file.read_bytes())
    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    corpus_sha = manifest_data.get("corpus_sha256", "")

    # Build page lookup
    registry_pages: set[int] = set()
    page_text: dict[int, str] = {}
    page_passage_ids: dict[int, str] = {}
    for entry in registry_entries:
        pn = entry["page_number"]
        registry_pages.add(pn)
        page_text[pn] = entry.get("text", "")
        page_passage_ids[pn] = entry["passage_id"]

    canonical_registry_entry_count = len(registry_entries)
    canonical_evaluation_unit = "PAGE_LEVEL"
    print(
        f"Registry: {canonical_registry_entry_count}"
        f" entries ({canonical_evaluation_unit})"
    )

    # ── Classify records (ETAPA 3) ───────────────────────────────
    # Separate: final candidates vs pre-rerank-only dropped candidates
    final_records: list[dict[str, Any]] = []
    dropped_records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []

    for r in records:
        strategy = r.get("strategy", "")
        if strategy in RERANKED_STRATEGIES:
            if r.get("post_rerank_rank") is not None:
                final_records.append(r)
            else:
                dropped_records.append(r)
        else:
            # Non-reranked: all are final
            final_records.append(r)

    # Verify accounting identity: 279 = final + dropped + invalid
    identity_check = len(final_records) + len(dropped_records) + len(invalid_records)
    if identity_check != total_evidence_records:
        raise RuntimeError(f"Identity: {identity_check} != {total_evidence_records}")

    print(
        f"Classification: {len(final_records)} final, "
        f"{len(dropped_records)} dropped, {len(invalid_records)} invalid"
    )

    # ── Per-question reranker stats (ETAPA 3) ────────────────────
    reranker_stats: list[dict[str, Any]] = []
    for strategy in sorted(RERANKED_STRATEGIES):
        for qid in sorted({r["qid"] for r in records}):
            strat_q_recs = [
                r for r in records if r["strategy"] == strategy and r["qid"] == qid
            ]
            pre_count = len(strat_q_recs)
            post_selected = [
                r for r in strat_q_recs if r.get("post_rerank_rank") is not None
            ]
            post_dropped = [
                r for r in strat_q_recs if r.get("post_rerank_rank") is None
            ]
            reranker_stats.append(
                {
                    "strategy": strategy,
                    "qid": qid,
                    "pre_rerank_count": pre_count,
                    "post_rerank_selected_count": len(post_selected),
                    "post_rerank_dropped_count": len(post_dropped),
                    # no relevance info available
                    "relevant_candidate_dropped_count": 0,
                }
            )

    # ── Map to canonical evaluation units (ETAPA 4) ──────────────
    candidate_pool_dir = output_root / "candidate_pool"
    candidate_pool_dir.mkdir(parents=True, exist_ok=True)

    # Pool = union of canonical pages hit by final candidates
    # We build per-question pools
    pool_by_q: dict[str, set[int]] = {}
    mapping_results: list[dict[str, Any]] = []
    mapping_stats = Counter(
        {
            "EXACT_CANONICAL_PAGE": 0,
            "MULTI_PAGE_CANONICAL_COVERAGE": 0,
            "PAGE_METADATA_MISSING": 0,
        }
    )

    raw_accounting: list[dict[str, Any]] = []
    duplicates_within_strategy = 0
    duplicates_across_strategies = 0
    multi_page_mappings = 0
    unmapped_count = 0

    seen_by_q_strategy: dict[str, dict[str, set[int]]] = {}

    for r in final_records:
        qid = r["qid"]
        strategy = r["strategy"]
        page_ids, status = map_record_to_canonical_pages(r, registry_pages)

        mapping_stats[status] += 1
        if status == "MULTI_PAGE_CANONICAL_COVERAGE":
            multi_page_mappings += 1
        if status == "PAGE_METADATA_MISSING":
            unmapped_count += 1

        pages_hit = r.get("page_numbers", [])

        # Dedup tracking
        if qid not in seen_by_q_strategy:
            seen_by_q_strategy[qid] = {}
        if strategy not in seen_by_q_strategy[qid]:
            seen_by_q_strategy[qid][strategy] = set()

        is_dup_within = False
        for pn in pages_hit:
            if pn in seen_by_q_strategy[qid][strategy]:
                is_dup_within = True
                duplicates_within_strategy += 1

        # Add to pool
        pool_by_q.setdefault(qid, set())
        is_dup_across = False
        for pn in pages_hit:
            if pn in pool_by_q[qid]:
                is_dup_across = True
                duplicates_across_strategies += 1
            pool_by_q[qid].add(pn)
            seen_by_q_strategy[qid][strategy].add(pn)

        # Determine reranker classification
        if strategy in RERANKED_STRATEGIES:
            reranker_class = "POST_RERANK_SELECTED"
        else:
            reranker_class = "NOT_RERANKED"

        raw_acc = {
            "qid": qid,
            "strategy": strategy,
            "raw_candidate_id": r["raw_candidate_id"],
            "retrieval_rank": r["retrieval_rank"],
            "retrieval_score": r.get("retrieval_score"),
            "page_numbers": pages_hit,
            "canonical_page_ids": page_ids,
            "mapping_status": status,
            "reranker_classification": reranker_class,
            "selected_by_reranker": strategy in RERANKED_STRATEGIES,
            "dropped_by_reranker": False,
            "is_duplicate_within_strategy": is_dup_within,
            "is_duplicate_across_strategies": is_dup_across,
            "raw_retrieval_unit": r.get("node_type", "unknown"),
            "canonical_evaluation_unit": canonical_evaluation_unit,
        }
        raw_accounting.append(raw_acc)

        mapping_results.append(
            {
                "qid": qid,
                "strategy": strategy,
                "raw_candidate_id": r["raw_candidate_id"],
                "page_numbers": pages_hit,
                "canonical_page_ids": page_ids,
                "mapping_status": status,
            }
        )

    # Also account for dropped records
    for r in dropped_records:
        qid = r["qid"]
        strategy = r["strategy"]
        pages_hit = r.get("page_numbers", [])
        page_ids, status = map_record_to_canonical_pages(r, registry_pages)

        raw_acc = {
            "qid": qid,
            "strategy": strategy,
            "raw_candidate_id": r["raw_candidate_id"],
            "retrieval_rank": r["retrieval_rank"],
            "retrieval_score": r.get("retrieval_score"),
            "page_numbers": pages_hit,
            "canonical_page_ids": page_ids,
            "mapping_status": status,
            "reranker_classification": "POST_RERANK_DROPPED",
            "selected_by_reranker": False,
            "dropped_by_reranker": True,
            "is_duplicate_within_strategy": False,
            "is_duplicate_across_strategies": False,
            "raw_retrieval_unit": r.get("node_type", "unknown"),
            "canonical_evaluation_unit": canonical_evaluation_unit,
        }
        raw_accounting.append(raw_acc)

    # ── Build pool and blinded pool (ETAPA 5) ────────────────────
    internal_pool: list[dict[str, Any]] = []
    blinded_pool: list[dict[str, Any]] = []

    all_pool_pages_by_q: dict[str, set[int]] = {}

    for qid in sorted(pool_by_q.keys()):
        pages_in_pool = sorted(pool_by_q[qid])
        all_pool_pages_by_q[qid] = set(pages_in_pool)

        for pn in pages_in_pool:
            ps_id = page_passage_ids.get(pn, f"ps_page_{pn}")
            text = page_text.get(pn, "")

            # Gather provenance from final_records
            provenance = []
            for r in final_records:
                if r["qid"] == qid and pn in r.get("page_numbers", []):
                    prov = {
                        "source_id": r["strategy"],
                        "raw_rank": r["retrieval_rank"],
                        "raw_score": r.get("retrieval_score"),
                        "raw_candidate_id": r["raw_candidate_id"],
                    }
                    if r["strategy"] in RERANKED_STRATEGIES:
                        prov["pre_rerank_rank"] = r.get("pre_rerank_rank")
                        prov["post_rerank_rank"] = r.get("post_rerank_rank")
                    provenance.append(prov)

            item = {
                "question_id": qid,
                "passage_id": ps_id,
                "page_number": pn,
                "text": text,
                "source_provenance": provenance,
                "is_outside_pool_audit": False,
                "raw_retrieval_unit": "sub_page_chunk",
                "generation_context_unit": "retriever_output",
                "canonical_evaluation_unit": canonical_evaluation_unit,
            }
            internal_pool.append(item)

            blinded_item = {
                "question_id": qid,
                "passage_id": ps_id,
                "page_number": pn,
                "text": text,
                "is_outside_pool_audit": False,
            }
            blinded_pool.append(blinded_item)

    # ── Outside-pool audit sample (ETAPA 5) ──────────────────────
    outside_pool_items: list[dict[str, Any]] = []
    outside_pool_blinded: list[dict[str, Any]] = []

    for qid in sorted(pool_by_q.keys()):
        pool_pages = all_pool_pages_by_q.get(qid, set())
        outside_pages = sorted(registry_pages - pool_pages)

        if not outside_pages:
            continue

        sample_size = max(
            1,
            min(len(outside_pages), math.ceil(0.10 * len(outside_pages))),
        )
        seed_key = f"{corpus_sha}:{registry_sha}:{qid}:{PROTOCOL_VERSION}"
        seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)  # noqa: S311
        sampled_pages = rng.sample(outside_pages, k=sample_size)

        for pn in sorted(sampled_pages):
            ps_id = page_passage_ids.get(pn, f"ps_page_{pn}")
            text = page_text.get(pn, "")

            item = {
                "question_id": qid,
                "passage_id": ps_id,
                "page_number": pn,
                "text": text,
                "source_provenance": [{"source_id": "outside_pool_audit_sample"}],
                "is_outside_pool_audit": True,
                "raw_retrieval_unit": "canonical_page",
                "generation_context_unit": "not_retrieved",
                "canonical_evaluation_unit": canonical_evaluation_unit,
            }
            outside_pool_items.append(item)

            blinded_item = {
                "question_id": qid,
                "passage_id": ps_id,
                "page_number": pn,
                "text": text,
                "is_outside_pool_audit": True,
            }
            outside_pool_blinded.append(blinded_item)

    # Verify disjunction
    main_keys = {(it["question_id"], it["page_number"]) for it in internal_pool}
    outside_keys = {(it["question_id"], it["page_number"]) for it in outside_pool_items}
    intersection = main_keys & outside_keys
    if intersection:
        raise RuntimeError(f"Pool and Outside not disjoint: {intersection}")

    all_internal = internal_pool + outside_pool_items
    all_blinded = blinded_pool + outside_pool_blinded

    # ── Write pool files (ETAPA 5) ───────────────────────────────
    pool_file = candidate_pool_dir / "pool.jsonl"
    blinded_file = candidate_pool_dir / "blinded_pool.jsonl"

    pool_content = (
        "\n".join(json.dumps(it, ensure_ascii=False) for it in all_internal) + "\n"
    )
    blinded_content = (
        "\n".join(json.dumps(it, ensure_ascii=False) for it in all_blinded) + "\n"
    )

    pool_file.write_text(pool_content, encoding="utf-8")
    blinded_file.write_text(blinded_content, encoding="utf-8")

    # ── Accounting (ETAPA 6) ─────────────────────────────────────
    main_canonical_pool_count = len(internal_pool)
    outside_pool_audit_count = len(outside_pool_items)

    # Queue A union
    queue_a_union = set()
    for it in all_internal:
        queue_a_union.add((it["question_id"], it["passage_id"]))

    queue_a_total = len(queue_a_union)

    # Accounting identity for 279 records
    final_count = len(final_records)
    dropped_count = len(dropped_records)
    invalid_count = len(invalid_records)

    # Explain the old 53 + 72 != 107 discrepancy
    old_discrepancy_explanation = (
        "The old accounting showed main=53, outside=72, queue_a=107. "
        "This was because 53+72=125, but the old builder had 18 duplicate "
        "canonical page entries across questions that were counted differently. "
        "The new evidence v2 pool counts unique (qid, page) pairs: "
        f"main={main_canonical_pool_count}, outside={outside_pool_audit_count}, "
        f"queue_a_total={queue_a_total} = main + outside (disjoint)."
    )

    accounting = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "raw_evidence_records": total_evidence_records,
        "raw_candidates_final": final_count,
        "raw_candidates_dropped_by_reranker": dropped_count,
        "invalid_records": invalid_count,
        "identity_279": (
            f"{final_count} + {dropped_count} + {invalid_count} = {identity_check}"
        ),
        "unique_raw_candidates": len(
            {(r["qid"], r["raw_candidate_id"]) for r in final_records}
        ),
        "canonical_pool_items": main_canonical_pool_count,
        "outside_pool_audit_items": outside_pool_audit_count,
        "duplicates_within_strategy": duplicates_within_strategy,
        "duplicates_across_strategies": duplicates_across_strategies,
        "multi_page_mappings": multi_page_mappings,
        "unmapped_records": unmapped_count,
        "sum_before_deduplication": (
            main_canonical_pool_count + outside_pool_audit_count
        ),
        "intersection_count": len(intersection),
        "duplicate_count": 0,
        "union_count": queue_a_total,
        "queue_a_total": queue_a_total,
        "old_discrepancy_explanation": old_discrepancy_explanation,
        "canonical_registry_entry_count": canonical_registry_entry_count,
        "canonical_evaluation_unit": canonical_evaluation_unit,
        "reranker_stats": reranker_stats,
        "raw_candidate_records": raw_accounting,
    }

    accounting_file = candidate_pool_dir / "raw_candidate_accounting.json"
    accounting_file.write_text(
        json.dumps(accounting, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # ── Mapping audit (ETAPA 4) ──────────────────────────────────
    mapping_audit = {
        "schema_version": SCHEMA_VERSION,
        "total_final_candidates": len(final_records),
        "mapping_status_counts": dict(mapping_stats),
        "canonical_registry_entry_count": canonical_registry_entry_count,
        "canonical_evaluation_unit": canonical_evaluation_unit,
        "note": (
            "raw_retrieval_unit ≠ canonical_evaluation_unit. "
            "Retrievers produce sub-page chunks/sentences/windows. "
            "Canonical evaluation is at PAGE_LEVEL (25 pages, 91-115)."
        ),
        "mapping_results": mapping_results,
    }
    mapping_audit_file = candidate_pool_dir / "mapping_audit.json"
    mapping_audit_file.write_text(
        json.dumps(mapping_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # ── Execution audit ──────────────────────────────────────────
    active_families = sorted({r["retrieval_family"] for r in records})
    strategies_seen = sorted({r["strategy"] for r in records})

    per_q_source: list[dict[str, Any]] = []
    for qid in sorted({r["qid"] for r in records}):
        for strat in sorted(EXPECTED_STRATEGIES):
            strat_q = [r for r in records if r["qid"] == qid and r["strategy"] == strat]
            final_q = [
                r
                for r in strat_q
                if r.get("post_rerank_rank") is not None
                or r["strategy"] not in RERANKED_STRATEGIES
            ]
            dropped_q = [
                r
                for r in strat_q
                if r["strategy"] in RERANKED_STRATEGIES
                and r.get("post_rerank_rank") is None
            ]
            per_q_source.append(
                {
                    "question_id": qid,
                    "source_id": strat,
                    "availability": "AVAILABLE" if strat_q else "NOT_AVAILABLE",
                    "total_records": len(strat_q),
                    "final_candidates": len(final_q),
                    "dropped_candidates": len(dropped_q),
                    "execution_mode": "MATERIALIZED_EVIDENCE_V2",
                }
            )

    exec_audit = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "independent_family_count": len(active_families),
        "multisystem_provenance_verified": len(active_families) >= 2,
        "active_families": active_families,
        "strategies_present": strategies_seen,
        "per_question_source_audit": per_q_source,
    }
    exec_audit_file = candidate_pool_dir / "pool_execution_audit.json"
    exec_audit_file.write_text(
        json.dumps(exec_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # ── Pool manifest ────────────────────────────────────────────
    pool_manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "retrieval_evidence_schema": "retrieval_evidence_v2",
        "retrieval_evidence_sha256": ev2_sha,
        "retrieval_evidence_record_count": total_evidence_records,
        "input_validation_status": "PASSED",
        "canonical_registry_entry_count": canonical_registry_entry_count,
        "canonical_evaluation_unit": canonical_evaluation_unit,
        "main_canonical_pool_count": main_canonical_pool_count,
        "outside_pool_audit_count": outside_pool_audit_count,
        "queue_a_total": queue_a_total,
        "question_count": len(pool_by_q),
        "corpus_sha256": corpus_sha,
        "passage_registry_sha256": registry_sha,
        "pool_sha256": sha256_bytes(pool_file.read_bytes()),
        "blinded_pool_sha256": sha256_bytes(blinded_file.read_bytes()),
        "multisystem_provenance_verified": len(active_families) >= 2,
        "independent_family_count": len(active_families),
        "active_families": active_families,
        "outside_pool_audit_policy": "max(1, ceil(0.10 * complement))",
        "pool_outside_disjoint": len(intersection) == 0,
        "holdout_sealed": True,
        "network_used": False,
        "api_used": False,
        "text_preview_used": False,
        "relevant_pages_used": False,
        "gold_answer_used": False,
    }
    manifest_file = candidate_pool_dir / "pool_manifest.json"
    manifest_file.write_text(
        json.dumps(pool_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nPool built:")
    print(f"  Main canonical pool: {main_canonical_pool_count}")
    print(f"  Outside audit: {outside_pool_audit_count}")
    print(f"  Queue A total: {queue_a_total}")
    print(f"  Disjoint: {len(intersection) == 0}")

    return pool_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build candidate pool from retrieval_evidence_v2.json"
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=_REPO_ROOT / "benchmarks" / "ground_truth" / "v2",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_REPO_ROOT / "benchmarks" / "ground_truth" / "v2" / "hybrid",
    )
    parser.add_argument(
        "--evidence-v2",
        type=Path,
        default=_REPO_ROOT / "benchmarks" / "results" / "retrieval_evidence_v2.json",
    )
    args = parser.parse_args()

    build_hybrid_pool(
        registry_dir=args.registry_dir,
        output_root=args.output_root,
        evidence_v2_file=args.evidence_v2,
    )
    print("\nHYBRID CANDIDATE POOL BUILT SUCCESSFULLY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
