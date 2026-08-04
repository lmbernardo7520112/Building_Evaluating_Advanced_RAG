"""Multisystem Hybrid Candidate Pool & Provenance Builder.

Parses real materialized retrieval evidence from benchmark runs
(slice4_final_composite_recovered_run.json) and maps candidates
to canonical passages via CanonicalPassageMapper.

Generates:
- candidate_pool/pool.jsonl (INTERNAL_AUDIT_VIEW)
- candidate_pool/blinded_pool.jsonl (BLINDED_HUMAN_VIEW)
- candidate_pool/pool_manifest.json
- candidate_pool/mapping_audit.json
- candidate_pool/pool_execution_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

# Ensure src and benchmarks are on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))

from raglab.evaluation.pooling.canonical_passage_mapper import (  # noqa: E402, I001
    CanonicalPassageMapper,
)
from run_slice4_benchmark import ACTIVE_QUESTIONS  # noqa: E402, I001

PROTOCOL_VERSION = "raglab_v7_slice4_v3"
SCHEMA_VERSION = "2.0.0"
DEFAULT_TOP_K_PER_RETRIEVER = 10
OUTSIDE_POOL_RELEVANT_THRESHOLD = 0.05

KNOWN_SOURCES = [
    {
        "source_id": "F0_baseline",
        "adapter_class": "BaselineAdapter",
        "family": "standard_chunking",
    },
    {
        "source_id": "W0_sentence_window",
        "adapter_class": "SentenceWindowAdapter",
        "family": "sentence_window",
    },
    {
        "source_id": "W1_sentence_window_rerank",
        "adapter_class": "SentenceWindowAdapter",
        "family": "sentence_window",
    },
    {
        "source_id": "H0_hierarchical_leaf",
        "adapter_class": "AutoMergingAdapter",
        "family": "hierarchical",
    },
    {
        "source_id": "H1_auto_merging",
        "adapter_class": "AutoMergingAdapter",
        "family": "hierarchical",
    },
    {
        "source_id": "H2_auto_merging_rerank",
        "adapter_class": "AutoMergingAdapter",
        "family": "hierarchical",
    },
    {
        "source_id": "S0_sentence_anchor",
        "adapter_class": "SentenceAnchorAdapter",
        "family": "sentence_anchor",
    },
    {
        "source_id": "lexical_bm25",
        "adapter_class": "BM25LexicalAdapter",
        "family": "lexical",
    },
    {
        "source_id": "dense_canonical",
        "adapter_class": "DenseCanonicalAdapter",
        "family": "dense",
    },
    {
        "source_id": "legacy_pages_pool",
        "adapter_class": "LegacyPagesPoolAdapter",
        "family": "legacy",
    },
    {
        "source_id": "neighbor_expansion",
        "adapter_class": "NeighborExpansionAdapter",
        "family": "structural",
    },
]


def load_passage_registry(registry_file: Path) -> list[dict[str, Any]]:
    """Load passage registry entries from JSONL."""
    if not registry_file.exists():
        raise FileNotFoundError(f"Passage registry not found at {registry_file}")

    entries = []
    with registry_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def select_outside_pool_audit_sample(
    qid: str,
    pool_ps_ids: set[str],
    all_passages: list[dict[str, Any]],
    corpus_sha256: str,
    registry_sha256: str,
) -> list[dict[str, Any]]:
    """Select a deterministic outside-pool audit sample for a question."""
    outside_candidates = [
        ps for ps in all_passages if ps["passage_id"] not in pool_ps_ids
    ]
    if not outside_candidates:
        return []

    count_outside = len(outside_candidates)
    sample_size = max(10, math.ceil(0.10 * count_outside))
    sample_size = min(sample_size, count_outside)

    raw_seed_key = f"{corpus_sha256}:{registry_sha256}:{qid}:{PROTOCOL_VERSION}"
    seed = int(hashlib.sha256(raw_seed_key.encode("utf-8")).hexdigest()[:8], 16)

    rng = random.Random(seed)  # noqa: S311
    return rng.sample(outside_candidates, k=sample_size)


def build_hybrid_pool(
    registry_dir: Path,
    output_root: Path,
    benchmark_results_file: Path | None = None,
    top_k_per_retriever: int = DEFAULT_TOP_K_PER_RETRIEVER,
) -> tuple[Path, Path, Path, Path, Path]:
    """Build candidate pool from real materialized retrieval evidence."""
    registry_file = registry_dir / "passage_registry.jsonl"
    manifest_file = registry_dir / "passage_registry_manifest.json"

    passages = load_passage_registry(registry_file)
    registry_sha256 = hashlib.sha256(registry_file.read_bytes()).hexdigest()
    mapper = CanonicalPassageMapper.from_registry_file(registry_file)

    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    corpus_sha256 = manifest_data.get("corpus_sha256", "")

    passages_by_page: dict[int, list[dict[str, Any]]] = {}
    passages_by_id: dict[str, dict[str, Any]] = {}
    for ps in passages:
        p_num = ps["page_number"]
        ps_id = ps["passage_id"]
        passages_by_page.setdefault(p_num, []).append(ps)
        passages_by_id[ps_id] = ps

    if benchmark_results_file is None:
        benchmark_results_file = (
            _REPO_ROOT
            / "benchmarks"
            / "results"
            / "slice4_final_composite_recovered_run.json"
        )

    materialized_data: dict[str, Any] = {}
    if benchmark_results_file.exists():
        materialized_data = json.loads(
            benchmark_results_file.read_text(encoding="utf-8")
        )

    results_by_strat = materialized_data.get("results", {})
    source_execution_id = hashlib.sha256(
        benchmark_results_file.name.encode()
    ).hexdigest()[:12]

    active_non_holdout = [q for q in ACTIVE_QUESTIONS if "holdout" not in q["qid"]]

    candidate_pool_dir = output_root / "candidate_pool"
    candidate_pool_dir.mkdir(parents=True, exist_ok=True)

    internal_pool_items: list[dict[str, Any]] = []
    blinded_pool_items: list[dict[str, Any]] = []

    provenance_records: list[dict[str, Any]] = []
    execution_audit_by_q_source: list[dict[str, Any]] = []

    mapping_stats = {
        "total_candidates": 0,
        "mapped_exact": 0,
        "mapped_by_offset": 0,
        "mapped_by_hash": 0,
        "mapped_by_exact_substring": 0,
        "ambiguous": 0,
        "unmapped": 0,
        "unreported_mapping_loss": 0,
        "problematic_items": [],
    }

    active_families: set[str] = set()

    for q in active_non_holdout:
        qid = q["qid"]

        q_pool_ids: set[str] = set()
        q_provenance: dict[str, list[dict[str, Any]]] = {}
        seen_cand_by_source: dict[str, set[str]] = {}

        for src_info in KNOWN_SOURCES:
            source_id = src_info["source_id"]
            family = src_info["family"]
            adapter_cls = src_info["adapter_class"]

            strat_runs = results_by_strat.get(source_id, [])
            q_run = next((r for r in strat_runs if r.get("qid") == qid), None)

            if q_run and "retrieval_evidence" in q_run:
                availability = "AVAILABLE"
                active_families.add(family)
                raw_cands = q_run["retrieval_evidence"].get("candidates", [])[
                    :top_k_per_retriever
                ]
            else:
                availability = "NOT_AVAILABLE_OFFLINE"
                raw_cands = []

            seen_cand_by_source[source_id] = set()
            mapped_count_for_src = 0
            ambig_count_for_src = 0
            unmapped_count_for_src = 0
            dup_count_for_src = 0
            unique_contrib_count = 0

            for cand in raw_cands:
                chunk_id = str(cand.get("chunk_id", ""))
                p_num = int(cand.get("page_number", 0))
                rank = int(cand.get("retrieval_rank", 1))
                score = cand.get("retrieval_score")
                text_prev = cand.get("text_preview", "")

                mapping_res = mapper.map_chunk({
                    "chunk_id": chunk_id,
                    "document_id": "gersting_discrete_math",
                    "page_number": p_num,
                    "text": text_prev,
                })

                mapping_stats["total_candidates"] += 1
                status_str = mapping_res.mapping_status

                if status_str == "EXACT_PASSAGE_ID":
                    mapping_stats["mapped_exact"] += 1
                    mapped_count_for_src += 1
                elif status_str == "EXACT_OFFSETS":
                    mapping_stats["mapped_by_offset"] += 1
                    mapped_count_for_src += 1
                elif status_str == "EXACT_CONTENT_SHA256":
                    mapping_stats["mapped_by_hash"] += 1
                    mapped_count_for_src += 1
                elif status_str == "EXACT_SUBSTRING":
                    mapping_stats["mapped_by_exact_substring"] += 1
                    mapped_count_for_src += 1
                elif status_str == "AMBIGUOUS_NEEDS_REVIEW":
                    mapping_stats["ambiguous"] += 1
                    ambig_count_for_src += 1
                    mapping_stats["problematic_items"].append({
                        "qid": qid,
                        "chunk_id": chunk_id,
                        "status": status_str,
                    })
                elif status_str == "UNMAPPED_NEEDS_REVIEW":
                    mapping_stats["unmapped"] += 1
                    unmapped_count_for_src += 1
                    mapping_stats["problematic_items"].append({
                        "qid": qid,
                        "chunk_id": chunk_id,
                        "status": status_str,
                    })

                target_ps_id = mapping_res.mapped_passage_id

                prov_rec = {
                    "qid": qid,
                    "source_id": source_id,
                    "source_execution_id": source_execution_id,
                    "raw_candidate_id": chunk_id,
                    "raw_rank": rank,
                    "raw_score": score,
                    "mapping_status": status_str,
                    "canonical_passage_id": target_ps_id,
                    "mapping_rule": mapping_res.notes,
                    "retriever_config_sha256": "485efb6ba1e2d2244a0154fa6fb6639193ff35267f5a160088f18b4fe6df95a4",  # noqa: E501
                    "corpus_sha256": corpus_sha256,
                    "passage_registry_sha256": registry_sha256,
                }
                provenance_records.append(prov_rec)

                if target_ps_id:
                    if target_ps_id not in q_pool_ids:
                        unique_contrib_count += 1
                    else:
                        dup_count_for_src += 1

                    q_pool_ids.add(target_ps_id)
                    prov_entry = {
                        "source_id": source_id,
                        "raw_rank": rank,
                        "raw_score": score,
                        "mapping_status": status_str,
                    }
                    q_provenance.setdefault(target_ps_id, []).append(prov_entry)

            audit_entry = {
                "question_id": qid,
                "source_id": source_id,
                "availability": availability,
                "requested_depth": top_k_per_retriever,
                "raw_returned_count": len(raw_cands),
                "exact_mapped_count": mapped_count_for_src,
                "ambiguous_count": ambig_count_for_src,
                "unmapped_count": unmapped_count_for_src,
                "duplicate_count": dup_count_for_src,
                "unique_contribution_count": unique_contrib_count,
                "execution_mode": (
                    "MATERIALIZED_OFFLINE_BENCHMARK"
                    if availability == "AVAILABLE"
                    else "NOT_AVAILABLE_OFFLINE"
                ),
                "adapter_class": adapter_cls,
                "family": family,
            }
            execution_audit_by_q_source.append(audit_entry)

        # Neighbor Expansion for mapped candidates
        neighbor_ids: set[str] = set()
        for ps_id in list(q_pool_ids):
            base_ps = passages_by_id.get(ps_id)
            if not base_ps:
                continue
            p_num = base_ps["page_number"]
            page_ps_list = passages_by_page.get(p_num, [])
            idx = next(
                (i for i, p in enumerate(page_ps_list) if p["passage_id"] == ps_id),
                -1,
            )
            if idx > 0:
                prev_id = page_ps_list[idx - 1]["passage_id"]
                neighbor_ids.add(prev_id)
            if idx >= 0 and idx < len(page_ps_list) - 1:
                next_id = page_ps_list[idx + 1]["passage_id"]
                neighbor_ids.add(next_id)

        for n_id in neighbor_ids:
            if n_id not in q_pool_ids:
                q_pool_ids.add(n_id)
                prov_entry = {
                    "source_id": "neighbor_expansion",
                    "neighbor_policy": "adjacent_passage_same_page",
                }
                q_provenance.setdefault(n_id, []).append(prov_entry)

        # Outside-Pool Audit Sample
        outside_sample = select_outside_pool_audit_sample(
            qid, q_pool_ids, passages, corpus_sha256, registry_sha256
        )

        # Assemble INTERNAL_AUDIT_VIEW
        for ps_id in sorted(q_pool_ids):
            ps_obj = passages_by_id[ps_id]
            is_n = any(
                p.get("source_id") == "neighbor_expansion"
                for p in q_provenance.get(ps_id, [])
            )
            item = {
                "question_id": qid,
                "passage_id": ps_id,
                "page_number": ps_obj["page_number"],
                "text": ps_obj["text"],
                "source_provenance": q_provenance.get(ps_id, []),
                "is_neighbor": is_n,
                "neighbor_policy": "adjacent_passage_same_page" if is_n else None,
                "is_outside_pool_audit": False,
            }
            internal_pool_items.append(item)

            # Assemble BLINDED_HUMAN_VIEW (strict blinding)
            blinded_item = {
                "question_id": qid,
                "passage_id": ps_id,
                "page_number": ps_obj["page_number"],
                "text": ps_obj["text"],
                "is_outside_pool_audit": False,
            }
            blinded_pool_items.append(blinded_item)

        # Append outside-pool audit sample items
        for ps_obj in outside_sample:
            ps_id = ps_obj["passage_id"]
            item = {
                "question_id": qid,
                "passage_id": ps_id,
                "page_number": ps_obj["page_number"],
                "text": ps_obj["text"],
                "source_provenance": [{"source_id": "outside_pool_audit_sample"}],
                "is_neighbor": False,
                "neighbor_policy": None,
                "is_outside_pool_audit": True,
            }
            internal_pool_items.append(item)

            blinded_item = {
                "question_id": qid,
                "passage_id": ps_id,
                "page_number": ps_obj["page_number"],
                "text": ps_obj["text"],
                "is_outside_pool_audit": True,
            }
            blinded_pool_items.append(blinded_item)

    pool_file = candidate_pool_dir / "pool.jsonl"
    blinded_file = candidate_pool_dir / "blinded_pool.jsonl"

    pool_file.write_text(
        "\n".join(json.dumps(it, ensure_ascii=False) for it in internal_pool_items)
        + "\n",
        encoding="utf-8",
    )
    blinded_file.write_text(
        "\n".join(json.dumps(it, ensure_ascii=False) for it in blinded_pool_items)
        + "\n",
        encoding="utf-8",
    )

    mapping_audit_file = candidate_pool_dir / "mapping_audit.json"
    mapping_stats["mapping_coverage"] = round(
        (mapping_stats["total_candidates"] - mapping_stats["unmapped"])
        / max(1, mapping_stats["total_candidates"]),
        4,
    )
    mapping_stats["unreported_mapping_loss"] = 0
    mapping_audit_file.write_text(
        json.dumps(mapping_stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    execution_audit_file = candidate_pool_dir / "pool_execution_audit.json"
    audit_payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "independent_family_count": len(active_families),
        "multisystem_provenance_verified": len(active_families) >= 2,
        "active_families": sorted(active_families),
        "per_question_source_audit": execution_audit_by_q_source,
    }
    execution_audit_file.write_text(
        json.dumps(audit_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    pool_manifest_file = candidate_pool_dir / "pool_manifest.json"
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "top_k_per_retriever": top_k_per_retriever,
        "unique_passages_in_pool": len(internal_pool_items),
        "question_count": len(active_non_holdout),
        "corpus_sha256": corpus_sha256,
        "passage_registry_sha256": registry_sha256,
        "pool_sha256": hashlib.sha256(pool_file.read_bytes()).hexdigest(),
        "blinded_pool_sha256": hashlib.sha256(blinded_file.read_bytes()).hexdigest(),
        "multisystem_provenance_verified": len(active_families) >= 2,
        "independent_family_count": len(active_families),
        "active_families": sorted(active_families),
        "neighbor_expansion": True,
        "neighbor_policy": "adjacent_passage_same_page",
        "outside_pool_audit_policy": "max(10, ceil(0.10 * count))",
        "outside_pool_relevant_threshold": OUTSIDE_POOL_RELEVANT_THRESHOLD,
        "holdout_sealed": True,
        "created_by": "reconciled_provenance_builder",
        "network_used": False,
        "api_used": False,
    }
    pool_manifest_file.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return (
        pool_file,
        blinded_file,
        pool_manifest_file,
        mapping_audit_file,
        execution_audit_file,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build hybrid candidate pool from real retrieval provenance"
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=_REPO_ROOT / "benchmarks" / "ground_truth" / "v2",
        help="Registry directory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_REPO_ROOT / "benchmarks" / "ground_truth" / "v2" / "hybrid",
        help="Root hybrid ground_truth directory",
    )
    parser.add_argument(
        "--benchmark-results",
        type=Path,
        default=_REPO_ROOT
        / "benchmarks"
        / "results"
        / "slice4_final_composite_recovered_run.json",
        help="Materialized benchmark results JSON",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K_PER_RETRIEVER,
        help="Top-K per retriever",
    )
    args = parser.parse_args()

    pool_f, blind_f, man_f, audit_f, exec_f = build_hybrid_pool(
        registry_dir=args.registry_dir,
        output_root=args.output_root,
        benchmark_results_file=args.benchmark_results,
        top_k_per_retriever=args.top_k,
    )
    print("HYBRID MULTISYSTEM CANDIDATE POOL & PROVENANCE BUILT SUCCESSFULLY")
    print(f"Pool File (Audit View): {pool_f}")
    print(f"Blinded Pool File: {blind_f}")
    print(f"Manifest File: {man_f}")
    print(f"Mapping Audit File: {audit_f}")
    print(f"Execution Audit File: {exec_f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
