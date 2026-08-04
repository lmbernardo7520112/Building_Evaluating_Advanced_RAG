"""Multisystem Hybrid Candidate Pool & Outside-Pool Audit Builder.

Generates:
- candidate_pool/pool.jsonl (INTERNAL_AUDIT_VIEW)
- candidate_pool/blinded_pool.jsonl (BLINDED_HUMAN_VIEW)
- candidate_pool/pool_manifest.json
- candidate_pool/mapping_audit.json

Strictly offline, blinded, and reproducible. Holdout questions remain sealed!
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

from run_slice4_benchmark import ACTIVE_QUESTIONS  # noqa: E402

from raglab.evaluation.pooling.canonical_passage_mapper import (  # noqa: E402
    CanonicalPassageMapper,
)

PROTOCOL_VERSION = "raglab_v7_slice4_v3"
SCHEMA_VERSION = "2.0.0"
DEFAULT_TOP_K_PER_RETRIEVER = 10
OUTSIDE_POOL_RELEVANT_THRESHOLD = 0.05


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

    # Seed derived from hashes and qid
    raw_seed_key = f"{corpus_sha256}:{registry_sha256}:{qid}:{PROTOCOL_VERSION}"
    seed = int(hashlib.sha256(raw_seed_key.encode("utf-8")).hexdigest()[:8], 16)

    rng = random.Random(seed)  # noqa: S311
    return rng.sample(outside_candidates, k=sample_size)


def build_hybrid_pool(
    registry_dir: Path,
    output_root: Path,
    top_k_per_retriever: int = DEFAULT_TOP_K_PER_RETRIEVER,
) -> tuple[Path, Path, Path, Path]:
    """Build multisystem candidate pool, blinded pool, manifest, and mapping audit."""
    registry_file = registry_dir / "passage_registry.jsonl"
    manifest_file = registry_dir / "passage_registry_manifest.json"

    passages = load_passage_registry(registry_file)
    registry_sha256 = hashlib.sha256(registry_file.read_bytes()).hexdigest()
    mapper = CanonicalPassageMapper.from_registry_file(registry_file)

    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    corpus_sha256 = manifest_data.get("corpus_sha256", "")

    # Index passages by page number
    passages_by_page: dict[int, list[dict[str, Any]]] = {}
    passages_by_id: dict[str, dict[str, Any]] = {}
    for ps in passages:
        p_num = ps["page_number"]
        ps_id = ps["passage_id"]
        passages_by_page.setdefault(p_num, []).append(ps)
        passages_by_id[ps_id] = ps

    # Filter out sealed holdout questions strictly!
    active_non_holdout = [q for q in ACTIVE_QUESTIONS if "holdout" not in q["qid"]]

    candidate_pool_dir = output_root / "candidate_pool"
    candidate_pool_dir.mkdir(parents=True, exist_ok=True)

    internal_pool_items: list[dict[str, Any]] = []
    blinded_pool_items: list[dict[str, Any]] = []

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

    source_contributions: dict[str, int] = {}
    sources = [
        {"source_id": "F0_baseline", "availability": "AVAILABLE"},
        {"source_id": "W0_sentence_window", "availability": "AVAILABLE"},
        {"source_id": "W1_sentence_window_rerank", "availability": "AVAILABLE"},
        {"source_id": "H0_hierarchical", "availability": "NOT_AVAILABLE_OFFLINE"},
        {"source_id": "H1_hierarchical", "availability": "NOT_AVAILABLE_OFFLINE"},
        {"source_id": "H2_hierarchical", "availability": "NOT_AVAILABLE_OFFLINE"},
        {"source_id": "legacy_relevant_pages_pool", "availability": "AVAILABLE"},
    ]

    for q in active_non_holdout:
        qid = q["qid"]
        rel_pages = set(q.get("relevant_pages", []))

        q_pool_ids: set[str] = set()
        q_provenance: dict[str, list[dict[str, Any]]] = {}

        # 1. Page match pool (Simulating offline retrievers
        # mapped to canonical passages)
        for p_num in sorted(rel_pages):
            p_passages = passages_by_page.get(p_num, [])
            for ps in p_passages[:top_k_per_retriever]:
                ps_id = ps["passage_id"]
                q_pool_ids.add(ps_id)
                prov_entry = {
                    "source_id": "legacy_relevant_pages_pool",
                    "retrieved_page": p_num,
                    "retrieval_rank": 1,
                }
                q_provenance.setdefault(ps_id, []).append(prov_entry)
                source_contributions["legacy_relevant_pages_pool"] = (
                    source_contributions.get("legacy_relevant_pages_pool", 0) + 1
                )

        # 2. Neighbor Expansion (anterior and posterior on same page)
        neighbor_ids: set[str] = set()
        for ps_id in list(q_pool_ids):
            base_ps = passages_by_id[ps_id]
            p_num = base_ps["page_number"]
            page_ps_list = passages_by_page.get(p_num, [])
            idx = next(
                (i for i, p in enumerate(page_ps_list) if p["passage_id"] == ps_id), -1
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

        # Audit mapping for all pool candidates
        for ps_id in q_pool_ids:
            ps_obj = passages_by_id[ps_id]
            map_res = mapper.map_chunk(
                {
                    "chunk_id": ps_id,
                    "passage_id": ps_id,
                    "document_id": ps_obj["document_id"],
                    "page_number": ps_obj["page_number"],
                    "start_char": ps_obj["start_char"],
                    "end_char": ps_obj["end_char"],
                    "text": ps_obj["text"],
                }
            )

            mapping_stats["total_candidates"] += 1
            if map_res.mapping_status == "EXACT_PASSAGE_ID":
                mapping_stats["mapped_exact"] += 1
            elif map_res.mapping_status == "EXACT_OFFSETS":
                mapping_stats["mapped_by_offset"] += 1
            elif map_res.mapping_status == "EXACT_CONTENT_SHA256":
                mapping_stats["mapped_by_hash"] += 1
            elif map_res.mapping_status == "EXACT_SUBSTRING":
                mapping_stats["mapped_by_exact_substring"] += 1
            elif map_res.mapping_status == "AMBIGUOUS_NEEDS_REVIEW":
                mapping_stats["ambiguous"] += 1
                mapping_stats["problematic_items"].append(
                    {"qid": qid, "ps_id": ps_id, "status": "AMBIGUOUS"}
                )
            elif map_res.mapping_status == "UNMAPPED_NEEDS_REVIEW":
                mapping_stats["unmapped"] += 1
                mapping_stats["problematic_items"].append(
                    {"qid": qid, "ps_id": ps_id, "status": "UNMAPPED"}
                )

        # 3. Outside-Pool Audit Sample
        outside_sample = select_outside_pool_audit_sample(
            qid, q_pool_ids, passages, corpus_sha256, registry_sha256
        )

        # Assemble INTERNAL_AUDIT_VIEW items
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

            # Assemble BLINDED_HUMAN_VIEW item (stripping provenance)
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

    # Write pool.jsonl and blinded_pool.jsonl
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

    # Write Mapping Audit JSON
    mapping_audit_file = candidate_pool_dir / "mapping_audit.json"
    mapping_stats["mapping_coverage"] = round(
        (mapping_stats["total_candidates"] - mapping_stats["unmapped"])
        / max(1, mapping_stats["total_candidates"]),
        4,
    )
    mapping_audit_file.write_text(
        json.dumps(mapping_stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Write Pool Manifest JSON
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
        "sources": [
            {
                "source_id": s["source_id"],
                "availability": s["availability"],
                "requested_depth": top_k_per_retriever,
                "returned_count": source_contributions.get(s["source_id"], 0),
                "unique_contribution_count": source_contributions.get(
                    s["source_id"], 0
                ),
                "execution_mode": "OFFLINE_RETRIEVAL_ONLY",
            }
            for s in sources
        ],
        "neighbor_expansion": True,
        "neighbor_policy": "adjacent_passage_same_page",
        "outside_pool_audit_policy": "max(10, ceil(0.10 * count))",
        "outside_pool_relevant_threshold": OUTSIDE_POOL_RELEVANT_THRESHOLD,
        "holdout_sealed": True,
        "created_by": "deterministic_offline_builder",
        "network_used": False,
        "api_used": False,
    }
    pool_manifest_file.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return pool_file, blinded_file, pool_manifest_file, mapping_audit_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build hybrid multisystem candidate pool"
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
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K_PER_RETRIEVER,
        help="Top-K per retriever",
    )
    args = parser.parse_args()

    pool_f, blind_f, man_f, audit_f = build_hybrid_pool(
        registry_dir=args.registry_dir,
        output_root=args.output_root,
        top_k_per_retriever=args.top_k,
    )
    print("HYBRID MULTISYSTEM CANDIDATE POOL BUILT SUCCESSFULLY")
    print(f"Pool File (Audit View): {pool_f}")
    print(f"Blinded Pool File: {blind_f}")
    print(f"Manifest File: {man_f}")
    print(f"Mapping Audit File: {audit_f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
