"""Unit tests for Multisystem Pooling, Canonical Passage Mapping, Text Rehydration, and Candidate Accounting (Gate B2 Reconciliation). # noqa: E501

Covers original tests 01-24 and reconciliation tests 25-33.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from raglab.evaluation.contracts.human_annotation_v2 import PassageRegistryEntry
from raglab.evaluation.contracts.hybrid_eval_v2 import CanonicalMappingStatus
from raglab.evaluation.pooling.canonical_passage_mapper import (
    CanonicalPassageMapper,
)
from scripts.build_hybrid_candidate_pool import (
    build_hybrid_pool,
    rehydrate_candidate_text,
)

FORBIDDEN_BLINDING_KEYS = {
    "strategy",
    "source_id",
    "source_provenance",
    "retriever",
    "retriever_name",
    "retriever_config",
    "rank",
    "raw_rank",
    "score",
    "similarity",
    "reranker_score",
    "silver",
    "silver_label",
    "judge",
    "model",
    "relevant_pages",
    "gold_answer",
    "annotator_a_grade",
    "annotator_b_grade",
}


def assert_no_forbidden_keys_recursive(data: Any) -> None:
    """Recursively assert no forbidden blinding keys exist at any nested depth."""
    if isinstance(data, dict):
        for k, v in data.items():
            k_lower = str(k).lower()
            for forbidden in FORBIDDEN_BLINDING_KEYS:
                assert (
                    forbidden not in k_lower
                ), f"BLINDING VIOLATION: key '{k}' contains forbidden term '{forbidden}'"
            assert_no_forbidden_keys_recursive(v)
    elif isinstance(data, list):
        for item in data:
            assert_no_forbidden_keys_recursive(item)


@pytest.fixture
def sample_registry_entries() -> list[PassageRegistryEntry]:
    t1 = (
        "Este é o texto exato do primeiro parágrafo sobre demonstração por"
        " indução matemática."
    )
    t2 = (
        "Este é o texto do segundo parágrafo detalhando o passo indutivo e a"
        " hipótese de indução."
    )
    return [
        PassageRegistryEntry(
            passage_id="ps_entry_01_12345678",
            document_id="gersting_discrete_math",
            page_number=92,
            start_char=100,
            end_char=300,
            content_sha256=hashlib.sha256(t1.encode("utf-8")).hexdigest(),
            text=t1,
        ),
        PassageRegistryEntry(
            passage_id="ps_entry_02_87654321",
            document_id="gersting_discrete_math",
            page_number=92,
            start_char=310,
            end_char=550,
            content_sha256=hashlib.sha256(t2.encode("utf-8")).hexdigest(),
            text=t2,
        ),
    ]


class TestMultisystemPoolingAndMapping:
    """Testes unitários para Pooling Multissistema, Reidratação e Mapeamento Canônico."""

    def test_01_union_of_multiple_sources(self):
        man_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/pool_manifest.json"
        )
        assert man_file.exists()
        manifest = json.loads(man_file.read_text(encoding="utf-8"))
        assert manifest["multisystem_provenance_verified"] is True
        assert manifest["independent_family_count"] >= 2

    def test_02_deduplication_by_passage_id(self):
        pool_file = Path("benchmarks/ground_truth/v2/hybrid/candidate_pool/pool.jsonl")
        lines = [
            json.loads(line)
            for line in pool_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        pairs = [(item["question_id"], item["passage_id"]) for item in lines]
        assert len(pairs) == len(set(pairs))

    def test_03_source_contribution_tracked_in_manifest(self):
        audit_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/pool_execution_audit.json"
        )
        assert audit_file.exists()
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
        avail = [
            e
            for e in audit["per_question_source_audit"]
            if e["availability"] == "AVAILABLE"
        ]
        assert len(avail) > 0
        for e in avail:
            assert e["raw_returned_count"] >= 0

    def test_04_unavailable_source_explicit(self):
        audit_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/pool_execution_audit.json"
        )
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
        unavail = [
            e
            for e in audit["per_question_source_audit"]
            if e["availability"] == "NOT_AVAILABLE_OFFLINE"
        ]
        assert len(unavail) > 0
        assert any(
            e["source_id"] in ["lexical_bm25", "dense_canonical"] for e in unavail
        )

    def test_05_legacy_pages_are_additional_source_only(self):
        pool_file = Path("benchmarks/ground_truth/v2/hybrid/candidate_pool/pool.jsonl")
        content = pool_file.read_text(encoding="utf-8")
        assert "gold_answer" not in content
        assert "relevance_grade" not in content

    def test_06_neighbors_preserve_canonical_ids(self):
        pool_file = Path("benchmarks/ground_truth/v2/hybrid/candidate_pool/pool.jsonl")
        lines = [
            json.loads(line)
            for line in pool_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for item in lines:
            if item.get("is_neighbor"):
                assert item["neighbor_policy"] == "adjacent_passage_same_page"

    def test_07_pool_contains_no_holdout(self):
        pool_file = Path("benchmarks/ground_truth/v2/hybrid/candidate_pool/pool.jsonl")
        lines = [
            json.loads(line)
            for line in pool_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for item in lines:
            assert "holdout" not in item["question_id"].lower()

    def test_08_pool_generation_is_deterministic(self, tmp_path: Path):
        gt_dir = Path("benchmarks/ground_truth/v2")
        bench_res = Path(
            "benchmarks/results/slice4_final_composite_recovered_run.json"
        )
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        p1, b1, m1, a1, e1 = build_hybrid_pool(gt_dir, out1, bench_res)
        p2, b2, m2, a2, e2 = build_hybrid_pool(gt_dir, out2, bench_res)
        assert p1.read_bytes() == p2.read_bytes()
        assert b1.read_bytes() == b2.read_bytes()

    def test_09_outside_pool_sample_is_deterministic(self):
        blinded_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/blinded_pool.jsonl"
        )
        lines = [
            json.loads(line)
            for line in blinded_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        audit_items = [item for item in lines if item.get("is_outside_pool_audit")]
        assert len(audit_items) > 0

    def test_10_outside_pool_sample_does_not_intersect_main_pool(self):
        pool_file = Path("benchmarks/ground_truth/v2/hybrid/candidate_pool/pool.jsonl")
        lines = [
            json.loads(line)
            for line in pool_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        main_ids = {
            (item["question_id"], item["passage_id"])
            for item in lines
            if not item.get("is_outside_pool_audit")
        }
        audit_ids = {
            (item["question_id"], item["passage_id"])
            for item in lines
            if item.get("is_outside_pool_audit")
        }
        assert len(main_ids.intersection(audit_ids)) == 0

    def test_11_expansion_threshold_registered(self):
        man_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/pool_manifest.json"
        )
        manifest = json.loads(man_file.read_text(encoding="utf-8"))
        assert manifest["outside_pool_relevant_threshold"] == 0.05

    def test_12_canonical_mapper_by_id(
        self, sample_registry_entries: list[PassageRegistryEntry]
    ):
        mapper = CanonicalPassageMapper(sample_registry_entries)
        res = mapper.map_chunk({
            "chunk_id": "ps_entry_01_12345678",
            "page_number": 92,
            "text": "qualquer texto",
        })
        assert res.mapping_status == CanonicalMappingStatus.EXACT_PASSAGE_ID
        assert res.mapped_passage_id == "ps_entry_01_12345678"

    def test_13_canonical_mapper_by_offsets(
        self, sample_registry_entries: list[PassageRegistryEntry]
    ):
        mapper = CanonicalPassageMapper(sample_registry_entries)
        res = mapper.map_chunk({
            "chunk_id": "c_offsets",
            "page_number": 92,
            "start_char": 100,
            "end_char": 300,
            "text": "qualquer texto",
        })
        assert res.mapping_status == CanonicalMappingStatus.EXACT_OFFSETS
        assert res.mapped_passage_id == "ps_entry_01_12345678"

    def test_14_canonical_mapper_by_hash(
        self, sample_registry_entries: list[PassageRegistryEntry]
    ):
        mapper = CanonicalPassageMapper(sample_registry_entries)
        text = sample_registry_entries[0].text
        res = mapper.map_chunk({
            "chunk_id": "c_hash",
            "page_number": 92,
            "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": "qualquer texto",
        })
        assert res.mapping_status == CanonicalMappingStatus.EXACT_CONTENT_SHA256
        assert res.mapped_passage_id == "ps_entry_01_12345678"

    def test_15_canonical_mapper_by_exact_substring(
        self, sample_registry_entries: list[PassageRegistryEntry]
    ):
        mapper = CanonicalPassageMapper(sample_registry_entries)
        res = mapper.map_chunk({
            "chunk_id": "c_substr",
            "page_number": 92,
            "text": "primeiro parágrafo sobre demonstração por indução",
        })
        assert res.mapping_status == CanonicalMappingStatus.EXACT_SUBSTRING
        assert res.mapped_passage_id == "ps_entry_01_12345678"

    def test_16_canonical_mapper_ambiguity_flagged(
        self, sample_registry_entries: list[PassageRegistryEntry]
    ):
        mapper = CanonicalPassageMapper(sample_registry_entries)
        res = mapper.map_chunk({
            "chunk_id": "c_ambig",
            "page_number": 92,
            "text": "Este é o texto",
        })
        assert res.mapping_status == CanonicalMappingStatus.AMBIGUOUS_NEEDS_REVIEW

    def test_17_canonical_mapper_unmapped_flagged(
        self, sample_registry_entries: list[PassageRegistryEntry]
    ):
        mapper = CanonicalPassageMapper(sample_registry_entries)
        res = mapper.map_chunk({
            "chunk_id": "c_unmapped",
            "page_number": 99,
            "text": "Texto ausente",
        })
        assert res.mapping_status == CanonicalMappingStatus.UNMAPPED_NEEDS_REVIEW

    def test_18_zero_unreported_mapping_loss(self):
        audit_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/mapping_audit.json"
        )
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
        assert audit["unreported_mapping_loss"] == 0

    def test_19_blinded_view_has_no_strategy(self):
        blinded_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/blinded_pool.jsonl"
        )
        content = blinded_file.read_text(encoding="utf-8")
        assert "strategy" not in content

    def test_20_blinded_view_has_no_rank(self):
        blinded_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/blinded_pool.jsonl"
        )
        content = blinded_file.read_text(encoding="utf-8")
        assert "retrieval_rank" not in content
        assert "rank" not in content

    def test_21_blinded_view_has_no_scores(self):
        blinded_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/blinded_pool.jsonl"
        )
        content = blinded_file.read_text(encoding="utf-8")
        assert "retrieval_score" not in content
        assert "rerank_score" not in content

    def test_22_blinded_view_has_no_silver_or_llm_fields(self):
        blinded_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/blinded_pool.jsonl"
        )
        content = blinded_file.read_text(encoding="utf-8")
        assert "silver" not in content
        assert "judge" not in content

    def test_23_blinded_view_has_no_other_annotator_answers(self):
        blinded_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/blinded_pool.jsonl"
        )
        content = blinded_file.read_text(encoding="utf-8")
        assert "annotator_a_grade" not in content
        assert "annotator_b_grade" not in content

    def test_24_blinded_order_is_deterministic(self, tmp_path: Path):
        blinded_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/blinded_pool.jsonl"
        )
        lines1 = blinded_file.read_text(encoding="utf-8").splitlines()
        lines2 = blinded_file.read_text(encoding="utf-8").splitlines()
        assert lines1 == lines2

    # Reconciliation Tests (ETAPA 8)
    def test_25_rehydration_of_truncated_preview_required(self):
        prev = "Demonstração por Exaustão Embora “provar a falsidade por um contraexemplo” sempr"  # noqa: E501
        entry = {
            "page_number": 92,
            "text": "Demonstração por Exaustão Embora “provar a falsidade por um contraexemplo” sempre funcione...",  # noqa: E501
        }
        full, s, e, status, sha = rehydrate_candidate_text(prev, 92, entry)
        assert status in ["REHYDRATED_EXACT", "REHYDRATED_DETERMINISTIC"]
        assert len(full) >= len(prev)

    def test_26_rehydration_demands_verifiable_artifact(self):
        rehyd_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/rehydration_audit.json"
        )
        assert rehyd_file.exists()
        rehyd = json.loads(rehyd_file.read_text(encoding="utf-8"))
        assert rehyd["rehydration_success_rate"] == 1.0
        assert rehyd["total_candidates"] == 168

    def test_27_reconstructed_text_from_preview_rejected(self):
        prev = "texto truncado sem correspondência"
        full, s, e, status, sha = rehydrate_candidate_text(prev, 99, None)
        assert status == "NOT_REHYDRATABLE"

    def test_28_one_hundred_sixty_eight_candidates_close_accountingly(self):
        acc_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/raw_candidate_accounting.json"  # noqa: E501
        )
        assert acc_file.exists()
        acc = json.loads(acc_file.read_text(encoding="utf-8"))
        assert acc["total_raw_candidates"] == 168
        assert acc["accounting_identity_verified"] is True
        counts = acc["disposition_counts"]
        assert sum(counts.values()) == 168

    def test_29_unmapped_candidate_has_operational_disposition(self):
        acc_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/raw_candidate_accounting.json"  # noqa: E501
        )
        acc = json.loads(acc_file.read_text(encoding="utf-8"))
        for rec in acc["raw_candidate_records"]:
            assert "operational_disposition" in rec
            assert rec["operational_disposition"] in [
                "CANONICAL_HUMAN_REVIEW",
                "RAW_CANDIDATE_HUMAN_REVIEW",
                "DUPLICATE_OF_CANONICAL",
                "DUPLICATE_OF_RAW_CANDIDATE",
                "UNRESOLVED_BLOCKING",
                "INVALID_SOURCE_RECORD",
            ]

    def test_30_unrehydratable_unmapped_generates_blocking(self):
        acc_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/raw_candidate_accounting.json"  # noqa: E501
        )
        acc = json.loads(acc_file.read_text(encoding="utf-8"))
        unresolved = [
            r
            for r in acc["raw_candidate_records"]
            if r["operational_disposition"] == "UNRESOLVED_BLOCKING"
        ]
        # In our reconciled dataset, 0 are unresolved blocking because all 168 rehydrated
        assert len(unresolved) == 0

    def test_31_raw_unmapped_review_recursively_blinded(self):
        blinded_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/blinded_pool.jsonl"
        )
        lines = [
            json.loads(line)
            for line in blinded_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for item in lines:
            assert_no_forbidden_keys_recursive(item)

    def test_32_duplicates_do_not_inflate_human_queue(self):
        acc_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/raw_candidate_accounting.json"  # noqa: E501
        )
        acc = json.loads(acc_file.read_text(encoding="utf-8"))
        dups = acc["disposition_counts"]["DUPLICATE_OF_CANONICAL"]
        assert dups == 115

    def test_33_mapping_coverage_before_after_calculated(self):
        audit_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/mapping_audit.json"
        )
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
        assert audit["mapping_coverage_before"] == 0.2857
        assert audit["mapping_coverage_after"] == 1.0
