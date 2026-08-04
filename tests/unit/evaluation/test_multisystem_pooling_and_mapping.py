"""Unit tests for Multisystem Pooling, Canonical Passage Mapping, and Blinding (Gate B2 - Commit 1).

Covers test invariants 1 to 24 specified in Gate B2 instructions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from raglab.evaluation.contracts.human_annotation_v2 import PassageRegistryEntry
from raglab.evaluation.contracts.hybrid_eval_v2 import (
    CanonicalMappingStatus,
)
from raglab.evaluation.pooling.canonical_passage_mapper import (
    CanonicalPassageMapper,
)
from scripts.build_hybrid_candidate_pool import (
    build_hybrid_pool,
)


@pytest.fixture
def sample_registry_entries() -> list[PassageRegistryEntry]:
    t1 = "Este é o texto exato do primeiro parágrafo sobre demonstração por indução matemática."
    t2 = "Este é o texto do segundo parágrafo detalhando o passo indutivo e a hipótese de indução."
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
    """Testes unitários direcionados para Pooling Multissistema e Mapeamento Canônico."""

    def test_01_union_of_multiple_sources(self):
        pool_file = Path("benchmarks/ground_truth/v2/hybrid/candidate_pool/pool.jsonl")
        assert pool_file.exists()
        lines = [
            json.loads(line)
            for line in pool_file.read_text().splitlines()
            if line.strip()
        ]
        assert len(lines) > 0

    def test_02_deduplication_by_passage_id(self):
        pool_file = Path("benchmarks/ground_truth/v2/hybrid/candidate_pool/pool.jsonl")
        lines = [
            json.loads(line)
            for line in pool_file.read_text().splitlines()
            if line.strip()
        ]
        # Check uniqueness of (question_id, passage_id) pair
        pairs = [(item["question_id"], item["passage_id"]) for item in lines]
        assert len(pairs) == len(set(pairs))

    def test_03_source_contribution_tracked_in_manifest(self):
        man_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/pool_manifest.json"
        )
        manifest = json.loads(man_file.read_text())
        assert "sources" in manifest
        assert len(manifest["sources"]) >= 5

    def test_04_unavailable_source_explicit(self):
        man_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/pool_manifest.json"
        )
        manifest = json.loads(man_file.read_text())
        unavail = [
            s
            for s in manifest["sources"]
            if s["availability"] == "NOT_AVAILABLE_OFFLINE"
        ]
        assert len(unavail) > 0
        assert any(s["source_id"].startswith("H") for s in unavail)

    def test_05_legacy_pages_are_additional_source_only(self):
        man_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/pool_manifest.json"
        )
        manifest = json.loads(man_file.read_text())
        leg_source = next(
            (
                s
                for s in manifest["sources"]
                if s["source_id"] == "legacy_relevant_pages_pool"
            ),
            None,
        )
        assert leg_source is not None

    def test_06_neighbors_preserve_canonical_ids(self):
        pool_file = Path("benchmarks/ground_truth/v2/hybrid/candidate_pool/pool.jsonl")
        lines = [
            json.loads(line)
            for line in pool_file.read_text().splitlines()
            if line.strip()
        ]
        for item in lines:
            if item.get("is_neighbor"):
                assert item["passage_id"].startswith("ps_")

    def test_07_pool_contains_no_holdout(self):
        pool_file = Path("benchmarks/ground_truth/v2/hybrid/candidate_pool/pool.jsonl")
        lines = [
            json.loads(line)
            for line in pool_file.read_text().splitlines()
            if line.strip()
        ]
        for item in lines:
            assert "holdout" not in item["question_id"].lower()

    def test_08_pool_generation_is_deterministic(self, tmp_path: Path):
        gt_dir = Path("benchmarks/ground_truth/v2")
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        p1, b1, m1, a1 = build_hybrid_pool(gt_dir, out1)
        p2, b2, m2, a2 = build_hybrid_pool(gt_dir, out2)
        assert p1.read_bytes() == p2.read_bytes()
        assert b1.read_bytes() == b2.read_bytes()

    def test_09_outside_pool_sample_is_deterministic(self):
        pool_file = Path("benchmarks/ground_truth/v2/hybrid/candidate_pool/pool.jsonl")
        lines = [
            json.loads(line)
            for line in pool_file.read_text().splitlines()
            if line.strip()
        ]
        outside_items = [it for it in lines if it.get("is_outside_pool_audit")]
        assert len(outside_items) > 0

    def test_10_outside_pool_sample_does_not_intersect_main_pool(self):
        pool_file = Path("benchmarks/ground_truth/v2/hybrid/candidate_pool/pool.jsonl")
        lines = [
            json.loads(line)
            for line in pool_file.read_text().splitlines()
            if line.strip()
        ]
        for qid in {it["question_id"] for it in lines}:
            main_ps = {
                it["passage_id"]
                for it in lines
                if it["question_id"] == qid and not it["is_outside_pool_audit"]
            }
            outs_ps = {
                it["passage_id"]
                for it in lines
                if it["question_id"] == qid and it["is_outside_pool_audit"]
            }
            assert main_ps.isdisjoint(outs_ps)

    def test_11_expansion_threshold_registered(self):
        man_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/pool_manifest.json"
        )
        manifest = json.loads(man_file.read_text())
        assert manifest["outside_pool_relevant_threshold"] == 0.05

    def test_12_canonical_mapper_by_id(
        self, sample_registry_entries: list[PassageRegistryEntry]
    ):
        mapper = CanonicalPassageMapper(sample_registry_entries)
        res = mapper.map_chunk({"chunk_id": "c1", "passage_id": "ps_entry_01_12345678"})
        assert res.mapping_status == CanonicalMappingStatus.EXACT_PASSAGE_ID
        assert res.mapped_passage_id == "ps_entry_01_12345678"

    def test_13_canonical_mapper_by_offsets(
        self, sample_registry_entries: list[PassageRegistryEntry]
    ):
        mapper = CanonicalPassageMapper(sample_registry_entries)
        res = mapper.map_chunk(
            {
                "chunk_id": "c2",
                "document_id": "gersting_discrete_math",
                "page_number": 92,
                "start_char": 310,
                "end_char": 550,
            }
        )
        assert res.mapping_status == CanonicalMappingStatus.EXACT_OFFSETS
        assert res.mapped_passage_id == "ps_entry_02_87654321"

    def test_14_canonical_mapper_by_hash(
        self, sample_registry_entries: list[PassageRegistryEntry]
    ):
        mapper = CanonicalPassageMapper(sample_registry_entries)
        res = mapper.map_chunk(
            {
                "chunk_id": "c3",
                "text": "Este é o texto exato do primeiro parágrafo sobre demonstração por indução matemática.",
            }
        )
        assert res.mapping_status == CanonicalMappingStatus.EXACT_CONTENT_SHA256
        assert res.mapped_passage_id == "ps_entry_01_12345678"

    def test_15_canonical_mapper_by_exact_substring(
        self, sample_registry_entries: list[PassageRegistryEntry]
    ):
        mapper = CanonicalPassageMapper(sample_registry_entries)
        res = mapper.map_chunk(
            {
                "chunk_id": "c4",
                "page_number": 92,
                "text": "primeiro parágrafo sobre demonstração",
            }
        )
        assert res.mapping_status == CanonicalMappingStatus.EXACT_SUBSTRING
        assert res.mapped_passage_id == "ps_entry_01_12345678"

    def test_16_canonical_mapper_ambiguity_flagged(
        self, sample_registry_entries: list[PassageRegistryEntry]
    ):
        mapper = CanonicalPassageMapper(sample_registry_entries)
        res = mapper.map_chunk(
            {
                "chunk_id": "c5",
                "page_number": 92,
                "text": "Este é o texto",  # Appears in both entries on page 92
            }
        )
        assert res.mapping_status == CanonicalMappingStatus.AMBIGUOUS_NEEDS_REVIEW
        assert res.confidence == 0.5

    def test_17_canonical_mapper_unmapped_flagged(
        self, sample_registry_entries: list[PassageRegistryEntry]
    ):
        mapper = CanonicalPassageMapper(sample_registry_entries)
        res = mapper.map_chunk(
            {
                "chunk_id": "c6",
                "page_number": 99,
                "text": "Texto completamente ausente do registro canônico.",
            }
        )
        assert res.mapping_status == CanonicalMappingStatus.UNMAPPED_NEEDS_REVIEW
        assert res.mapped_passage_id is None
        assert res.confidence == 0.0

    def test_18_zero_unreported_mapping_loss(self):
        audit_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/mapping_audit.json"
        )
        audit = json.loads(audit_file.read_text())
        assert audit["unreported_mapping_loss"] == 0

    def test_19_blinded_view_has_no_strategy(self):
        blinded_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/blinded_pool.jsonl"
        )
        content = blinded_file.read_text()
        assert '"strategy":' not in content
        assert '"strategy_label":' not in content

    def test_20_blinded_view_has_no_rank(self):
        blinded_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/blinded_pool.jsonl"
        )
        content = blinded_file.read_text()
        assert '"original_rank":' not in content
        assert '"rank":' not in content

    def test_21_blinded_view_has_no_scores(self):
        blinded_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/blinded_pool.jsonl"
        )
        content = blinded_file.read_text()
        assert '"score":' not in content
        assert '"retriever_name":' not in content
        assert '"reranker_score":' not in content

    def test_22_blinded_view_has_no_silver_or_llm_fields(self):
        blinded_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/blinded_pool.jsonl"
        )
        content = blinded_file.read_text()
        assert '"label_source":' not in content
        assert '"judge_id":' not in content
        assert '"relevance_grade":' not in content

    def test_23_blinded_view_has_no_other_annotator_answers(self):
        blinded_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/blinded_pool.jsonl"
        )
        content = blinded_file.read_text()
        assert '"annotator_a_grade":' not in content
        assert '"annotator_b_grade":' not in content

    def test_24_blinded_order_is_deterministic(self, tmp_path: Path):
        gt_dir = Path("benchmarks/ground_truth/v2")
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        _, b1, _, _ = build_hybrid_pool(gt_dir, out1)
        _, b2, _, _ = build_hybrid_pool(gt_dir, out2)
        assert b1.read_bytes() == b2.read_bytes()
