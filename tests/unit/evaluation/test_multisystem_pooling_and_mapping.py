"""Unit tests for Multisystem Pooling, Canonical Passage Mapping, and Blinding (Gate B2 Reconciliation). # noqa: E501

Covers reconciliation invariants 1-8, 14-15.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from raglab.evaluation.contracts.human_annotation_v2 import PassageRegistryEntry
from raglab.evaluation.contracts.hybrid_eval_v2 import (
    CanonicalMappingStatus,
)
from raglab.evaluation.pooling.canonical_passage_mapper import (
    CanonicalPassageMapper,
)
from scripts.build_hybrid_candidate_pool import build_hybrid_pool

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
    """Testes unitários direcionados para Pooling Multissistema e Mapeamento Canônico."""

    def test_01_direct_registry_selection_does_not_count_as_retriever(self):
        man_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/pool_manifest.json"
        )
        assert man_file.exists()
        manifest = json.loads(man_file.read_text())
        assert manifest["multisystem_provenance_verified"] is True
        assert manifest["independent_family_count"] >= 2

    def test_02_source_id_requires_real_provenance(self):
        audit_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/pool_execution_audit.json"
        )
        assert audit_file.exists()
        audit = json.loads(audit_file.read_text())
        avail = [
            e
            for e in audit["per_question_source_audit"]
            if e["availability"] == "AVAILABLE"
        ]
        assert len(avail) > 0
        for e in avail:
            assert e["raw_returned_count"] >= 0
            assert e["execution_mode"] == "MATERIALIZED_OFFLINE_BENCHMARK"

    def test_03_unavailable_source_explicit(self):
        audit_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/pool_execution_audit.json"
        )
        audit = json.loads(audit_file.read_text())
        unavail = [
            e
            for e in audit["per_question_source_audit"]
            if e["availability"] == "NOT_AVAILABLE_OFFLINE"
        ]
        assert len(unavail) > 0
        assert any(e["source_id"] in ["lexical_bm25", "dense_canonical"] for e in unavail)

    def test_04_unique_contribution_calculated_after_deduplication(self):
        pool_file = Path("benchmarks/ground_truth/v2/hybrid/candidate_pool/pool.jsonl")
        lines = [
            json.loads(line)
            for line in pool_file.read_text().splitlines()
            if line.strip()
        ]
        pairs = [(item["question_id"], item["passage_id"]) for item in lines]
        assert len(pairs) == len(set(pairs))

    def test_05_ground_truth_not_used_in_retrieval(self):
        pool_file = Path("benchmarks/ground_truth/v2/hybrid/candidate_pool/pool.jsonl")
        content = pool_file.read_text()
        assert "gold_answer" not in content
        assert "relevance_grade" not in content

    def test_06_ambiguous_mapping_not_discarded(
        self, sample_registry_entries: list[PassageRegistryEntry]
    ):
        mapper = CanonicalPassageMapper(sample_registry_entries)
        res = mapper.map_chunk({
            "chunk_id": "c_ambig",
            "page_number": 92,
            "text": "Este é o texto",
        })
        assert res.mapping_status == CanonicalMappingStatus.AMBIGUOUS_NEEDS_REVIEW
        assert res.mapped_passage_id is not None

    def test_07_unmapped_mapping_not_discarded(
        self, sample_registry_entries: list[PassageRegistryEntry]
    ):
        mapper = CanonicalPassageMapper(sample_registry_entries)
        res = mapper.map_chunk({
            "chunk_id": "c_unmapped",
            "page_number": 99,
            "text": "Texto ausente no registro",
        })
        assert res.mapping_status == CanonicalMappingStatus.UNMAPPED_NEEDS_REVIEW
        assert res.mapped_passage_id is None

    def test_08_recursive_blinding_audit(self):
        blinded_file = Path(
            "benchmarks/ground_truth/v2/hybrid/candidate_pool/blinded_pool.jsonl"
        )
        lines = [
            json.loads(line)
            for line in blinded_file.read_text().splitlines()
            if line.strip()
        ]
        for item in lines:
            assert_no_forbidden_keys_recursive(item)

    def test_14_holdout_rejected_in_pooling(self):
        pool_file = Path("benchmarks/ground_truth/v2/hybrid/candidate_pool/pool.jsonl")
        lines = [
            json.loads(line)
            for line in pool_file.read_text().splitlines()
            if line.strip()
        ]
        for item in lines:
            assert "holdout" not in item["question_id"].lower()

    def test_15_rebuild_is_deterministic(self, tmp_path: Path):
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
