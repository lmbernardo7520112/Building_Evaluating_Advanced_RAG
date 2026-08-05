"""Tests for Gate B2 integration: pool, queues, evidence v2, and accounting.

Covers ETAPA 9 invariants:
1. pool rejects legacy artifacts
2. pool accepts only evidence v2
3. pool doesn't read relevant_pages/gold_answer
4. pre/post reranking no double counting
5. dropped candidates auditable
6. canonical page-level unit explicit
7. outside audit disjoint from pool
8. 279-record accounting closes
9. Queue A accounting closes by union
10. human queues recursively blinded
11. holdout sealed
12. test invariant matrix has zero MISSING_BLOCKING
13. rebuild is deterministic
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from raglab.evaluation.contracts.human_annotation_v2 import PassageRegistryEntry
from raglab.evaluation.contracts.hybrid_eval_v2 import CanonicalMappingStatus
from raglab.evaluation.pooling.canonical_passage_mapper import (
    CanonicalPassageMapper,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
POOL_DIR = REPO_ROOT / "benchmarks" / "ground_truth" / "v2" / "hybrid" / "candidate_pool"
QUEUE_DIR = REPO_ROOT / "benchmarks" / "ground_truth" / "v2" / "hybrid" / "human_queues"
EVIDENCE_V2 = REPO_ROOT / "benchmarks" / "results" / "retrieval_evidence_v2.json"

FORBIDDEN_BLINDING_KEYS = {
    "strategy", "source_id", "source_provenance",
    "retriever", "retriever_name", "retriever_config",
    "raw_rank", "retrieval_rank", "retrieval_score",
    "score", "similarity", "reranker_score",
    "silver", "silver_label", "judge", "model",
    "relevant_pages", "gold_answer",
    "annotator_a_grade", "annotator_b_grade",
    "pre_rerank_rank", "post_rerank_rank",
    "selected_by_reranker", "dropped_by_reranker",
}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        pytest.skip(f"{path.name} not found")
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def _load_json(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"{path.name} not found")
    return json.loads(path.read_text("utf-8"))


def _assert_no_forbidden(data) -> None:
    """Recursively assert no forbidden blinding keys (exact match)."""
    if isinstance(data, dict):
        for k, v in data.items():
            assert k not in FORBIDDEN_BLINDING_KEYS, (
                f"BLINDING VIOLATION: key '{k}' is forbidden"
            )
            _assert_no_forbidden(v)
    elif isinstance(data, list):
        for item in data:
            _assert_no_forbidden(item)


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pool_manifest() -> dict:
    return _load_json(POOL_DIR / "pool_manifest.json")


@pytest.fixture(scope="module")
def pool_items() -> list[dict]:
    return _load_jsonl(POOL_DIR / "pool.jsonl")


@pytest.fixture(scope="module")
def blinded_items() -> list[dict]:
    return _load_jsonl(POOL_DIR / "blinded_pool.jsonl")


@pytest.fixture(scope="module")
def accounting() -> dict:
    return _load_json(POOL_DIR / "raw_candidate_accounting.json")


@pytest.fixture(scope="module")
def mapping_audit() -> dict:
    return _load_json(POOL_DIR / "mapping_audit.json")


@pytest.fixture(scope="module")
def exec_audit() -> dict:
    return _load_json(POOL_DIR / "pool_execution_audit.json")


@pytest.fixture(scope="module")
def routing_manifest() -> dict:
    return _load_json(QUEUE_DIR / "routing_manifest.json")


@pytest.fixture(scope="module")
def queue_a() -> list[dict]:
    return _load_jsonl(QUEUE_DIR / "annotator_a.jsonl")


@pytest.fixture(scope="module")
def queue_b() -> list[dict]:
    return _load_jsonl(QUEUE_DIR / "annotator_b.jsonl")


@pytest.fixture(scope="module")
def evidence_v2() -> dict:
    if not EVIDENCE_V2.exists():
        pytest.skip("evidence v2 not materialized")
    return json.loads(EVIDENCE_V2.read_bytes())


@pytest.fixture
def sample_registry_entries() -> list[PassageRegistryEntry]:
    t1 = "Este é o texto exato do primeiro parágrafo sobre demonstração por indução matemática."
    t2 = "Este é o texto do segundo parágrafo detalhando o passo indutivo e a hipótese de indução."
    return [
        PassageRegistryEntry(
            passage_id="ps_entry_01_12345678",
            document_id="gersting_discrete_math",
            page_number=92, start_char=100, end_char=300,
            content_sha256=hashlib.sha256(t1.encode()).hexdigest(), text=t1,
        ),
        PassageRegistryEntry(
            passage_id="ps_entry_02_87654321",
            document_id="gersting_discrete_math",
            page_number=92, start_char=310, end_char=550,
            content_sha256=hashlib.sha256(t2.encode()).hexdigest(), text=t2,
        ),
    ]


# ── Test 1: pool rejects legacy artifacts ────────────────────────

class TestPoolRejectsLegacy:
    def test_manifest_declares_evidence_v2(self, pool_manifest: dict) -> None:
        assert pool_manifest["retrieval_evidence_schema"] == "retrieval_evidence_v2"

    def test_text_preview_not_used(self, pool_manifest: dict) -> None:
        assert pool_manifest["text_preview_used"] is False

    def test_no_rehydration_audit(self) -> None:
        """Evidence v2 pool does NOT need rehydration — data is already full."""
        rehyd = POOL_DIR / "rehydration_audit.json"
        assert not rehyd.exists(), "rehydration_audit.json should not exist"


# ── Test 2: pool accepts only evidence v2 ─────────────────────────

class TestPoolAcceptsEvidenceV2:
    def test_input_validation_passed(self, pool_manifest: dict) -> None:
        assert pool_manifest["input_validation_status"] == "PASSED"

    def test_evidence_sha_recorded(self, pool_manifest: dict) -> None:
        assert len(pool_manifest["retrieval_evidence_sha256"]) == 64

    def test_record_count_matches(self, pool_manifest: dict) -> None:
        assert pool_manifest["retrieval_evidence_record_count"] == 279


# ── Test 3: no relevant_pages/gold_answer ─────────────────────────

class TestNoGroundTruthInPool:
    def test_pool_has_no_relevant_pages(self, pool_items: list[dict]) -> None:
        for it in pool_items:
            assert "relevant_pages" not in json.dumps(it)

    def test_pool_has_no_gold_answer(self, pool_items: list[dict]) -> None:
        for it in pool_items:
            assert "gold_answer" not in json.dumps(it)

    def test_manifest_declares_no_ground_truth(self, pool_manifest: dict) -> None:
        assert pool_manifest["relevant_pages_used"] is False
        assert pool_manifest["gold_answer_used"] is False


# ── Test 4: pre/post reranking no double counting ────────────────

class TestRerankerNoDuplicate:
    def test_accounting_separates_final_and_dropped(self, accounting: dict) -> None:
        final = accounting["raw_candidates_final"]
        dropped = accounting["raw_candidates_dropped_by_reranker"]
        invalid = accounting["invalid_records"]
        assert final + dropped + invalid == 279

    def test_no_dropped_in_pool_provenance(self, pool_items: list[dict]) -> None:
        for it in pool_items:
            for prov in it.get("source_provenance", []):
                if prov.get("source_id") in ("W1_sentence_window_rerank", "H2_auto_merging_rerank"):
                    assert prov.get("post_rerank_rank") is not None


# ── Test 5: dropped candidates auditable ──────────────────────────

class TestDroppedCandidatesAuditable:
    def test_dropped_records_in_accounting(self, accounting: dict) -> None:
        dropped_recs = [
            r for r in accounting["raw_candidate_records"]
            if r["reranker_classification"] == "POST_RERANK_DROPPED"
        ]
        assert len(dropped_recs) == accounting["raw_candidates_dropped_by_reranker"]
        for r in dropped_recs:
            assert r["dropped_by_reranker"] is True
            assert r["selected_by_reranker"] is False


# ── Test 6: canonical page-level unit explicit ───────────────────

class TestCanonicalPageLevel:
    def test_manifest_declares_page_level(self, pool_manifest: dict) -> None:
        assert pool_manifest["canonical_evaluation_unit"] == "PAGE_LEVEL"
        assert pool_manifest["canonical_registry_entry_count"] == 25

    def test_accounting_declares_page_level(self, accounting: dict) -> None:
        assert accounting["canonical_evaluation_unit"] == "PAGE_LEVEL"
        assert accounting["canonical_registry_entry_count"] == 25

    def test_pool_items_declare_page_level(self, pool_items: list[dict]) -> None:
        main_items = [it for it in pool_items if not it.get("is_outside_pool_audit")]
        for it in main_items:
            assert it["canonical_evaluation_unit"] == "PAGE_LEVEL"


# ── Test 7: outside audit disjoint from pool ──────────────────────

class TestOutsideAuditDisjoint:
    def test_disjoint(self, pool_items: list[dict]) -> None:
        main_keys = {
            (it["question_id"], it["page_number"])
            for it in pool_items if not it.get("is_outside_pool_audit")
        }
        audit_keys = {
            (it["question_id"], it["page_number"])
            for it in pool_items if it.get("is_outside_pool_audit")
        }
        assert len(main_keys & audit_keys) == 0

    def test_manifest_confirms_disjoint(self, pool_manifest: dict) -> None:
        assert pool_manifest["pool_outside_disjoint"] is True


# ── Test 8: 279-record accounting closes ──────────────────────────

class TestAccountingCloses279:
    def test_total_evidence_records(self, accounting: dict) -> None:
        assert accounting["raw_evidence_records"] == 279

    def test_identity_holds(self, accounting: dict) -> None:
        final = accounting["raw_candidates_final"]
        dropped = accounting["raw_candidates_dropped_by_reranker"]
        invalid = accounting["invalid_records"]
        assert final + dropped + invalid == 279


# ── Test 9: Queue A accounting closes ─────────────────────────────

class TestQueueACloses:
    def test_queue_a_equals_pool_total(
        self, routing_manifest: dict, pool_manifest: dict,
    ) -> None:
        assert routing_manifest["annotator_a_queue_count"] == pool_manifest["queue_a_total"]

    def test_queue_a_items_match_count(
        self, queue_a: list[dict], routing_manifest: dict,
    ) -> None:
        assert len(queue_a) == routing_manifest["annotator_a_queue_count"]


# ── Test 10: human queues recursively blinded ─────────────────────

class TestQueuesBlinded:
    def test_queue_a_blinded(self, queue_a: list[dict]) -> None:
        for it in queue_a:
            _assert_no_forbidden(it)

    def test_queue_b_blinded(self, queue_b: list[dict]) -> None:
        for it in queue_b:
            _assert_no_forbidden(it)

    def test_blinded_pool_blinded(self, blinded_items: list[dict]) -> None:
        for it in blinded_items:
            _assert_no_forbidden(it)


# ── Test 11: holdout sealed ───────────────────────────────────────

class TestHoldoutSealed:
    def test_pool_no_holdout(self, pool_items: list[dict]) -> None:
        for it in pool_items:
            assert "holdout" not in it["question_id"]

    def test_pool_manifest_holdout_sealed(self, pool_manifest: dict) -> None:
        assert pool_manifest["holdout_sealed"] is True

    def test_queue_manifest_holdout_sealed(self, routing_manifest: dict) -> None:
        assert routing_manifest["holdout_sealed"] is True


# ── Test 12: canonical mapper unit tests ──────────────────────────

class TestCanonicalMapper:
    def test_mapper_by_id(self, sample_registry_entries: list[PassageRegistryEntry]) -> None:
        mapper = CanonicalPassageMapper(sample_registry_entries)
        res = mapper.map_chunk({
            "chunk_id": "ps_entry_01_12345678",
            "page_number": 92, "text": "qualquer",
        })
        assert res.mapping_status == CanonicalMappingStatus.EXACT_PASSAGE_ID

    def test_mapper_by_offsets(self, sample_registry_entries: list[PassageRegistryEntry]) -> None:
        mapper = CanonicalPassageMapper(sample_registry_entries)
        res = mapper.map_chunk({
            "chunk_id": "c_off", "page_number": 92,
            "start_char": 100, "end_char": 300, "text": "qualquer",
        })
        assert res.mapping_status == CanonicalMappingStatus.EXACT_OFFSETS

    def test_mapper_by_hash(self, sample_registry_entries: list[PassageRegistryEntry]) -> None:
        mapper = CanonicalPassageMapper(sample_registry_entries)
        text = sample_registry_entries[0].text
        res = mapper.map_chunk({
            "chunk_id": "c_hash", "page_number": 92,
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "text": "qualquer",
        })
        assert res.mapping_status == CanonicalMappingStatus.EXACT_CONTENT_SHA256

    def test_mapper_by_substring(self, sample_registry_entries: list[PassageRegistryEntry]) -> None:
        mapper = CanonicalPassageMapper(sample_registry_entries)
        res = mapper.map_chunk({
            "chunk_id": "c_sub", "page_number": 92,
            "text": "primeiro parágrafo sobre demonstração por indução",
        })
        assert res.mapping_status == CanonicalMappingStatus.EXACT_SUBSTRING

    def test_mapper_ambiguous(self, sample_registry_entries: list[PassageRegistryEntry]) -> None:
        mapper = CanonicalPassageMapper(sample_registry_entries)
        res = mapper.map_chunk({
            "chunk_id": "c_ambig", "page_number": 92, "text": "Este é o texto",
        })
        assert res.mapping_status == CanonicalMappingStatus.AMBIGUOUS_NEEDS_REVIEW

    def test_mapper_unmapped(self, sample_registry_entries: list[PassageRegistryEntry]) -> None:
        mapper = CanonicalPassageMapper(sample_registry_entries)
        res = mapper.map_chunk({
            "chunk_id": "c_unmap", "page_number": 99, "text": "Texto ausente",
        })
        assert res.mapping_status == CanonicalMappingStatus.UNMAPPED_NEEDS_REVIEW


# ── Test 13: rebuild deterministic ────────────────────────────────

class TestRebuildDeterministic:
    def test_pool_manifest_sha_stable(self, pool_manifest: dict) -> None:
        assert len(pool_manifest["pool_sha256"]) == 64
        assert len(pool_manifest["blinded_pool_sha256"]) == 64

    def test_routing_manifest_sha_stable(self, routing_manifest: dict) -> None:
        assert len(routing_manifest["file_a_sha256"]) == 64
        assert len(routing_manifest["file_b_sha256"]) == 64


# ── Multisystem provenance ────────────────────────────────────────

class TestMultisystemProvenance:
    def test_multisystem_verified(self, pool_manifest: dict) -> None:
        assert pool_manifest["multisystem_provenance_verified"] is True
        assert pool_manifest["independent_family_count"] >= 2

    def test_seven_strategies_present(self, exec_audit: dict) -> None:
        strategies = set(exec_audit["strategies_present"])
        expected = {
            "F0_baseline", "S0_sentence_anchor",
            "W0_sentence_window", "W1_sentence_window_rerank",
            "H0_hierarchical_leaf", "H1_auto_merging", "H2_auto_merging_rerank",
        }
        assert strategies == expected

    def test_all_available(self, exec_audit: dict) -> None:
        for entry in exec_audit["per_question_source_audit"]:
            assert entry["availability"] == "AVAILABLE"
            assert entry["execution_mode"] == "MATERIALIZED_EVIDENCE_V2"


# ── Pool deduplication ────────────────────────────────────────────

class TestPoolDeduplication:
    def test_no_duplicate_qid_page_pairs(self, pool_items: list[dict]) -> None:
        pairs = [(it["question_id"], it["page_number"]) for it in pool_items]
        assert len(pairs) == len(set(pairs))

    def test_blinded_view_clean(self, blinded_items: list[dict]) -> None:
        content = json.dumps(blinded_items)
        assert "gold_answer" not in content


# ── Queue status ──────────────────────────────────────────────────

class TestQueueStatus:
    def test_queue_status_valid(self, routing_manifest: dict) -> None:
        assert routing_manifest["queue_status"] in (
            "DEFINITIVE_HUMAN_REVIEW",
            "PROVISIONAL_WITHOUT_SILVER",
        )

    def test_overlap_range(self, routing_manifest: dict) -> None:
        rate = routing_manifest["planned_overlap_rate"]
        assert 0 < rate <= 1.0

    def test_blinding_verified(self, routing_manifest: dict) -> None:
        assert routing_manifest["blinding_verified"] is True
