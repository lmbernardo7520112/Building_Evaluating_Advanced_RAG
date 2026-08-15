"""Tests for RetrievalToolAdapter — bridge from RetrievalPort to ToolObservation."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from raglab.agentic.enums import InvocationStatus
from raglab.agentic.errors import NonCanonicalIdError, PassageIntegrityError
from raglab.agentic.runtime.passage_resolver import PassagePayload
from raglab.agentic.runtime.retrieval_tool_adapter import RetrievalToolAdapter
from raglab.domain.entities import RetrievedEvidence
from raglab.domain.value_objects import ChunkId

# ── Fake RetrievalPort ──────────────────────────────────────────


@dataclass
class FakeRetrievalPort:
    """Minimal fake satisfying RetrievalPort protocol."""

    results: list[RetrievedEvidence]

    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedEvidence]:
        return self.results[:top_k]


class FailingRetrievalPort:
    """Port that always raises."""

    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedEvidence]:
        raise RuntimeError("backend failure simulated")


def _make_evidence(
    rank: int,
    passage_id: str = "ps_001",
    doc_id: str = "doc_01",
    text: str | None = None,
    page_number: int | None = None,
    content_sha256: str | None = None,
) -> RetrievedEvidence:
    txt = text if text is not None else f"evidence text {rank}"
    return RetrievedEvidence(
        chunk_id=ChunkId(value=passage_id.replace("ps_", "")),
        document_id=doc_id,
        text=txt,
        rank=rank,
        score=1.0 / rank,
        passage_id=passage_id,
        page_number=page_number,
        content_sha256=content_sha256,
    )


# ── Spy PassageCaptureStore ─────────────────────────────────────


class SpyPassageCaptureStore:
    """Spy satisfying PassageCapturePort for verifying retrieval capture."""

    def __init__(self, *, raise_on_record: Exception | None = None) -> None:
        self.calls: list[tuple[PassagePayload, ...]] = []
        self._raise_on_record = raise_on_record

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def recorded_payloads(self) -> tuple[PassagePayload, ...]:
        if not self.calls:
            return ()
        return self.calls[-1]

    def record_passages(self, payloads: Sequence[PassagePayload]) -> None:
        if self._raise_on_record is not None:
            raise self._raise_on_record
        self.calls.append(tuple(payloads))


# ── Tests ────────────────────────────────────────────────────────


class TestRetrievalToolAdapter:
    def test_successful_retrieval(self) -> None:
        evidence = [_make_evidence(1, "ps_a1"), _make_evidence(2, "ps_a2")]
        port = FakeRetrievalPort(results=evidence)
        adapter = RetrievalToolAdapter(strategy="baseline", retrieval_port=port)

        obs = adapter.retrieve(query="test query", strategy="baseline", top_k=3)

        assert obs.passage_ids == ("ps_a1", "ps_a2")
        assert obs.document_ids == ("doc_01", "doc_01")
        assert obs.ranks == (1, 2)
        assert len(obs.scores) == 2
        assert len(obs.content_hashes) == 2
        assert obs.status == InvocationStatus.EXECUTED
        assert obs.failure_code is None
        assert obs.latency_ms >= 0

    def test_strategy_mismatch_raises(self) -> None:
        port = FakeRetrievalPort(results=[])
        adapter = RetrievalToolAdapter(strategy="baseline", retrieval_port=port)

        with pytest.raises(ValueError, match="Strategy mismatch"):
            adapter.retrieve(query="q", strategy="auto_merging", top_k=3)

    def test_backend_failure_captured(self) -> None:
        port = FailingRetrievalPort()
        adapter = RetrievalToolAdapter(strategy="baseline", retrieval_port=port)

        obs = adapter.retrieve(query="q", strategy="baseline", top_k=3)

        assert obs.status == InvocationStatus.FAILED
        assert obs.failure_code is not None
        assert "RuntimeError" in obs.failure_code
        assert obs.passage_ids == ()

    def test_preserves_canonical_ids(self) -> None:
        evidence = [_make_evidence(1, "ps_canonical_001")]
        port = FakeRetrievalPort(results=evidence)
        adapter = RetrievalToolAdapter(strategy="baseline", retrieval_port=port)

        obs = adapter.retrieve(query="q", strategy="baseline", top_k=3)

        assert obs.passage_ids == ("ps_canonical_001",)

    def test_non_canonical_id_raises(self) -> None:
        """Evidence with non-ps_ passage_id must raise NonCanonicalIdError."""
        ev = RetrievedEvidence(
            chunk_id=ChunkId(value="bad"),
            document_id="doc",
            text="text",
            rank=1,
            score=1.0,
            passage_id="bad_id_no_prefix",
        )
        port = FakeRetrievalPort(results=[ev])
        adapter = RetrievalToolAdapter(strategy="baseline", retrieval_port=port)

        with pytest.raises(NonCanonicalIdError):
            adapter.retrieve(query="q", strategy="baseline", top_k=3)

    def test_config_hash_is_stable(self) -> None:
        port = FakeRetrievalPort(results=[])
        a1 = RetrievalToolAdapter(strategy="baseline", retrieval_port=port)
        a2 = RetrievalToolAdapter(strategy="baseline", retrieval_port=port)
        assert a1.config_hash == a2.config_hash

    def test_top_k_respected(self) -> None:
        evidence = [_make_evidence(i, f"ps_{i:03d}") for i in range(1, 6)]
        port = FakeRetrievalPort(results=evidence)
        adapter = RetrievalToolAdapter(strategy="baseline", retrieval_port=port)

        obs = adapter.retrieve(query="q", strategy="baseline", top_k=2)
        assert len(obs.passage_ids) == 2

    def test_08_adapter_works_without_passage_store(self) -> None:
        evidence = [_make_evidence(1, "ps_001")]
        port = FakeRetrievalPort(results=evidence)
        adapter = RetrievalToolAdapter(strategy="baseline", retrieval_port=port)

        obs = adapter.retrieve(query="q", strategy="baseline", top_k=3)

        assert obs.passage_ids == ("ps_001",)
        assert obs.status == InvocationStatus.EXECUTED

    def test_09_captures_explicit_and_fallback_passage_ids(self) -> None:
        ev_explicit = RetrievedEvidence(
            chunk_id=ChunkId("chk_01"),
            document_id="doc_01",
            text="text 1",
            rank=1,
            score=1.0,
            passage_id="ps_custom_01",
        )
        ev_fallback = RetrievedEvidence(
            chunk_id=ChunkId("chapter_1_chunk_42"),
            document_id="doc_01",
            text="text 2",
            rank=2,
            score=0.5,
            passage_id=None,
        )
        port = FakeRetrievalPort(results=[ev_explicit, ev_fallback])
        store = SpyPassageCaptureStore()
        adapter = RetrievalToolAdapter(
            strategy="baseline",
            retrieval_port=port,
            passage_store=store,
        )

        obs = adapter.retrieve(query="q", strategy="baseline", top_k=3)

        assert obs.passage_ids == ("ps_custom_01", "ps_chapter_1_chunk_42")
        assert store.call_count == 1
        assert len(store.recorded_payloads) == 2
        assert store.recorded_payloads[0].passage_id == "ps_custom_01"
        assert (
            store.recorded_payloads[1].passage_id == "ps_chapter_1_chunk_42"
        )

    def test_10_captures_explicit_and_computed_sha256(self) -> None:
        explicit_hash = "a" * 64
        ev1 = RetrievedEvidence(
            chunk_id=ChunkId("chk_01"),
            document_id="doc_01",
            text="text 1",
            rank=1,
            score=1.0,
            passage_id="ps_01",
            content_sha256=explicit_hash,
        )
        computed_hash = hashlib.sha256(b"text 2").hexdigest()
        ev2 = RetrievedEvidence(
            chunk_id=ChunkId("chk_02"),
            document_id="doc_01",
            text="text 2",
            rank=2,
            score=0.5,
            passage_id="ps_02",
            content_sha256=None,
        )
        port = FakeRetrievalPort(results=[ev1, ev2])
        store = SpyPassageCaptureStore()
        adapter = RetrievalToolAdapter(
            strategy="baseline",
            retrieval_port=port,
            passage_store=store,
        )

        obs = adapter.retrieve(query="q", strategy="baseline", top_k=3)

        assert store.call_count == 1
        assert store.recorded_payloads[0].content_sha256 == explicit_hash
        assert store.recorded_payloads[1].content_sha256 == computed_hash
        assert obs.content_hashes == (explicit_hash, computed_hash)

    def test_11_preserves_chunk_document_text_and_page_number(self) -> None:
        ev_none = RetrievedEvidence(
            chunk_id=ChunkId("c1"),
            document_id="doc_alpha",
            text="text alpha",
            rank=1,
            score=0.9,
            passage_id="ps_01",
            page_number=None,
        )
        ev_zero = RetrievedEvidence(
            chunk_id=ChunkId("c2"),
            document_id="doc_beta",
            text="text beta",
            rank=2,
            score=0.8,
            passage_id="ps_02",
            page_number=0,
        )
        ev_pos = RetrievedEvidence(
            chunk_id=ChunkId("c3"),
            document_id="doc_gamma",
            text="text gamma",
            rank=3,
            score=0.7,
            passage_id="ps_03",
            page_number=17,
        )
        port = FakeRetrievalPort(results=[ev_none, ev_zero, ev_pos])
        store = SpyPassageCaptureStore()
        adapter = RetrievalToolAdapter(
            strategy="baseline",
            retrieval_port=port,
            passage_store=store,
        )

        adapter.retrieve(query="q", strategy="baseline", top_k=3)

        assert store.call_count == 1
        p1, p2, p3 = store.recorded_payloads
        assert p1.chunk_id == ChunkId("c1")
        assert p1.document_id == "doc_alpha"
        assert p1.text == "text alpha"
        assert p1.page_number is None

        assert p2.chunk_id == ChunkId("c2")
        assert p2.document_id == "doc_beta"
        assert p2.text == "text beta"
        assert p2.page_number == 0

        assert p3.chunk_id == ChunkId("c3")
        assert p3.document_id == "doc_gamma"
        assert p3.text == "text gamma"
        assert p3.page_number == 17

    def test_12_records_once_only_after_entire_batch_validates(self) -> None:
        evidence_valid = [
            _make_evidence(1, "ps_01"),
            _make_evidence(2, "ps_02"),
            _make_evidence(3, "ps_03"),
        ]
        port = FakeRetrievalPort(results=evidence_valid)
        store = SpyPassageCaptureStore()
        adapter = RetrievalToolAdapter(
            strategy="baseline",
            retrieval_port=port,
            passage_store=store,
        )

        obs = adapter.retrieve(query="q", strategy="baseline", top_k=3)
        assert obs.status == InvocationStatus.EXECUTED
        assert store.call_count == 1
        assert len(store.recorded_payloads) == 3

        ev_valid = _make_evidence(1, "ps_valid_01")
        ev_invalid = RetrievedEvidence(
            chunk_id=ChunkId("bad"),
            document_id="doc",
            text="txt",
            rank=2,
            score=0.5,
            passage_id="invalid_prefix_02",
        )
        port_bad = FakeRetrievalPort(results=[ev_valid, ev_invalid])
        store_bad = SpyPassageCaptureStore()
        adapter_bad = RetrievalToolAdapter(
            strategy="baseline",
            retrieval_port=port_bad,
            passage_store=store_bad,
        )

        with pytest.raises(NonCanonicalIdError):
            adapter_bad.retrieve(query="q", strategy="baseline", top_k=3)

        assert store_bad.call_count == 0

    def test_13_store_integrity_error_propagates_fail_closed(self) -> None:
        evidence = [_make_evidence(1, "ps_01")]
        port = FakeRetrievalPort(results=evidence)
        store = SpyPassageCaptureStore(
            raise_on_record=PassageIntegrityError("conflict simulated")
        )
        adapter = RetrievalToolAdapter(
            strategy="baseline",
            retrieval_port=port,
            passage_store=store,
        )

        with pytest.raises(PassageIntegrityError, match="conflict simulated"):
            adapter.retrieve(query="q", strategy="baseline", top_k=3)

    def test_14_retrieval_failure_returns_failed_observation_without_store_call(
        self,
    ) -> None:
        port = FailingRetrievalPort()
        store = SpyPassageCaptureStore()
        adapter = RetrievalToolAdapter(
            strategy="baseline",
            retrieval_port=port,
            passage_store=store,
        )

        obs = adapter.retrieve(query="q", strategy="baseline", top_k=3)

        assert obs.status == InvocationStatus.FAILED
        assert "RuntimeError" in (obs.failure_code or "")
        assert store.call_count == 0

    def test_15_empty_evidence_does_not_call_store(self) -> None:
        port = FakeRetrievalPort(results=[])
        store = SpyPassageCaptureStore()
        adapter = RetrievalToolAdapter(
            strategy="baseline",
            retrieval_port=port,
            passage_store=store,
        )

        obs = adapter.retrieve(query="q", strategy="baseline", top_k=3)

        assert obs.status == InvocationStatus.EXECUTED
        assert obs.passage_ids == ()
        assert store.call_count == 0
