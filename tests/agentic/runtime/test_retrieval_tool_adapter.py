"""Tests for RetrievalToolAdapter — bridge from RetrievalPort to ToolObservation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from raglab.agentic.enums import InvocationStatus
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
    rank: int, passage_id: str = "ps_001", doc_id: str = "doc_01"
) -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id=ChunkId(value=passage_id.replace("ps_", "")),
        document_id=doc_id,
        text=f"evidence text {rank}",
        rank=rank,
        score=1.0 / rank,
        passage_id=passage_id,
    )


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

        from raglab.agentic.errors import NonCanonicalIdError

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
