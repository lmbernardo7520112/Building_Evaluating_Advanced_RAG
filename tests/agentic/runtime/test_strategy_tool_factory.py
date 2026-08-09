"""Tests for StrategyToolFactory — maps PipelineStrategy to tool specs + adapters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from raglab.agentic.runtime.strategy_tool_factory import (
    ALL_STRATEGIES,
    build_adapter,
    build_registry_with_adapters,
    make_tool_spec,
    tool_id_for_strategy,
)
from raglab.domain.entities import RetrievedEvidence
from raglab.domain.enums import PipelineStrategy
from raglab.domain.value_objects import ChunkId


@dataclass
class StubPort:
    """Stub satisfying RetrievalPort."""

    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedEvidence]:
        return [
            RetrievedEvidence(
                chunk_id=ChunkId(value="stub_001"),
                document_id="doc_01",
                text="stub evidence",
                rank=1,
                score=0.9,
                passage_id="ps_stub_001",
            )
        ]


class TestStrategyToolFactory:
    def test_all_seven_strategies_produce_valid_specs(self) -> None:
        for strategy in ALL_STRATEGIES:
            spec = make_tool_spec(strategy)
            assert spec.read_only is True
            assert spec.network_access is False
            assert spec.deterministic is True
            assert spec.tool_id == f"retrieve_{strategy.value}"
            assert spec.version == "1.0.0"
            assert spec.max_top_k == 10

    def test_tool_ids_are_canonical(self) -> None:
        expected_ids = {
            "retrieve_baseline",
            "retrieve_sentence_anchor",
            "retrieve_sentence_window",
            "retrieve_sentence_window_rerank",
            "retrieve_hierarchical_leaf",
            "retrieve_auto_merging",
            "retrieve_auto_merging_rerank",
        }
        actual_ids = {tool_id_for_strategy(s) for s in ALL_STRATEGIES}
        assert actual_ids == expected_ids

    def test_build_adapter_returns_correct_strategy(self) -> None:
        port = StubPort()
        adapter = build_adapter(PipelineStrategy.BASELINE, port)
        assert adapter.strategy == "baseline"

    def test_build_registry_with_adapters_freezes_registry(self) -> None:
        ports = {s: StubPort() for s in ALL_STRATEGIES}
        registry, adapters = build_registry_with_adapters(ports)

        assert registry.is_frozen
        assert len(adapters) == 7
        assert registry.tool_ids == {f"retrieve_{s.value}" for s in ALL_STRATEGIES}

    def test_registry_rejects_duplicate_registration(self) -> None:
        ports = {s: StubPort() for s in ALL_STRATEGIES}
        registry, _ = build_registry_with_adapters(ports)

        with pytest.raises(ValueError, match="frozen"):
            registry.register(make_tool_spec(PipelineStrategy.BASELINE))

    def test_adapter_dispatch_produces_observation(self) -> None:
        port = StubPort()
        adapter = build_adapter(PipelineStrategy.BASELINE, port)
        obs = adapter.retrieve(query="test", strategy="baseline", top_k=3)
        assert obs.passage_ids == ("ps_stub_001",)

    def test_specs_are_deterministic(self) -> None:
        s1 = make_tool_spec(PipelineStrategy.AUTO_MERGING)
        s2 = make_tool_spec(PipelineStrategy.AUTO_MERGING)
        assert s1.implementation_sha256 == s2.implementation_sha256
