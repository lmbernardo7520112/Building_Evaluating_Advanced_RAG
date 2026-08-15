"""Tests for StrategyToolFactory — maps PipelineStrategy to tool specs + adapters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from raglab.agentic.runtime.passage_resolver import PassagePayload
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

    passage_id: str = "ps_stub_001"

    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedEvidence]:
        return [
            RetrievedEvidence(
                chunk_id=ChunkId(value=self.passage_id.replace("ps_", "")),
                document_id="doc_01",
                text=f"stub evidence for {self.passage_id}",
                rank=1,
                score=0.9,
                passage_id=self.passage_id,
            )
        ]


class SpyPassageCaptureStore:
    """Spy satisfying PassageCapturePort for test validation."""

    def __init__(self) -> None:
        self.calls: list[tuple[PassagePayload, ...]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def recorded_payloads(self) -> tuple[PassagePayload, ...]:
        if not self.calls:
            return ()
        return self.calls[-1]

    def record_passages(self, payloads: Sequence[PassagePayload]) -> None:
        self.calls.append(tuple(payloads))


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

    def test_08_build_adapter_accepts_and_wires_passage_store(self) -> None:
        spy = SpyPassageCaptureStore()
        port = StubPort(passage_id="ps_custom_008")
        adapter = build_adapter(
            PipelineStrategy.BASELINE,
            port,
            passage_store=spy,
        )
        obs = adapter.retrieve(query="test", strategy="baseline", top_k=3)

        assert obs.passage_ids == ("ps_custom_008",)
        assert spy.call_count == 1
        assert len(spy.recorded_payloads) == 1
        assert spy.recorded_payloads[0].passage_id == "ps_custom_008"
        assert spy.recorded_payloads[0].chunk_id == ChunkId("custom_008")

    def test_09_build_registry_wires_same_passage_store_instance_to_all_adapters(
        self,
    ) -> None:
        spy = SpyPassageCaptureStore()
        ports = {
            PipelineStrategy.BASELINE: StubPort("ps_base_09"),
            PipelineStrategy.SENTENCE_WINDOW: StubPort("ps_window_09"),
        }
        registry, adapters = build_registry_with_adapters(
            ports,
            passage_store=spy,
        )

        obs_base = adapters["retrieve_baseline"].retrieve(
            query="q",
            strategy="baseline",
            top_k=1,
        )
        obs_win = adapters["retrieve_sentence_window"].retrieve(
            query="q",
            strategy="sentence_window",
            top_k=1,
        )

        assert obs_base.passage_ids == ("ps_base_09",)
        assert obs_win.passage_ids == ("ps_window_09",)
        assert spy.call_count == 2
        assert spy.calls[0][0].passage_id == "ps_base_09"
        assert spy.calls[1][0].passage_id == "ps_window_09"

    def test_10_build_registry_works_without_passage_store(self) -> None:
        ports = {PipelineStrategy.BASELINE: StubPort("ps_legacy_10")}
        registry, adapters = build_registry_with_adapters(ports)

        obs = adapters["retrieve_baseline"].retrieve(
            query="q",
            strategy="baseline",
            top_k=1,
        )
        assert obs.passage_ids == ("ps_legacy_10",)
        assert obs.status.name == "EXECUTED"
