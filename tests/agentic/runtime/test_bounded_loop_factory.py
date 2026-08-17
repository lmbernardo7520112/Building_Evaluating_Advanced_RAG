"""Contract tests for public build_bounded_loop_coordinator factory."""

from __future__ import annotations

import unittest
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from raglab.agentic.runtime.bounded_loop_runner import BoundedLoopRunner
from raglab.agentic.runtime.passage_resolver import PassagePayload
from raglab.domain.entities import RetrievedEvidence
from raglab.domain.enums import PipelineStrategy
from raglab.domain.value_objects import ChunkId


def _get_factory_func() -> Any:
    """Import build_bounded_loop_coordinator at test execution time."""
    from raglab.agentic.runtime.bounded_loop_factory import (
        build_bounded_loop_coordinator,
    )

    return build_bounded_loop_coordinator


@dataclass
class FakeRetrievalPort:
    """Hermetic FakeRetrievalPort returning valid RetrievedEvidence."""

    passage_prefix: str

    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedEvidence]:
        return [
            RetrievedEvidence(
                chunk_id=ChunkId(value=f"{self.passage_prefix}_001"),
                document_id="doc_100",
                text=f"Sample evidence text for {query}",
                rank=1,
                score=0.95,
                passage_id=f"ps_{self.passage_prefix}_001",
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


class TestBoundedLoopFactory(unittest.TestCase):
    """Contract tests for build_bounded_loop_coordinator factory."""

    def setUp(self) -> None:
        self.port_baseline = FakeRetrievalPort("base")
        self.port_window = FakeRetrievalPort("window")
        self.ports = {
            PipelineStrategy.BASELINE: self.port_baseline,
            PipelineStrategy.SENTENCE_WINDOW_RERANK: self.port_window,
        }

    def _build_coordinator(
        self,
        ports: dict[Any, Any],
        run_id: str,
        **kwargs: Any,
    ) -> Any:
        factory = _get_factory_func()
        return factory(ports, run_id=run_id, **kwargs)

    def test_01_factory_returns_functional_bounded_loop_runner(self) -> None:
        runner = self._build_coordinator(self.ports, run_id="run_l3_factory_01")
        self.assertIsInstance(runner, BoundedLoopRunner)
        self.assertEqual(runner.run_id, "run_l3_factory_01")
        self.assertEqual(runner.budget.remaining()["logical_calls"], 2)

    def test_02_factory_requires_mandatory_baseline_and_sentence_window(
        self,
    ) -> None:
        runner = self._build_coordinator(self.ports, run_id="run_l3_factory_02")
        self.assertTrue(runner.registry.has("retrieve_baseline"))
        self.assertTrue(runner.registry.has("retrieve_sentence_window_rerank"))

    def test_03_each_factory_call_produces_independent_budget_registry_and_backend(
        self,
    ) -> None:
        runner1 = self._build_coordinator(self.ports, run_id="run_1")
        runner2 = self._build_coordinator(self.ports, run_id="run_2")

        self.assertIsNot(runner1.budget, runner2.budget)
        self.assertIsNot(runner1.registry, runner2.registry)
        self.assertIsNot(runner1.backend, runner2.backend)

    def test_04_missing_required_strategy_fails_during_construction(
        self,
    ) -> None:
        incomplete_ports = {
            PipelineStrategy.BASELINE: self.port_baseline,
        }
        with self.assertRaises((ValueError, KeyError)):
            self._build_coordinator(incomplete_ports, run_id="run_test_04")

    def test_05_two_factory_calls_allow_independent_queries_starting_with_two_logical_calls(
        self,
    ) -> None:
        """Each factory call creates a new Budget instance with remaining()["logical_calls"] == 2.

        Executing a query on runner1 consumes runner1's Budget, while runner2's Budget
        remains untouched at 2 logical calls.
        """
        runner1 = self._build_coordinator(self.ports, run_id="run_q1")
        runner2 = self._build_coordinator(self.ports, run_id="run_q2")

        self.assertEqual(runner1.budget.remaining()["logical_calls"], 2)
        self.assertEqual(runner2.budget.remaining()["logical_calls"], 2)

        runner1.execute("q1", "Compare A vs B", top_k=3)

        self.assertLessEqual(runner1.budget.remaining()["logical_calls"], 1)
        self.assertEqual(runner2.budget.remaining()["logical_calls"], 2)

    def test_06_factory_exposes_fixed_budget_contract(self) -> None:
        """Confirm factory builds frozen 2/2/0 Budget and signature excludes budget overrides."""
        import inspect

        runner = self._build_coordinator(self.ports, run_id="run_test_06")
        self.assertEqual(
            runner.budget.remaining(),
            {
                "logical_calls": 2,
                "physical_attempts": 2,
                "retries": 0,
            },
        )

        sig = inspect.signature(_get_factory_func())
        self.assertNotIn("max_logical_calls", sig.parameters)
        self.assertNotIn("max_physical_attempts", sig.parameters)
        self.assertNotIn("max_retries", sig.parameters)

    def test_07_factory_accepts_and_propagates_passage_store_to_adapters(
        self,
    ) -> None:
        spy = SpyPassageCaptureStore()
        runner = self._build_coordinator(
            self.ports,
            run_id="run_l3_factory_07",
            passage_store=spy,
        )
        result = runner.execute("q_07", "Query for test 07", top_k=1)

        self.assertEqual(result.evidence_count, 1)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(
            spy.recorded_payloads[0].passage_id,
            result.evidence_items[0].passage_id,
        )

    def test_08_factory_without_passage_store_retains_legacy_behavior(
        self,
    ) -> None:
        runner = self._build_coordinator(
            self.ports,
            run_id="run_l3_factory_08",
        )
        result = runner.execute("q_08", "Query for test 08", top_k=1)

        self.assertEqual(result.evidence_count, 1)
        self.assertTrue(result.evidence_items[0].passage_id.startswith("ps_"))


if __name__ == "__main__":
    unittest.main()
