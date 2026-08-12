"""Contract tests for public build_one_shot_coordinator factory."""

import unittest
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from raglab.agentic.one_shot_runner import OneShotRunner
from raglab.domain.entities import RetrievedEvidence
from raglab.domain.enums import PipelineStrategy
from raglab.domain.value_objects import ChunkId


def _get_factory_func() -> Any:
    """Import and return build_one_shot_coordinator at test execution time."""
    from raglab.agentic.runtime.coordinator_factory import (
        build_one_shot_coordinator,
    )

    return build_one_shot_coordinator


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


class TestOneShotCoordinatorFactory(unittest.TestCase):
    """Contract tests for build_one_shot_coordinator factory."""

    def setUp(self) -> None:
        self.port_baseline = FakeRetrievalPort("base")
        self.port_window = FakeRetrievalPort("window")
        self.ports = {
            PipelineStrategy.BASELINE: self.port_baseline,
            PipelineStrategy.SENTENCE_WINDOW_RERANK: self.port_window,
        }

    def _build_coordinator(self, ports: dict[Any, Any], run_id: str) -> Any:
        factory = _get_factory_func()
        return factory(ports, run_id=run_id)

    def test_01_factory_returns_functional_public_coordinator(self) -> None:
        runner = self._build_coordinator(self.ports, run_id="run_test_01")
        self.assertIsInstance(runner, OneShotRunner)
        self.assertEqual(runner.run_id, "run_test_01")

    def test_02_comparative_query_dispatches_sentence_window_rerank_with_evidence(
        self,
    ) -> None:
        runner = self._build_coordinator(self.ports, run_id="run_test_02")
        res = runner.execute(
            query_id="q_comp_01",
            query_text="Compare sentence window vs baseline",
            top_k=3,
        )

        self.assertEqual(
            res.routing_decision.selected_strategy,
            "sentence_window_rerank",
        )
        self.assertEqual(len(res.trajectory.steps), 1)
        self.assertEqual(
            res.trajectory.steps[0].action,
            "retrieve:retrieve_sentence_window_rerank",
        )
        self.assertGreater(res.evidence_count, 0)
        self.assertIsNone(res.error)

    def test_03_baseline_query_dispatches_baseline_with_evidence(self) -> None:
        runner = self._build_coordinator(self.ports, run_id="run_test_03")
        res = runner.execute(
            query_id="q_base_01",
            query_text="What is unanswerable in the text?",
            top_k=3,
        )

        self.assertEqual(res.routing_decision.selected_strategy, "baseline")
        self.assertEqual(len(res.trajectory.steps), 1)
        self.assertEqual(
            res.trajectory.steps[0].action,
            "retrieve:retrieve_baseline",
        )
        self.assertGreater(res.evidence_count, 0)
        self.assertIsNone(res.error)

    def test_04_execution_consumes_at_most_one_logical_call(self) -> None:
        runner = self._build_coordinator(self.ports, run_id="run_test_04")
        res = runner.execute(
            query_id="q_budget_01",
            query_text="Compare X and Y",
            top_k=3,
        )

        self.assertIsNotNone(res)
        remaining = runner.budget.remaining()
        self.assertEqual(remaining.get("logical_call"), 0)

    def test_05_missing_required_strategy_fails_during_construction(
        self,
    ) -> None:
        incomplete_ports = {
            PipelineStrategy.BASELINE: self.port_baseline,
        }
        with self.assertRaises((ValueError, KeyError)):
            self._build_coordinator(incomplete_ports, run_id="run_test_05")

    def test_06_two_factory_calls_do_not_share_budget_or_mutable_state(
        self,
    ) -> None:
        runner1 = self._build_coordinator(self.ports, run_id="run_01")
        runner2 = self._build_coordinator(self.ports, run_id="run_02")

        runner1.execute("q1", "Compare A and B", top_k=3)

        self.assertEqual(runner1.budget.remaining().get("logical_call"), 0)
        self.assertEqual(runner2.budget.remaining().get("logical_call"), 1)
        self.assertIsNot(runner1.budget, runner2.budget)
        self.assertIsNot(runner1.registry, runner2.registry)


if __name__ == "__main__":
    unittest.main()
