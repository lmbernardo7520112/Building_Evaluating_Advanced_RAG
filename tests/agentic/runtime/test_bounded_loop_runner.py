"""Contract tests for BoundedLoopRunner — 2-step governed retrieval loop."""

import unittest
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from raglab.agentic.budget import Budget
from raglab.agentic.enums import StopReason
from raglab.agentic.runtime.dispatching_backend import (
    DispatchingRetrievalBackend,
)
from raglab.agentic.runtime.strategy_tool_factory import (
    build_registry_with_adapters,
)
from raglab.domain.entities import RetrievedEvidence
from raglab.domain.enums import PipelineStrategy
from raglab.domain.value_objects import ChunkId


def _get_runner_and_result_classes() -> tuple[Any, Any]:
    """Import BoundedLoopRunner and BoundedLoopResult at test execution time."""
    from raglab.agentic.runtime.bounded_loop_runner import (
        BoundedLoopResult,
        BoundedLoopRunner,
    )

    return BoundedLoopRunner, BoundedLoopResult


@dataclass
class FakeRetrievalPort:
    """Hermetic FakeRetrievalPort returning configurable evidence."""

    passage_prefix: str
    results: list[RetrievedEvidence] | None = None

    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedEvidence]:
        if self.results is not None:
            return self.results[:top_k]
        return [
            RetrievedEvidence(
                chunk_id=ChunkId(value=f"{self.passage_prefix}_001"),
                document_id="doc_100",
                text=f"Sample evidence text for {query}",
                rank=1,
                score=0.95,
                passage_id=f"ps_{self.passage_prefix}_001",
            )
        ][:top_k]


class FailingRetrievalPort:
    """Port that raises a runtime backend error."""

    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedEvidence]:
        raise RuntimeError("simulated backend retrieval failure")


class TestBoundedLoopRunner(unittest.TestCase):
    """Contract tests for 2-step BoundedLoopRunner."""

    def _build_runner(
        self,
        ports: dict[PipelineStrategy, Any],
        run_id: str,
        budget: Budget | None = None,
    ) -> Any:
        runner_class, _ = _get_runner_and_result_classes()
        registry, adapters = build_registry_with_adapters(ports)
        dispatcher = DispatchingRetrievalBackend(adapters)
        b = budget or Budget(
            max_logical_calls=2, max_physical_attempts=2, max_retries=0
        )
        return runner_class(
            registry=registry,
            budget=b,
            backend=dispatcher,
            run_id=run_id,
        )

    def test_01_step_0_with_evidence_stops_immediately(self) -> None:
        ports = {
            PipelineStrategy.BASELINE: FakeRetrievalPort("base"),
            PipelineStrategy.SENTENCE_WINDOW_RERANK: FakeRetrievalPort(
                "window"
            ),
        }
        runner = self._build_runner(ports, run_id="run_l3_01")
        res = runner.execute("q1", "Compare sentence window vs baseline", top_k=3)

        self.assertEqual(res.evidence_count, 1)
        self.assertEqual(len(res.trajectory.steps), 1)
        self.assertEqual(
            res.trajectory.steps[0].action,
            "retrieve:retrieve_sentence_window_rerank",
        )
        self.assertEqual(
            res.stop_decision.reason, StopReason.SUFFICIENT_EVIDENCE
        )
        self.assertIsNone(res.error)
        self.assertEqual(runner.budget.remaining()["logical_calls"], 1)

    def test_02_sentence_window_no_evidence_fallbacks_to_baseline(
        self,
    ) -> None:
        ports = {
            PipelineStrategy.BASELINE: FakeRetrievalPort("base"),
            PipelineStrategy.SENTENCE_WINDOW_RERANK: FakeRetrievalPort(
                "window", results=[]
            ),
        }
        runner = self._build_runner(ports, run_id="run_l3_02")
        res = runner.execute("q2", "Compare X vs Y", top_k=3)

        self.assertEqual(res.evidence_count, 1)
        self.assertEqual(len(res.trajectory.steps), 2)
        self.assertEqual(res.trajectory.steps[0].step_index, 0)
        self.assertEqual(
            res.trajectory.steps[0].action,
            "retrieve:retrieve_sentence_window_rerank",
        )
        self.assertEqual(res.trajectory.steps[1].step_index, 1)
        self.assertEqual(
            res.trajectory.steps[1].action, "retrieve:retrieve_baseline"
        )
        self.assertEqual(
            res.stop_decision.reason, StopReason.SUFFICIENT_EVIDENCE
        )
        self.assertIsNone(res.error)
        self.assertEqual(runner.budget.remaining()["logical_calls"], 0)

    def test_03_baseline_no_evidence_fallbacks_to_sentence_window(
        self,
    ) -> None:
        ports = {
            PipelineStrategy.BASELINE: FakeRetrievalPort("base", results=[]),
            PipelineStrategy.SENTENCE_WINDOW_RERANK: FakeRetrievalPort(
                "window"
            ),
        }
        runner = self._build_runner(ports, run_id="run_l3_03")
        res = runner.execute("q3", "What is unanswerable in the text?", top_k=3)

        self.assertEqual(res.evidence_count, 1)
        self.assertEqual(len(res.trajectory.steps), 2)
        self.assertEqual(res.trajectory.steps[0].step_index, 0)
        self.assertEqual(
            res.trajectory.steps[0].action, "retrieve:retrieve_baseline"
        )
        self.assertEqual(res.trajectory.steps[1].step_index, 1)
        self.assertEqual(
            res.trajectory.steps[1].action,
            "retrieve:retrieve_sentence_window_rerank",
        )
        self.assertEqual(
            res.stop_decision.reason, StopReason.SUFFICIENT_EVIDENCE
        )
        self.assertIsNone(res.error)
        self.assertEqual(runner.budget.remaining()["logical_calls"], 0)

    def test_04_zero_evidence_both_steps_stops_with_no_evidence(self) -> None:
        ports = {
            PipelineStrategy.BASELINE: FakeRetrievalPort("base", results=[]),
            PipelineStrategy.SENTENCE_WINDOW_RERANK: FakeRetrievalPort(
                "window", results=[]
            ),
        }
        runner = self._build_runner(ports, run_id="run_l3_04")
        res = runner.execute("q4", "Compare X vs Y", top_k=3)

        self.assertEqual(res.evidence_count, 0)
        self.assertEqual(len(res.trajectory.steps), 2)
        self.assertEqual(res.stop_decision.reason, StopReason.NO_EVIDENCE)
        self.assertIsNone(res.error)
        self.assertEqual(runner.budget.remaining()["logical_calls"], 0)

    def test_05_exhausted_budget_blocks_new_retrieval(self) -> None:
        ports = {
            PipelineStrategy.BASELINE: FakeRetrievalPort("base"),
            PipelineStrategy.SENTENCE_WINDOW_RERANK: FakeRetrievalPort(
                "window"
            ),
        }
        budget = Budget(
            max_logical_calls=1, max_physical_attempts=1, max_retries=0
        )
        budget.consume_logical_call()

        runner = self._build_runner(ports, run_id="run_l3_05", budget=budget)
        res = runner.execute("q5", "Compare X vs Y", top_k=3)

        self.assertEqual(
            res.stop_decision.reason, StopReason.BUDGET_EXHAUSTED
        )
        self.assertEqual(runner.budget.remaining()["logical_calls"], 0)

    def test_06_same_tool_repetition_blocked(self) -> None:
        ports = {
            PipelineStrategy.BASELINE: FakeRetrievalPort("base", results=[]),
            PipelineStrategy.SENTENCE_WINDOW_RERANK: FakeRetrievalPort(
                "window", results=[]
            ),
        }
        runner = self._build_runner(ports, run_id="run_l3_06")
        res = runner.execute("q6", "Compare X vs Y", top_k=3)

        tool_ids = [step.action for step in res.trajectory.steps]
        self.assertEqual(len(tool_ids), len(set(tool_ids)))

    def test_07_typed_domain_error_step_0_stops_with_error_not_sufficient(
        self,
    ) -> None:
        ports = {
            PipelineStrategy.BASELINE: FakeRetrievalPort("base"),
            PipelineStrategy.SENTENCE_WINDOW_RERANK: FailingRetrievalPort(),
        }
        runner = self._build_runner(ports, run_id="run_l3_07")
        res = runner.execute("q7", "Compare A vs B", top_k=3)

        self.assertIsNotNone(res.error)
        self.assertNotEqual(
            res.stop_decision.reason, StopReason.SUFFICIENT_EVIDENCE
        )
        self.assertEqual(len(res.trajectory.steps), 1)


if __name__ == "__main__":
    unittest.main()
