"""Contract tests for dual query channels (routing vs retrieval) in BoundedLoopRunner."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

from raglab.agentic.budget import Budget
from raglab.agentic.enums import StopReason
from raglab.agentic.runtime.bounded_loop_runner import BoundedLoopRunner
from raglab.agentic.runtime.dispatching_backend import (
    DispatchingRetrievalBackend,
)
from raglab.agentic.runtime.strategy_tool_factory import (
    build_registry_with_adapters,
)
from raglab.domain.entities import RetrievedEvidence
from raglab.domain.enums import PipelineStrategy
from raglab.domain.value_objects import ChunkId


@dataclass
class CapturingRetrievalPort:
    """Hermetic retrieval port capturing all queries received by the tool adapter."""

    passage_prefix: str
    results: list[RetrievedEvidence] | None = None
    captured_queries: list[str] = field(default_factory=list)
    call_count: int = 0

    def retrieve(self, query: str, top_k: int) -> list[RetrievedEvidence]:
        self.call_count += 1
        self.captured_queries.append(query)
        if self.results is not None:
            return self.results[:top_k]
        return [
            RetrievedEvidence(
                chunk_id=ChunkId(value=f"{self.passage_prefix}_001"),
                document_id="doc_100",
                text=f"Evidence text for {query}",
                rank=1,
                score=0.95,
                passage_id=f"ps_{self.passage_prefix}_001",
            )
        ][:top_k]


class TestBoundedLoopDualQuery(unittest.TestCase):
    """RED contract tests proving separation between routing query and retrieval query."""

    def _build_runner(
        self,
        ports: dict[PipelineStrategy, Any],
        run_id: str = "test_dual_query_run",
    ) -> BoundedLoopRunner:
        registry, adapters = build_registry_with_adapters(ports)
        dispatcher = DispatchingRetrievalBackend(adapters)
        budget = Budget(
            max_logical_calls=2,
            max_physical_attempts=2,
            max_retries=0,
        )
        return BoundedLoopRunner(
            registry=registry,
            budget=budget,
            backend=dispatcher,
            run_id=run_id,
        )

    def test_01_unanswerable_raw_query_routes_to_baseline_with_comparative_retrieval_query(
        self,
    ) -> None:
        base_port = CapturingRetrievalPort("base")
        window_port = CapturingRetrievalPort("window")
        ports = {
            PipelineStrategy.BASELINE: base_port,
            PipelineStrategy.SENTENCE_WINDOW_RERANK: window_port,
        }
        runner = self._build_runner(ports)

        raw_query = "Why is this topic not addressed in the book?"
        retrieval_query = (
            "Compare BM25 versus dense vector embeddings.\n\n"
            "Why is this topic not addressed in the book?"
        )

        res = runner.execute(
            query_id="q_dual_01",
            query_text=raw_query,
            top_k=3,
            retrieval_query_text=retrieval_query,
        )

        # 1. Routing decision must be governed strictly by raw_query (POSSIBLY_UNANSWERABLE -> baseline)
        self.assertEqual(
            res.routing_decision.selected_strategy, "baseline"
        )
        self.assertEqual(len(res.trajectory.steps), 1)
        self.assertEqual(
            res.trajectory.steps[0].action, "retrieve:retrieve_baseline"
        )
        # 2. Tool argument must receive retrieval_query_text
        self.assertEqual(base_port.call_count, 1)
        self.assertEqual(base_port.captured_queries, [retrieval_query])
        self.assertEqual(window_port.call_count, 0)
        self.assertEqual(
            res.stop_decision.reason, StopReason.SUFFICIENT_EVIDENCE
        )

    def test_02_comparative_raw_query_routes_to_sentence_window_with_unanswerable_retrieval_query(
        self,
    ) -> None:
        base_port = CapturingRetrievalPort("base")
        window_port = CapturingRetrievalPort("window")
        ports = {
            PipelineStrategy.BASELINE: base_port,
            PipelineStrategy.SENTENCE_WINDOW_RERANK: window_port,
        }
        runner = self._build_runner(ports)

        raw_query = "Compare sentence window retrieval with baseline chunking."
        retrieval_query = (
            "Why is this topic not addressed in the book?\n\n"
            "Compare sentence window retrieval with baseline chunking."
        )

        res = runner.execute(
            query_id="q_dual_02",
            query_text=raw_query,
            top_k=3,
            retrieval_query_text=retrieval_query,
        )

        # 1. Routing decision must be governed strictly by raw_query (COMPARISON -> sentence_window_rerank)
        self.assertEqual(
            res.routing_decision.selected_strategy, "sentence_window_rerank"
        )
        self.assertEqual(len(res.trajectory.steps), 1)
        self.assertEqual(
            res.trajectory.steps[0].action,
            "retrieve:retrieve_sentence_window_rerank",
        )
        # 2. Tool argument must receive retrieval_query_text
        self.assertEqual(window_port.call_count, 1)
        self.assertEqual(window_port.captured_queries, [retrieval_query])
        self.assertEqual(base_port.call_count, 0)
        self.assertEqual(
            res.stop_decision.reason, StopReason.SUFFICIENT_EVIDENCE
        )

    def test_03_complementary_fallback_passes_identical_retrieval_query_to_both_steps(
        self,
    ) -> None:
        # Step 0 tool (sentence_window_rerank) returns 0 evidence -> triggers Step 1 fallback (baseline)
        window_port = CapturingRetrievalPort("window", results=[])
        base_port = CapturingRetrievalPort("base")
        ports = {
            PipelineStrategy.BASELINE: base_port,
            PipelineStrategy.SENTENCE_WINDOW_RERANK: window_port,
        }
        runner = self._build_runner(ports)

        raw_query = "Compare sentence window vs baseline"
        retrieval_query = "Previous question context\n\nCompare sentence window vs baseline"

        res = runner.execute(
            query_id="q_dual_03",
            query_text=raw_query,
            top_k=3,
            retrieval_query_text=retrieval_query,
        )

        # Both steps must have been executed
        self.assertEqual(len(res.trajectory.steps), 2)
        self.assertEqual(
            res.trajectory.steps[0].action,
            "retrieve:retrieve_sentence_window_rerank",
        )
        self.assertEqual(
            res.trajectory.steps[1].action, "retrieve:retrieve_baseline"
        )

        # Both ports must have received the exact same retrieval_query
        self.assertEqual(window_port.call_count, 1)
        self.assertEqual(window_port.captured_queries, [retrieval_query])
        self.assertEqual(base_port.call_count, 1)
        self.assertEqual(base_port.captured_queries, [retrieval_query])
        self.assertEqual(res.evidence_count, 1)

    def test_04_forbidden_token_in_retrieval_query_triggers_leakage_rejection_before_port_call(
        self,
    ) -> None:
        base_port = CapturingRetrievalPort("base")
        window_port = CapturingRetrievalPort("window")
        ports = {
            PipelineStrategy.BASELINE: base_port,
            PipelineStrategy.SENTENCE_WINDOW_RERANK: window_port,
        }
        runner = self._build_runner(ports)

        clean_raw_query = "What is chunking in RAG?"
        leaked_retrieval_query = (
            "What are the best chunks in qrels?\n\nWhat is chunking in RAG?"
        )

        res = runner.execute(
            query_id="q_dual_04",
            query_text=clean_raw_query,
            top_k=3,
            retrieval_query_text=leaked_retrieval_query,
        )

        # 1. Stop decision must be TOOL_FAILURE with LeakageDetectedError detail
        self.assertEqual(
            res.stop_decision.reason, StopReason.TOOL_FAILURE
        )
        self.assertIsNotNone(res.error)
        self.assertIn("LeakageDetectedError", res.error or "")
        self.assertIn("LeakageDetectedError", res.stop_decision.detail or "")

        # 2. Ports must NOT have been called
        self.assertEqual(base_port.call_count, 0)
        self.assertEqual(window_port.call_count, 0)

    def test_05_forbidden_token_in_raw_routing_query_is_rejected_with_clean_retrieval_query(
        self,
    ) -> None:
        base_port = CapturingRetrievalPort("base")
        window_port = CapturingRetrievalPort("window")
        ports = {
            PipelineStrategy.BASELINE: base_port,
            PipelineStrategy.SENTENCE_WINDOW_RERANK: window_port,
        }
        runner = self._build_runner(ports)

        leaked_raw_query = "Compare qrels vs gold_answer for search evaluation"
        clean_retrieval_query = "Compare vector vs keyword search"

        res = runner.execute(
            query_id="q_dual_05",
            query_text=leaked_raw_query,
            top_k=3,
            retrieval_query_text=clean_retrieval_query,
        )

        # 1. Stop decision must be TOOL_FAILURE with LeakageDetectedError detail
        self.assertEqual(
            res.stop_decision.reason, StopReason.TOOL_FAILURE
        )
        self.assertIsNotNone(res.error)
        self.assertIn("LeakageDetectedError", res.error or "")
        self.assertIn("LeakageDetectedError", res.stop_decision.detail or "")

        # 2. Zero port calls on all adapters
        self.assertEqual(base_port.call_count, 0)
        self.assertEqual(window_port.call_count, 0)


if __name__ == "__main__":
    unittest.main()
