"""Contract tests for BoundedLoopResult evidence export."""

from __future__ import annotations

import hashlib
import unittest
from collections.abc import Sequence
from dataclasses import dataclass

from raglab.agentic.contracts import EvidenceItem
from raglab.agentic.runtime.bounded_loop_factory import (
    build_bounded_loop_coordinator,
)
from raglab.agentic.runtime.bounded_loop_runner import BoundedLoopResult
from raglab.domain.entities import RetrievedEvidence
from raglab.domain.enums import PipelineStrategy
from raglab.domain.value_objects import ChunkId


@dataclass
class FakeRetrievalPort:
    """Hermetic in-memory FakeRetrievalPort returning configurable evidence."""

    passage_prefix: str
    results: list[RetrievedEvidence] | None = None

    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedEvidence]:
        if self.results is not None:
            return self.results[:top_k]
        text = f"Sample text for {self.passage_prefix} query: {query}"
        return [
            RetrievedEvidence(
                chunk_id=ChunkId(value=f"{self.passage_prefix}_001"),
                document_id="doc_100",
                text=text,
                rank=1,
                score=0.95,
                passage_id=f"ps_{self.passage_prefix}_001",
                content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        ][:top_k]


class FailingRetrievalPort:
    """Port that raises a runtime backend error."""

    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedEvidence]:
        raise RuntimeError("simulated backend retrieval failure")


class TestBoundedLoopEvidenceExport(unittest.TestCase):
    """RED contract tests for immutable evidence export in BoundedLoopResult."""

    def test_01_success_exports_exact_evidence_metadata(self) -> None:
        text1 = "Sample text for window query: Compare X vs Y"
        text2 = "Second sample text for window query: Compare X vs Y"
        sha1 = hashlib.sha256(text1.encode("utf-8")).hexdigest()
        sha2 = hashlib.sha256(text2.encode("utf-8")).hexdigest()
        custom_results = [
            RetrievedEvidence(
                chunk_id=ChunkId(value="window_001"),
                document_id="doc_100",
                text=text1,
                rank=1,
                score=0.95,
                passage_id="ps_window_001",
                content_sha256=sha1,
            ),
            RetrievedEvidence(
                chunk_id=ChunkId(value="window_002"),
                document_id="doc_101",
                text=text2,
                rank=2,
                score=0.85,
                passage_id="ps_window_002",
                content_sha256=sha2,
            ),
        ]
        ports = {
            PipelineStrategy.BASELINE: FakeRetrievalPort("base"),
            PipelineStrategy.SENTENCE_WINDOW_RERANK: FakeRetrievalPort(
                "window", results=custom_results
            ),
        }
        runner = build_bounded_loop_coordinator(ports, run_id="run_exp_01")
        res: BoundedLoopResult = runner.execute(
            "q1", "Compare sentence window vs baseline", top_k=3
        )

        self.assertEqual(res.evidence_count, 2)
        self.assertIsInstance(res.evidence_items, tuple)
        self.assertEqual(len(res.evidence_items), 2)

        item1 = res.evidence_items[0]
        self.assertIsInstance(item1, EvidenceItem)
        self.assertEqual(item1.passage_id, "ps_window_001")
        self.assertEqual(item1.document_id, "doc_100")
        self.assertEqual(item1.rank, 1)
        self.assertEqual(item1.score, 0.95)
        self.assertEqual(item1.content_sha256, sha1)
        self.assertEqual(item1.source_tool_id, "retrieve_sentence_window_rerank")

        item2 = res.evidence_items[1]
        self.assertIsInstance(item2, EvidenceItem)
        self.assertEqual(item2.passage_id, "ps_window_002")
        self.assertEqual(item2.document_id, "doc_101")
        self.assertEqual(item2.rank, 2)
        self.assertEqual(item2.score, 0.85)
        self.assertEqual(item2.content_sha256, sha2)
        self.assertEqual(item2.source_tool_id, "retrieve_sentence_window_rerank")

    def test_02_fallback_exports_accumulated_second_step_evidence(self) -> None:
        text = "Sample text for base query: Compare X vs Y"
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        ports = {
            PipelineStrategy.BASELINE: FakeRetrievalPort("base"),
            PipelineStrategy.SENTENCE_WINDOW_RERANK: FakeRetrievalPort(
                "window", results=[]
            ),
        }
        runner = build_bounded_loop_coordinator(ports, run_id="run_exp_02")
        res: BoundedLoopResult = runner.execute("q2", "Compare X vs Y", top_k=3)

        self.assertEqual(res.evidence_count, 1)
        self.assertIsInstance(res.evidence_items, tuple)
        self.assertEqual(len(res.evidence_items), 1)

        item = res.evidence_items[0]
        self.assertIsInstance(item, EvidenceItem)
        self.assertEqual(item.passage_id, "ps_base_001")
        self.assertEqual(item.document_id, "doc_100")
        self.assertEqual(item.rank, 1)
        self.assertEqual(item.score, 0.95)
        self.assertEqual(item.content_sha256, sha)
        self.assertEqual(item.source_tool_id, "retrieve_baseline")

    def test_03_zero_evidence_exports_empty_tuple(self) -> None:
        ports = {
            PipelineStrategy.BASELINE: FakeRetrievalPort("base", results=[]),
            PipelineStrategy.SENTENCE_WINDOW_RERANK: FakeRetrievalPort(
                "window", results=[]
            ),
        }
        runner = build_bounded_loop_coordinator(ports, run_id="run_exp_03")
        res: BoundedLoopResult = runner.execute("q3", "Compare X vs Y", top_k=3)

        self.assertEqual(res.evidence_count, 0)
        self.assertIsInstance(res.evidence_items, tuple)
        self.assertEqual(res.evidence_items, ())

    def test_04_tool_failure_exports_no_evidence(self) -> None:
        ports = {
            PipelineStrategy.BASELINE: FakeRetrievalPort("base"),
            PipelineStrategy.SENTENCE_WINDOW_RERANK: FailingRetrievalPort(),
        }
        runner = build_bounded_loop_coordinator(ports, run_id="run_exp_04")
        res: BoundedLoopResult = runner.execute("q4", "Compare A vs B", top_k=3)

        self.assertIsNotNone(res.error)
        self.assertEqual(res.evidence_count, 0)
        self.assertIsInstance(res.evidence_items, tuple)
        self.assertEqual(res.evidence_items, ())

    def test_05_export_is_immutable_and_contains_no_text(self) -> None:
        ports = {
            PipelineStrategy.BASELINE: FakeRetrievalPort("base"),
            PipelineStrategy.SENTENCE_WINDOW_RERANK: FakeRetrievalPort(
                "window"
            ),
        }
        runner = build_bounded_loop_coordinator(ports, run_id="run_exp_05")
        res: BoundedLoopResult = runner.execute("q5", "Compare X vs Y", top_k=3)

        self.assertIsInstance(res.evidence_items, tuple)
        self.assertGreater(len(res.evidence_items), 0)

        for item in res.evidence_items:
            self.assertIsInstance(item, EvidenceItem)
            self.assertFalse(hasattr(item, "text"))

            with self.assertRaises((AttributeError, TypeError)):
                item.passage_id = "ps_mutated"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
