"""RED contract tests for AgenticGenerationBridge.

Defines the contract between the agentic orchestrator, VerifiedPassageResolver,
and GenerationPort.
"""

from __future__ import annotations

import hashlib
import unittest
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from raglab.agentic.contracts import EvidenceItem
from raglab.agentic.errors import PassageIntegrityError, PassageNotFoundError
from raglab.domain.entities import GeneratedAnswer, RetrievedEvidence
from raglab.domain.errors import (
    CitationProvenanceMismatchError,
    InvalidIdentifierError,
)
from raglab.domain.value_objects import ChunkId, Citation


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class MockResolver:
    """Mock implementation of VerifiedPassageResolver for bridge testing."""

    resolved_returns: Sequence[RetrievedEvidence] | None = None
    resolve_calls: list[Sequence[EvidenceItem]] = field(default_factory=list)
    raise_error: Exception | None = None

    def resolve(self, items: Sequence[EvidenceItem]) -> Sequence[RetrievedEvidence]:
        self.resolve_calls.append(items)
        if self.raise_error is not None:
            raise self.raise_error
        if self.resolved_returns is not None:
            return self.resolved_returns
        return tuple(
            RetrievedEvidence(
                chunk_id=ChunkId(f"chunk_{item.passage_id}"),
                document_id=item.document_id,
                text=f"Text for {item.passage_id}",
                rank=item.rank,
                score=item.score,
                passage_id=item.passage_id,
                content_sha256=item.content_sha256,
                page_number=getattr(item, "page_number", None),
            )
            for item in items
        )


@dataclass
class MockGenerator:
    """Mock implementation of GenerationPort for bridge testing."""

    model_id_value: str = "mock-generator-v1"
    generate_calls: list[tuple[str, str, Sequence[RetrievedEvidence]]] = field(
        default_factory=list
    )
    answer_return: GeneratedAnswer | None = None
    raise_error: Exception | None = None

    @property
    def model_id(self) -> str:
        return self.model_id_value

    def generate(
        self,
        query_id: str,
        query: str,
        evidence: Sequence[RetrievedEvidence],
    ) -> GeneratedAnswer:
        self.generate_calls.append((query_id, query, evidence))
        if self.raise_error is not None:
            raise self.raise_error
        if self.answer_return is not None:
            return self.answer_return
        citations = tuple(
            Citation(
                document_id=ev.document_id,
                page_number=ev.page_number if ev.page_number is not None else 0,
                chunk_id=ev.chunk_id,
                text_span=ev.text[:40],
                evidence_id=f"E{idx + 1}",
                passage_id=ev.passage_id,
                content_sha256=ev.content_sha256,
                retrieval_rank=ev.rank,
            )
            for idx, ev in enumerate(evidence)
        )
        return GeneratedAnswer(
            query_id=query_id,
            text=f"Answer for {query}",
            abstained=False,
            citations=citations,
        )


class TestAgenticGenerationBridge(unittest.TestCase):
    """RED test suite defining contract invariants for AgenticGenerationBridge."""

    def test_01_invalid_query_id_rejected_before_dependencies(self) -> None:
        from raglab.agentic.runtime.agentic_generation_bridge import (
            AgenticGenerationBridge,
        )

        resolver = MockResolver()
        generator = MockGenerator()
        bridge = AgenticGenerationBridge(resolver, generator)

        item = EvidenceItem(
            "ps_001", "doc_1", 1, 0.9, _sha256("t1"), "tool_1", "inv_1"
        )
        for bad_qid in ("", "   ", None):  # type: ignore[arg-type]
            with self.assertRaises(InvalidIdentifierError):
                bridge.generate(bad_qid, "Valid query?", (item,))

        self.assertEqual(len(resolver.resolve_calls), 0)
        self.assertEqual(len(generator.generate_calls), 0)

    def test_02_empty_evidence_abstains_without_dependency_calls(self) -> None:
        from raglab.agentic.runtime.agentic_generation_bridge import (
            AgenticGenerationBridge,
        )

        resolver = MockResolver()
        generator = MockGenerator()
        bridge = AgenticGenerationBridge(resolver, generator)

        answer = bridge.generate("q_001", "Valid query?", ())

        self.assertEqual(answer.query_id, "q_001")
        self.assertEqual(answer.text, "")
        self.assertTrue(answer.abstained)
        self.assertEqual(answer.citations, ())
        self.assertEqual(len(resolver.resolve_calls), 0)
        self.assertEqual(len(generator.generate_calls), 0)

    def test_03_resolves_and_delegates_in_exact_input_order(self) -> None:
        from raglab.agentic.runtime.agentic_generation_bridge import (
            AgenticGenerationBridge,
        )

        item1 = EvidenceItem(
            "ps_001", "doc_1", 1, 0.95, _sha256("t1"), "tool_1", "inv_1"
        )
        item2 = EvidenceItem(
            "ps_002", "doc_1", 2, 0.85, _sha256("t2"), "tool_1", "inv_1"
        )
        item3 = EvidenceItem(
            "ps_003", "doc_1", 3, 0.75, _sha256("t3"), "tool_1", "inv_1"
        )

        ev1 = RetrievedEvidence(
            ChunkId("c1"), "doc_1", "t1", 1, 0.95, "ps_001", _sha256("t1"), 10
        )
        ev2 = RetrievedEvidence(
            ChunkId("c2"), "doc_1", "t2", 2, 0.85, "ps_002", _sha256("t2"), 20
        )
        ev3 = RetrievedEvidence(
            ChunkId("c3"), "doc_1", "t3", 3, 0.75, "ps_003", _sha256("t3"), 30
        )

        resolver = MockResolver(resolved_returns=(ev1, ev2, ev3))
        generator = MockGenerator()
        bridge = AgenticGenerationBridge(resolver, generator)

        items = (item1, item2, item3)
        answer = bridge.generate("q_001", "What is RAG?", items)

        self.assertEqual(len(resolver.resolve_calls), 1)
        self.assertEqual(resolver.resolve_calls[0], items)

        self.assertEqual(len(generator.generate_calls), 1)
        qid_arg, q_arg, ev_arg = generator.generate_calls[0]
        self.assertEqual(qid_arg, "q_001")
        self.assertEqual(q_arg, "What is RAG?")
        self.assertEqual(tuple(ev_arg), (ev1, ev2, ev3))

        self.assertEqual(answer.query_id, "q_001")
        self.assertFalse(answer.abstained)

    def test_04_model_id_delegates_to_generator(self) -> None:
        from raglab.agentic.runtime.agentic_generation_bridge import (
            AgenticGenerationBridge,
        )

        resolver = MockResolver()
        generator = MockGenerator(model_id_value="gemini-3.1-flash-lite-agentic-v1")
        bridge = AgenticGenerationBridge(resolver, generator)

        self.assertEqual(bridge.model_id, "gemini-3.1-flash-lite-agentic-v1")

    def test_05_passage_integrity_error_propagates_before_generation(self) -> None:
        from raglab.agentic.runtime.agentic_generation_bridge import (
            AgenticGenerationBridge,
        )

        resolver = MockResolver(
            raise_error=PassageIntegrityError("SHA-256 mismatch detected")
        )
        generator = MockGenerator()
        bridge = AgenticGenerationBridge(resolver, generator)

        item = EvidenceItem(
            "ps_001", "doc_1", 1, 0.9, _sha256("t1"), "tool_1", "inv_1"
        )
        with self.assertRaises(PassageIntegrityError):
            bridge.generate("q_001", "Query?", (item,))

        self.assertEqual(len(generator.generate_calls), 0)

    def test_06_passage_not_found_error_propagates_before_generation(self) -> None:
        from raglab.agentic.runtime.agentic_generation_bridge import (
            AgenticGenerationBridge,
        )

        resolver = MockResolver(
            raise_error=PassageNotFoundError("ps_missing_999")
        )
        generator = MockGenerator()
        bridge = AgenticGenerationBridge(resolver, generator)

        item = EvidenceItem(
            "ps_missing_999", "doc_1", 1, 0.9, _sha256("t1"), "tool_1", "inv_1"
        )
        with self.assertRaises(PassageNotFoundError):
            bridge.generate("q_001", "Query?", (item,))

        self.assertEqual(len(generator.generate_calls), 0)

    def test_07_generator_domain_error_propagates(self) -> None:
        from raglab.agentic.runtime.agentic_generation_bridge import (
            AgenticGenerationBridge,
        )

        resolver = MockResolver()
        generator = MockGenerator(
            raise_error=CitationProvenanceMismatchError("E_UNKNOWN")
        )
        bridge = AgenticGenerationBridge(resolver, generator)

        item = EvidenceItem(
            "ps_001", "doc_1", 1, 0.9, _sha256("t1"), "tool_1", "inv_1"
        )
        with self.assertRaises(CitationProvenanceMismatchError):
            bridge.generate("q_001", "Query?", (item,))

        self.assertEqual(len(resolver.resolve_calls), 1)

    def test_08_accepts_answer_with_matching_citation_provenance(self) -> None:
        from raglab.agentic.runtime.agentic_generation_bridge import (
            AgenticGenerationBridge,
        )

        t1 = "Evidence text one"
        sha1 = _sha256(t1)
        ev1 = RetrievedEvidence(
            chunk_id=ChunkId("real_chunk_001"),
            document_id="doc_101",
            text=t1,
            rank=1,
            score=0.95,
            passage_id="ps_001",
            content_sha256=sha1,
            page_number=15,
        )
        resolver = MockResolver(resolved_returns=(ev1,))
        expected_citation = Citation(
            document_id="doc_101",
            page_number=15,
            chunk_id=ChunkId("real_chunk_001"),
            text_span=t1[:40],
            evidence_id="E1",
            passage_id="ps_001",
            content_sha256=sha1,
            retrieval_rank=1,
        )
        expected_answer = GeneratedAnswer(
            query_id="q_001",
            text="Valid answer text referencing E1",
            abstained=False,
            citations=(expected_citation,),
        )
        generator = MockGenerator(answer_return=expected_answer)
        bridge = AgenticGenerationBridge(resolver, generator)

        item1 = EvidenceItem("ps_001", "doc_101", 1, 0.95, sha1, "tool_1", "inv_1")
        answer = bridge.generate("q_001", "Question?", (item1,))

        self.assertEqual(answer, expected_answer)
        self.assertEqual(len(answer.citations), 1)
        c = answer.citations[0]
        self.assertEqual(c.passage_id, "ps_001")
        self.assertEqual(c.document_id, "doc_101")
        self.assertEqual(c.chunk_id, ChunkId("real_chunk_001"))
        self.assertEqual(c.content_sha256, sha1)
        self.assertEqual(c.retrieval_rank, 1)
        self.assertEqual(c.page_number, 15)

    def test_09_rejects_generated_answer_query_id_mismatch(self) -> None:
        from raglab.agentic.runtime.agentic_generation_bridge import (
            AgenticGenerationBridge,
        )

        t1 = "Evidence text one"
        sha1 = _sha256(t1)
        ev1 = RetrievedEvidence(
            chunk_id=ChunkId("chunk_1"),
            document_id="doc_1",
            text=t1,
            rank=1,
            score=0.9,
            passage_id="ps_001",
            content_sha256=sha1,
            page_number=5,
        )
        resolver = MockResolver(resolved_returns=(ev1,))
        mismatched_answer = GeneratedAnswer(
            query_id="q_DIFFERENT_999",
            text="Answer text",
            abstained=False,
            citations=(
                Citation(
                    document_id="doc_1",
                    page_number=5,
                    chunk_id=ChunkId("chunk_1"),
                    text_span=t1[:40],
                    evidence_id="E1",
                    passage_id="ps_001",
                    content_sha256=sha1,
                    retrieval_rank=1,
                ),
            ),
        )
        generator = MockGenerator(answer_return=mismatched_answer)
        bridge = AgenticGenerationBridge(resolver, generator)

        item = EvidenceItem("ps_001", "doc_1", 1, 0.9, sha1, "t1", "inv_1")
        with pytest.raises(InvalidIdentifierError):
            bridge.generate("q_001", "Question?", (item,))

        self.assertEqual(len(resolver.resolve_calls), 1)
        self.assertEqual(len(generator.generate_calls), 1)

    def test_10_rejects_citation_identity_or_explicit_page_mismatch(self) -> None:
        from raglab.agentic.runtime.agentic_generation_bridge import (
            AgenticGenerationBridge,
        )

        t1 = "Evidence text one"
        sha1 = _sha256(t1)
        ev1 = RetrievedEvidence(
            chunk_id=ChunkId("c1"),
            document_id="doc_1",
            text=t1,
            rank=1,
            score=0.9,
            passage_id="ps_001",
            content_sha256=sha1,
            page_number=10,
        )
        resolver = MockResolver(resolved_returns=(ev1,))

        # Sub-case A: passage_id mismatch
        bad_passage_answer = GeneratedAnswer(
            query_id="q_001",
            text="Ans",
            abstained=False,
            citations=(
                Citation(
                    document_id="doc_1",
                    page_number=10,
                    chunk_id=ChunkId("c1"),
                    text_span=t1[:40],
                    evidence_id="E1",
                    passage_id="ps_WRONG_999",
                    content_sha256=sha1,
                    retrieval_rank=1,
                ),
            ),
        )
        generator_bad_p = MockGenerator(answer_return=bad_passage_answer)
        bridge_bad_p = AgenticGenerationBridge(resolver, generator_bad_p)
        item = EvidenceItem("ps_001", "doc_1", 1, 0.9, sha1, "t1", "inv_1")
        with self.assertRaises(CitationProvenanceMismatchError):
            bridge_bad_p.generate("q_001", "Query?", (item,))

        # Sub-case B: explicit page mismatch when RetrievedEvidence.page_number is 10 but Citation has 99
        bad_page_answer = GeneratedAnswer(
            query_id="q_001",
            text="Ans",
            abstained=False,
            citations=(
                Citation(
                    document_id="doc_1",
                    page_number=99,
                    chunk_id=ChunkId("c1"),
                    text_span=t1[:40],
                    evidence_id="E1",
                    passage_id="ps_001",
                    content_sha256=sha1,
                    retrieval_rank=1,
                ),
            ),
        )
        generator_bad_page = MockGenerator(answer_return=bad_page_answer)
        bridge_bad_page = AgenticGenerationBridge(resolver, generator_bad_page)
        with self.assertRaises(CitationProvenanceMismatchError):
            bridge_bad_page.generate("q_001", "Query?", (item,))

        # Sub-case C: content_sha256 mismatch
        bad_sha_answer = GeneratedAnswer(
            query_id="q_001",
            text="Ans",
            abstained=False,
            citations=(
                Citation(
                    document_id="doc_1",
                    page_number=10,
                    chunk_id=ChunkId("c1"),
                    text_span=t1[:40],
                    evidence_id="E1",
                    passage_id="ps_001",
                    content_sha256=_sha256("TAMPERED"),
                    retrieval_rank=1,
                ),
            ),
        )
        generator_bad_sha = MockGenerator(answer_return=bad_sha_answer)
        bridge_bad_sha = AgenticGenerationBridge(resolver, generator_bad_sha)
        with self.assertRaises(CitationProvenanceMismatchError):
            bridge_bad_sha.generate("q_001", "Query?", (item,))
