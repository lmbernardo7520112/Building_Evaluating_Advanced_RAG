"""Offline end-to-end canary tests for the agentic conversational runtime.

Validates the full pipeline:
ConversationSessionRunner -> BoundedLoopCoordinator -> RetrievalToolAdapter
-> RuntimePassageStore -> VerifiedPassageResolver -> AgenticGenerationBridge
-> FakeGeneratorAdapter -> ConversationState
"""

from __future__ import annotations

import hashlib
import unittest
from collections.abc import Sequence
from dataclasses import dataclass, field

from raglab.agentic.runtime.conversation_session_runner import (
    ConversationSessionRunner,
)
from raglab.application.ports.generation import GenerationPort
from raglab.domain.entities import GeneratedAnswer, RetrievedEvidence
from raglab.domain.enums import PipelineStrategy
from raglab.domain.value_objects import ChunkId
from raglab.infrastructure.fakes.fake_generator_adapter import (
    FakeGeneratorAdapter,
)


@dataclass
class RecordingRetrievalPort:
    """Hermetic retrieval port recording dispatched queries."""

    prefix: str
    empty: bool = False
    custom_passages: dict[str, Sequence[RetrievedEvidence]] | None = None
    captured_queries: list[str] = field(default_factory=list)

    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedEvidence]:
        self.captured_queries.append(query)
        if self.empty:
            return []
        if self.custom_passages is not None:
            matched_key = None
            max_pos = -1
            for query_key in self.custom_passages:
                pos = query.rfind(query_key)
                if pos > max_pos:
                    max_pos = pos
                    matched_key = query_key
            if matched_key is not None:
                return self.custom_passages[matched_key]
        text = f"Synthetic evidence for query: {query}"
        text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return [
            RetrievedEvidence(
                chunk_id=ChunkId(value=f"chk_{self.prefix}_001"),
                document_id=f"doc_{self.prefix}_001",
                text=text,
                rank=1,
                score=0.95,
                passage_id=f"ps_{self.prefix}_001",
                content_sha256=text_sha,
                page_number=10,
            )
        ]


class RecordingGenerationPort:
    """GenerationPort proxy recording all generator invocations."""

    def __init__(self, delegate: GenerationPort | None = None) -> None:
        self._delegate = delegate if delegate is not None else FakeGeneratorAdapter()
        self.captured_calls: list[tuple[str, str, Sequence[RetrievedEvidence]]] = []

    @property
    def model_id(self) -> str:
        return self._delegate.model_id

    def generate(
        self,
        query_id: str,
        query: str,
        evidence: Sequence[RetrievedEvidence],
    ) -> GeneratedAnswer:
        self.captured_calls.append((query_id, query, tuple(evidence)))
        return self._delegate.generate(query_id, query, evidence)


class TestConversationGenerationE2E(unittest.TestCase):
    """End-to-end canary tests with FakeGeneratorAdapter."""

    def test_01_direct_supported_fact_end_to_end(self) -> None:
        """Canary 1: single turn with direct supporting evidence produces verified answer."""
        fact_text = "A velocidade da luz no vácuo é exatamente 299.792.458 metros por segundo."
        fact_sha = hashlib.sha256(fact_text.encode("utf-8")).hexdigest()

        evidence_item = RetrievedEvidence(
            chunk_id=ChunkId(value="chk_physics_001"),
            document_id="doc_physics_001",
            text=fact_text,
            rank=1,
            score=0.98,
            passage_id="ps_fact_001",
            content_sha256=fact_sha,
            page_number=42,
        )

        custom_mapping = {"velocidade da luz": [evidence_item]}
        port_base = RecordingRetrievalPort("base", custom_passages=custom_mapping)
        port_window = RecordingRetrievalPort("window", custom_passages=custom_mapping)
        ports = {
            PipelineStrategy.BASELINE: port_base,
            PipelineStrategy.SENTENCE_WINDOW_RERANK: port_window,
        }

        generator_spy = RecordingGenerationPort()
        runner = ConversationSessionRunner(ports=ports, generator=generator_spy)
        runner.create_conversation("conv_canary1")

        turn = runner.execute_turn("conv_canary1", "Qual é a velocidade da luz no vácuo?", top_k=3)

        # 1. Answer verification
        self.assertIsNotNone(turn.generated_answer)
        self.assertFalse(turn.generated_answer.abstained)
        self.assertEqual(turn.generated_answer.query_id, "conv_canary1_turn_0_q")
        self.assertEqual(turn.generation_model_id, "fake-generator-v1-no-network")

        # 2. Generator spy verification
        self.assertEqual(len(generator_spy.captured_calls), 1)
        call_qid, call_query, call_evidence = generator_spy.captured_calls[0]
        self.assertEqual(call_qid, "conv_canary1_turn_0_q")
        self.assertEqual(call_query, "Qual é a velocidade da luz no vácuo?")
        self.assertEqual(len(call_evidence), 1)

        # 3. Citation exact provenance
        self.assertEqual(len(turn.generated_answer.citations), 1)
        citation = turn.generated_answer.citations[0]
        self.assertEqual(citation.passage_id, "ps_fact_001")
        self.assertEqual(citation.document_id, "doc_physics_001")
        self.assertEqual(citation.chunk_id.value, "chk_physics_001")
        self.assertEqual(citation.page_number, 42)
        self.assertEqual(citation.content_sha256, fact_sha)
        self.assertEqual(citation.retrieval_rank, 1)

        # 4. State persistence
        state = runner.get_conversation("conv_canary1")
        self.assertEqual(len(state.turns), 1)
        self.assertEqual(state.turns[0].state_hash, turn.state_hash)

    def test_02_conversational_ellipsis_uses_contextual_retrieval(self) -> None:
        """Canary 2: multi-turn with elliptical second query routes contextual query to retrieval."""
        text_t0 = "A relatividade geral é a teoria geométrica da gravitação publicada por Einstein."
        sha_t0 = hashlib.sha256(text_t0.encode("utf-8")).hexdigest()
        ev_t0 = RetrievedEvidence(
            chunk_id=ChunkId(value="chk_relativity_001"),
            document_id="doc_relativity_001",
            text=text_t0,
            rank=1,
            score=0.92,
            passage_id="ps_relativity_001",
            content_sha256=sha_t0,
            page_number=1,
        )

        text_t1 = "As comprovações da relatividade incluem o desvio da luz e a precessão do periélio."
        sha_t1 = hashlib.sha256(text_t1.encode("utf-8")).hexdigest()
        ev_t1 = RetrievedEvidence(
            chunk_id=ChunkId(value="chk_proofs_001"),
            document_id="doc_proofs_001",
            text=text_t1,
            rank=1,
            score=0.96,
            passage_id="ps_proofs_001",
            content_sha256=sha_t1,
            page_number=25,
        )

        custom_mapping = {
            "relatividade geral": [ev_t0],
            "comprovações": [ev_t1],
        }

        port_base = RecordingRetrievalPort("base", custom_passages=custom_mapping)
        port_window = RecordingRetrievalPort("window", custom_passages=custom_mapping)
        ports = {
            PipelineStrategy.BASELINE: port_base,
            PipelineStrategy.SENTENCE_WINDOW_RERANK: port_window,
        }

        generator_spy = RecordingGenerationPort()
        runner = ConversationSessionRunner(ports=ports, generator=generator_spy)
        runner.create_conversation("conv_canary2")

        # Turn 0
        turn0 = runner.execute_turn("conv_canary2", "O que é a teoria da relatividade geral?")
        self.assertFalse(turn0.generated_answer.abstained)
        self.assertEqual(turn0.generated_answer.citations[0].passage_id, "ps_relativity_001")

        queries_before_t1_base = len(port_base.captured_queries)
        queries_before_t1_window = len(port_window.captured_queries)

        # Turn 1 (elliptical query)
        turn1 = runner.execute_turn("conv_canary2", "Quais são as suas principais comprovações?")

        # 1. Retrieval query context check
        self.assertIn("O que é a teoria da relatividade geral?", turn1.retrieval_query)
        self.assertIn("Quais são as suas principais comprovações?", turn1.retrieval_query)

        # All retrieval port calls during turn 1 received exactly the contextual retrieval_query
        t1_queries_base = port_base.captured_queries[queries_before_t1_base:]
        t1_queries_window = port_window.captured_queries[queries_before_t1_window:]
        for q in t1_queries_base + t1_queries_window:
            self.assertEqual(q, turn1.retrieval_query)

        # 2. Generator query check: generator received ONLY raw second user query
        self.assertEqual(len(generator_spy.captured_calls), 2)
        call1_qid, call1_query, call1_evidence = generator_spy.captured_calls[1]
        self.assertEqual(call1_qid, "conv_canary2_turn_1_q")
        self.assertEqual(call1_query, "Quais são as suas principais comprovações?")
        self.assertEqual(call1_evidence[0].passage_id, "ps_proofs_001")

        # 3. Turn 1 citations do NOT reuse turn 0 passages
        self.assertEqual(len(turn1.generated_answer.citations), 1)
        self.assertEqual(turn1.generated_answer.citations[0].passage_id, "ps_proofs_001")
        self.assertEqual(turn1.generated_answer.citations[0].page_number, 25)

        # 4. State contains both turns with distinct state hashes
        state = runner.get_conversation("conv_canary2")
        self.assertEqual(len(state.turns), 2)
        self.assertEqual(state.turns[0].user_query, "O que é a teoria da relatividade geral?")
        self.assertEqual(state.turns[1].user_query, "Quais são as suas principais comprovações?")
        self.assertNotEqual(state.turns[0].state_hash, state.turns[1].state_hash)

    def test_03_unsupported_query_abstains_without_generator_call(self) -> None:
        """Canary 3: unsupported query with empty retrieval produces abstention without generator call."""
        port_base = RecordingRetrievalPort("base", empty=True)
        port_window = RecordingRetrievalPort("window", empty=True)
        ports = {
            PipelineStrategy.BASELINE: port_base,
            PipelineStrategy.SENTENCE_WINDOW_RERANK: port_window,
        }

        generator_spy = RecordingGenerationPort()
        runner = ConversationSessionRunner(ports=ports, generator=generator_spy)
        runner.create_conversation("conv_canary3")

        turn = runner.execute_turn("conv_canary3", "Pergunta sobre evento inexistente no corpus")

        # 1. Abstention verification
        self.assertIsNotNone(turn.generated_answer)
        self.assertTrue(turn.generated_answer.abstained)
        self.assertEqual(turn.generated_answer.text, "")
        self.assertEqual(turn.generated_answer.citations, ())
        self.assertIsNone(turn.generation_model_id)

        # 2. Generator spy: zero calls made
        self.assertEqual(len(generator_spy.captured_calls), 0)

        # 3. State persistence
        state = runner.get_conversation("conv_canary3")
        self.assertEqual(len(state.turns), 1)
        self.assertEqual(state.turns[0].state_hash, turn.state_hash)
        self.assertEqual(state.status.value, "ACTIVE")


if __name__ == "__main__":
    unittest.main()
