"""Contract tests for L3B.1 / L3B.2 ConversationSessionRunner."""

from __future__ import annotations

import hashlib
import json
import unittest
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from raglab.agentic.runtime.passage_resolver import PassageIntegrityError
from raglab.domain.entities import GeneratedAnswer, RetrievedEvidence
from raglab.domain.enums import PipelineStrategy
from raglab.domain.value_objects import ChunkId, Citation


def _get_runner_types() -> tuple[Any, Any, Any, Any]:
    """Import conversation_session_runner components at test execution time."""
    from raglab.agentic.runtime.conversation_session_runner import (
        ConversationSessionRunner,
        ConversationState,
        ConversationStatus,
        ConversationTurnResult,
    )

    return (
        ConversationSessionRunner,
        ConversationState,
        ConversationStatus,
        ConversationTurnResult,
    )


@dataclass
class HermeticRetrievalPort:
    """Hermetic retrieval port for L3B contract tests."""

    prefix: str
    should_fail: bool = False
    empty: bool = False
    corrupt_hash: bool = False
    custom_text: str | None = None
    page_number: int | None = 1

    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedEvidence]:
        if self.should_fail:
            raise RuntimeError("Hermetic retrieval port simulated failure")
        if self.empty:
            return []
        text = (
            self.custom_text
            if self.custom_text is not None
            else f"Evidence for query: {query}"
        )
        text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        sha = (
            "bad_hash_0000000000000000000000000000000000000000000000000000000000"
            if self.corrupt_hash
            else text_sha
        )
        return [
            RetrievedEvidence(
                chunk_id=ChunkId(value=f"{self.prefix}_001"),
                document_id="doc_100",
                text=text,
                rank=1,
                score=0.9,
                passage_id=f"ps_{self.prefix}_001",
                content_sha256=sha,
                page_number=self.page_number,
            )
        ]


class HermeticGenerator:
    """Hermetic generator double satisfying GenerationPort."""

    def __init__(
        self,
        model_id: str = "mock-agentic-gen-v1",
        should_fail: bool = False,
        custom_text: str = "Generated answer text",
        abstained: bool = False,
        sentinel_marker: str | None = None,
        custom_evidence_id: str | None = None,
        custom_text_span: str | None = None,
        custom_page_number: int | None = None,
    ) -> None:
        self._model_id = model_id
        self.should_fail = should_fail
        self.custom_text = custom_text
        self.abstained = abstained
        self.sentinel_marker = sentinel_marker
        self.custom_evidence_id = custom_evidence_id
        self.custom_text_span = custom_text_span
        self.custom_page_number = custom_page_number
        self.captured_queries: list[tuple[str, str, Sequence[RetrievedEvidence]]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def __repr__(self) -> str:
        return f"HermeticGenerator(model_id={self._model_id!r}, marker={self.sentinel_marker!r})"

    def generate(
        self,
        query_id: str,
        query: str,
        evidence: Sequence[RetrievedEvidence],
    ) -> GeneratedAnswer:
        if self.should_fail:
            raise RuntimeError("Hermetic generator simulated failure")
        self.captured_queries.append((query_id, query, tuple(evidence)))
        citations: list[Citation] = []
        if not self.abstained:
            for ev in evidence:
                page_no = (
                    self.custom_page_number
                    if self.custom_page_number is not None
                    else (ev.page_number if ev.page_number is not None else 1)
                )
                span = (
                    self.custom_text_span
                    if self.custom_text_span is not None
                    else (ev.text[:20] if ev.text else "span")
                )
                ev_id = (
                    self.custom_evidence_id
                    if self.custom_evidence_id is not None
                    else f"ev_{ev.chunk_id.value}"
                )
                citations.append(
                    Citation(
                        document_id=ev.document_id,
                        page_number=page_no,
                        chunk_id=ev.chunk_id,
                        text_span=span,
                        evidence_id=ev_id,
                        passage_id=ev.passage_id,
                        content_sha256=ev.content_sha256,
                        retrieval_rank=ev.rank,
                    )
                )
        return GeneratedAnswer(
            query_id=query_id,
            text="" if self.abstained else self.custom_text,
            abstained=self.abstained,
            citations=tuple(citations),
        )


class TestConversationSessionRunner(unittest.TestCase):
    """Contract tests for ConversationSessionRunner."""

    def setUp(self) -> None:
        self.port_base = HermeticRetrievalPort("base")
        self.port_window = HermeticRetrievalPort("window")
        self.ports = {
            PipelineStrategy.BASELINE: self.port_base,
            PipelineStrategy.SENTENCE_WINDOW_RERANK: self.port_window,
        }

    def _make_runner(
        self,
        ports: dict[Any, Any] | None = None,
        max_turns: int = 3,
        generator: Any = None,
        clock: Any = None,
    ) -> Any:
        runner_cls, _, _, _ = _get_runner_types()
        kwargs: dict[str, Any] = {
            "ports": ports if ports is not None else self.ports,
            "max_turns": max_turns,
        }
        if generator is not None:
            kwargs["generator"] = generator
        if clock is not None:
            kwargs["clock"] = clock
        return runner_cls(**kwargs)

    def test_01_creates_active_empty_conversation(self) -> None:
        _, state_cls, status_enum, _ = _get_runner_types()
        runner = self._make_runner()
        state = runner.create_conversation("conv_01")

        self.assertIsInstance(state, state_cls)
        self.assertEqual(state.conversation_id, "conv_01")
        self.assertEqual(state.turns, ())
        self.assertEqual(state.status, status_enum.ACTIVE)
        self.assertTrue(state.created_at)
        self.assertTrue(state.updated_at)

    def test_02_rejects_empty_and_duplicate_conversation_id(self) -> None:
        runner = self._make_runner()
        with self.assertRaises(ValueError):
            runner.create_conversation("")

        runner.create_conversation("conv_dup")
        with self.assertRaises(ValueError):
            runner.create_conversation("conv_dup")

        with self.assertRaises(ValueError):
            runner.execute_turn("unknown_conv", "query")

        with self.assertRaises((KeyError, ValueError)):
            runner.get_conversation("unknown_conv")

    def test_03_executes_turn_0_with_fresh_l3a_composition(self) -> None:
        _, _, _, turn_result_cls = _get_runner_types()
        runner = self._make_runner()
        runner.create_conversation("conv_03")

        turn0 = runner.execute_turn("conv_03", "Compare A vs B", top_k=3)
        self.assertIsInstance(turn0, turn_result_cls)
        self.assertEqual(turn0.conversation_id, "conv_03")
        self.assertEqual(turn0.turn_index, 0)
        self.assertEqual(turn0.user_query, "Compare A vs B")
        self.assertTrue(turn0.bounded_loop_result.trajectory.steps)
        self.assertTrue(turn0.state_hash)

    def test_04_registers_sequential_turn_indices_0_1_2(self) -> None:
        runner = self._make_runner()
        runner.create_conversation("conv_seq")

        turn0 = runner.execute_turn("conv_seq", "Turn 0 query")
        turn1 = runner.execute_turn("conv_seq", "Turn 1 query")
        turn2 = runner.execute_turn("conv_seq", "Turn 2 query")

        self.assertEqual(turn0.turn_index, 0)
        self.assertEqual(turn1.turn_index, 1)
        self.assertEqual(turn2.turn_index, 2)

        state = runner.get_conversation("conv_seq")
        self.assertEqual(len(state.turns), 3)

    def test_05_third_turn_preserved_and_status_changes_to_exhausted(self) -> None:
        _, _, status_enum, _ = _get_runner_types()
        runner = self._make_runner()
        runner.create_conversation("conv_exh")

        runner.execute_turn("conv_exh", "Q0")
        runner.execute_turn("conv_exh", "Q1")
        turn2 = runner.execute_turn("conv_exh", "Q2")

        self.assertEqual(turn2.turn_index, 2)
        state = runner.get_conversation("conv_exh")
        self.assertEqual(state.status, status_enum.EXHAUSTED)
        self.assertEqual(len(state.turns), 3)

    def test_06_fourth_turn_rejected_before_retrieval(self) -> None:
        call_count = 0

        @dataclass
        class TrackingPort:
            def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedEvidence]:
                nonlocal call_count
                call_count += 1
                return [
                    RetrievedEvidence(
                        chunk_id=ChunkId(value="t_001"),
                        document_id="doc_1",
                        text="txt",
                        rank=1,
                        score=0.9,
                        passage_id="ps_t_001",
                    )
                ]

        tracking_ports = {
            PipelineStrategy.BASELINE: TrackingPort(),
            PipelineStrategy.SENTENCE_WINDOW_RERANK: TrackingPort(),
        }

        runner = self._make_runner(ports=tracking_ports)
        runner.create_conversation("conv_max")

        runner.execute_turn("conv_max", "Q0")
        runner.execute_turn("conv_max", "Q1")
        runner.execute_turn("conv_max", "Q2")

        calls_before = call_count
        with self.assertRaises(ValueError):
            runner.execute_turn("conv_max", "Q3")

        self.assertEqual(call_count, calls_before)

    def test_07_two_conversations_maintain_strict_state_and_budget_isolation(self) -> None:
        _, _, status_enum, _ = _get_runner_types()
        runner = self._make_runner()

        runner.create_conversation("conv_a")
        runner.create_conversation("conv_b")

        runner.execute_turn("conv_a", "Query A1")
        runner.execute_turn("conv_a", "Query A2")

        state_a = runner.get_conversation("conv_a")
        state_b = runner.get_conversation("conv_b")

        self.assertEqual(len(state_a.turns), 2)
        self.assertEqual(state_a.status, status_enum.ACTIVE)
        self.assertEqual(state_b.turns, ())
        self.assertEqual(state_b.status, status_enum.ACTIVE)

        turn_b0 = runner.execute_turn("conv_b", "Query B1")
        self.assertEqual(turn_b0.turn_index, 0)
        self.assertEqual(len(runner.get_conversation("conv_b").turns), 1)

    def test_08_unexpected_exception_propagates_without_partial_turn_write(self) -> None:
        from unittest.mock import MagicMock, patch

        runner = self._make_runner()
        state_before = runner.create_conversation("conv_panic")

        mock_coordinator = MagicMock()
        mock_coordinator.execute.side_effect = RuntimeError("Unexpected bounded loop failure")

        with (
            patch(
                "raglab.agentic.runtime.conversation_session_runner.build_bounded_loop_coordinator",
                return_value=mock_coordinator,
            ),
            self.assertRaises(RuntimeError),
        ):
            runner.execute_turn("conv_panic", "Panic query")

        state_after = runner.get_conversation("conv_panic")
        self.assertEqual(state_after.turns, state_before.turns)
        self.assertEqual(state_after.turns, ())
        self.assertEqual(state_after.status, state_before.status)
        self.assertEqual(state_after.updated_at, state_before.updated_at)

    def test_09_execute_turn_with_generator_produces_verified_generated_answer(self) -> None:
        generator = HermeticGenerator(model_id="gen-test-09", custom_text="Answer for test 09")
        runner = self._make_runner(generator=generator)
        runner.create_conversation("conv_09")

        turn = runner.execute_turn("conv_09", "Explain pipeline architecture")

        self.assertIsNotNone(turn.generated_answer)
        self.assertEqual(turn.generated_answer.text, "Answer for test 09")
        self.assertFalse(turn.generated_answer.abstained)
        self.assertEqual(turn.generated_answer.query_id, "conv_09_turn_0_q")
        self.assertEqual(turn.generation_model_id, "gen-test-09")
        self.assertTrue(len(turn.generated_answer.citations) >= 1)
        self.assertTrue(turn.generated_answer.citations[0].passage_id.startswith("ps_"))

        # Session state committed only after successful generation
        state = runner.get_conversation("conv_09")
        self.assertEqual(len(state.turns), 1)
        self.assertEqual(state.turns[0].state_hash, turn.state_hash)

    def test_10_execute_turn_without_generator_preserves_none_answer_and_legacy_hash(self) -> None:
        runner = self._make_runner()
        runner.create_conversation("conv_10")

        turn = runner.execute_turn("conv_10", "Legacy query without generator")

        self.assertIsNone(turn.generated_answer)
        self.assertIsNone(turn.generation_model_id)

        legacy_payload = {
            "conversation_id": "conv_10",
            "turn_index": 0,
            "user_query": "Legacy query without generator",
            "config_sha256": turn.bounded_loop_result.trajectory.config_sha256,
            "stop_reason": turn.bounded_loop_result.stop_decision.reason.value,
            "evidence_count": turn.bounded_loop_result.evidence_count,
        }
        canonical_json = json.dumps(legacy_payload, sort_keys=True, separators=(",", ":"))
        expected_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        self.assertEqual(turn.state_hash, expected_hash)

    def test_11_empty_evidence_produces_bridge_abstention_without_generator_call(self) -> None:
        empty_ports = {
            PipelineStrategy.BASELINE: HermeticRetrievalPort("base", empty=True),
            PipelineStrategy.SENTENCE_WINDOW_RERANK: HermeticRetrievalPort("window", empty=True),
        }
        generator = HermeticGenerator(model_id="gen-test-11")
        runner = self._make_runner(ports=empty_ports, generator=generator)
        runner.create_conversation("conv_11")

        turn = runner.execute_turn("conv_11", "Query yielding empty evidence")

        self.assertIsNotNone(turn.generated_answer)
        self.assertTrue(turn.generated_answer.abstained)
        self.assertEqual(turn.generated_answer.text, "")
        self.assertEqual(turn.generated_answer.citations, ())
        self.assertIsNone(turn.generation_model_id)
        self.assertEqual(len(generator.captured_queries), 0)

        # State committed with abstained turn
        state = runner.get_conversation("conv_11")
        self.assertEqual(len(state.turns), 1)

    def test_12_resolution_and_generation_failures_leave_session_unchanged(self) -> None:
        # Subcase a: Passage integrity error (corrupt hash in port)
        corrupt_ports = {
            PipelineStrategy.BASELINE: HermeticRetrievalPort("base", corrupt_hash=True),
            PipelineStrategy.SENTENCE_WINDOW_RERANK: HermeticRetrievalPort("window", corrupt_hash=True),
        }
        gen_a = HermeticGenerator(model_id="gen-12a")
        runner_a = self._make_runner(ports=corrupt_ports, generator=gen_a)
        state_before_a = runner_a.create_conversation("conv_corrupt")

        with self.assertRaises(PassageIntegrityError):
            runner_a.execute_turn("conv_corrupt", "Query targeting corrupt passage")

        state_after_a = runner_a.get_conversation("conv_corrupt")
        self.assertEqual(state_after_a.turns, state_before_a.turns)
        self.assertEqual(state_after_a.turns, ())
        self.assertEqual(state_after_a.status, state_before_a.status)
        self.assertEqual(state_after_a.updated_at, state_before_a.updated_at)

        # Subcase b: Generator failure
        failing_gen = HermeticGenerator(model_id="gen-12b", should_fail=True)
        runner_b = self._make_runner(generator=failing_gen)
        state_before_b = runner_b.create_conversation("conv_gen_fail")

        with self.assertRaises(RuntimeError):
            runner_b.execute_turn("conv_gen_fail", "Query with failing generator")

        state_after_b = runner_b.get_conversation("conv_gen_fail")
        self.assertEqual(state_after_b.turns, state_before_b.turns)
        self.assertEqual(state_after_b.turns, ())
        self.assertEqual(state_after_b.status, state_before_b.status)
        self.assertEqual(state_after_b.updated_at, state_before_b.updated_at)

    def test_13_turns_and_conversations_have_strict_passage_store_isolation(self) -> None:
        """Sequential turns with differing text for the same passage_id must succeed without conflict."""
        port_base = HermeticRetrievalPort("base", custom_text="Version 1 payload")
        port_window = HermeticRetrievalPort("window", custom_text="Version 1 payload")
        ports = {
            PipelineStrategy.BASELINE: port_base,
            PipelineStrategy.SENTENCE_WINDOW_RERANK: port_window,
        }
        generator = HermeticGenerator(model_id="gen-test-13")
        runner = self._make_runner(ports=ports, generator=generator)
        runner.create_conversation("conv_13")

        turn0 = runner.execute_turn("conv_13", "Turn 0")
        self.assertIsNotNone(turn0.generated_answer)

        # Mutate port text for turn 1 (same passage_id ps_window_001 / ps_base_001)
        port_base.custom_text = "Version 2 conflicting payload"
        port_window.custom_text = "Version 2 conflicting payload"

        turn1 = runner.execute_turn("conv_13", "Turn 1")
        self.assertIsNotNone(turn1.generated_answer)
        self.assertEqual(len(runner.get_conversation("conv_13").turns), 2)

    def test_14_state_hash_binds_answer_model_and_every_citation_field(self) -> None:
        fixed_clock = lambda: "2026-08-15T12:00:00Z"  # noqa: E731

        def run_with_gen(
            gen: HermeticGenerator,
            ports: dict[Any, Any] | None = None,
        ) -> str:
            runner = self._make_runner(ports=ports, generator=gen, clock=fixed_clock)
            runner.create_conversation("conv_hash")
            turn = runner.execute_turn("conv_hash", "Hash test query")
            return turn.state_hash

        base_hash = run_with_gen(HermeticGenerator(model_id="base-model", custom_text="Base text"))

        # Variation 1: changed answer text
        hash_text = run_with_gen(HermeticGenerator(model_id="base-model", custom_text="Modified text"))
        self.assertNotEqual(base_hash, hash_text)

        # Variation 2: changed model_id
        hash_model = run_with_gen(HermeticGenerator(model_id="other-model", custom_text="Base text"))
        self.assertNotEqual(base_hash, hash_model)

        # Variation 3: changed text_span
        hash_span = run_with_gen(
            HermeticGenerator(
                model_id="base-model",
                custom_text="Base text",
                custom_text_span="different span text",
            )
        )
        self.assertNotEqual(base_hash, hash_span)

        # Variation 4: changed evidence_id
        hash_ev_id = run_with_gen(
            HermeticGenerator(
                model_id="base-model",
                custom_text="Base text",
                custom_evidence_id="custom_ev_999",
            )
        )
        self.assertNotEqual(base_hash, hash_ev_id)

        # Variation 5: changed page_number
        page_42_ports = {
            PipelineStrategy.BASELINE: HermeticRetrievalPort("base", page_number=42),
            PipelineStrategy.SENTENCE_WINDOW_RERANK: HermeticRetrievalPort("window", page_number=42),
        }
        hash_page = run_with_gen(
            HermeticGenerator(
                model_id="base-model",
                custom_text="Base text",
                custom_page_number=42,
            ),
            ports=page_42_ports,
        )
        self.assertNotEqual(base_hash, hash_page)

    def test_15_runner_repr_excludes_generator_and_result_exposes_model_id(self) -> None:
        sentinel_pattern = "SENTINEL_PATTERN_987654"
        generator = HermeticGenerator(model_id="safe-model-v1", sentinel_marker=sentinel_pattern)
        runner = self._make_runner(generator=generator)

        runner_repr = repr(runner)
        self.assertNotIn(sentinel_pattern, runner_repr)

        runner.create_conversation("conv_15")
        turn = runner.execute_turn("conv_15", "Check model_id exposure")
        self.assertEqual(turn.generation_model_id, "safe-model-v1")

    def test_16_generation_uses_raw_query_while_retrieval_uses_contextual_query(self) -> None:
        generator = HermeticGenerator(model_id="gen-test-16")
        runner = self._make_runner(generator=generator)
        runner.create_conversation("conv_16")

        runner.execute_turn("conv_16", "First user query")
        turn1 = runner.execute_turn("conv_16", "Second user query")

        self.assertIn("First user query", turn1.retrieval_query)
        self.assertIn("Second user query", turn1.retrieval_query)
        self.assertEqual(turn1.user_query, "Second user query")

        # Generator received raw user_query and canonical query_id
        self.assertEqual(len(generator.captured_queries), 2)
        query_id, query_text, evidence = generator.captured_queries[1]
        self.assertEqual(query_id, "conv_16_turn_1_q")
        self.assertEqual(query_text, "Second user query")
        self.assertTrue(len(evidence) >= 1)


if __name__ == "__main__":
    unittest.main()
