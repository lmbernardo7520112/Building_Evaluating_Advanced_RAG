"""Contract tests for L3B.1 ConversationSessionRunner."""

import unittest
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from raglab.domain.entities import RetrievedEvidence
from raglab.domain.enums import PipelineStrategy
from raglab.domain.value_objects import ChunkId


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
    """Hermetic retrieval port for L3B.1 contract tests."""

    prefix: str
    should_fail: bool = False

    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedEvidence]:
        if self.should_fail:
            raise RuntimeError("Hermetic retrieval port simulated failure")
        return [
            RetrievedEvidence(
                chunk_id=ChunkId(value=f"{self.prefix}_001"),
                document_id="doc_100",
                text=f"Evidence for query: {query}",
                rank=1,
                score=0.9,
                passage_id=f"ps_{self.prefix}_001",
            )
        ]


class TestConversationSessionRunner(unittest.TestCase):
    """Contract tests for ConversationSessionRunner."""

    def setUp(self) -> None:
        self.port_base = HermeticRetrievalPort("base")
        self.port_window = HermeticRetrievalPort("window")
        self.ports = {
            PipelineStrategy.BASELINE: self.port_base,
            PipelineStrategy.SENTENCE_WINDOW_RERANK: self.port_window,
        }

    def _make_runner(self, ports: dict[Any, Any] | None = None, max_turns: int = 3) -> Any:
        runner_cls, _, _, _ = _get_runner_types()
        return runner_cls(ports if ports is not None else self.ports, max_turns=max_turns)

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


if __name__ == "__main__":
    unittest.main()
