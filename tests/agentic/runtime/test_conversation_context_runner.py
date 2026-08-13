"""Unit tests for L3B.2-B bounded conversation context in session runner.

Verifies:
1. Turn 0 uses raw user query as retrieval query and computes context_sha256.
2. Turn 1 prepends previous user query prefix with separator and preserves raw query.
3. Overlong history is truncated to ensure final retrieval query length <= 512.
4. Zero history capacity (query length >= 510) omits history prefix and separator.
5. Queries longer than 512 characters are rejected before factory invocation.
6. Safe query from previous port failure is the only history input (no errors/metadata).
7. Current raw query controls routing decision while combined context reaches retrieval port.
8. Concurrent conversation sessions do not cross-contaminate context.
9. context_sha256 is separate from legacy state_hash and legacy hash is unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from raglab.agentic.enums import StopReason
from raglab.agentic.runtime.conversation_session_runner import (
    ConversationSessionRunner,
)
from raglab.domain.entities import RetrievedEvidence
from raglab.domain.enums import PipelineStrategy
from raglab.domain.value_objects import ChunkId

MAX_RETRIEVAL_QUERY_CHARS = 512
MAX_HISTORY_QUERY_CHARS = 256
SEPARATOR = "\n\n"


@dataclass
class CapturingRetrievalPort:
    """Hermetic retrieval port capturing all queries sent to retrieve()."""

    prefix: str
    captured_queries: list[str] = field(default_factory=list)
    call_count: int = 0
    should_fail: bool = False
    results: list[RetrievedEvidence] | None = None

    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedEvidence]:
        self.call_count += 1
        self.captured_queries.append(query)
        if self.should_fail:
            raise RuntimeError(f"Simulated port failure on {self.prefix}")
        if self.results is not None:
            return self.results[:top_k]
        return [
            RetrievedEvidence(
                chunk_id=ChunkId(value=f"{self.prefix}_001"),
                document_id="doc_100",
                text=f"Evidence from {self.prefix} for query: {query}",
                rank=1,
                score=0.95,
                passage_id=f"ps_{self.prefix}_001",
            )
        ][:top_k]


def _build_runner(
    base_port: CapturingRetrievalPort | None = None,
    window_port: CapturingRetrievalPort | None = None,
    max_turns: int = 3,
) -> tuple[
    ConversationSessionRunner,
    CapturingRetrievalPort,
    CapturingRetrievalPort,
]:
    bp = base_port or CapturingRetrievalPort("base")
    wp = window_port or CapturingRetrievalPort("window")
    ports = {
        PipelineStrategy.BASELINE: bp,
        PipelineStrategy.SENTENCE_WINDOW_RERANK: wp,
    }
    runner = ConversationSessionRunner(ports=ports, max_turns=max_turns)
    return runner, bp, wp


def test_01_turn_zero_records_raw_retrieval_query_and_context_hash() -> None:
    """Turn 0 sets retrieval_query equal to raw user_query and hashes it."""
    runner, _bp, wp = _build_runner()
    runner.create_conversation("conv_01")

    query = "What is chunking in RAG?"
    turn0 = runner.execute_turn("conv_01", query)

    assert turn0.user_query == query
    assert turn0.retrieval_query == query
    expected_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
    assert turn0.context_sha256 == expected_hash
    assert wp.captured_queries == [query]


def test_02_turn_one_prepends_previous_query_and_preserves_current_raw() -> None:
    """Turn 1 prepends previous user query and passes combined context to port."""
    runner, _bp, wp = _build_runner()
    runner.create_conversation("conv_02")

    q0 = "What is chunking in RAG?"
    q1 = "How does it compare to sentence window?"
    turn0 = runner.execute_turn("conv_02", q0)
    turn1 = runner.execute_turn("conv_02", q1)

    assert turn0.user_query == q0
    assert turn1.user_query == q1

    expected_combined = f"{q0}{SEPARATOR}{q1}"
    assert turn1.retrieval_query == expected_combined
    assert len(turn1.retrieval_query) <= MAX_RETRIEVAL_QUERY_CHARS

    expected_hash = hashlib.sha256(expected_combined.encode("utf-8")).hexdigest()
    assert turn1.context_sha256 == expected_hash
    assert wp.captured_queries[-1] == expected_combined


def test_03_history_prefix_is_truncated_to_exact_512_char_limit() -> None:
    """Overlong history prefix is truncated so total retrieval query length <= 512."""
    runner, _bp, wp = _build_runner()
    runner.create_conversation("conv_03")

    q0 = "A" * 300
    q1 = "B" * 300
    runner.execute_turn("conv_03", q0)
    turn1 = runner.execute_turn("conv_03", q1)

    available_history = MAX_RETRIEVAL_QUERY_CHARS - len(q1) - len(SEPARATOR)
    history_chars = min(MAX_HISTORY_QUERY_CHARS, max(0, available_history))
    assert history_chars == 210

    expected_retrieval = f"{'A' * 210}{SEPARATOR}{'B' * 300}"
    assert len(expected_retrieval) == 512
    assert turn1.user_query == q1
    assert turn1.retrieval_query == expected_retrieval
    assert len(turn1.retrieval_query) == 512

    expected_hash = hashlib.sha256(expected_retrieval.encode("utf-8")).hexdigest()
    assert turn1.context_sha256 == expected_hash
    assert wp.captured_queries[-1] == expected_retrieval


def test_04_zero_history_capacity_omits_history_and_separator() -> None:
    """When query is 510+ characters, history capacity is 0 and separator is omitted."""
    runner, _bp, wp = _build_runner()
    runner.create_conversation("conv_04")

    q0 = "Short previous question"
    q1 = "C" * 510
    runner.execute_turn("conv_04", q0)
    turn1 = runner.execute_turn("conv_04", q1)

    assert turn1.user_query == q1
    assert turn1.retrieval_query == q1
    assert len(turn1.retrieval_query) == 510

    expected_hash = hashlib.sha256(q1.encode("utf-8")).hexdigest()
    assert turn1.context_sha256 == expected_hash
    assert wp.captured_queries[-1] == q1


def test_05_overlong_query_is_rejected_before_factory_without_state_change() -> None:
    """Query exceeding 512 characters is rejected with ValueError before coordinator build."""
    runner, bp, wp = _build_runner()
    runner.create_conversation("conv_05")

    overlong_query = "D" * 513
    with pytest.raises(ValueError, match="512"):
        runner.execute_turn("conv_05", overlong_query)

    state = runner.get_conversation("conv_05")
    assert len(state.turns) == 0
    assert bp.call_count == 0
    assert wp.call_count == 0


def test_06_safe_query_from_previous_port_failure_is_the_only_history_input() -> None:
    """Previous turn TOOL_FAILURE supplies only its clean raw query into next turn's context."""
    bp = CapturingRetrievalPort("base", should_fail=True)
    wp = CapturingRetrievalPort("window", should_fail=True)
    runner, _bp, wp = _build_runner(base_port=bp, window_port=wp)
    runner.create_conversation("conv_06")

    q0 = "What is chunking?"
    turn0 = runner.execute_turn("conv_06", q0)
    assert turn0.bounded_loop_result.stop_decision.reason == StopReason.TOOL_FAILURE

    # Unset failure for turn 1
    bp.should_fail = False
    wp.should_fail = False

    q1 = "Can we try retrieval again?"
    turn1 = runner.execute_turn("conv_06", q1)

    expected_retrieval = f"{q0}{SEPARATOR}{q1}"
    assert turn1.retrieval_query == expected_retrieval
    assert "Simulated port failure" not in turn1.retrieval_query
    assert "TOOL_FAILURE" not in turn1.retrieval_query
    assert "error" not in turn1.retrieval_query.lower()


def test_07_current_raw_query_controls_routing_while_context_reaches_port() -> None:
    """Current raw query controls deterministic routing while retrieval query reaches port."""
    runner, bp, _wp = _build_runner()
    runner.create_conversation("conv_07")

    # Turn 0: comparative query (routes to sentence_window_rerank)
    q0 = "Compare baseline vs sentence window reranking"
    turn0 = runner.execute_turn("conv_07", q0)
    assert (
        turn0.bounded_loop_result.routing_decision.selected_strategy
        == "sentence_window_rerank"
    )

    # Turn 1: unanswerable query (must route to baseline based on raw query)
    q1 = "Why is this unanswerable from the documentation?"
    turn1 = runner.execute_turn("conv_07", q1)
    assert (
        turn1.bounded_loop_result.routing_decision.selected_strategy
        == "baseline"
    )

    expected_port_query = f"{q0}{SEPARATOR}{q1}"
    assert turn1.retrieval_query == expected_port_query
    assert bp.captured_queries == [expected_port_query]


def test_08_conversations_do_not_share_context() -> None:
    """Different conversation IDs maintain isolated context history."""
    runner, _bp, wp = _build_runner()
    runner.create_conversation("conv_A")
    runner.create_conversation("conv_B")

    runner.execute_turn("conv_A", "Question for session A")
    runner.execute_turn("conv_B", "Question for session B")

    turn_a1 = runner.execute_turn("conv_A", "Followup for session A")

    expected_a1 = f"Question for session A{SEPARATOR}Followup for session A"
    assert turn_a1.retrieval_query == expected_a1
    assert "session B" not in turn_a1.retrieval_query
    assert wp.captured_queries[-1] == expected_a1


def test_09_context_hash_is_separate_and_legacy_state_hash_is_unchanged() -> None:
    """context_sha256 reflects retrieval_query while state_hash retains L3B.1 payload."""
    runner, _bp, _wp = _build_runner()
    runner.create_conversation("conv_09")

    q0 = "Query zero"
    q1 = "Query one"
    runner.execute_turn("conv_09", q0)
    turn1 = runner.execute_turn("conv_09", q1)

    expected_retrieval = f"{q0}{SEPARATOR}{q1}"
    expected_context_hash = hashlib.sha256(
        expected_retrieval.encode("utf-8")
    ).hexdigest()
    assert turn1.context_sha256 == expected_context_hash

    # Recompute legacy L3B.1 state_hash payload
    legacy_payload = {
        "conversation_id": "conv_09",
        "turn_index": 1,
        "user_query": q1,
        "config_sha256": turn1.bounded_loop_result.trajectory.config_sha256,
        "stop_reason": turn1.bounded_loop_result.stop_decision.reason.value,
        "evidence_count": turn1.bounded_loop_result.evidence_count,
    }
    canonical_json = json.dumps(
        legacy_payload, sort_keys=True, separators=(",", ":")
    )
    expected_state_hash = hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()

    assert turn1.state_hash == expected_state_hash
    assert turn1.state_hash != turn1.context_sha256
