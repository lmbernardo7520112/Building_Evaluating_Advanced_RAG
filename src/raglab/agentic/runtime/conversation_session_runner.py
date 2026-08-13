"""Public conversation session runtime — manages multi-turn L3B sessions.

Composes isolated 2-step L3A BoundedLoopRunner per turn with strict in-memory state.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from raglab.agentic.runtime.bounded_loop_factory import (
    build_bounded_loop_coordinator,
)
from raglab.agentic.runtime.bounded_loop_runner import BoundedLoopResult
from raglab.domain.enums import PipelineStrategy

MAX_RETRIEVAL_QUERY_CHARS = 512
MAX_HISTORY_QUERY_CHARS = 256
SEPARATOR = "\n\n"


def _default_clock() -> str:
    return datetime.now(UTC).isoformat()


def _build_bounded_retrieval_query(
    current_query: str,
    previous_query: str | None,
) -> str:
    """Derive bounded retrieval query from previous turn and current query."""
    if previous_query is None:
        return current_query

    available_history_chars = (
        MAX_RETRIEVAL_QUERY_CHARS - len(current_query) - len(SEPARATOR)
    )
    history_chars = min(
        MAX_HISTORY_QUERY_CHARS, max(0, available_history_chars)
    )

    if history_chars == 0:
        return current_query

    history_prefix = previous_query[:history_chars]
    return f"{history_prefix}{SEPARATOR}{current_query}"


class ConversationStatus(str, Enum):  # noqa: UP042
    """Lifecycle status for a conversation session."""

    ACTIVE = "ACTIVE"
    EXHAUSTED = "EXHAUSTED"


@dataclass(frozen=True, slots=True)
class ConversationTurnResult:
    """Structured result of a single conversation turn."""

    conversation_id: str
    turn_index: int
    user_query: str
    bounded_loop_result: BoundedLoopResult
    created_at: str
    state_hash: str
    retrieval_query: str
    context_sha256: str


@dataclass(frozen=True, slots=True)
class ConversationState:
    """Immutable snapshot of conversation session state."""

    conversation_id: str
    turns: tuple[ConversationTurnResult, ...]
    status: ConversationStatus
    created_at: str
    updated_at: str


@dataclass
class ConversationSessionRunner:
    """Runtime managing multi-turn conversation sessions with L3A per-turn isolation."""

    ports: Mapping[PipelineStrategy, Any]
    max_turns: int = 3
    clock: Callable[[], str] = field(default_factory=lambda: _default_clock)
    _sessions: dict[str, ConversationState] = field(
        default_factory=dict, init=False
    )

    def create_conversation(self, conversation_id: str) -> ConversationState:
        """Create a new ACTIVE empty conversation session."""
        if not conversation_id or not conversation_id.strip():
            raise ValueError("conversation_id must be a non-empty string")

        if conversation_id in self._sessions:
            raise ValueError(f"Conversation '{conversation_id}' already exists")

        now = self.clock()
        state = ConversationState(
            conversation_id=conversation_id,
            turns=(),
            status=ConversationStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
        self._sessions[conversation_id] = state
        return state

    def get_conversation(self, conversation_id: str) -> ConversationState:
        """Retrieve current ConversationState snapshot."""
        if conversation_id not in self._sessions:
            raise ValueError(f"Unknown conversation_id '{conversation_id}'")
        return self._sessions[conversation_id]

    def execute_turn(
        self,
        conversation_id: str,
        user_query: str,
        top_k: int = 3,
    ) -> ConversationTurnResult:
        """Execute a single turn using a fresh L3A composition."""
        if conversation_id not in self._sessions:
            raise ValueError(f"Unknown conversation_id '{conversation_id}'")

        if not user_query or not user_query.strip():
            raise ValueError("user_query must be a non-empty string")

        if len(user_query) > MAX_RETRIEVAL_QUERY_CHARS:
            raise ValueError(
                f"user_query exceeds maximum length of "
                f"{MAX_RETRIEVAL_QUERY_CHARS} characters"
            )

        session = self._sessions[conversation_id]
        if session.status == ConversationStatus.EXHAUSTED:
            raise ValueError(
                f"Conversation '{conversation_id}' is EXHAUSTED (max turns reached)"
            )

        turn_index = len(session.turns)
        if turn_index >= self.max_turns:
            raise ValueError(
                f"Conversation turn limit reached ({turn_index}/{self.max_turns})"
            )

        previous_query = (
            session.turns[-1].user_query if session.turns else None
        )
        retrieval_query = _build_bounded_retrieval_query(
            user_query, previous_query
        )
        context_sha256 = hashlib.sha256(
            retrieval_query.encode("utf-8")
        ).hexdigest()

        run_id = f"{conversation_id}_turn_{turn_index}"

        # 1. Build a fresh L3A coordinator for this turn
        coordinator = build_bounded_loop_coordinator(
            self.ports,
            run_id=run_id,
            clock=self.clock,
        )

        # 2. Execute 2-step bounded loop
        # (unexpected exceptions propagate without state mutation)
        bounded_loop_result = coordinator.execute(
            query_id=f"{run_id}_q",
            query_text=user_query,
            top_k=top_k,
            retrieval_query_text=retrieval_query,
        )

        # 3. Calculate canonical SHA-256 turn state hash
        payload = {
            "conversation_id": conversation_id,
            "turn_index": turn_index,
            "user_query": user_query,
            "config_sha256": bounded_loop_result.trajectory.config_sha256,
            "stop_reason": bounded_loop_result.stop_decision.reason.value,
            "evidence_count": bounded_loop_result.evidence_count,
        }
        canonical_json = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        )
        state_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

        now = self.clock()
        turn_result = ConversationTurnResult(
            conversation_id=conversation_id,
            turn_index=turn_index,
            user_query=user_query,
            bounded_loop_result=bounded_loop_result,
            created_at=now,
            state_hash=state_hash,
            retrieval_query=retrieval_query,
            context_sha256=context_sha256,
        )

        # 4. Atomic state update
        new_turns = session.turns + (turn_result,)
        new_status = (
            ConversationStatus.EXHAUSTED
            if len(new_turns) >= self.max_turns
            else ConversationStatus.ACTIVE
        )

        updated_state = ConversationState(
            conversation_id=conversation_id,
            turns=new_turns,
            status=new_status,
            created_at=session.created_at,
            updated_at=now,
        )
        self._sessions[conversation_id] = updated_state

        return turn_result
