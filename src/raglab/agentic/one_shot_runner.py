"""One-shot runner — minimal governed coordinator for a single query.

Composes the full agentic pipeline:

    Query → Router → RoutingDecision → ToolRegistry → ToolExecutor
    → ToolObservation → EvidenceAccumulator → StopPolicy → StopDecision
    → AgentTrajectory

Constraints:
- Maximum ONE logical tool call per query (no loop).
- No Gemini / LLM execution.
- No network access.
- No LlamaIndex imports (framework-neutral).
- All dependencies injected.
- Clock and ID generation injectable for determinism.
- Trajectory is complete and auditable.
- Failures produce governed stop decisions (never silent).
- Memory is isolated per query_id.
- No chain-of-thought persisted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from raglab.agentic.budget import Budget
from raglab.agentic.contracts import (
    SCHEMA_VERSION,
    AgentTrajectory,
    RoutingDecision,
    StopDecision,
    ToolArguments,
    ToolInvocation,
    TrajectoryStep,
)
from raglab.agentic.enums import (
    CallType,
    InvocationStatus,
)
from raglab.agentic.errors import (
    BudgetExhaustedError,
    InvalidToolArgumentsError,
    LeakageDetectedError,
    NonCanonicalIdError,
    UnauthorizedToolError,
    UnknownToolError,
)
from raglab.agentic.evidence_state import EvidenceAccumulator
from raglab.agentic.router import (
    get_deterministic_policy_metadata,
    route_deterministic,
)
from raglab.agentic.stop_policy import StopPolicy
from raglab.agentic.tool_executor import RetrievalBackend, ToolExecutor
from raglab.agentic.tool_registry import ToolRegistry


def _default_clock() -> str:
    """ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def _default_id_factory(prefix: str) -> Callable[[], str]:
    """Sequential ID generator for deterministic testing."""
    counter = 0

    def _next() -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}_{counter:04d}"

    return _next


@dataclass(frozen=True, slots=True)
class OneShotResult:
    """Outcome of a single one-shot run."""

    trajectory: AgentTrajectory
    routing_decision: RoutingDecision
    stop_decision: StopDecision
    evidence_count: int
    error: str | None = None


@dataclass
class OneShotRunner:
    """Framework-neutral one-shot coordinator.

    All dependencies are injected; no global state is touched.
    """

    registry: ToolRegistry
    budget: Budget
    backend: RetrievalBackend
    run_id: str
    clock: Callable[[], str] = field(default_factory=lambda: _default_clock)
    invocation_id_gen: Callable[[], str] = field(
        default_factory=lambda: _default_id_factory("inv")
    )

    def execute(
        self,
        query_id: str,
        query_text: str,
        top_k: int = 3,
    ) -> OneShotResult:
        """Run the full one-shot pipeline for a single query.

        Returns a OneShotResult with the complete trajectory,
        routing decision, stop decision, and evidence count.

        OneShotRunner captures expected domain failures (AgenticError
        subclasses) and fails visibly on unexpected internal errors.
        RuntimeError, AssertionError, and other non-domain exceptions
        propagate as bugs — they are never silently converted.
        """
        evidence = EvidenceAccumulator()
        stop_policy = StopPolicy()
        executor = ToolExecutor(self.registry, self.budget)
        started_at = self.clock()
        policy_meta = get_deterministic_policy_metadata()

        # 1. Route
        decision = route_deterministic(query_id, query_text)
        tool_id = f"retrieve_{decision.selected_strategy}"

        # 2. Prepare invocation
        invocation_id = self.invocation_id_gen()
        error_msg: str | None = None
        observation_hash = ""
        evidence_delta = 0
        state_before_hash = evidence.snapshot_hash()
        args_sha256 = "error"

        try:
            args = ToolArguments(
                query=query_text,
                strategy=decision.selected_strategy,
                top_k=top_k,
            )
            args_sha256 = args.sha256
            invocation = ToolInvocation(
                invocation_id=invocation_id,
                query_id=query_id,
                step_index=0,
                tool_id=tool_id,
                tool_version="1.0.0",
                arguments=args,
                arguments_sha256=args.sha256,
                authorization_status=InvocationStatus.AUTHORIZED,
                call_type=CallType.LOGICAL_CALL,
                logical_call_index=0,
                started_at=started_at,
            )

            # 3. Execute
            observation = executor.validate_and_execute(invocation, self.backend)
            observation_hash = observation.invocation_id

            # 4. Accumulate evidence
            evidence_delta = evidence.add_from_observation(
                passage_ids=observation.passage_ids,
                document_ids=observation.document_ids,
                ranks=observation.ranks,
                scores=observation.scores,
                content_hashes=observation.content_hashes,
                source_tool_id=tool_id,
                source_invocation_id=observation.invocation_id,
            )

        except (
            UnknownToolError,
            UnauthorizedToolError,
            InvalidToolArgumentsError,
            LeakageDetectedError,
            BudgetExhaustedError,
            NonCanonicalIdError,
        ) as exc:
            error_msg = f"{type(exc).__name__}: {exc}"

        # 5. Evaluate stop
        state_after_hash = evidence.snapshot_hash()
        stop_decision = stop_policy.evaluate_one_shot(evidence, self.budget)

        finished_at = self.clock()

        # 6. Build trajectory step
        step = TrajectoryStep(
            step_index=0,
            state_before_hash=state_before_hash,
            action=f"retrieve:{tool_id}",
            arguments_sha256=args_sha256,
            observation_hash=observation_hash or "none",
            evidence_delta_count=evidence_delta,
            decision_code=decision.decision_code,
            state_after_hash=state_after_hash,
            budget_remaining=self.budget.remaining(),
            stop_reason=stop_decision.reason,
        )

        # 7. Build trajectory
        config_hash = self.registry.registry_hash()
        trajectory = AgentTrajectory(
            schema_version=SCHEMA_VERSION,
            run_id=self.run_id,
            query_id=query_id,
            policy_id=policy_meta.policy_id,
            policy_sha256=policy_meta.policy_sha256,
            config_sha256=config_hash,
            steps=(step,),
            stop_decision=stop_decision,
            routing_decision=decision,
            created_at=finished_at,
        )

        return OneShotResult(
            trajectory=trajectory,
            routing_decision=decision,
            stop_decision=stop_decision,
            evidence_count=len(evidence.items_in_order()),
            error=error_msg,
        )
