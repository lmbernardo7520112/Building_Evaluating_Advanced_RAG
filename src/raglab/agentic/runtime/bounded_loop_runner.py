"""Bounded 2-step retrieval loop runner — governed multi-step coordinator.

Composes the agentic pipeline for up to 2 governed retrieval steps:
    Step 0: Initial query -> route_deterministic() -> tool_0 -> evidence
    Step 1: If 0 evidence -> complementary tool_1 -> evidence

Constraints:
- Maximum TWO logical tool calls per query.
- Baseline <-> Sentence Window Rerank complementary fallback.
- Same tool repetition is strictly forbidden.
- No LLM execution or text synthesis.
- Governed execution via ToolExecutor and Budget.
- Complete auditable AgentTrajectory.
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
    DecisionCode,
    InvocationStatus,
    StopReason,
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
from raglab.agentic.tool_executor import (
    RetrievalBackend,
    ToolExecutor,
    validate_query_safety,
)
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


def _get_complementary_strategy(strategy: str) -> str:
    """Map strategy to its exact complementary counterpart for Step 1."""
    if strategy == "baseline":
        return "sentence_window_rerank"
    elif strategy == "sentence_window_rerank":
        return "baseline"

    raise ValueError(f"No complementary strategy mapped for '{strategy}'")


@dataclass(frozen=True, slots=True)
class BoundedLoopResult:
    """Outcome of a governed 2-step bounded loop run."""

    trajectory: AgentTrajectory
    routing_decision: RoutingDecision
    stop_decision: StopDecision
    evidence_count: int
    error: str | None = None


@dataclass
class BoundedLoopRunner:
    """Framework-neutral 2-step bounded loop coordinator."""

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
        *,
        retrieval_query_text: str | None = None,
    ) -> BoundedLoopResult:
        """Run up to 2 retrieval steps for a single query."""
        evidence = EvidenceAccumulator()
        executor = ToolExecutor(self.registry, self.budget)
        policy_meta = get_deterministic_policy_metadata()

        effective_retrieval_query = (
            retrieval_query_text
            if retrieval_query_text is not None
            else query_text
        )

        steps: list[TrajectoryStep] = []
        executed_tools: set[str] = set()

        # Initial Routing
        decision = route_deterministic(query_id, query_text)
        initial_strategy = decision.selected_strategy
        current_strategy = initial_strategy

        final_stop_decision: StopDecision | None = None
        error_msg: str | None = None

        # Check initial budget before any execution
        if not self.budget.can_consume_logical_call():
            final_stop_decision = StopDecision(
                reason=StopReason.BUDGET_EXHAUSTED,
                detail="Logical calls exhausted before execution",
                evidence_count=0,
                budget_remaining=self.budget.remaining(),
            )
            config_hash = self.registry.registry_hash()
            trajectory = AgentTrajectory(
                schema_version=SCHEMA_VERSION,
                run_id=self.run_id,
                query_id=query_id,
                policy_id=policy_meta.policy_id,
                policy_sha256=policy_meta.policy_sha256,
                config_sha256=config_hash,
                steps=(),
                stop_decision=final_stop_decision,
                routing_decision=decision,
                created_at=self.clock(),
            )
            return BoundedLoopResult(
                trajectory=trajectory,
                routing_decision=decision,
                stop_decision=final_stop_decision,
                evidence_count=0,
                error=None,
            )

        # Bounded Loop (Step 0 and optional Step 1)
        for step_index in range(2):
            tool_id = f"retrieve_{current_strategy}"
            decision_code = (
                DecisionCode.SELECTED
                if step_index == 0
                else DecisionCode.FALLBACK
            )

            # Check for forbidden tool repetition
            if tool_id in executed_tools:
                final_stop_decision = StopDecision(
                    reason=StopReason.REPEATED_CALL_NO_NEW_EVIDENCE,
                    detail=f"Tool '{tool_id}' repetition blocked",
                    evidence_count=evidence.count,
                    budget_remaining=self.budget.remaining(),
                )
                break

            # Check budget availability for this step
            if not self.budget.can_consume_logical_call():
                final_stop_decision = StopDecision(
                    reason=StopReason.BUDGET_EXHAUSTED,
                    detail=f"Logical calls exhausted at step {step_index}",
                    evidence_count=evidence.count,
                    budget_remaining=self.budget.remaining(),
                )
                break

            state_before_hash = evidence.snapshot_hash()
            invocation_id = self.invocation_id_gen()
            observation_hash = ""
            evidence_delta = 0
            args_sha256 = "error"
            step_error: str | None = None

            try:
                if step_index == 0 and retrieval_query_text is not None:
                    validate_query_safety(query_text)

                args = ToolArguments(
                    query=effective_retrieval_query,
                    strategy=current_strategy,
                    top_k=top_k,
                )
                args_sha256 = args.sha256
                invocation = ToolInvocation(
                    invocation_id=invocation_id,
                    query_id=query_id,
                    step_index=step_index,
                    tool_id=tool_id,
                    tool_version="1.0.0",
                    arguments=args,
                    arguments_sha256=args.sha256,
                    authorization_status=InvocationStatus.AUTHORIZED,
                    call_type=CallType.LOGICAL_CALL,
                    logical_call_index=step_index,
                    started_at=self.clock(),
                )

                # Execute step
                observation = executor.validate_and_execute(
                    invocation, self.backend
                )
                observation_hash = observation.invocation_id
                executed_tools.add(tool_id)

                # Check if observation returned FAILED status
                if (
                    observation.status == InvocationStatus.FAILED
                    or observation.failure_code is not None
                ):
                    step_error = observation.failure_code or "Tool execution failed"
                    error_msg = step_error
                else:
                    # Accumulate evidence
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
                step_error = f"{type(exc).__name__}: {exc}"
                error_msg = step_error

            state_after_hash = evidence.snapshot_hash()

            # Handle step failure
            if step_error is not None:
                if "UnauthorizedToolError" in step_error:
                    stop_reason = StopReason.UNAUTHORIZED_TOOL
                elif "InvalidToolArgumentsError" in step_error:
                    stop_reason = StopReason.INVALID_TOOL_ARGUMENTS
                elif "NonCanonicalIdError" in step_error:
                    stop_reason = StopReason.CANONICAL_ID_FAILURE
                else:
                    stop_reason = StopReason.TOOL_FAILURE

                final_stop_decision = StopDecision(
                    reason=stop_reason,
                    detail=step_error,
                    evidence_count=evidence.count,
                    budget_remaining=self.budget.remaining(),
                )

                step = TrajectoryStep(
                    step_index=step_index,
                    state_before_hash=state_before_hash,
                    action=f"retrieve:{tool_id}",
                    arguments_sha256=args_sha256,
                    observation_hash=observation_hash or "none",
                    evidence_delta_count=evidence_delta,
                    decision_code=decision_code,
                    state_after_hash=state_after_hash,
                    budget_remaining=self.budget.remaining(),
                    stop_reason=stop_reason,
                )
                steps.append(step)
                break

            # If evidence was retrieved in step 0 or 1, stop immediately
            if evidence.count > 0:
                final_stop_decision = StopDecision(
                    reason=StopReason.SUFFICIENT_EVIDENCE,
                    detail=f"Retrieved {evidence.count} evidence items",
                    evidence_count=evidence.count,
                    budget_remaining=self.budget.remaining(),
                )
                step = TrajectoryStep(
                    step_index=step_index,
                    state_before_hash=state_before_hash,
                    action=f"retrieve:{tool_id}",
                    arguments_sha256=args_sha256,
                    observation_hash=observation_hash or "none",
                    evidence_delta_count=evidence_delta,
                    decision_code=decision_code,
                    state_after_hash=state_after_hash,
                    budget_remaining=self.budget.remaining(),
                    stop_reason=StopReason.SUFFICIENT_EVIDENCE,
                )
                steps.append(step)
                break

            # If 0 evidence in step 0, record step and switch strategy
            step = TrajectoryStep(
                step_index=step_index,
                state_before_hash=state_before_hash,
                action=f"retrieve:{tool_id}",
                arguments_sha256=args_sha256,
                observation_hash=observation_hash or "none",
                evidence_delta_count=evidence_delta,
                decision_code=decision_code,
                state_after_hash=state_after_hash,
                budget_remaining=self.budget.remaining(),
                stop_reason=None,
            )
            steps.append(step)

            # Prepare complementary strategy for Step 1
            if step_index == 0:
                current_strategy = _get_complementary_strategy(
                    initial_strategy
                )

        # Fallback if loop ended with 0 evidence after step 1
        if final_stop_decision is None:
            if evidence.count == 0:
                final_stop_decision = StopDecision(
                    reason=StopReason.NO_EVIDENCE,
                    detail="Zero evidence retrieved across bounded steps",
                    evidence_count=0,
                    budget_remaining=self.budget.remaining(),
                )
            else:
                final_stop_decision = StopDecision(
                    reason=StopReason.SUFFICIENT_EVIDENCE,
                    detail=f"Retrieved {evidence.count} evidence items",
                    evidence_count=evidence.count,
                    budget_remaining=self.budget.remaining(),
                )

        finished_at = self.clock()
        config_hash = self.registry.registry_hash()

        trajectory = AgentTrajectory(
            schema_version=SCHEMA_VERSION,
            run_id=self.run_id,
            query_id=query_id,
            policy_id=policy_meta.policy_id,
            policy_sha256=policy_meta.policy_sha256,
            config_sha256=config_hash,
            steps=tuple(steps),
            stop_decision=final_stop_decision,
            routing_decision=decision,
            created_at=finished_at,
        )

        return BoundedLoopResult(
            trajectory=trajectory,
            routing_decision=decision,
            stop_decision=final_stop_decision,
            evidence_count=len(evidence.items_in_order()),
            error=error_msg,
        )
