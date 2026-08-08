"""Stop policy — evaluates state and produces versioned stop decisions.

The StopPolicy decides whether to continue, complete, abstain, or escalate.
Budget only tracks consumption; StopPolicy interprets it.
"""

from __future__ import annotations

from raglab.agentic.budget import Budget
from raglab.agentic.contracts import StopDecision
from raglab.agentic.enums import StopReason
from raglab.agentic.evidence_state import EvidenceAccumulator


class StopPolicy:
    """Evaluates agent state and produces authoritative stop decisions.

    All stop reasons are from the versioned StopReason enum.
    No free-text stop reasons are permitted.

    Compatible with Slice 5B bounded reasoning loop (future)
    without implementing the loop itself.
    """

    def __init__(
        self,
        min_evidence: int = 1,
        max_repeated_no_new: int = 2,
    ) -> None:
        self._min_evidence = min_evidence
        self._max_repeated_no_new = max_repeated_no_new
        self._consecutive_no_new: int = 0

    def evaluate_one_shot(
        self,
        evidence: EvidenceAccumulator,
        budget: Budget,
        tool_failed: bool = False,
        invalid_args: bool = False,
        unauthorized_tool: bool = False,
        canonical_id_failure: bool = False,
        timeout: bool = False,
    ) -> StopDecision:
        """Evaluate stop condition for one-shot mode.

        Returns a StopDecision with an authoritative reason.
        """
        remaining = budget.remaining()

        # Priority-ordered evaluation
        if timeout:
            return StopDecision(
                reason=StopReason.TIMEOUT,
                detail="Execution timeout reached",
                evidence_count=evidence.count,
                budget_remaining=remaining,
            )

        if canonical_id_failure:
            return StopDecision(
                reason=StopReason.CANONICAL_ID_FAILURE,
                detail="Non-canonical passage ID detected in observation",
                evidence_count=evidence.count,
                budget_remaining=remaining,
            )

        if unauthorized_tool:
            return StopDecision(
                reason=StopReason.UNAUTHORIZED_TOOL,
                detail="Tool not authorized for execution",
                evidence_count=evidence.count,
                budget_remaining=remaining,
            )

        if invalid_args:
            return StopDecision(
                reason=StopReason.INVALID_TOOL_ARGUMENTS,
                detail="Tool arguments failed validation",
                evidence_count=evidence.count,
                budget_remaining=remaining,
            )

        if tool_failed:
            return StopDecision(
                reason=StopReason.TOOL_FAILURE,
                detail="Tool execution failed",
                evidence_count=evidence.count,
                budget_remaining=remaining,
            )

        if not budget.can_consume_logical_call():
            return StopDecision(
                reason=StopReason.BUDGET_EXHAUSTED,
                detail=(
                    f"Logical calls: {budget.logical_calls_consumed}"
                    f"/{budget.max_logical_calls}"
                ),
                evidence_count=evidence.count,
                budget_remaining=remaining,
            )

        if evidence.count == 0:
            return StopDecision(
                reason=StopReason.NO_EVIDENCE,
                detail="No evidence retrieved",
                evidence_count=0,
                budget_remaining=remaining,
            )

        if evidence.count >= self._min_evidence:
            return StopDecision(
                reason=StopReason.COMPLETED_ONE_SHOT,
                detail=f"One-shot completed with {evidence.count} evidence items",
                evidence_count=evidence.count,
                budget_remaining=remaining,
            )

        return StopDecision(
            reason=StopReason.COMPLETED_ONE_SHOT,
            detail="One-shot completed",
            evidence_count=evidence.count,
            budget_remaining=remaining,
        )

    def record_no_new_evidence(self) -> bool:
        """Record that a call produced no new evidence.

        Returns True if the threshold for repeated-no-new is reached.
        """
        self._consecutive_no_new += 1
        return self._consecutive_no_new >= self._max_repeated_no_new

    def reset_no_new_counter(self) -> None:
        """Reset when new evidence is found."""
        self._consecutive_no_new = 0
