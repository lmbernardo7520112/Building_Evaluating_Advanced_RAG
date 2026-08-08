"""Agentic track domain contracts — typed, immutable, serializable, stdlib-only.

No LlamaIndex, no Gemini, no global state, no chain-of-thought.
All contracts are versioned under schema 'slice5a_agentic_v1'.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from raglab.agentic.enums import (
    CallType,
    DecisionCode,
    InvocationStatus,
    StopReason,
    ValidationStatus,
)
from raglab.agentic.errors import NonCanonicalIdError

SCHEMA_VERSION = "slice5a_agentic_v1"

# Canonical passage ID prefix — all passage IDs must start with this.
_CANONICAL_PASSAGE_PREFIX = "ps_"


def _canonical_json(obj: dict[str, Any]) -> str:
    """Produce deterministic JSON for hashing."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _sha256(data: str) -> str:
    """Compute SHA-256 hex digest of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def is_canonical_passage_id(pid: str) -> bool:
    """Check whether a passage ID follows the canonical ``ps_*`` pattern."""
    return isinstance(pid, str) and pid.startswith(_CANONICAL_PASSAGE_PREFIX)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PolicyMetadata:
    """Frozen metadata for a routing policy."""

    policy_id: str
    policy_version: str
    policy_sha256: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id must be non-empty")
        if not self.policy_version:
            raise ValueError("policy_version must be non-empty")
        if not self.policy_sha256:
            raise ValueError("policy_sha256 must be non-empty")


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Structured decision from a routing policy."""

    schema_version: str
    query_id: str
    policy_id: str
    policy_version: str
    policy_sha256: str
    selected_strategy: str
    decision_code: DecisionCode
    public_features_used: tuple[str, ...]
    validation_status: ValidationStatus
    fallback_used: bool = False
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Schema mismatch: expected '{SCHEMA_VERSION}', "
                f"got '{self.schema_version}'"
            )
        if not self.query_id:
            raise ValueError("query_id must be non-empty")
        if not self.selected_strategy:
            raise ValueError("selected_strategy must be non-empty")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")


# ---------------------------------------------------------------------------
# Tool specification and invocation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolSpecification:
    """Typed specification of a governed tool."""

    tool_id: str
    version: str
    description: str
    read_only: bool
    network_access: bool
    deterministic: bool
    max_top_k: int
    timeout_seconds: float
    allowed_strategies: tuple[str, ...]
    implementation_sha256: str

    def __post_init__(self) -> None:
        if not self.tool_id:
            raise ValueError("tool_id must be non-empty")
        if not self.version:
            raise ValueError("version must be non-empty")
        if self.max_top_k < 1:
            raise ValueError(f"max_top_k must be >= 1, got {self.max_top_k}")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not self.read_only:
            raise ValueError("Only read-only tools are permitted")
        if self.network_access:
            raise ValueError("Tools with network access are not permitted")


@dataclass(frozen=True, slots=True)
class ToolArguments:
    """Normalized, validated arguments for a tool invocation."""

    query: str
    strategy: str
    top_k: int
    allowed_document_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.query or not self.query.strip():
            raise ValueError("query must be non-empty")
        if len(self.query) > 10000:
            raise ValueError(f"query length {len(self.query)} exceeds maximum 10000")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")

    def to_dict(self) -> dict[str, Any]:
        """Deterministic dict for hashing."""
        d: dict[str, Any] = {
            "query": self.query,
            "strategy": self.strategy,
            "top_k": self.top_k,
        }
        if self.allowed_document_ids is not None:
            d["allowed_document_ids"] = list(self.allowed_document_ids)
        return d

    @property
    def sha256(self) -> str:
        """SHA-256 of the canonical JSON representation."""
        return _sha256(_canonical_json(self.to_dict()))


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """Record of a tool invocation attempt."""

    invocation_id: str
    query_id: str
    step_index: int
    tool_id: str
    tool_version: str
    arguments: ToolArguments
    arguments_sha256: str
    authorization_status: InvocationStatus
    call_type: CallType
    logical_call_index: int
    started_at: str  # ISO 8601

    def __post_init__(self) -> None:
        if not self.invocation_id:
            raise ValueError("invocation_id must be non-empty")
        if not self.query_id:
            raise ValueError("query_id must be non-empty")
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")


@dataclass(frozen=True, slots=True)
class ToolObservation:
    """Observation from a tool execution."""

    invocation_id: str
    status: InvocationStatus
    passage_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    ranks: tuple[int, ...]
    scores: tuple[float, ...]
    content_hashes: tuple[str, ...]
    retrieval_config_hash: str
    latency_ms: float
    failure_code: str | None = None
    physical_attempts: int = 1
    retries: int = 0

    def __post_init__(self) -> None:
        if not self.invocation_id:
            raise ValueError("invocation_id must be non-empty")
        # Validate all passage IDs are canonical
        for pid in self.passage_ids:
            if not is_canonical_passage_id(pid):
                raise NonCanonicalIdError(
                    "passage_id",
                    f"{pid} (must start with '{_CANONICAL_PASSAGE_PREFIX}')",
                )


# ---------------------------------------------------------------------------
# Evidence accumulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """A single piece of accumulated evidence with provenance."""

    passage_id: str
    document_id: str
    rank: int
    score: float
    content_sha256: str
    source_tool_id: str
    source_invocation_id: str
    is_new: bool = True

    def __post_init__(self) -> None:
        if not is_canonical_passage_id(self.passage_id):
            raise NonCanonicalIdError(
                "passage_id",
                f"{self.passage_id} (must start with '{_CANONICAL_PASSAGE_PREFIX}')",
            )
        if not self.document_id:
            raise ValueError("document_id must be non-empty")
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """A single step in an agent trajectory."""

    step_index: int
    state_before_hash: str
    action: str
    arguments_sha256: str
    observation_hash: str
    evidence_delta_count: int
    decision_code: DecisionCode
    state_after_hash: str
    budget_remaining: dict[str, int]
    stop_reason: StopReason | None = None

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Deterministic dict for serialization."""
        d: dict[str, Any] = {
            "step_index": self.step_index,
            "state_before_hash": self.state_before_hash,
            "action": self.action,
            "arguments_sha256": self.arguments_sha256,
            "observation_hash": self.observation_hash,
            "evidence_delta_count": self.evidence_delta_count,
            "decision_code": self.decision_code.value,
            "state_after_hash": self.state_after_hash,
            "budget_remaining": self.budget_remaining,
        }
        if self.stop_reason is not None:
            d["stop_reason"] = self.stop_reason.value
        return d


@dataclass(frozen=True, slots=True)
class StopDecision:
    """Authoritative decision to stop the agent."""

    reason: StopReason
    detail: str
    evidence_count: int
    budget_remaining: dict[str, int]

    def __post_init__(self) -> None:
        if not self.detail:
            raise ValueError("detail must be non-empty")


@dataclass(frozen=True, slots=True)
class AgentTrajectory:
    """Complete agent trajectory for a single query."""

    schema_version: str
    run_id: str
    query_id: str
    policy_id: str
    policy_sha256: str
    config_sha256: str
    steps: tuple[TrajectoryStep, ...]
    stop_decision: StopDecision
    routing_decision: RoutingDecision
    created_at: str  # ISO 8601

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Schema mismatch: expected '{SCHEMA_VERSION}', "
                f"got '{self.schema_version}'"
            )
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not self.query_id:
            raise ValueError("query_id must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Full deterministic dict for serialization."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "query_id": self.query_id,
            "policy_id": self.policy_id,
            "policy_sha256": self.policy_sha256,
            "config_sha256": self.config_sha256,
            "steps": [s.to_dict() for s in self.steps],
            "stop_decision": {
                "reason": self.stop_decision.reason.value,
                "detail": self.stop_decision.detail,
                "evidence_count": self.stop_decision.evidence_count,
                "budget_remaining": self.stop_decision.budget_remaining,
            },
            "routing_decision": {
                "schema_version": self.routing_decision.schema_version,
                "query_id": self.routing_decision.query_id,
                "policy_id": self.routing_decision.policy_id,
                "policy_version": self.routing_decision.policy_version,
                "policy_sha256": self.routing_decision.policy_sha256,
                "selected_strategy": self.routing_decision.selected_strategy,
                "decision_code": self.routing_decision.decision_code.value,
                "public_features_used": list(
                    self.routing_decision.public_features_used
                ),
                "validation_status": (self.routing_decision.validation_status.value),
                "fallback_used": self.routing_decision.fallback_used,
                "confidence": self.routing_decision.confidence,
            },
            "created_at": self.created_at,
        }

    @property
    def sha256(self) -> str:
        """SHA-256 of the canonical JSON."""
        return _sha256(_canonical_json(self.to_dict()))
