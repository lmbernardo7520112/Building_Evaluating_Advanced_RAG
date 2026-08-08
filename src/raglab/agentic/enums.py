"""Agentic track enumerations — authoritative, versioned, stdlib-only."""

from __future__ import annotations

from enum import Enum, unique


@unique
class QueryClass(Enum):
    """Public classification of a query for routing purposes.

    These classes represent a CONTROL POLICY, not human ground truth.
    They are derived exclusively from the public text of the question.
    """

    GLOBAL_SUMMARY = "GLOBAL_SUMMARY"
    LOCALIZED_FACT = "LOCALIZED_FACT"
    COMPARISON = "COMPARISON"
    PROCEDURAL_EXPLANATION = "PROCEDURAL_EXPLANATION"
    POSSIBLY_UNANSWERABLE = "POSSIBLY_UNANSWERABLE"
    UNCLASSIFIED = "UNCLASSIFIED"


@unique
class StopReason(Enum):
    """Versioned enumeration of agent termination reasons.

    No free-text is permitted as an authoritative stop reason.
    """

    SUFFICIENT_EVIDENCE = "SUFFICIENT_EVIDENCE"
    NO_EVIDENCE = "NO_EVIDENCE"
    INVALID_TOOL_ARGUMENTS = "INVALID_TOOL_ARGUMENTS"
    UNAUTHORIZED_TOOL = "UNAUTHORIZED_TOOL"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    TIMEOUT = "TIMEOUT"
    REPEATED_CALL_NO_NEW_EVIDENCE = "REPEATED_CALL_NO_NEW_EVIDENCE"
    CANONICAL_ID_FAILURE = "CANONICAL_ID_FAILURE"
    TOOL_FAILURE = "TOOL_FAILURE"
    POSSIBLY_UNANSWERABLE = "POSSIBLY_UNANSWERABLE"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    COMPLETED_ONE_SHOT = "COMPLETED_ONE_SHOT"


@unique
class DecisionCode(Enum):
    """Routing decision outcome codes."""

    SELECTED = "SELECTED"
    FALLBACK = "FALLBACK"
    REJECTED = "REJECTED"
    ABSTAIN = "ABSTAIN"


@unique
class ValidationStatus(Enum):
    """Result of validating a routing decision or tool invocation."""

    VALID = "VALID"
    INVALID_STRATEGY = "INVALID_STRATEGY"
    INVALID_CAPABILITY = "INVALID_CAPABILITY"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    FORBIDDEN_FIELD = "FORBIDDEN_FIELD"
    POLICY_HASH_MISSING = "POLICY_HASH_MISSING"
    CONFIDENCE_OUT_OF_RANGE = "CONFIDENCE_OUT_OF_RANGE"
    LEAKAGE_DETECTED = "LEAKAGE_DETECTED"


@unique
class InvocationStatus(Enum):
    """Status of a tool invocation."""

    AUTHORIZED = "AUTHORIZED"
    EXECUTED = "EXECUTED"
    REJECTED_UNKNOWN_TOOL = "REJECTED_UNKNOWN_TOOL"
    REJECTED_UNAUTHORIZED = "REJECTED_UNAUTHORIZED"
    REJECTED_NETWORK_REQUIRED = "REJECTED_NETWORK_REQUIRED"
    REJECTED_WRITE_REQUIRED = "REJECTED_WRITE_REQUIRED"
    REJECTED_INVALID_ARGUMENTS = "REJECTED_INVALID_ARGUMENTS"
    REJECTED_BUDGET_EXHAUSTED = "REJECTED_BUDGET_EXHAUSTED"
    REJECTED_NON_CANONICAL_ID = "REJECTED_NON_CANONICAL_ID"
    FAILED = "FAILED"


@unique
class OracleLabel(Enum):
    """Labels for oracle analyses — never operational."""

    POST_HOC_ORACLE = "POST_HOC_ORACLE"
    NON_OPERATIONAL = "NON_OPERATIONAL"
    USES_EVALUATION_OUTCOMES = "USES_EVALUATION_OUTCOMES"
    NOT_A_DEPLOYABLE_POLICY = "NOT_A_DEPLOYABLE_POLICY"


@unique
class CallType(Enum):
    """Distinguishes call types for accounting."""

    LOGICAL_CALL = "LOGICAL_CALL"
    PHYSICAL_ATTEMPT = "PHYSICAL_ATTEMPT"
    RETRY = "RETRY"
    FALLBACK = "FALLBACK"
