"""Agentic track domain errors — structured, no silent fallbacks."""

from __future__ import annotations


class AgenticError(Exception):
    """Base for all agentic track errors."""


class UnknownToolError(AgenticError):
    """Raised when a tool ID is not in the registry."""

    def __init__(self, tool_id: str) -> None:
        self.tool_id = tool_id
        super().__init__(f"Unknown tool: '{tool_id}'")


class UnauthorizedToolError(AgenticError):
    """Raised when a tool is not authorized for the current context."""

    def __init__(self, tool_id: str, reason: str) -> None:
        self.tool_id = tool_id
        self.reason = reason
        super().__init__(f"Unauthorized tool '{tool_id}': {reason}")


class InvalidToolArgumentsError(AgenticError):
    """Raised when tool arguments fail validation."""

    def __init__(self, tool_id: str, detail: str) -> None:
        self.tool_id = tool_id
        self.detail = detail
        super().__init__(f"Invalid arguments for '{tool_id}': {detail}")


class BudgetExhaustedError(AgenticError):
    """Raised when the execution budget is exceeded."""

    def __init__(self, resource: str, limit: int, consumed: int) -> None:
        self.resource = resource
        self.limit = limit
        self.consumed = consumed
        super().__init__(
            f"Budget exhausted for {resource}: limit={limit}, consumed={consumed}"
        )


class NonCanonicalIdError(AgenticError):
    """Raised when an ID does not follow the canonical pattern."""

    def __init__(self, field: str, value: str) -> None:
        self.field = field
        self.value = value
        super().__init__(f"Non-canonical ID in '{field}': '{value}'")


class LeakageDetectedError(AgenticError):
    """Raised when qrels, gold answers, or holdout data is accessed."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"Leakage detected: {detail}")


class LedgerCorruptionError(AgenticError):
    """Raised when the trajectory ledger fails integrity checks."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"Ledger corruption: {detail}")


class LedgerConflictError(AgenticError):
    """Raised when a conflicting duplicate entry is detected."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"Ledger conflict: {detail}")


class IncompatibleRunError(AgenticError):
    """Raised when run ID, policy hash, or config hash mismatch."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"Incompatible run: {detail}")


class PolicyValidationError(AgenticError):
    """Raised when a routing policy output fails validation."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"Policy validation failed: {detail}")


class OptionalBackendNotAvailableError(AgenticError):
    """Raised when an optional agentic backend is not installed."""

    def __init__(self, backend: str) -> None:
        self.backend = backend
        super().__init__(f"OPTIONAL_AGENTIC_BACKEND_NOT_AVAILABLE: '{backend}'")
