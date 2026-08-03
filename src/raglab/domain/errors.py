"""Domain errors — specific, non-leaking, no secrets in messages."""

from __future__ import annotations


class RagLabDomainError(Exception):
    """Base for all domain errors."""


class InvalidIdentifierError(RagLabDomainError):
    """Raised when an identifier is empty or malformed."""

    def __init__(self, field: str) -> None:
        super().__init__(f"Invalid identifier: '{field}' must be non-empty")


class InvalidScoreError(RagLabDomainError):
    """Raised when a metric score is not finite."""

    def __init__(self, metric: str, value: float) -> None:
        # Never include raw values that could leak context
        super().__init__(
            f"Invalid score for '{metric}': must be finite, got non-finite value"
            if not _is_finite(value)
            else f"Invalid score for '{metric}': out of valid range"
        )


class NormalizedScoreOutOfRangeError(RagLabDomainError):
    """Raised when a normalized score is outside [0, 1]."""

    def __init__(self, metric: str) -> None:
        super().__init__(
            f"Normalized score for '{metric}' must be in [0.0, 1.0]"
        )


class InvalidFingerprintError(RagLabDomainError):
    """Raised when a SHA-256 fingerprint has wrong format."""

    def __init__(self) -> None:
        super().__init__(
            "Fingerprint must be a 64-character lowercase hexadecimal string"
        )


class NegativePositionError(RagLabDomainError):
    """Raised when a page number or position is negative."""

    def __init__(self, field: str) -> None:
        super().__init__(f"'{field}' must be non-negative")


class MissingProvenanceError(RagLabDomainError):
    """Raised when evidence lacks source provenance."""

    def __init__(self) -> None:
        super().__init__("Evidence must have identifiable source provenance")


class HoldoutAccessViolationError(RagLabDomainError):
    """Raised on unauthorized access to holdout data."""

    def __init__(self, split_name: str) -> None:
        super().__init__(
            f"Holdout '{split_name}' requires explicit authorization before access"
        )


class CheckpointMismatchError(RagLabDomainError):
    """Raised when checkpoint config/corpus doesn't match current run."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Checkpoint incompatible: {reason}")


class ConfigurationError(RagLabDomainError):
    """Raised when configuration is invalid and must fail early."""

    def __init__(self, field: str, reason: str) -> None:
        # Never include actual values — they could be secrets
        super().__init__(f"Configuration error in '{field}': {reason}")


class CitationProvenanceMismatchError(RagLabDomainError):
    """Raised when a cited evidence_id cannot be found in the evidence snapshot."""

    def __init__(self, citation_id: str) -> None:
        super().__init__(
            f"CITATION_PROVENANCE_MISMATCH: cited evidence_id '{citation_id}' "
            "not present in prompt snapshot"
        )


def _is_finite(value: float) -> bool:
    """Check finiteness without importing math in the public API."""
    import math

    return math.isfinite(value)
