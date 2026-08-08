"""Budget — resource limits and consumption tracking.

Budget only represents and consumes limits.
It does NOT decide whether to stop — that is the StopPolicy's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Budget:
    """Tracks resource consumption against declared limits.

    All limits are set at construction and cannot be increased.
    Consumption is monotonically increasing.
    """

    max_logical_calls: int = 10
    max_physical_attempts: int = 20
    max_retries: int = 5
    max_evidence_items: int = 50
    max_top_k: int = 10
    timeout_seconds: float = 60.0

    logical_calls_consumed: int = field(default=0, init=False)
    physical_attempts_consumed: int = field(default=0, init=False)
    retries_consumed: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.max_logical_calls < 1:
            raise ValueError("max_logical_calls must be >= 1")
        if self.max_physical_attempts < 1:
            raise ValueError("max_physical_attempts must be >= 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.max_evidence_items < 1:
            raise ValueError("max_evidence_items must be >= 1")
        if self.max_top_k < 1:
            raise ValueError("max_top_k must be >= 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    def can_consume_logical_call(self) -> bool:
        return self.logical_calls_consumed < self.max_logical_calls

    def can_consume_physical_attempt(self) -> bool:
        return self.physical_attempts_consumed < self.max_physical_attempts

    def can_retry(self) -> bool:
        return self.retries_consumed < self.max_retries

    def consume_logical_call(self) -> None:
        """Record a logical call. Raises if budget exhausted."""
        if not self.can_consume_logical_call():
            raise ValueError(
                f"Logical calls exhausted: {self.logical_calls_consumed}"
                f"/{self.max_logical_calls}"
            )
        self.logical_calls_consumed += 1

    def consume_physical_attempt(self) -> None:
        """Record a physical attempt."""
        if not self.can_consume_physical_attempt():
            raise ValueError(
                f"Physical attempts exhausted: "
                f"{self.physical_attempts_consumed}"
                f"/{self.max_physical_attempts}"
            )
        self.physical_attempts_consumed += 1

    def consume_retry(self) -> None:
        """Record a retry."""
        if not self.can_retry():
            raise ValueError(
                f"Retries exhausted: {self.retries_consumed}/{self.max_retries}"
            )
        self.retries_consumed += 1

    def remaining(self) -> dict[str, int]:
        """Return remaining budget as a dict."""
        return {
            "logical_calls": self.max_logical_calls - self.logical_calls_consumed,
            "physical_attempts": (
                self.max_physical_attempts - self.physical_attempts_consumed
            ),
            "retries": self.max_retries - self.retries_consumed,
        }
