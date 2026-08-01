"""Retry policy for external API calls with 429 / transient error handling.

This module is pure domain logic (no network, no credentials).

Design:
  - Exponential backoff with jitter for 429 (rate limit exceeded)
  - Separate handling for 5xx transient errors
  - Maximum attempts are configurable
  - Wait times are observable for logging/checkpointing
  - Raises a non-retryable error on terminal conditions (4xx except 429)
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Final

_MAX_ATTEMPTS: Final[int] = 5
_BASE_DELAY_SECONDS: Final[float] = 2.0
_MAX_DELAY_SECONDS: Final[float] = 60.0
_JITTER_FACTOR: Final[float] = 0.25


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, attempts: int, last_error: Exception) -> None:
        super().__init__(
            f"Exhausted {attempts} retry attempts. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        )
        self.attempts = attempts
        self.last_error = last_error


class NonRetryableError(Exception):
    """Raised for terminal errors that should not be retried (e.g. 400, 403)."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"Non-retryable error {status_code}: {message}")
        self.status_code = status_code


@dataclass
class RetryPolicy:
    """Exponential backoff retry policy for Gemini API calls.

    Args:
        max_attempts: Maximum number of total attempts (including first).
        base_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay cap in seconds.
        jitter_factor: Random jitter range as fraction of delay.

    Usage:
        policy = RetryPolicy()
        for attempt in policy.attempts():
            try:
                result = call_api()
                break
            except RateLimitError as e:
                policy.handle_rate_limit(attempt, e)
            except TransientError as e:
                policy.handle_transient(attempt, e)
    """

    max_attempts: int = _MAX_ATTEMPTS
    base_delay: float = _BASE_DELAY_SECONDS
    max_delay: float = _MAX_DELAY_SECONDS
    jitter_factor: float = _JITTER_FACTOR

    def backoff_seconds(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter.

        Args:
            attempt: 0-indexed attempt number (0 = first retry).

        Returns:
            Seconds to wait before the next attempt.
        """
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        jitter = delay * self.jitter_factor * (random.random() * 2 - 1)  # noqa: S311
        return float(max(0.0, delay + jitter))

    def sleep_for_retry(self, attempt: int) -> float:
        """Sleep for the computed backoff. Returns actual wait time."""
        wait = self.backoff_seconds(attempt)
        time.sleep(wait)
        return wait
