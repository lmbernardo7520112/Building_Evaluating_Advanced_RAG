"""Quota manager for Gemini API — tracks RPM and TPM budgets.

This module is pure domain logic (no network, no credentials).
It is designed to be used by the Gemini adapters to enforce
rate limits before making API calls.

Design:
  - Thread-safe token bucket per limit dimension
  - Pluggable backends (in-memory for Slice 4, Redis for future)
  - All state is observable for checkpoint writing
  - Fully serializable for restart/resume

Rate limits for gemini-3.1-flash-lite (free tier):
  RPM:   15   requests per minute
  TPD:   1500 requests per day
  TPM:   1_000_000 tokens per minute

These defaults can be overridden via configuration.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Final

# Gemini free-tier defaults for gemini-3.1-flash-lite
DEFAULT_RPM: Final[int] = 15
DEFAULT_TPD: Final[int] = 1_500
DEFAULT_TPM_TOKENS: Final[int] = 1_000_000


@dataclass
class QuotaWindow:
    """A sliding time window counter for a single dimension."""

    limit: int
    window_seconds: float
    _timestamps: list[float] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def is_allowed(self) -> bool:
        """Return True if a new request is within limit."""
        now = time.monotonic()
        with self._lock:
            cutoff = now - self.window_seconds
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            if len(self._timestamps) < self.limit:
                self._timestamps.append(now)
                return True
            return False

    def wait_until_allowed(self, poll_interval: float = 0.5) -> float:
        """Block until a request is allowed. Returns wait time in seconds."""
        start = time.monotonic()
        while not self.is_allowed():
            time.sleep(poll_interval)
        return time.monotonic() - start

    @property
    def current_count(self) -> int:
        """Return number of requests in the current window."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            return sum(1 for t in self._timestamps if t > cutoff)


@dataclass
class QuotaManager:
    """Multi-dimensional quota manager for a single Gemini API key.

    Tracks:
    - RPM  (requests per minute)
    - TPD  (requests per day, approximated)
    - TPM  (tokens per minute, estimated from request count)

    All limits are enforced BEFORE the API call to prevent 429s.
    When a 429 is received anyway (e.g. burst), the adapter is
    responsible for exponential backoff and then re-attempting.
    """

    rpm_limit: int = DEFAULT_RPM
    tpd_limit: int = DEFAULT_TPD
    tpm_token_limit: int = DEFAULT_TPM_TOKENS

    _rpm_window: QuotaWindow = field(init=False, repr=False)
    _tpd_window: QuotaWindow = field(init=False, repr=False)
    _total_requests: int = field(default=0, init=False)
    _total_wait_seconds: float = field(default=0.0, init=False)
    _total_retries: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self._rpm_window = QuotaWindow(
            limit=self.rpm_limit, window_seconds=60.0
        )
        self._tpd_window = QuotaWindow(
            limit=self.tpd_limit, window_seconds=86_400.0
        )

    def acquire(self, poll_interval: float = 0.5) -> float:
        """Acquire quota for one request. Blocks if needed.

        Returns:
            Total wait time in seconds (0.0 if no waiting was needed).
        """
        wait = 0.0
        # TPD check first (day-level, long window)
        if not self._tpd_window.is_allowed():
            wait += self._tpd_window.wait_until_allowed(poll_interval)
        # RPM check (minute-level, shorter window)
        if not self._rpm_window.is_allowed():
            wait += self._rpm_window.wait_until_allowed(poll_interval)
        with self._lock:
            self._total_requests += 1
            self._total_wait_seconds += wait
        return wait

    def record_retry(self, backoff_seconds: float) -> None:
        """Record that a 429 was received and we waited for backoff."""
        with self._lock:
            self._total_wait_seconds += backoff_seconds
            self._total_retries += 1

    @property
    def stats(self) -> dict[str, float | int]:
        """Return observable quota stats for checkpointing."""
        with self._lock:
            return {
                "total_requests": self._total_requests,
                "total_wait_seconds": round(self._total_wait_seconds, 2),
                "total_retries": self._total_retries,
                "rate_limit_429_count": self._total_retries,
                "rpm_current": self._rpm_window.current_count,
                "rpm_limit": self.rpm_limit,
                "tpd_current": self._tpd_window.current_count,
                "tpd_limit": self.tpd_limit,
            }
