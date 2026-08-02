"""Tests for RetryPolicy — pure domain, no network, no credentials."""

from __future__ import annotations

import pytest

from raglab.domain.retry import (
    NonRetryableError,
    RetryExhaustedError,
    RetryPolicy,
)


class TestRetryPolicy:
    def test_backoff_increases_exponentially(self):
        policy = RetryPolicy(max_attempts=5, base_delay=1.0, jitter_factor=0.0)
        delays = [policy.backoff_seconds(i) for i in range(4)]
        # Jitter=0 so should be exactly 1, 2, 4, 8
        assert delays[0] == pytest.approx(1.0)
        assert delays[1] == pytest.approx(2.0)
        assert delays[2] == pytest.approx(4.0)
        assert delays[3] == pytest.approx(8.0)

    def test_backoff_capped_at_max_delay(self):
        policy = RetryPolicy(max_attempts=5, base_delay=1.0, max_delay=5.0, jitter_factor=0.0)
        delay = policy.backoff_seconds(10)  # Would be 1024 without cap
        assert delay == pytest.approx(5.0)

    def test_backoff_never_negative(self):
        policy = RetryPolicy(jitter_factor=10.0)  # Extreme jitter
        for i in range(5):
            assert policy.backoff_seconds(i) >= 0.0

    def test_retry_exhausted_error_has_attempt_count(self):
        exc = ValueError("last error")
        err = RetryExhaustedError(attempts=5, last_error=exc)
        assert err.attempts == 5
        assert err.last_error is exc
        assert "5" in str(err)

    def test_non_retryable_error_has_status_code(self):
        err = NonRetryableError(status_code=403, message="Forbidden")
        assert err.status_code == 403
        assert "403" in str(err)

    def test_retry_policy_defaults(self):
        policy = RetryPolicy()
        assert policy.max_attempts == 5
        assert policy.base_delay == 2.0
        assert policy.max_delay == 60.0
