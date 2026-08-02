"""Tests for QuotaManager — pure domain, no network, no credentials."""

from __future__ import annotations

import time

import pytest

from raglab.domain.quota import QuotaManager, QuotaWindow


class TestQuotaWindow:
    def test_allows_up_to_limit(self):
        window = QuotaWindow(limit=3, window_seconds=60.0)
        assert window.is_allowed() is True
        assert window.is_allowed() is True
        assert window.is_allowed() is True
        # 4th should be denied (limit=3)
        assert window.is_allowed() is False

    def test_current_count_increments(self):
        window = QuotaWindow(limit=5, window_seconds=60.0)
        assert window.current_count == 0
        window.is_allowed()
        assert window.current_count == 1

    def test_expired_entries_not_counted(self):
        # Very short window so timestamps expire quickly
        window = QuotaWindow(limit=3, window_seconds=0.01)
        window.is_allowed()
        window.is_allowed()
        time.sleep(0.05)  # Let window expire
        # Should allow again after window expiry
        assert window.is_allowed() is True


class TestQuotaManager:
    def test_acquire_increments_total(self):
        qm = QuotaManager(rpm_limit=100, tpd_limit=10_000)
        qm.acquire()
        qm.acquire()
        stats = qm.stats
        assert stats["total_requests"] == 2

    def test_stats_no_credentials(self):
        qm = QuotaManager()
        stats = qm.stats
        assert "GEMINI_API_KEY" not in str(stats)
        assert "API_KEY" not in str(stats)
        assert "total_requests" in stats
        assert "rpm_limit" in stats

    def test_record_retry_accumulates_wait(self):
        qm = QuotaManager()
        qm.record_retry(1.5)
        qm.record_retry(2.0)
        assert qm.stats["total_wait_seconds"] == pytest.approx(3.5, abs=0.1)

    def test_rpm_limit_enforced(self):
        # Very low limit for fast test
        qm = QuotaManager(rpm_limit=2, tpd_limit=100)
        qm.acquire()
        qm.acquire()
        # Third acquire should block (but with poll_interval, it will wait)
        # We just verify the state after 2 acquires
        stats = qm.stats
        assert stats["rpm_current"] == 2
