"""Tests for domain policies — holdout protection and abstention."""

import logging
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from raglab.domain.enums import DatasetSplit
from raglab.domain.errors import HoldoutAccessViolationError
from raglab.domain.policies import AbstentionPolicy, HoldoutPolicy


class TestHoldoutPolicy(unittest.TestCase):
    def test_development_access_always_ok(self) -> None:
        policy = HoldoutPolicy.development_only()
        policy.check_access(DatasetSplit.DEVELOPMENT)  # No exception

    def test_test_access_always_ok(self) -> None:
        policy = HoldoutPolicy.development_only()
        policy.check_access(DatasetSplit.TEST)  # No exception

    def test_query_holdout_denied_by_default(self) -> None:
        policy = HoldoutPolicy.development_only()
        with self.assertRaises(HoldoutAccessViolationError):
            policy.check_access(DatasetSplit.QUERY_HOLDOUT)

    def test_corpus_holdout_denied_by_default(self) -> None:
        policy = HoldoutPolicy.development_only()
        with self.assertRaises(HoldoutAccessViolationError):
            policy.check_access(DatasetSplit.CORPUS_HOLDOUT)

    def test_explicit_authorization_grants_access(self) -> None:
        policy = HoldoutPolicy.with_holdout_authorization(
            DatasetSplit.QUERY_HOLDOUT
        )
        policy.check_access(DatasetSplit.QUERY_HOLDOUT)  # No exception

    def test_partial_authorization(self) -> None:
        policy = HoldoutPolicy.with_holdout_authorization(
            DatasetSplit.QUERY_HOLDOUT
        )
        # Corpus holdout still denied
        with self.assertRaises(HoldoutAccessViolationError):
            policy.check_access(DatasetSplit.CORPUS_HOLDOUT)

    def test_holdout_access_produces_log(self) -> None:
        policy = HoldoutPolicy.with_holdout_authorization(
            DatasetSplit.QUERY_HOLDOUT
        )
        with self.assertLogs("raglab.domain.policies", level="INFO") as cm:
            policy.check_access(DatasetSplit.QUERY_HOLDOUT)
        self.assertTrue(any("HOLDOUT_ACCESS_GRANTED" in msg for msg in cm.output))

    def test_holdout_denial_produces_warning(self) -> None:
        policy = HoldoutPolicy.development_only()
        with self.assertLogs("raglab.domain.policies", level="WARNING") as cm:
            with self.assertRaises(HoldoutAccessViolationError):
                policy.check_access(DatasetSplit.QUERY_HOLDOUT)
        self.assertTrue(any("HOLDOUT_ACCESS_DENIED" in msg for msg in cm.output))


class TestAbstentionPolicy(unittest.TestCase):
    def test_abstain_on_low_scores(self) -> None:
        policy = AbstentionPolicy(min_retrieval_score=0.5, min_evidence_count=1)
        self.assertTrue(policy.should_abstain([0.1, 0.2, 0.3], 3))

    def test_no_abstention_on_high_scores(self) -> None:
        policy = AbstentionPolicy(min_retrieval_score=0.5, min_evidence_count=1)
        self.assertFalse(policy.should_abstain([0.8, 0.6], 2))

    def test_abstain_on_insufficient_evidence(self) -> None:
        policy = AbstentionPolicy(min_retrieval_score=0.1, min_evidence_count=3)
        self.assertTrue(policy.should_abstain([0.9, 0.8], 2))

    def test_abstain_on_empty_scores(self) -> None:
        policy = AbstentionPolicy(min_retrieval_score=0.1, min_evidence_count=0)
        self.assertTrue(policy.should_abstain([], 0))

    def test_invalid_min_score_raises(self) -> None:
        with self.assertRaises(ValueError):
            AbstentionPolicy(min_retrieval_score=float("nan"), min_evidence_count=1)

    def test_negative_evidence_count_raises(self) -> None:
        with self.assertRaises(ValueError):
            AbstentionPolicy(min_retrieval_score=0.5, min_evidence_count=-1)


if __name__ == "__main__":
    unittest.main()
