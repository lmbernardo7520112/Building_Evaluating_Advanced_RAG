"""Tests for deterministic Recall@k and MRR metrics.

Includes manually computed examples, edge cases, and property-like
determinism checks. Does NOT test RAG Triad metrics.
"""

from __future__ import annotations

import unittest

from raglab.domain.metrics import (
    compute_mrr,
    compute_recall_at_k,
    compute_reciprocal_rank,
)


class TestRecallAtK(unittest.TestCase):
    """Test Recall@k computation."""

    def test_perfect_recall(self) -> None:
        """All relevant items retrieved → Recall = 1.0."""
        result = compute_recall_at_k(
            retrieved_ids=["a", "b", "c"],
            relevant_ids={"a", "b"},
            k=3,
        )
        self.assertAlmostEqual(result.value, 1.0)
        self.assertEqual(result.retrieved_relevant, 2)
        self.assertEqual(result.total_relevant, 2)
        self.assertFalse(result.skipped)

    def test_partial_recall(self) -> None:
        """Manual: 1 of 2 relevant in top-2 → Recall@2 = 0.5."""
        result = compute_recall_at_k(
            retrieved_ids=["a", "x", "b"],
            relevant_ids={"a", "b"},
            k=2,
        )
        self.assertAlmostEqual(result.value, 0.5)  # only "a" in top-2
        self.assertEqual(result.retrieved_relevant, 1)

    def test_zero_recall(self) -> None:
        """No relevant items in top-k → Recall = 0.0."""
        result = compute_recall_at_k(
            retrieved_ids=["x", "y", "z"],
            relevant_ids={"a", "b"},
            k=3,
        )
        self.assertAlmostEqual(result.value, 0.0)
        self.assertEqual(result.retrieved_relevant, 0)

    def test_empty_ground_truth_skip(self) -> None:
        """Empty ground truth with skip policy → None value, skipped=True."""
        result = compute_recall_at_k(
            retrieved_ids=["a", "b"],
            relevant_ids=set(),
            k=3,
            empty_gt_policy="skip",
        )
        self.assertIsNone(result.value)
        self.assertTrue(result.skipped)

    def test_empty_ground_truth_zero(self) -> None:
        """Empty ground truth with zero policy → 0.0."""
        result = compute_recall_at_k(
            retrieved_ids=["a"],
            relevant_ids=set(),
            k=3,
            empty_gt_policy="zero",
        )
        self.assertAlmostEqual(result.value, 0.0)
        self.assertFalse(result.skipped)

    def test_duplicates_do_not_inflate(self) -> None:
        """Duplicate IDs in retrieved should not inflate recall."""
        result = compute_recall_at_k(
            retrieved_ids=["a", "a", "a"],
            relevant_ids={"a", "b"},
            k=3,
        )
        # "a" appears 3 times but is counted once → 1/2 = 0.5
        self.assertAlmostEqual(result.value, 0.5)
        self.assertEqual(result.retrieved_relevant, 1)

    def test_k_less_than_one_raises(self) -> None:
        """k < 1 should raise ValueError."""
        with self.assertRaises(ValueError, msg="k must be >= 1"):
            compute_recall_at_k(["a"], {"a"}, k=0)

    def test_unknown_ids_treated_as_irrelevant(self) -> None:
        """IDs not in ground truth are non-relevant."""
        result = compute_recall_at_k(
            retrieved_ids=["unknown1", "unknown2", "a"],
            relevant_ids={"a", "b"},
            k=3,
        )
        self.assertAlmostEqual(result.value, 0.5)

    def test_value_in_unit_interval(self) -> None:
        """Recall must always be in [0, 1]."""
        result = compute_recall_at_k(
            retrieved_ids=["a", "b", "c", "d"],
            relevant_ids={"a", "b"},
            k=4,
        )
        self.assertGreaterEqual(result.value, 0.0)
        self.assertLessEqual(result.value, 1.0)

    def test_empty_retrieved_list(self) -> None:
        """No retrieved items → recall = 0."""
        result = compute_recall_at_k(
            retrieved_ids=[],
            relevant_ids={"a", "b"},
            k=5,
        )
        self.assertAlmostEqual(result.value, 0.0)

    def test_unknown_policy_raises(self) -> None:
        """Unknown empty_gt_policy should raise."""
        with self.assertRaises(ValueError):
            compute_recall_at_k(["a"], set(), k=1, empty_gt_policy="invalid")

    def test_k_larger_than_list(self) -> None:
        """k > len(retrieved) → uses all available."""
        result = compute_recall_at_k(
            retrieved_ids=["a"],
            relevant_ids={"a", "b"},
            k=100,
        )
        self.assertAlmostEqual(result.value, 0.5)

    def test_determinism(self) -> None:
        """Same inputs always produce same outputs."""
        for _ in range(10):
            r = compute_recall_at_k(["a", "b", "c"], {"a", "c"}, k=3)
            self.assertAlmostEqual(r.value, 1.0)


class TestReciprocalRank(unittest.TestCase):
    """Test single-query reciprocal rank."""

    def test_first_position(self) -> None:
        """Relevant at rank 1 → RR = 1.0."""
        rr = compute_reciprocal_rank(["a", "b", "c"], {"a"})
        self.assertAlmostEqual(rr, 1.0)

    def test_second_position(self) -> None:
        """Relevant at rank 2 → RR = 0.5."""
        rr = compute_reciprocal_rank(["x", "a", "c"], {"a"})
        self.assertAlmostEqual(rr, 0.5)

    def test_third_position(self) -> None:
        """Relevant at rank 3 → RR = 1/3."""
        rr = compute_reciprocal_rank(["x", "y", "a"], {"a"})
        self.assertAlmostEqual(rr, 1.0 / 3.0)

    def test_not_found(self) -> None:
        """No relevant results → RR = 0.0."""
        rr = compute_reciprocal_rank(["x", "y", "z"], {"a"})
        self.assertAlmostEqual(rr, 0.0)

    def test_duplicates_dont_affect_rank(self) -> None:
        """Duplicates are skipped; rank counts unique items only."""
        # ["x", "x", "a"] → unique order: ["x", "a"] → "a" at rank 2
        rr = compute_reciprocal_rank(["x", "x", "a"], {"a"})
        self.assertAlmostEqual(rr, 0.5)

    def test_multiple_relevant_uses_first(self) -> None:
        """Multiple relevant items: RR is based on the first one found."""
        rr = compute_reciprocal_rank(["x", "a", "b"], {"a", "b"})
        self.assertAlmostEqual(rr, 0.5)  # "a" at rank 2

    def test_empty_retrieved(self) -> None:
        """Empty retrieved → RR = 0.0."""
        rr = compute_reciprocal_rank([], {"a"})
        self.assertAlmostEqual(rr, 0.0)


class TestMRR(unittest.TestCase):
    """Test Mean Reciprocal Rank across queries."""

    def test_manual_computation(self) -> None:
        """Manual: RRs = [1.0, 0.5, 1/3] → MRR = (1 + 0.5 + 0.333)/3 ≈ 0.611."""
        queries = [
            (["a", "b", "c"], {"a"}),      # RR = 1.0
            (["x", "a", "c"], {"a"}),      # RR = 0.5
            (["x", "y", "a"], {"a"}),      # RR = 1/3
        ]
        result = compute_mrr(queries)
        expected = (1.0 + 0.5 + 1.0 / 3.0) / 3.0
        self.assertAlmostEqual(result.value, expected, places=6)
        self.assertEqual(result.num_queries, 3)

    def test_perfect_mrr(self) -> None:
        """All at rank 1 → MRR = 1.0."""
        queries = [
            (["a"], {"a"}),
            (["b"], {"b"}),
        ]
        result = compute_mrr(queries)
        self.assertAlmostEqual(result.value, 1.0)

    def test_empty_queries_list(self) -> None:
        """No queries → MRR = None."""
        result = compute_mrr([])
        self.assertIsNone(result.value)
        self.assertEqual(result.num_queries, 0)

    def test_skip_empty_gt(self) -> None:
        """Queries with empty GT are excluded when policy=skip."""
        queries = [
            (["a"], {"a"}),        # RR = 1.0
            (["a"], set()),        # skipped
            (["x", "b"], {"b"}),   # RR = 0.5
        ]
        result = compute_mrr(queries, empty_gt_policy="skip")
        expected = (1.0 + 0.5) / 2.0
        self.assertAlmostEqual(result.value, expected)
        self.assertEqual(result.num_queries, 2)

    def test_zero_empty_gt(self) -> None:
        """Queries with empty GT get RR=0 when policy=zero."""
        queries = [
            (["a"], {"a"}),        # RR = 1.0
            (["a"], set()),        # RR = 0.0
        ]
        result = compute_mrr(queries, empty_gt_policy="zero")
        expected = (1.0 + 0.0) / 2.0
        self.assertAlmostEqual(result.value, expected)
        self.assertEqual(result.num_queries, 2)

    def test_all_empty_gt_skip(self) -> None:
        """All queries have empty GT with skip → None."""
        queries = [
            (["a"], set()),
            (["b"], set()),
        ]
        result = compute_mrr(queries, empty_gt_policy="skip")
        self.assertIsNone(result.value)
        self.assertEqual(result.num_queries, 0)

    def test_mrr_in_unit_interval(self) -> None:
        """MRR must be in [0, 1]."""
        queries = [
            (["a", "b"], {"a"}),
            (["x", "y", "z"], {"a"}),  # not found → RR = 0
        ]
        result = compute_mrr(queries)
        self.assertGreaterEqual(result.value, 0.0)
        self.assertLessEqual(result.value, 1.0)

    def test_unknown_policy_raises(self) -> None:
        """Unknown policy raises."""
        with self.assertRaises(ValueError):
            compute_mrr([], empty_gt_policy="invalid")

    def test_determinism(self) -> None:
        """Same inputs produce same MRR every time."""
        queries = [
            (["a", "b"], {"a"}),
            (["x", "a"], {"a"}),
        ]
        values = [compute_mrr(queries).value for _ in range(10)]
        self.assertTrue(all(v == values[0] for v in values))

    def test_reciprocal_ranks_stored(self) -> None:
        """Individual RRs are stored for audit trail."""
        queries = [
            (["a"], {"a"}),      # RR = 1.0
            (["x", "b"], {"b"}), # RR = 0.5
        ]
        result = compute_mrr(queries)
        self.assertEqual(len(result.reciprocal_ranks), 2)
        self.assertAlmostEqual(result.reciprocal_ranks[0], 1.0)
        self.assertAlmostEqual(result.reciprocal_ranks[1], 0.5)


if __name__ == "__main__":
    unittest.main()
