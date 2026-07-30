"""Tests for domain enums."""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from raglab.domain.enums import DatasetSplit, MetricName, PipelineStrategy, QuestionState


class TestPipelineStrategy(unittest.TestCase):
    def test_three_strategies(self) -> None:
        strategies = list(PipelineStrategy)
        self.assertEqual(len(strategies), 3)
        self.assertIn(PipelineStrategy.BASELINE, strategies)
        self.assertIn(PipelineStrategy.SENTENCE_WINDOW, strategies)
        self.assertIn(PipelineStrategy.AUTO_MERGING, strategies)

    def test_values_are_strings(self) -> None:
        for s in PipelineStrategy:
            self.assertIsInstance(s.value, str)


class TestDatasetSplit(unittest.TestCase):
    def test_four_splits(self) -> None:
        self.assertEqual(len(list(DatasetSplit)), 4)

    def test_holdout_detection(self) -> None:
        self.assertFalse(DatasetSplit.DEVELOPMENT.is_holdout)
        self.assertFalse(DatasetSplit.TEST.is_holdout)
        self.assertTrue(DatasetSplit.QUERY_HOLDOUT.is_holdout)
        self.assertTrue(DatasetSplit.CORPUS_HOLDOUT.is_holdout)


class TestQuestionState(unittest.TestCase):
    def test_six_states(self) -> None:
        self.assertEqual(len(list(QuestionState)), 6)

    def test_transactional_states(self) -> None:
        names = {s.value for s in QuestionState}
        self.assertIn("pending", names)
        self.assertIn("running", names)
        self.assertIn("complete", names)
        self.assertIn("retryable", names)
        self.assertIn("terminal", names)
        self.assertIn("blocked_by_quota", names)


if __name__ == "__main__":
    unittest.main()
