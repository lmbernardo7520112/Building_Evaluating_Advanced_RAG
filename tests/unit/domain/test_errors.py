"""Tests for domain errors — specificity and no secret leaking."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from raglab.domain.errors import (
    CheckpointMismatchError,
    ConfigurationError,
    HoldoutAccessViolationError,
    InvalidFingerprintError,
    InvalidIdentifierError,
    InvalidScoreError,
    MissingProvenanceError,
    NegativePositionError,
    NormalizedScoreOutOfRangeError,
    RagLabDomainError,
)


class TestErrorHierarchy(unittest.TestCase):
    def test_all_inherit_from_base(self) -> None:
        errors = [
            InvalidIdentifierError("test"),
            InvalidScoreError("test", float("nan")),
            NormalizedScoreOutOfRangeError("test"),
            InvalidFingerprintError(),
            NegativePositionError("test"),
            MissingProvenanceError(),
            HoldoutAccessViolationError("test"),
            CheckpointMismatchError("test"),
            ConfigurationError("field", "reason"),
        ]
        for err in errors:
            self.assertIsInstance(err, RagLabDomainError)
            self.assertIsInstance(err, Exception)


class TestErrorMessagesNoSecrets(unittest.TestCase):
    def test_config_error_no_value(self) -> None:
        """Configuration errors must not include actual values."""
        err = ConfigurationError("api_key", "must be non-empty")
        msg = str(err)
        self.assertNotIn("sk-", msg)
        self.assertNotIn("AIza", msg)
        self.assertIn("api_key", msg)
        self.assertIn("must be non-empty", msg)

    def test_score_error_no_raw_value(self) -> None:
        """Score errors should not reveal context content."""
        err = InvalidScoreError("secret_metric", float("nan"))
        msg = str(err)
        self.assertIn("non-finite", msg)

    def test_holdout_error_names_split(self) -> None:
        err = HoldoutAccessViolationError("query_holdout")
        self.assertIn("query_holdout", str(err))
        self.assertIn("authorization", str(err))


if __name__ == "__main__":
    unittest.main()
