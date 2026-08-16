"""Unit tests for the agentic conversational smoke harness.

Validates the high-level canary runner without external network calls or credentials.
"""

from __future__ import annotations

import json
import unittest
from collections.abc import Sequence

from raglab.domain.entities import GeneratedAnswer, RetrievedEvidence
from raglab.domain.errors import CitationProvenanceMismatchError
from raglab.infrastructure.fakes.fake_generator_adapter import (
    FakeGeneratorAdapter,
)


class TestSmokeHarness(unittest.TestCase):
    """Test suite for run_known_canaries execution and report generation."""

    def test_01_fake_generator_returns_pass_report_for_three_canaries(self) -> None:
        """Verify that FakeGeneratorAdapter executes all 3 canaries and produces PASS report."""
        from raglab.agentic.smoke_harness import run_known_canaries

        generator = FakeGeneratorAdapter()
        report = run_known_canaries(generator, backend="fake")

        # Ensure full JSON serializability
        serialized = json.dumps(report)
        self.assertIsInstance(serialized, str)

        self.assertEqual(report.get("backend"), "fake")
        self.assertEqual(report.get("model_id"), generator.model_id)
        self.assertEqual(report.get("status"), "PASS")
        self.assertIsNone(report.get("error"))

        canaries = report.get("canaries")
        self.assertIsInstance(canaries, dict)
        self.assertEqual(
            set(canaries.keys()),
            {
                "direct_supported_fact",
                "conversational_ellipsis",
                "unsupported_abstention",
            },
        )
        self.assertEqual(canaries["direct_supported_fact"]["status"], "PASS")
        self.assertEqual(canaries["conversational_ellipsis"]["status"], "PASS")
        self.assertEqual(canaries["unsupported_abstention"]["status"], "PASS")

    def test_02_provenance_failure_returns_sanitized_fail_report(self) -> None:
        """Verify that provenance mismatch raises typed failure and emits sanitized report."""
        from raglab.agentic.smoke_harness import run_known_canaries

        class FailingProvenanceGenerator:
            def __init__(self) -> None:
                self.captured_calls: list[
                    tuple[str, str, tuple[RetrievedEvidence, ...]]
                ] = []

            @property
            def model_id(self) -> str:
                return "failing-provenance-generator"

            def generate(
                self,
                query_id: str,
                query: str,
                evidence: Sequence[RetrievedEvidence],
            ) -> GeneratedAnswer:
                self.captured_calls.append((query_id, query, tuple(evidence)))
                raise CitationProvenanceMismatchError("E1")

        generator = FailingProvenanceGenerator()
        report = run_known_canaries(generator, backend="fake")

        # Ensure full JSON serializability
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertIsInstance(serialized, str)

        self.assertEqual(report.get("status"), "FAIL")
        error_info = report.get("error")
        self.assertIsInstance(error_info, dict)
        self.assertEqual(error_info.get("type"), "CitationProvenanceMismatchError")
        self.assertTrue(set(error_info.keys()).issubset({"type", "message"}))

        # Sanitization verification: ensure no raw query or passage text leaks
        self.assertGreater(len(generator.captured_calls), 0)
        for _, call_query, call_evidence in generator.captured_calls:
            self.assertNotIn(call_query, serialized)
            for ev in call_evidence:
                self.assertNotIn(ev.text, serialized)

        # Ensure no sensitive keywords leak
        serialized_lower = serialized.lower()
        self.assertNotIn("secret", serialized_lower)
        self.assertNotIn("password", serialized_lower)
        self.assertNotIn("gemini_api_key", serialized_lower)


if __name__ == "__main__":
    unittest.main()
