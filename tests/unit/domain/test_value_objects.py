"""Tests for domain value objects — invariants and edge cases."""

import math
import unittest

# These tests use unittest (stdlib) to avoid requiring pytest installation.
# When pytest is authorized (Slice 1 supply chain), migrate to pytest style.

import sys
import os

# Add src to path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from raglab.domain.value_objects import (
    ChunkId,
    Citation,
    DocumentPage,
    IntegrityDigest,
    MetricResult,
    RunId,
)
from raglab.domain.errors import (
    InvalidFingerprintError,
    InvalidIdentifierError,
    InvalidScoreError,
    NegativePositionError,
    NormalizedScoreOutOfRangeError,
)


class TestChunkId(unittest.TestCase):
    def test_valid_chunk_id(self) -> None:
        cid = ChunkId("chunk-001")
        self.assertEqual(cid.value, "chunk-001")

    def test_empty_chunk_id_raises(self) -> None:
        with self.assertRaises(InvalidIdentifierError):
            ChunkId("")

    def test_whitespace_only_chunk_id_raises(self) -> None:
        with self.assertRaises(InvalidIdentifierError):
            ChunkId("   ")

    def test_chunk_id_immutable(self) -> None:
        cid = ChunkId("test")
        with self.assertRaises(AttributeError):
            cid.value = "modified"  # type: ignore[misc]


class TestRunId(unittest.TestCase):
    def test_valid_run_id(self) -> None:
        rid = RunId("run-2026-07-30-001")
        self.assertEqual(rid.value, "run-2026-07-30-001")

    def test_empty_run_id_raises(self) -> None:
        with self.assertRaises(InvalidIdentifierError):
            RunId("")


class TestIntegrityDigest(unittest.TestCase):
    VALID_HASH = "c11c323e9d5362d4706c3fbbe4b11a107e7c4648407399186aef64fc1fb14db3"

    def test_valid_digest(self) -> None:
        digest = IntegrityDigest(self.VALID_HASH)
        self.assertEqual(digest.hex_digest, self.VALID_HASH)

    def test_short_hash_raises(self) -> None:
        with self.assertRaises(InvalidFingerprintError):
            IntegrityDigest("abc123")

    def test_uppercase_hash_raises(self) -> None:
        with self.assertRaises(InvalidFingerprintError):
            IntegrityDigest(self.VALID_HASH.upper())

    def test_non_hex_raises(self) -> None:
        with self.assertRaises(InvalidFingerprintError):
            IntegrityDigest("g" * 64)

    def test_empty_raises(self) -> None:
        with self.assertRaises(InvalidFingerprintError):
            IntegrityDigest("")


class TestMetricResult(unittest.TestCase):
    def test_valid_metric(self) -> None:
        m = MetricResult(name="recall_at_k", value=0.85, k=5)
        self.assertEqual(m.value, 0.85)
        self.assertFalse(m.is_absent)

    def test_absent_metric_distinct_from_zero(self) -> None:
        absent = MetricResult(name="mrr", value=None)
        zero = MetricResult(name="mrr", value=0.0)
        self.assertTrue(absent.is_absent)
        self.assertFalse(zero.is_absent)
        self.assertNotEqual(absent, zero)

    def test_nan_score_raises(self) -> None:
        with self.assertRaises(InvalidScoreError):
            MetricResult(name="test", value=float("nan"))

    def test_inf_score_raises(self) -> None:
        with self.assertRaises(InvalidScoreError):
            MetricResult(name="test", value=float("inf"))

    def test_normalized_in_range(self) -> None:
        MetricResult(name="test", value=0.5, normalized=True)  # OK

    def test_normalized_above_one_raises(self) -> None:
        with self.assertRaises(NormalizedScoreOutOfRangeError):
            MetricResult(name="test", value=1.1, normalized=True)

    def test_normalized_below_zero_raises(self) -> None:
        with self.assertRaises(NormalizedScoreOutOfRangeError):
            MetricResult(name="test", value=-0.1, normalized=True)

    def test_empty_name_raises(self) -> None:
        with self.assertRaises(InvalidIdentifierError):
            MetricResult(name="", value=0.5)

    def test_k_zero_raises(self) -> None:
        with self.assertRaises(NegativePositionError):
            MetricResult(name="recall", value=0.5, k=0)


class TestDocumentPage(unittest.TestCase):
    def test_valid_page(self) -> None:
        p = DocumentPage(document_id="doc-1", page_number=0, text="Hello")
        self.assertEqual(p.page_number, 0)

    def test_negative_page_raises(self) -> None:
        with self.assertRaises(NegativePositionError):
            DocumentPage(document_id="doc-1", page_number=-1, text="Hello")

    def test_empty_doc_id_raises(self) -> None:
        with self.assertRaises(InvalidIdentifierError):
            DocumentPage(document_id="", page_number=0, text="Hello")


class TestCitation(unittest.TestCase):
    def test_valid_citation(self) -> None:
        c = Citation(
            document_id="doc-1",
            page_number=5,
            chunk_id=ChunkId("c1"),
            text_span="some text",
        )
        self.assertEqual(c.page_number, 5)

    def test_negative_page_raises(self) -> None:
        with self.assertRaises(NegativePositionError):
            Citation(
                document_id="doc-1",
                page_number=-1,
                chunk_id=ChunkId("c1"),
                text_span="text",
            )


if __name__ == "__main__":
    unittest.main()
