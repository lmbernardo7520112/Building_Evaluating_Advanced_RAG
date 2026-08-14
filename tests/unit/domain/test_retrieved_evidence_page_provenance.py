"""Tests for explicit page_number provenance in RetrievedEvidence."""

from __future__ import annotations

import unittest

from raglab.domain.entities import RetrievedEvidence
from raglab.domain.errors import NegativePositionError
from raglab.domain.value_objects import ChunkId


class TestRetrievedEvidencePageProvenance(unittest.TestCase):
    """Test suite for page_number field invariants on RetrievedEvidence."""

    def test_retrieved_evidence_accepts_valid_page_number_and_none(self) -> None:
        # Explicit positive page_number
        ev_p15 = RetrievedEvidence(
            chunk_id=ChunkId("c1"),
            document_id="doc_1",
            text="sample text",
            rank=1,
            score=0.9,
            page_number=15,
        )
        self.assertEqual(ev_p15.page_number, 15)

        # Explicit zero page_number
        ev_p0 = RetrievedEvidence(
            chunk_id=ChunkId("c2"),
            document_id="doc_1",
            text="sample text",
            rank=2,
            score=0.8,
            page_number=0,
        )
        self.assertEqual(ev_p0.page_number, 0)

        # Default / None page_number
        ev_none = RetrievedEvidence(
            chunk_id=ChunkId("c3"),
            document_id="doc_1",
            text="sample text",
            rank=3,
            score=0.7,
            page_number=None,
        )
        self.assertIsNone(ev_none.page_number)

    def test_retrieved_evidence_rejects_negative_page_number(self) -> None:
        with self.assertRaises(NegativePositionError):
            RetrievedEvidence(
                chunk_id=ChunkId("c1"),
                document_id="doc_1",
                text="sample text",
                rank=1,
                score=0.9,
                page_number=-1,
            )
