"""Unit tests for FakeGeneratorAdapter page precedence and offline generation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from raglab.domain.entities import RetrievedEvidence
from raglab.domain.value_objects import ChunkId
from raglab.infrastructure.fakes.fake_generator_adapter import FakeGeneratorAdapter


@dataclass
class LegacyEvidenceDouble:
    """Test double representing legacy evidence with start_page/page attributes."""

    chunk_id: ChunkId
    document_id: str
    text: str
    rank: int
    score: float
    page_number: int | None = None
    start_page: int | None = None
    page: int | None = None
    passage_id: str | None = None
    content_sha256: str | None = None


_PAGE_PRECEDENCE_CASES = [
    # 1. page_number=91, doc_p7 -> 91
    (
        RetrievedEvidence(
            chunk_id=ChunkId("c1"),
            document_id="doc_p7",
            text="Evidence text for case 1",
            rank=1,
            score=0.9,
            page_number=91,
        ),
        91,
    ),
    # 2. page_number=0, doc_p7 -> 0
    (
        RetrievedEvidence(
            chunk_id=ChunkId("c2"),
            document_id="doc_p7",
            text="Evidence text for case 2",
            rank=1,
            score=0.9,
            page_number=0,
        ),
        0,
    ),
    # 3. page_number=None, start_page=33, doc_p7 -> 33
    (
        LegacyEvidenceDouble(
            chunk_id=ChunkId("c3"),
            document_id="doc_p7",
            text="Evidence text for case 3",
            rank=1,
            score=0.9,
            page_number=None,
            start_page=33,
        ),
        33,
    ),
    # 4. page_number=None, start_page=None, page=44, doc_p7 -> 44
    (
        LegacyEvidenceDouble(
            chunk_id=ChunkId("c4"),
            document_id="doc_p7",
            text="Evidence text for case 4",
            rank=1,
            score=0.9,
            page_number=None,
            start_page=None,
            page=44,
        ),
        44,
    ),
    # 5. todos explícitos/legados None, doc_p7 -> 7
    (
        RetrievedEvidence(
            chunk_id=ChunkId("c5"),
            document_id="doc_p7",
            text="Evidence text for case 5",
            rank=1,
            score=0.9,
            page_number=None,
        ),
        7,
    ),
    # 6. todos explícitos/legados None, unpaginated_doc -> 0
    (
        RetrievedEvidence(
            chunk_id=ChunkId("c6"),
            document_id="unpaginated_doc",
            text="Evidence text for case 6",
            rank=1,
            score=0.9,
            page_number=None,
        ),
        0,
    ),
]


class TestFakeGeneratorPagePrecedence:
    """Matrix tests for page number resolution precedence in FakeGeneratorAdapter."""

    @pytest.mark.parametrize("evidence_item, expected_page", _PAGE_PRECEDENCE_CASES)
    def test_page_precedence_matrix(self, evidence_item, expected_page):
        adapter = FakeGeneratorAdapter()
        answer = adapter.generate(
            query_id="q_test_prec",
            query="Qual é a técnica de demonstração?",
            evidence=[evidence_item],
        )
        assert answer.abstained is False
        assert len(answer.citations) >= 1
        assert answer.citations[0].page_number == expected_page
