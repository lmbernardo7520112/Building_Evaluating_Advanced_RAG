"""Tests for domain entities — invariants and construction."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from raglab.domain.entities import (
    Checkpoint,
    Chunk,
    Corpus,
    ExperimentRun,
    GeneratedAnswer,
    Query,
    QuestionSet,
    RetrievedEvidence,
)
from raglab.domain.enums import DatasetSplit, PipelineStrategy
from raglab.domain.errors import (
    InvalidIdentifierError,
    MissingProvenanceError,
)
from raglab.domain.value_objects import ChunkId, IntegrityDigest, RunId

VALID_HASH = "c11c323e9d5362d4706c3fbbe4b11a107e7c4648407399186aef64fc1fb14db3"


class TestCorpus(unittest.TestCase):
    def test_valid_corpus(self) -> None:
        c = Corpus(
            corpus_id="corpus-001",
            name="Test Corpus",
            fingerprint=IntegrityDigest(VALID_HASH),
            total_pages=100,
            total_documents=5,
        )
        self.assertEqual(c.corpus_id, "corpus-001")

    def test_empty_id_raises(self) -> None:
        with self.assertRaises(InvalidIdentifierError):
            Corpus("", "name", IntegrityDigest(VALID_HASH), 10, 1)

    def test_negative_pages_raises(self) -> None:
        with self.assertRaises(ValueError):
            Corpus("c1", "name", IntegrityDigest(VALID_HASH), -1, 1)


class TestChunk(unittest.TestCase):
    def test_valid_chunk(self) -> None:
        c = Chunk(
            chunk_id=ChunkId("ch-1"),
            document_id="doc-1",
            text="some content",
            start_page=0,
            end_page=1,
        )
        self.assertEqual(c.document_id, "doc-1")

    def test_missing_provenance_raises(self) -> None:
        with self.assertRaises(MissingProvenanceError):
            Chunk(ChunkId("ch-1"), "", "text", 0, 0)

    def test_negative_page_raises(self) -> None:
        with self.assertRaises(ValueError):
            Chunk(ChunkId("ch-1"), "doc-1", "text", -1, 0)

    def test_end_before_start_raises(self) -> None:
        with self.assertRaises(ValueError):
            Chunk(ChunkId("ch-1"), "doc-1", "text", 5, 3)


class TestQuery(unittest.TestCase):
    def test_valid_query(self) -> None:
        q = Query("q-1", "What is RAG?", DatasetSplit.DEVELOPMENT)
        self.assertEqual(q.split, DatasetSplit.DEVELOPMENT)

    def test_empty_text_raises(self) -> None:
        with self.assertRaises(InvalidIdentifierError):
            Query("q-1", "", DatasetSplit.DEVELOPMENT)


class TestQuestionSet(unittest.TestCase):
    def test_mismatched_split_raises(self) -> None:
        q = Query("q-1", "What?", DatasetSplit.TEST)
        with self.assertRaises(ValueError):
            QuestionSet(DatasetSplit.DEVELOPMENT, (q,))


class TestRetrievedEvidence(unittest.TestCase):
    def test_valid_evidence(self) -> None:
        e = RetrievedEvidence(
            chunk_id=ChunkId("ch-1"),
            document_id="doc-1",
            text="evidence text",
            rank=1,
            score=0.95,
        )
        self.assertEqual(e.rank, 1)

    def test_missing_doc_id_raises(self) -> None:
        with self.assertRaises(MissingProvenanceError):
            RetrievedEvidence(ChunkId("ch-1"), "", "text", 1, 0.9)

    def test_zero_rank_raises(self) -> None:
        with self.assertRaises(ValueError):
            RetrievedEvidence(ChunkId("ch-1"), "doc-1", "text", 0, 0.9)

    def test_inf_score_raises(self) -> None:
        with self.assertRaises(ValueError):
            RetrievedEvidence(ChunkId("ch-1"), "doc-1", "text", 1, float("inf"))


class TestGeneratedAnswer(unittest.TestCase):
    def test_abstained_distinguishable(self) -> None:
        normal = GeneratedAnswer("q-1", "Answer text", False, ())
        abstained = GeneratedAnswer("q-1", "", True, ())
        self.assertFalse(normal.abstained)
        self.assertTrue(abstained.abstained)
        self.assertNotEqual(normal, abstained)


class TestCheckpoint(unittest.TestCase):
    def test_compatible(self) -> None:
        fp1 = IntegrityDigest(VALID_HASH)
        fp2 = IntegrityDigest("a" * 64)
        cp = Checkpoint(
            run_id=RunId("run-1"),
            corpus_fingerprint=fp1,
            config_fingerprint=fp2,
        )
        self.assertTrue(cp.is_compatible(fp1, fp2))

    def test_incompatible_corpus(self) -> None:
        fp1 = IntegrityDigest(VALID_HASH)
        fp2 = IntegrityDigest("a" * 64)
        fp3 = IntegrityDigest("b" * 64)
        cp = Checkpoint(RunId("run-1"), fp1, fp2)
        self.assertFalse(cp.is_compatible(fp3, fp2))

    def test_incompatible_config(self) -> None:
        fp1 = IntegrityDigest(VALID_HASH)
        fp2 = IntegrityDigest("a" * 64)
        fp3 = IntegrityDigest("b" * 64)
        cp = Checkpoint(RunId("run-1"), fp1, fp2)
        self.assertFalse(cp.is_compatible(fp1, fp3))


class TestExperimentRun(unittest.TestCase):
    def test_valid_run(self) -> None:
        run = ExperimentRun(
            run_id=RunId("run-1"),
            strategy=PipelineStrategy.BASELINE,
            corpus_id="corpus-1",
            split=DatasetSplit.DEVELOPMENT,
        )
        self.assertEqual(run.strategy, PipelineStrategy.BASELINE)

    def test_empty_corpus_raises(self) -> None:
        with self.assertRaises(InvalidIdentifierError):
            ExperimentRun(
                RunId("run-1"), PipelineStrategy.BASELINE, "", DatasetSplit.TEST
            )


if __name__ == "__main__":
    unittest.main()
