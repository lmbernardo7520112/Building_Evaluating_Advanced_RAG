"""Tests for InMemoryBaselineAdapter.

Covers: retrieval, top-k, provenance, determinism,
empty query, offline operation.
"""

from __future__ import annotations

import unittest

from raglab.domain.entities import Chunk
from raglab.domain.value_objects import ChunkId
from raglab.infrastructure.retrieval.baseline_adapter import (
    DeterministicEmbedding,
    InMemoryBaselineAdapter,
    load_tiny_corpus,
)


def _make_chunks() -> list[Chunk]:
    """Create test chunks with provenance."""
    return [
        Chunk(
            chunk_id=ChunkId("c1"),
            document_id="doc-1",
            text="Information retrieval measures precision and recall.",
            start_page=0,
            end_page=0,
        ),
        Chunk(
            chunk_id=ChunkId("c2"),
            document_id="doc-1",
            text="RAG combines retrieval with language model generation.",
            start_page=1,
            end_page=1,
        ),
        Chunk(
            chunk_id=ChunkId("c3"),
            document_id="doc-2",
            text="Statistical significance tests verify observed differences.",
            start_page=0,
            end_page=0,
        ),
    ]


class TestDeterministicEmbedding(unittest.TestCase):
    """Test the hash-based deterministic embedding."""

    def test_determinism(self) -> None:
        """Same text always produces same embedding."""
        emb = DeterministicEmbedding()
        v1 = emb.embed("hello world")
        v2 = emb.embed("hello world")
        self.assertEqual(v1, v2)

    def test_different_texts_differ(self) -> None:
        """Different texts produce different embeddings."""
        emb = DeterministicEmbedding()
        v1 = emb.embed("hello")
        v2 = emb.embed("world")
        self.assertNotEqual(v1, v2)

    def test_dimension(self) -> None:
        """Embedding has correct dimension."""
        emb = DeterministicEmbedding(dimension=128)
        v = emb.embed("test")
        self.assertEqual(len(v), 128)

    def test_unit_vector(self) -> None:
        """Embedding is approximately unit length."""
        emb = DeterministicEmbedding()
        v = emb.embed("test")
        import math
        magnitude = math.sqrt(sum(x * x for x in v))
        self.assertAlmostEqual(magnitude, 1.0, places=6)

    def test_model_id(self) -> None:
        """Model ID identifies the deterministic method."""
        emb = DeterministicEmbedding()
        self.assertEqual(emb.model_id, "deterministic-sha256-hash")


class TestInMemoryBaselineAdapter(unittest.TestCase):
    """Test baseline retrieval adapter."""

    def setUp(self) -> None:
        self.adapter = InMemoryBaselineAdapter()
        self.chunks = _make_chunks()
        self.adapter.index_chunks(self.chunks)

    def test_retrieval_returns_results(self) -> None:
        """Retrieval returns non-empty results."""
        results = self.adapter.retrieve("retrieval", top_k=3)
        self.assertGreater(len(results), 0)

    def test_top_k_limits_results(self) -> None:
        """top_k limits the number of results."""
        results = self.adapter.retrieve("test", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_top_k_one(self) -> None:
        """top_k=1 returns exactly 1 result."""
        results = self.adapter.retrieve("test", top_k=1)
        self.assertEqual(len(results), 1)

    def test_results_have_provenance(self) -> None:
        """Each result has document_id and chunk_id (provenance)."""
        results = self.adapter.retrieve("retrieval", top_k=3)
        for r in results:
            self.assertTrue(r.document_id)
            self.assertTrue(r.chunk_id.value)

    def test_results_ranked(self) -> None:
        """Results have sequential ranks starting from 1."""
        results = self.adapter.retrieve("test", top_k=3)
        for i, r in enumerate(results, start=1):
            self.assertEqual(r.rank, i)

    def test_scores_finite(self) -> None:
        """All scores are finite."""
        import math
        results = self.adapter.retrieve("test", top_k=3)
        for r in results:
            self.assertTrue(math.isfinite(r.score))

    def test_deterministic_ordering(self) -> None:
        """Same query always produces same order."""
        r1 = self.adapter.retrieve("retrieval", top_k=3)
        r2 = self.adapter.retrieve("retrieval", top_k=3)
        ids1 = [r.chunk_id.value for r in r1]
        ids2 = [r.chunk_id.value for r in r2]
        self.assertEqual(ids1, ids2)

    def test_empty_query_returns_empty(self) -> None:
        """Empty query returns no results."""
        results = self.adapter.retrieve("", top_k=3)
        self.assertEqual(len(results), 0)

    def test_whitespace_query_returns_empty(self) -> None:
        """Whitespace-only query returns no results."""
        results = self.adapter.retrieve("   ", top_k=3)
        self.assertEqual(len(results), 0)

    def test_empty_index_returns_empty(self) -> None:
        """Query on empty index returns no results."""
        adapter = InMemoryBaselineAdapter()
        results = adapter.retrieve("test", top_k=3)
        self.assertEqual(len(results), 0)

    def test_clear_removes_data(self) -> None:
        """clear() removes all indexed data."""
        self.adapter.clear()
        results = self.adapter.retrieve("test", top_k=3)
        self.assertEqual(len(results), 0)

    def test_zero_top_k_returns_empty(self) -> None:
        """top_k < 1 returns empty."""
        results = self.adapter.retrieve("test", top_k=0)
        self.assertEqual(len(results), 0)

    def test_scores_descending(self) -> None:
        """Results are sorted by score descending."""
        results = self.adapter.retrieve("retrieval", top_k=3)
        scores = [r.score for r in results]
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i], scores[i + 1])


class TestLoadTinyCorpus(unittest.TestCase):
    """Test tiny corpus loading."""

    def test_load_corpus(self) -> None:
        """Can load the tiny corpus and get chunks."""
        import os
        corpus_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "data", "tiny_corpus", "corpus.json",
        )
        corpus_path = os.path.normpath(corpus_path)
        if not os.path.exists(corpus_path):
            # Try from project root
            corpus_path = os.path.join(
                os.path.dirname(__file__),
                *([".." ] * 4),
                "data", "tiny_corpus", "corpus.json",
            )
            corpus_path = os.path.normpath(corpus_path)

        if os.path.exists(corpus_path):
            chunks, data = load_tiny_corpus(corpus_path)
            self.assertGreater(len(chunks), 0)
            self.assertIn("documents", data)
            self.assertIn("questions", data)
            # Verify all chunks have provenance
            for chunk in chunks:
                self.assertTrue(chunk.document_id)
                self.assertTrue(chunk.chunk_id.value)
        else:
            self.skipTest("tiny corpus not found at expected path")


if __name__ == "__main__":
    unittest.main()
