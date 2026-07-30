"""Shared contract tests for RetrievalPort implementations.

Verifies that both InMemoryBaselineAdapter and LlamaIndexBaselineAdapter
strictly comply with RetrievalPort invariants:
- top_k limiting
- provenance (document_id, chunk_id)
- finite score
- 1-based sequential rank
- empty query returns empty list
- deterministic ordering
- offline execution
"""

from __future__ import annotations

import math
import unittest

from raglab.domain.entities import Chunk
from raglab.domain.value_objects import ChunkId
from raglab.infrastructure.retrieval.baseline_adapter import (
    InMemoryBaselineAdapter,
)
from raglab.infrastructure.retrieval.llamaindex_adapter import (
    LlamaIndexBaselineAdapter,
)


def _make_test_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id=ChunkId("c1"),
            document_id="doc-alpha",
            text="Information retrieval systems measure recall and precision.",
            start_page=0,
            end_page=0,
        ),
        Chunk(
            chunk_id=ChunkId("c2"),
            document_id="doc-alpha",
            text="Vector space models represent documents as high-dimensional vectors.",
            start_page=1,
            end_page=1,
        ),
        Chunk(
            chunk_id=ChunkId("c3"),
            document_id="doc-beta",
            text="RAG architecture combines retrieval with LLM generation.",
            start_page=0,
            end_page=0,
        ),
    ]


class BaseRetrievalContractTests:
    """Base mixin for testing any RetrievalPort implementation."""

    __test__ = False

    def create_adapter(self) -> object:
        raise NotImplementedError

    def test_retrieval_basic(self) -> None:
        adapter: object = self.create_adapter()
        chunks = _make_test_chunks()
        adapter.index_chunks(chunks)  # type: ignore[attr-defined]

        results = adapter.retrieve("retrieval", top_k=3)  # type: ignore[attr-defined]
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)

    def test_provenance_and_ranking(self) -> None:
        adapter: object = self.create_adapter()
        chunks = _make_test_chunks()
        adapter.index_chunks(chunks)  # type: ignore[attr-defined]

        results = adapter.retrieve("vector", top_k=2)  # type: ignore[attr-defined]
        for idx, item in enumerate(results, start=1):
            self.assertEqual(item.rank, idx)
            self.assertTrue(item.document_id)
            self.assertTrue(item.chunk_id.value)
            self.assertTrue(math.isfinite(item.score))

    def test_empty_query_returns_empty(self) -> None:
        adapter: object = self.create_adapter()
        chunks = _make_test_chunks()
        adapter.index_chunks(chunks)  # type: ignore[attr-defined]

        results = adapter.retrieve("", top_k=3)  # type: ignore[attr-defined]
        self.assertEqual(len(results), 0)

    def test_top_k_zero_returns_empty(self) -> None:
        adapter: object = self.create_adapter()
        chunks = _make_test_chunks()
        adapter.index_chunks(chunks)  # type: ignore[attr-defined]

        results = adapter.retrieve("query", top_k=0)  # type: ignore[attr-defined]
        self.assertEqual(len(results), 0)

    def test_determinism(self) -> None:
        adapter: object = self.create_adapter()
        chunks = _make_test_chunks()
        adapter.index_chunks(chunks)  # type: ignore[attr-defined]

        r1 = adapter.retrieve("retrieval", top_k=3)  # type: ignore[attr-defined]
        r2 = adapter.retrieve("retrieval", top_k=3)  # type: ignore[attr-defined]

        ids1 = [item.chunk_id.value for item in r1]
        ids2 = [item.chunk_id.value for item in r2]
        self.assertEqual(ids1, ids2)

    def test_clear(self) -> None:
        adapter: object = self.create_adapter()
        chunks = _make_test_chunks()
        adapter.index_chunks(chunks)  # type: ignore[attr-defined]
        adapter.clear()  # type: ignore[attr-defined]

        results = adapter.retrieve("retrieval", top_k=3)  # type: ignore[attr-defined]
        self.assertEqual(len(results), 0)


class TestInMemoryBaselineAdapterContract(
    BaseRetrievalContractTests, unittest.TestCase
):
    """Contract tests for InMemoryBaselineAdapter."""

    __test__ = True

    def create_adapter(self) -> InMemoryBaselineAdapter:
        return InMemoryBaselineAdapter()


class TestLlamaIndexBaselineAdapterContract(
    BaseRetrievalContractTests, unittest.TestCase
):
    """Contract tests for LlamaIndexBaselineAdapter."""

    __test__ = True

    def create_adapter(self) -> LlamaIndexBaselineAdapter:
        return LlamaIndexBaselineAdapter()


if __name__ == "__main__":
    unittest.main()
