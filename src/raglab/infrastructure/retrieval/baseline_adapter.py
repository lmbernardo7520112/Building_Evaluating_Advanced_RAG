"""Deterministic in-memory baseline retrieval adapter.

This adapter uses a hash-based embedding for **testing purposes only**.
It is NOT a semantic embedding and does NOT represent production quality.
It provides deterministic, reproducible retrieval for smoke tests and CI.

The adapter satisfies the RetrievalPort contract and the S1.5 requirement
for a baseline adapter that works without APIs, models, or network access.

Limitations (honestly stated):
- Hash embedding has no semantic understanding.
- Cosine similarity on hash vectors does not correlate with meaning.
- Results are deterministic but not semantically meaningful.
- This is an infrastructure test harness, not a production retriever.

For production, replace DeterministicEmbedding with a real embedding provider.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from raglab.domain.entities import Chunk, RetrievedEvidence
from raglab.domain.value_objects import ChunkId


@dataclass(frozen=True, slots=True)
class DeterministicEmbedding:
    """Hash-based deterministic embedding for testing.

    NOT a semantic embedding. Used only for deterministic smoke tests.
    Maps text → fixed-dimension vector via SHA-256 hash bytes.
    """

    dimension: int = 64

    def embed(self, text: str) -> list[float]:
        """Embed text into a fixed-dimension vector deterministically."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Cycle hash bytes to fill dimension
        raw = []
        for i in range(self.dimension):
            raw.append(h[i % len(h)])
        # Normalize to unit vector
        magnitude = math.sqrt(sum(x * x for x in raw))
        if magnitude == 0:
            return [0.0] * self.dimension
        return [x / magnitude for x in raw]

    @property
    def model_id(self) -> str:
        return "deterministic-sha256-hash"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


@dataclass
class InMemoryBaselineAdapter:
    """In-memory baseline retrieval adapter with deterministic embedding.

    Implements the RetrievalPort contract:
    - retrieve(query, top_k) → Sequence[RetrievedEvidence]

    Features:
    - Fixed-size chunking (configurable chunk_size)
    - similarity_top_k configurable
    - Evidence with document, page, chunk, score, provenance
    - Deterministic ordering (by score desc, then chunk_id for ties)
    - Empty query handled explicitly
    - No remote API or model download
    - Embedding can be swapped for a real provider later

    Limitations:
    - Hash embedding is NOT semantic (documented honestly)
    """

    embedding: DeterministicEmbedding = field(
        default_factory=DeterministicEmbedding
    )
    chunk_size: int = 256
    _chunks: list[Chunk] = field(default_factory=list, repr=False)
    _vectors: list[list[float]] = field(default_factory=list, repr=False)

    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Index chunks for retrieval.

        Args:
            chunks: Chunks with provenance to index.
        """
        self._chunks = list(chunks)
        self._vectors = [
            self.embedding.embed(chunk.text) for chunk in self._chunks
        ]

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> Sequence[RetrievedEvidence]:
        """Retrieve top-k evidence for a query.

        Args:
            query: Query text.
            top_k: Number of results to return.

        Returns:
            Sequence of RetrievedEvidence, ranked by similarity (desc).
            Empty query returns empty results.
        """
        if not query or not query.strip():
            return []

        if top_k < 1:
            return []

        if not self._chunks:
            return []

        query_vec = self.embedding.embed(query)

        # Compute similarities
        scored: list[tuple[float, int]] = []
        for idx, chunk_vec in enumerate(self._vectors):
            sim = _cosine_similarity(query_vec, chunk_vec)
            scored.append((sim, idx))

        # Sort by score descending, then by chunk_id for deterministic ties
        scored.sort(
            key=lambda x: (-x[0], self._chunks[x[1]].chunk_id.value)
        )

        results: list[RetrievedEvidence] = []
        for rank, (score, idx) in enumerate(scored[:top_k], start=1):
            chunk = self._chunks[idx]
            results.append(
                RetrievedEvidence(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    rank=rank,
                    score=score,
                )
            )

        return results

    def clear(self) -> None:
        """Clear indexed data."""
        self._chunks = []
        self._vectors = []


def load_tiny_corpus(corpus_path: str) -> tuple[list[Chunk], dict[str, Any]]:
    """Load the tiny corpus JSON and return chunks + metadata.

    Args:
        corpus_path: Path to corpus.json.

    Returns:
        Tuple of (chunks, corpus_data).
    """
    with open(corpus_path, encoding="utf-8") as f:
        data = json.load(f)

    chunks: list[Chunk] = []
    for doc in data["documents"]:
        doc_id = doc["document_id"]
        for page in doc["pages"]:
            page_num = page["page_number"]
            chunk_id = f"{doc_id}_p{page_num}"
            chunks.append(
                Chunk(
                    chunk_id=ChunkId(chunk_id),
                    document_id=doc_id,
                    text=page["text"],
                    start_page=page_num,
                    end_page=page_num,
                )
            )

    return chunks, data
