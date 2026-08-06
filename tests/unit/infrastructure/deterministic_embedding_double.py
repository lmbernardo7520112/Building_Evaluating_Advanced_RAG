"""Deterministic embedding test double for unit tests.

Zero-IO, zero-network, zero-cache 384-dimensional dense vector double.
Uses sha256 with domain separation to generate stable, reproducible
dense vectors that preserve word-overlap similarity for unit tests.
"""

from __future__ import annotations

import hashlib
import string
from collections.abc import Sequence


class DeterministicTestEmbeddingAdapter:
    """Deterministic test double for FastEmbedEmbeddingAdapter."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dimension: int = 384,
    ) -> None:
        self._model_name_str = model_name
        self._embedding_dim = dimension
        self._cache: dict[str, list[float]] = {}

    @property
    def model_id(self) -> str:
        return self._model_name_str

    @property
    def dimension(self) -> int:
        return self._embedding_dim

    @property
    def cache_dir_path(self) -> str:
        return "/tmp/synthetic_test_cache"  # noqa: S108


    def _word_vector(self, word: str) -> list[float]:
        vec = [0.0] * self._embedding_dim
        for i in range(self._embedding_dim):
            seed = f"w:{word}:{i}"
            h_hex = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8]
            vec[i] = (int(h_hex, 16) / 0xFFFFFFFF) * 2.0 - 1.0
        return vec

    def _embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * self._embedding_dim

        cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        raw_words = text.lower().split()
        words = [w.strip(string.punctuation) for w in raw_words]
        words = [w for w in words if w]
        if not words:
            words = [text.lower().strip()]

        vec = [0.0] * self._embedding_dim
        for word in words:
            w_vec = self._word_vector(word)
            for i in range(self._embedding_dim):
                vec[i] += w_vec[i]


        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]

        self._cache[cache_key] = vec
        return vec

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [self._embed(t) for t in texts]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    def get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    def get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)
