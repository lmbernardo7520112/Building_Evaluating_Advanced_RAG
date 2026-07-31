"""FastEmbed local embedding adapter for RAGLab v7.

Uses ONNX-accelerated CPU inference via FastEmbed with zero remote API calls.
Subclasses LlamaIndex BaseEmbedding and implements EmbeddingPort.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from fastembed import TextEmbedding
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr


class FastEmbedEmbeddingAdapter(BaseEmbedding):
    """Local ONNX-accelerated embedding adapter for Portuguese & Multilingual text.

    Default model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    (Dimension: 384, CPU-optimized, MIT/Apache-2.0 license).
    Implements EmbeddingPort protocol via structural typing.
    """

    _model_name_str: str = PrivateAttr(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    _embedding_dim: int = PrivateAttr(default=384)
    _fastembed_instance: Any = PrivateAttr(default=None)
    _cache: dict[str, list[float]] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dimension: int = 384,
    ) -> None:
        super().__init__(model_name=model_name)
        self._model_name_str = model_name
        self._embedding_dim = dimension
        self._fastembed_instance = TextEmbedding(model_name=model_name)
        self._cache = {}

    @property
    def model_id(self) -> str:
        return self._model_name_str

    @property
    def dimension(self) -> int:
        return self._embedding_dim

    def _embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * self._embedding_dim

        cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        embeddings_gen = self._fastembed_instance.embed([text])
        vec = list(next(embeddings_gen))
        # Ensure list float representation
        float_vec = [float(x) for x in vec]
        self._cache[cache_key] = float_vec
        return float_vec

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        results: list[list[float]] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        for idx, text in enumerate(texts):
            if not text or not text.strip():
                results.append([0.0] * self._embedding_dim)
                continue
            cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if cache_key in self._cache:
                results.append(self._cache[cache_key])
            else:
                results.append([])  # placeholder
                uncached_texts.append(text)
                uncached_indices.append(idx)

        if uncached_texts:
            embeddings_gen = self._fastembed_instance.embed(uncached_texts)
            for sub_idx, vec_arr in enumerate(embeddings_gen):
                orig_idx = uncached_indices[sub_idx]
                text = uncached_texts[sub_idx]
                cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
                float_vec = [float(x) for x in vec_arr]
                self._cache[cache_key] = float_vec
                results[orig_idx] = float_vec

        return results

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)
