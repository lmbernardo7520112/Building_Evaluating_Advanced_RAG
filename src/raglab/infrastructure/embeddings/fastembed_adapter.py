"""FastEmbed local embedding adapter for RAGLab v7.

Uses ONNX-accelerated CPU inference via FastEmbed with zero remote API calls.
Subclasses LlamaIndex BaseEmbedding and implements EmbeddingPort.

Cache policy (ADR-001):
- cache_dir must be passed explicitly in scientific mode
- local_files_only=True in scientific mode prevents implicit downloads
- Provisioning (network allowed) is a separate phase
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr

# Default persistent cache directory (relative to repo root, outside Git).
# Override via RAGLAB_MODEL_CACHE environment variable.
_DEFAULT_CACHE_DIR = ".model_cache"

# Transient directories that must NOT be used as authoritative cache.
# We reject /tmp itself and /tmp/fastembed_cache (FastEmbed's default)
# but NOT arbitrary /tmp/user-controlled-subdir paths (e.g. pytest tmp_path).
_REJECTED_CACHE_EXACT = frozenset({"/tmp"})
_REJECTED_CACHE_PREFIXES = frozenset({"/tmp/fastembed_cache"})


def resolve_cache_dir(explicit: str | None = None) -> Path:
    """Resolve the model cache directory.

    Priority:
    1. explicit parameter
    2. RAGLAB_MODEL_CACHE environment variable
    3. _DEFAULT_CACHE_DIR (relative to repo root)

    Raises ValueError if the resolved path is a known transient directory.
    """
    if explicit:
        resolved = Path(explicit).resolve()
    elif env_val := os.environ.get("RAGLAB_MODEL_CACHE"):
        resolved = Path(env_val).resolve()
    else:
        # Resolve relative to the repo root (4 levels up from this file)
        repo_root = Path(__file__).resolve().parents[4]
        resolved = (repo_root / _DEFAULT_CACHE_DIR).resolve()

    resolved_str = str(resolved)

    # Reject exact matches (e.g. /tmp)
    if resolved_str in _REJECTED_CACHE_EXACT:
        raise ValueError(
            f"Transient directory '{resolved}' cannot be used as authoritative "
            f"model cache. Set RAGLAB_MODEL_CACHE to a persistent path."
        )

    # Reject known transient prefixes (e.g. /tmp/fastembed_cache/...)
    for rejected_prefix in _REJECTED_CACHE_PREFIXES:
        if resolved_str == rejected_prefix or resolved_str.startswith(
            rejected_prefix + "/"
        ):
            raise ValueError(
                f"Transient directory '{resolved}' cannot be used as authoritative "
                f"model cache. Set RAGLAB_MODEL_CACHE to a persistent path."
            )

    return resolved


class FastEmbedEmbeddingAdapter(BaseEmbedding):
    """Local ONNX-accelerated embedding adapter for Portuguese & Multilingual text.

    Default model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    (Dimension: 384, CPU-optimized, MIT/Apache-2.0 license).
    Implements EmbeddingPort protocol via structural typing.

    Cache policy: cache_dir and local_files_only are explicit parameters.
    In scientific (offline) mode, local_files_only=True prevents downloads.
    """

    _model_name_str: str = PrivateAttr(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    _embedding_dim: int = PrivateAttr(default=384)
    _fastembed_instance: Any = PrivateAttr(default=None)
    _cache: dict[str, list[float]] = PrivateAttr(default_factory=dict)
    _cache_dir: str = PrivateAttr(default="")
    _local_files_only: bool = PrivateAttr(default=False)

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dimension: int = 384,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__(model_name=model_name)
        self._model_name_str = model_name
        self._embedding_dim = dimension
        self._local_files_only = local_files_only

        resolved = resolve_cache_dir(cache_dir)
        self._cache_dir = str(resolved)

        # Build kwargs for TextEmbedding
        te_kwargs: dict[str, Any] = {
            "model_name": model_name,
            "cache_dir": self._cache_dir,
        }

        # Set local_files_only via environment for fastembed's huggingface_hub
        if local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"

        self._fastembed_instance = TextEmbedding(**te_kwargs)
        self._cache = {}

    @property
    def model_id(self) -> str:
        return self._model_name_str

    @property
    def dimension(self) -> int:
        return self._embedding_dim

    @property
    def cache_dir_path(self) -> str:
        return self._cache_dir

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
