"""Productive composition root for Slice 5A.2 deterministic pilot.

Connects the DeterministicPilot to the frozen retrieval pipelines
from the Slice 4 benchmark (build_retrievers) via RetrievalToolAdapter.

Requirements:
- RAGLAB_PDF_PATH environment variable set
- Embedding cache provisioned (.model_cache)
- No Gemini API key (generation = NOT_EXECUTED)
- No qrels during routing/retrieval

Does NOT:
- Alter retrieval strategies
- Use Settings global
- Call Gemini or any external API
- Import qrels during routing
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from raglab.domain.enums import PipelineStrategy  # noqa: E402

logger = logging.getLogger("slice5a_productive_pilot")

# Constants copied from benchmark (frozen)
CHUNK_SIZE = 512
WINDOW_SIZE = 3
TOP_K = 3
CANDIDATE_K = 10
MERGE_THRESHOLD = 0.5
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PDF_SHA256_EXPECTED = "33e2e9f1e190158b3e99c19fced1acd050720247c7556780bad82b2f93bf1254"
PAGES_START = 22
PAGES_END = 235


def sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_pdf_path() -> Path | None:
    """Resolve the corpus PDF path from env."""
    raw = os.environ.get("RAGLAB_PDF_PATH")
    if not raw:
        return None
    p = Path(raw)
    if not p.exists():
        return None
    return p


def verify_pdf(pdf_path: Path) -> bool:
    """Verify PDF SHA-256 matches expected."""
    actual = sha256_file(pdf_path)
    return actual == PDF_SHA256_EXPECTED


def load_pages(pdf_path: Path) -> list[Any]:
    """Load PDF pages using the benchmark loader."""
    from raglab.infrastructure.pdf_parsers.pdf_parser_adapter import (
        PyPdfExtractorAdapter,
    )

    adapter = PyPdfExtractorAdapter()
    pages: list[Any] = adapter.read_document(
        str(pdf_path),
        page_start=PAGES_START,
        page_end=PAGES_END,
    )
    return pages


def load_embedding_model() -> Any:
    """Load the frozen embedding model."""
    from raglab.infrastructure.embeddings.fastembed_adapter import (
        FastEmbedEmbeddingAdapter,
    )

    return FastEmbedEmbeddingAdapter(
        model_name=EMBEDDING_MODEL,
        local_files_only=True,
    )


def build_productive_retrievers(
    pages: list[Any],
    embed_model: Any,
) -> dict[PipelineStrategy, object]:
    """Build all 7 retrievers (frozen config).

    Reuses the exact same builder logic as
    benchmarks/run_slice4_benchmark.py.
    """
    from raglab.domain.entities import Chunk
    from raglab.domain.value_objects import ChunkId

    # Helpers (identical to benchmark)
    def _pages_to_chunks(
        pages_list: list[Any],
        chunk_size: int = CHUNK_SIZE,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in pages_list:
            doc_id = page.document_id
            page_num = page.page_number
            text = page.text
            for i in range(0, len(text), chunk_size):
                chunk_text = text[i : i + chunk_size]
                cid = f"{doc_id}_p{page_num}_c{i // chunk_size}"
                chunks.append(
                    Chunk(
                        chunk_id=ChunkId(cid),
                        document_id=doc_id,
                        text=chunk_text,
                        start_page=page_num,
                        end_page=page_num,
                    )
                )
        return chunks

    class _RerankedRetriever:
        """Reranker wrapper (same as benchmark)."""

        def __init__(
            self,
            base_retriever: Any,
            reranker: Any,
            candidate_k: int,
            top_n: int,
        ) -> None:
            self._base = base_retriever
            self._reranker = reranker
            self._candidate_k = candidate_k
            self._top_n = top_n

        def retrieve(
            self,
            query: str,
            top_k: int = 3,
        ) -> list[Any]:
            candidates = self._base.retrieve(query, top_k=self._candidate_k)
            reranked, _ = self._reranker.rerank(query, candidates, top_n=self._top_n)
            return list(reranked)

    def _build_f0() -> Any:
        from raglab.infrastructure.retrieval.baseline_adapter import (
            InMemoryBaselineAdapter,
        )

        class _EmbeddingShim:
            def __init__(self, adapter: Any) -> None:
                self._adapter = adapter

            def embed(self, text: str) -> list[float]:
                return list(
                    self._adapter._embed(text)  # noqa: SLF001
                )

            @property
            def model_id(self) -> str:
                mid: str = self._adapter.model_id
                return mid

        adapter = InMemoryBaselineAdapter(
            embedding=_EmbeddingShim(embed_model),
        )
        chunks = _pages_to_chunks(pages)
        adapter.index_chunks(chunks)
        return adapter

    def _build_s0() -> Any:
        from raglab.infrastructure.retrieval.sentence_anchor_adapter import (
            SentenceAnchorAdapter,
        )

        adapter = SentenceAnchorAdapter(embedding_adapter=embed_model)
        adapter.index_pages(pages)
        return adapter

    def _build_w0() -> Any:
        from raglab.infrastructure.retrieval.sentence_window_adapter import (
            SentenceWindowAdapter,
        )

        adapter = SentenceWindowAdapter(
            embedding_adapter=embed_model,
            window_size=WINDOW_SIZE,
        )
        adapter.index_pages(pages)
        return adapter

    def _build_w1() -> Any:
        from raglab.infrastructure.retrieval.reranker_adapter import (
            LocalRerankerAdapter,
        )
        from raglab.infrastructure.retrieval.sentence_window_adapter import (
            SentenceWindowAdapter,
        )

        base = SentenceWindowAdapter(
            embedding_adapter=embed_model,
            window_size=WINDOW_SIZE,
        )
        base.index_pages(pages)
        reranker = LocalRerankerAdapter(embedding_adapter=embed_model)
        return _RerankedRetriever(
            base_retriever=base,
            reranker=reranker,
            candidate_k=CANDIDATE_K,
            top_n=TOP_K,
        )

    def _build_h0() -> Any:
        from llama_index.core.embeddings import BaseEmbedding

        from raglab.infrastructure.retrieval.auto_merging_adapter import (
            HierarchicalRetrievalAdapter,
        )
        from raglab.infrastructure.retrieval.llamaindex_adapter import (
            LlamaIndexEmbeddingBridge,
        )

        bridge = (
            embed_model
            if (
                hasattr(embed_model, "_get_query_embedding")
                and isinstance(embed_model, BaseEmbedding)
            )
            else LlamaIndexEmbeddingBridge(embed_model)
        )
        adapter = HierarchicalRetrievalAdapter(
            embed_model=bridge,
            chunk_sizes=[1024, 512, 256],
            merge_threshold=MERGE_THRESHOLD,
            auto_merge=False,
            top_k=TOP_K,
        )
        adapter.index_pages(pages)
        return adapter

    def _build_h1() -> Any:
        from llama_index.core.embeddings import BaseEmbedding

        from raglab.infrastructure.retrieval.auto_merging_adapter import (
            HierarchicalRetrievalAdapter,
        )
        from raglab.infrastructure.retrieval.llamaindex_adapter import (
            LlamaIndexEmbeddingBridge,
        )

        bridge = (
            embed_model
            if (
                hasattr(embed_model, "_get_query_embedding")
                and isinstance(embed_model, BaseEmbedding)
            )
            else LlamaIndexEmbeddingBridge(embed_model)
        )
        adapter = HierarchicalRetrievalAdapter(
            embed_model=bridge,
            chunk_sizes=[1024, 512, 256],
            merge_threshold=MERGE_THRESHOLD,
            auto_merge=True,
            top_k=TOP_K,
        )
        adapter.index_pages(pages)
        return adapter

    def _build_h2() -> Any:
        from llama_index.core.embeddings import BaseEmbedding

        from raglab.infrastructure.retrieval.auto_merging_adapter import (
            HierarchicalRetrievalAdapter,
        )
        from raglab.infrastructure.retrieval.llamaindex_adapter import (
            LlamaIndexEmbeddingBridge,
        )
        from raglab.infrastructure.retrieval.reranker_adapter import (
            LocalRerankerAdapter,
        )

        bridge = (
            embed_model
            if (
                hasattr(embed_model, "_get_query_embedding")
                and isinstance(embed_model, BaseEmbedding)
            )
            else LlamaIndexEmbeddingBridge(embed_model)
        )
        base = HierarchicalRetrievalAdapter(
            embed_model=bridge,
            chunk_sizes=[1024, 512, 256],
            merge_threshold=MERGE_THRESHOLD,
            auto_merge=True,
            top_k=CANDIDATE_K,
        )
        base.index_pages(pages)
        reranker = LocalRerankerAdapter(embedding_adapter=embed_model)
        return _RerankedRetriever(
            base_retriever=base,
            reranker=reranker,
            candidate_k=CANDIDATE_K,
            top_n=TOP_K,
        )

    from raglab.domain.enums import PipelineStrategy

    builders: dict[PipelineStrategy, Any] = {
        PipelineStrategy.BASELINE: _build_f0,
        PipelineStrategy.SENTENCE_ANCHOR: _build_s0,
        PipelineStrategy.SENTENCE_WINDOW: _build_w0,
        PipelineStrategy.SENTENCE_WINDOW_RERANK: _build_w1,
        PipelineStrategy.HIERARCHICAL_LEAF: _build_h0,
        PipelineStrategy.AUTO_MERGING: _build_h1,
        PipelineStrategy.AUTO_MERGING_RERANK: _build_h2,
    }

    result: dict[PipelineStrategy, object] = {}
    for strategy, builder in builders.items():
        logger.info("Building retriever: %s", strategy.value)
        result[strategy] = builder()

    return result


def check_all_assets() -> dict[str, Any]:
    """Check all assets needed for productive pilot."""
    status: dict[str, Any] = {}

    # PDF
    pdf = resolve_pdf_path()
    status["pdf_path"] = str(pdf) if pdf else None
    status["pdf_exists"] = pdf is not None
    status["pdf_hash_ok"] = verify_pdf(pdf) if pdf else False

    # Embedding cache
    try:
        from raglab.infrastructure.embeddings.fastembed_adapter import (
            resolve_cache_dir,
        )

        cache = resolve_cache_dir()
        status["cache_dir"] = str(cache)
        status["cache_exists"] = cache.exists()
    except Exception as e:
        status["cache_dir"] = None
        status["cache_exists"] = False
        status["cache_error"] = str(e)

    # Provision manifest
    manifest_path = _REPO_ROOT / "benchmarks" / "provision_manifest.json"
    status["provision_manifest_exists"] = manifest_path.exists()

    # LlamaIndex
    try:
        import llama_index.core

        status["llamaindex_version"] = llama_index.core.__version__
    except ImportError:
        status["llamaindex_version"] = None

    # Overall
    status["all_ready"] = all(
        [
            status["pdf_exists"],
            status["pdf_hash_ok"],
            status["cache_exists"],
            status["provision_manifest_exists"],
            status.get("llamaindex_version") is not None,
        ]
    )

    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    status = check_all_assets()
    print(json.dumps(status, indent=2, default=str))
    if status["all_ready"]:
        print("\nPRODUCTIVE_PILOT_ASSETS_READY")
    else:
        print("\nPRODUCTIVE_PILOT_ASSETS_MISSING")
        missing = [
            k
            for k in (
                "pdf_exists",
                "pdf_hash_ok",
                "cache_exists",
                "provision_manifest_exists",
            )
            if not status.get(k)
        ]
        print(f"Missing: {missing}")
