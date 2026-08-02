#!/usr/bin/env python3
"""RAGLab v7 Slice 4 Benchmark — RAG Generation + RAG Triad Evaluation.

EXECUTION ENVIRONMENT: Ambiente B (human terminal only).

SECURITY:
- GEMINI_API_KEY must be exported before running (never read by Antigravity)
- No key is logged, stored, or printed
- All artifacts are sanitized via sanitize_*_for_artifact()
- set +x must be active in the calling shell

TWO-PHASE EXECUTION:
- Phase A (no Gemini): provision-embedding → preflight
- Phase B (with Gemini): smoke → full

HOLDOUT: SEALED — q_holdout_01, q_holdout_02 are never in ACTIVE_QUESTIONS.

EVALUATION CONTRACT (schema=slice4_v2):
- Each metric has a typed status: COMPUTED, NOT_APPLICABLE, FAILED, NOT_EXECUTED
- score is non-null only when status=COMPUTED
- FAILED/NOT_EXECUTED block SMOKE_*_OK
- Abstention uses abstention_correctness (deterministic)
- Groundedness is NOT_APPLICABLE for ABSTAIN answers

USAGE:

  # Show help (no key, no PDF, no model loaded)
  .venv/bin/python benchmarks/run_slice4_benchmark.py --help

  # Preflight: validate embedding cache offline (no Gemini key needed)
  .venv/bin/python benchmarks/run_slice4_benchmark.py --mode preflight

  # Smoke positive: answerable question, must produce evaluated answer
  .venv/bin/python benchmarks/run_slice4_benchmark.py \\
      --mode smoke \\
      --smoke-strategy W0_sentence_window \\
      --smoke-question q_dev_01

  # Smoke abstention: unanswerable question, must produce correct ABSTAIN
  .venv/bin/python benchmarks/run_slice4_benchmark.py \\
      --mode smoke \\
      --smoke-strategy F0_baseline \\
      --smoke-question q_test_04

  # Full benchmark (requires explicit confirmation flag)
  .venv/bin/python benchmarks/run_slice4_benchmark.py \\
      --mode full \\
      --confirm-full-benchmark

  # Resume an interrupted run by RUN_ID
  .venv/bin/python benchmarks/run_slice4_benchmark.py \\
      --mode resume \\
      --run-id raglab_v7_slice4_v1_20260731T1230UTC
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

# ─── Path setup ───────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ─── Constants ───────────────────────────────────────────────────
PROTOCOL_VERSION = "raglab_v7_slice4_v2"
EXPERIMENT_ID = "raglab_v7_slice4_v2_20260731T1230UTC"
PDF_SHA256_EXPECTED = (
    "33e2e9f1e190158b3e99c19fced1acd050720247c7556780bad82b2f93bf1254"
)
PAGES_START = 91
PAGES_END = 115
CHUNK_SIZE = 512
WINDOW_SIZE = 3
TOP_K = 3
CANDIDATE_K = 10
MERGE_THRESHOLD = 0.5
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL = "gemini-3.1-flash-lite"

RESULTS_DIR = _REPO_ROOT / "benchmarks" / "results"
CHECKPOINT_DIR = _REPO_ROOT / "checkpoints"
PROVISION_MANIFEST_PATH = _REPO_ROOT / "benchmarks" / "provision_manifest.json"

_EVAL_SCHEMA_VERSION = "slice4_v2"

# ─── Evaluation metric status enum ───────────────────────────────
# Typed states for each metric — replaces bare null
METRIC_COMPUTED = "COMPUTED"
METRIC_NOT_APPLICABLE = "NOT_APPLICABLE"
METRIC_FAILED = "FAILED"
METRIC_NOT_EXECUTED = "NOT_EXECUTED"
_VALID_METRIC_STATUSES = frozenset(
    {METRIC_COMPUTED, METRIC_NOT_APPLICABLE, METRIC_FAILED, METRIC_NOT_EXECUTED}
)

_SECRET_PATTERNS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "AIza", "ya29.")

_NO_CREDENTIAL_VARS: Final[tuple[str, ...]] = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "LANGSMITH_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
)


def _validate_no_credentials_for_preflight(logger: logging.Logger) -> None:
    found = [var for var in _NO_CREDENTIAL_VARS if os.environ.get(var)]
    if found:
        logger.error(
            "PREFLIGHT_FAILED: Credential/token variable(s) present in environment during preflight: %s. "
            "Unset all credential/token variables before running preflight.",
            found,
        )
        sys.exit(1)

# Valid strategy labels — used for CLI validation
VALID_STRATEGIES: tuple[str, ...] = (
    "F0_baseline",
    "S0_sentence_anchor",
    "W0_sentence_window",
    "W1_sentence_window_rerank",
    "H0_hierarchical_leaf",
    "H1_auto_merging",
    "H2_auto_merging_rerank",
)

# ─── Active questions (holdout sealed) ───────────────────────────
# HOLDOUT: q_holdout_01, q_holdout_02 — NEVER ADD HERE
ACTIVE_QUESTIONS: list[dict] = [
    {
        "qid": "q_dev_01",
        "split": "development",
        "query": "O que é demonstração por exaustão e quando é aplicável?",
        "relevant_pages": [92],
    },
    {
        "qid": "q_dev_02",
        "split": "development",
        "query": "Como o método de prova por contradição funciona em matemática discreta?",
        "relevant_pages": [95],
    },
    {
        "qid": "q_dev_03",
        "split": "development",
        "query": "Quais são as etapas do princípio da indução matemática?",
        "relevant_pages": [97],
    },
    {
        "qid": "q_dev_04",
        "split": "development",
        "query": "Como funciona a indução forte comparada à indução fraca?",
        "relevant_pages": [101, 102],
    },
    {
        "qid": "q_test_01",
        "split": "test",
        "query": "Qual é a diferença entre indução fraca e indução forte?",
        "relevant_pages": [101, 102],
    },
    {
        "qid": "q_test_02",
        "split": "test",
        "query": (
            "Como se define a base e o passo indutivo em demonstração por indução?"
        ),
        "relevant_pages": [95],
    },
    {
        "qid": "q_test_03",
        "split": "test",
        "query": "Quais são os passos para provar uma afirmação usando indução completa?",
        "relevant_pages": [101, 102, 103],
    },
    {
        "qid": "q_test_04",
        "split": "test",
        "query": "Qual é a capital da França?",
        "relevant_pages": [],
        "abstention_expected": True,
    },
]

# Non-holdout question IDs valid for smoke test
_NON_HOLDOUT_QIDS: frozenset[str] = frozenset(
    q["qid"] for q in ACTIVE_QUESTIONS if "holdout" not in q["qid"]
)


# ─── Logging (deferred until after argparse) ─────────────────────

def _configure_logging(run_id: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    return logging.getLogger(f"slice4_benchmark.{run_id[:16]}")


# ─── CLI ─────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_slice4_benchmark.py",
        description=(
            "RAGLab v7 Slice 4 Benchmark — Ambiente B (human terminal only).\n\n"
            "SECURITY: GEMINI_API_KEY must be injected by the human operator.\n"
            "          This script never prints or logs the key value.\n\n"
            "TWO-PHASE EXECUTION:\n"
            "  Phase A (no Gemini): provision → preflight\n"
            "  Phase B (with Gemini): smoke → full\n\n"
            "Run --help without any credentials or PDF loaded.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Preflight (validates cache, no Gemini key needed)\n"
            "  .venv/bin/python benchmarks/run_slice4_benchmark.py --mode preflight\n\n"
            "  # Smoke test (mandatory first run)\n"
            "  .venv/bin/python benchmarks/run_slice4_benchmark.py \\\n"
            "      --mode smoke --smoke-strategy F0_baseline --smoke-question q_dev_01\n\n"
            "  # Full benchmark (requires explicit confirmation)\n"
            "  .venv/bin/python benchmarks/run_slice4_benchmark.py \\\n"
            "      --mode full --confirm-full-benchmark\n\n"
            "  # Resume interrupted run\n"
            "  .venv/bin/python benchmarks/run_slice4_benchmark.py \\\n"
            "      --mode resume --run-id raglab_v7_slice4_v1_20260731T1230UTC\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["preflight", "preflight-retrievers", "smoke", "full", "resume"],
        required=True,
        help=(
            "preflight: validate embedding cache offline (no Gemini key). "
            "preflight-retrievers: validate all 7 retriever builders structurally "
            "(no Gemini, no real corpus). "
            "smoke: 1 strategy x 1 question (mandatory before full). "
            "full: all 7 strategies x 8 questions (requires --confirm-full-benchmark). "
            "resume: continue an interrupted full run (requires --run-id)."
        ),
    )
    parser.add_argument(
        "--confirm-full-benchmark",
        action="store_true",
        default=False,
        help="Required flag to authorize the full 7x8 run. Absent → fail-closed.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID for --mode resume. Must match an existing checkpoint file.",
    )
    parser.add_argument(
        "--smoke-strategy",
        default="F0_baseline",
        choices=VALID_STRATEGIES,
        help="Strategy for smoke test (default: F0_baseline).",
    )
    parser.add_argument(
        "--smoke-question",
        default="q_dev_01",
        choices=sorted(_NON_HOLDOUT_QIDS),
        help="Question ID for smoke test (default: q_dev_01, must not be holdout).",
    )
    return parser


# ─── Helpers ─────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_credential(logger: logging.Logger) -> None:
    """Verify GEMINI_API_KEY is present. Never logs the value."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        logger.error(
            "GEMINI_API_KEY not found. "
            "Export it before running. "
            "See docs/runbooks/slice4_human_execution.md"
        )
        sys.exit(1)
    logger.info("GEMINI_API_KEY detected (value not logged)")


def verify_pdf(pdf_path: Path, logger: logging.Logger) -> None:
    if not pdf_path.exists():
        logger.error("PDF not found: %s", pdf_path)
        sys.exit(1)
    actual_sha = sha256_file(pdf_path)
    if actual_sha != PDF_SHA256_EXPECTED:
        logger.error(
            "PDF SHA-256 mismatch. Expected %s, got %s",
            PDF_SHA256_EXPECTED,
            actual_sha,
        )
        sys.exit(1)
    logger.info("PDF SHA-256 verified: %s", actual_sha)


def load_pdf_pages(pdf_path: Path, logger: logging.Logger) -> list:
    from raglab.infrastructure.pdf_parsers.pdf_parser_adapter import (
        PyPdfExtractorAdapter,
    )

    adapter = PyPdfExtractorAdapter()
    pages = adapter.read_document(
        str(pdf_path), page_start=PAGES_START, page_end=PAGES_END
    )
    logger.info(
        "Extracted %d pages (pages %d–%d) from PDF",
        len(pages),
        PAGES_START,
        PAGES_END,
    )
    return pages


def load_embedding_model(
    logger: logging.Logger, *, local_files_only: bool = True
) -> object:
    # CR-1 fix: canonical class name is FastEmbedEmbeddingAdapter
    from raglab.infrastructure.embeddings.fastembed_adapter import (
        FastEmbedEmbeddingAdapter,
        resolve_cache_dir,
    )

    cache_dir = resolve_cache_dir()
    logger.info("Loading FastEmbed model: %s (cache=%s, offline=%s)",
                EMBEDDING_MODEL, cache_dir, local_files_only)
    adapter = FastEmbedEmbeddingAdapter(
        model_name=EMBEDDING_MODEL,
        cache_dir=str(cache_dir),
        local_files_only=local_files_only,
    )
    # CR-2: .dimension property is the canonical name; .embedding_dim also works
    # but .dimension is what InMemoryBaselineAdapter uses internally.
    logger.info("Model loaded (dim=%d)", adapter.dimension)
    return adapter


def build_retrievers(
    pages: list,
    embed_model: object,
    strategies: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Build retrievers for requested strategies only (lazy per-strategy).

    Each builder imports only the modules it needs.  For ``strategies=("F0_baseline",)``
    the auto-merging or reranker modules are **never** imported.
    """
    from raglab.domain.entities import Chunk
    from raglab.domain.value_objects import ChunkId

    # ── helpers ────────────────────────────────────────────────────
    def _pages_to_chunks(pages_list: list, chunk_size: int = CHUNK_SIZE) -> list:
        """Convert DocumentPage list → Chunk list for InMemoryBaselineAdapter."""
        chunks: list[Chunk] = []
        for page in pages_list:
            doc_id = page.document_id
            page_num = page.page_number
            text = page.text
            # Fixed-size chunking
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
        """Wrapper that composes a base retriever + reranker into RetrievalPort."""

        def __init__(
            self, base_retriever: object, reranker: object,
            candidate_k: int, top_n: int,
        ) -> None:
            self._base = base_retriever
            self._reranker = reranker
            self._candidate_k = candidate_k
            self._top_n = top_n

        def retrieve(self, query: str, top_k: int = 3) -> list:
            candidates = self._base.retrieve(query, top_k=self._candidate_k)
            reranked, _ = self._reranker.rerank(query, candidates, top_n=self._top_n)
            return reranked

    # ── per-strategy builders ─────────────────────────────────────
    def _build_f0() -> object:
        from raglab.infrastructure.retrieval.baseline_adapter import (
            InMemoryBaselineAdapter,
        )

        # InMemoryBaselineAdapter uses its own DeterministicEmbedding by default;
        # for the benchmark we override with the real embed_model via
        # a minimal shim that delegates to the FastEmbed adapter.
        class _EmbeddingShim:
            """Adapts FastEmbedEmbeddingAdapter to the embed(text) interface."""
            def __init__(self, adapter: object) -> None:
                self._adapter = adapter
            def embed(self, text: str) -> list[float]:
                return list(self._adapter._embed(text))  # noqa: SLF001
            @property
            def model_id(self) -> str:
                return self._adapter.model_id

        adapter = InMemoryBaselineAdapter(embedding=_EmbeddingShim(embed_model))
        chunks = _pages_to_chunks(pages)
        adapter.index_chunks(chunks)
        return adapter

    def _build_s0() -> object:
        from raglab.infrastructure.retrieval.sentence_anchor_adapter import (
            SentenceAnchorAdapter,
        )
        adapter = SentenceAnchorAdapter(embedding_adapter=embed_model)
        adapter.index_pages(pages)
        return adapter

    def _build_w0() -> object:
        from raglab.infrastructure.retrieval.sentence_window_adapter import (
            SentenceWindowAdapter,
        )
        adapter = SentenceWindowAdapter(
            embedding_adapter=embed_model, window_size=WINDOW_SIZE,
        )
        adapter.index_pages(pages)
        return adapter

    def _build_w1() -> object:
        from raglab.infrastructure.retrieval.reranker_adapter import (
            LocalRerankerAdapter,
        )
        from raglab.infrastructure.retrieval.sentence_window_adapter import (
            SentenceWindowAdapter,
        )
        base = SentenceWindowAdapter(
            embedding_adapter=embed_model, window_size=WINDOW_SIZE,
        )
        base.index_pages(pages)
        reranker = LocalRerankerAdapter(embedding_adapter=embed_model)
        return _RerankedRetriever(
            base_retriever=base, reranker=reranker,
            candidate_k=CANDIDATE_K, top_n=TOP_K,
        )

    def _build_h0() -> object:
        from llama_index.core.embeddings import BaseEmbedding

        from raglab.infrastructure.retrieval.auto_merging_adapter import (
            HierarchicalRetrievalAdapter,
        )
        from raglab.infrastructure.retrieval.llamaindex_adapter import (
            LlamaIndexEmbeddingBridge,
        )

        bridge = (
            embed_model
            if hasattr(embed_model, "_get_query_embedding")
            and isinstance(embed_model, BaseEmbedding)
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

    def _build_h1() -> object:
        from llama_index.core.embeddings import BaseEmbedding

        from raglab.infrastructure.retrieval.auto_merging_adapter import (
            HierarchicalRetrievalAdapter,
        )
        from raglab.infrastructure.retrieval.llamaindex_adapter import (
            LlamaIndexEmbeddingBridge,
        )

        bridge = (
            embed_model
            if hasattr(embed_model, "_get_query_embedding")
            and isinstance(embed_model, BaseEmbedding)
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

    def _build_h2() -> object:
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
            if hasattr(embed_model, "_get_query_embedding")
            and isinstance(embed_model, BaseEmbedding)
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

    # ── registry ──────────────────────────────────────────────────
    builders: dict[str, object] = {
        "F0_baseline": _build_f0,
        "S0_sentence_anchor": _build_s0,
        "W0_sentence_window": _build_w0,
        "W1_sentence_window_rerank": _build_w1,
        "H0_hierarchical_leaf": _build_h0,
        "H1_auto_merging": _build_h1,
        "H2_auto_merging_rerank": _build_h2,
    }

    assert set(builders.keys()) == set(VALID_STRATEGIES), (
        f"Builder registry incomplete: {set(builders.keys())} != {set(VALID_STRATEGIES)}"
    )

    requested = list(strategies) if strategies is not None else list(VALID_STRATEGIES)

    # Validate: no unknowns, no duplicates
    unknown = set(requested) - set(builders.keys())
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}")
    if len(requested) != len(set(requested)):
        raise ValueError(f"Duplicate strategies: {requested}")

    # Build only requested
    all_retrievers: dict[str, object] = {}
    for label in requested:
        all_retrievers[label] = builders[label]()  # type: ignore[operator]

    return all_retrievers


# ─── Manifest Attestation ────────────────────────────────────────

def load_provision_manifest(
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Load the canonical provision manifest.

    Raises ValueError if manifest is missing, invalid, or incompatible.
    """
    path = manifest_path or PROVISION_MANIFEST_PATH
    if not path.exists():
        raise ValueError(
            f"EMBEDDING_ATTESTATION_FAILED: Manifest not found: {path}. "
            "Run: python scripts/provision_embedding_model.py --attest-existing"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"EMBEDDING_ATTESTATION_FAILED: Invalid manifest {path}: {exc}"
        ) from exc

    required_fields = (
        "model_id", "fastembed_version", "onnxruntime_version",
        "dimension", "pooling", "normalization", "cache_tree_sha256",
    )
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(
            f"EMBEDDING_ATTESTATION_FAILED: Missing fields {missing} in {path}"
        )

    cache_sha = data["cache_tree_sha256"]
    if not cache_sha or cache_sha == "UNRESOLVED" or len(cache_sha) != 64:
        raise ValueError(
            f"EMBEDDING_ATTESTATION_FAILED: Invalid cache_tree_sha256={cache_sha!r}"
        )

    if data.get("model_id") != EMBEDDING_MODEL:
        raise ValueError(
            f"EMBEDDING_ATTESTATION_FAILED: Model mismatch: "
            f"manifest={data.get('model_id')}, expected={EMBEDDING_MODEL}"
        )

    if not data.get("canary_dim_ok") or not data.get("canary_finite_ok"):
        raise ValueError(
            "EMBEDDING_ATTESTATION_FAILED: Canary validation not passed in manifest"
        )

    return data


# ─── Embedding Parity Verification ───────────────────────────────

def extract_underlying_embedding_adapter(retriever: object) -> object:
    """Unpack composite retrievers and bridges to obtain the root embedding adapter."""
    obj = retriever
    if hasattr(obj, "_base"):
        obj = obj._base
    if hasattr(obj, "embedding_adapter"):
        obj = obj.embedding_adapter
    elif hasattr(obj, "embedding"):
        obj = obj.embedding
    elif hasattr(obj, "embed_model"):
        obj = obj.embed_model

    while hasattr(obj, "_adapter") or hasattr(obj, "underlying_adapter"):
        if hasattr(obj, "underlying_adapter"):
            obj = obj.underlying_adapter
        elif hasattr(obj, "_adapter"):
            obj = obj._adapter

    return obj


def get_embedding_fingerprint(
    adapter_or_retriever: object,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract standard 7-field embedding fingerprint from a retriever or adapter.

    Uses the attested provision manifest as the canonical source for
    cache_tree_sha256, avoiding UNRESOLVED values.
    """
    root = extract_underlying_embedding_adapter(adapter_or_retriever)

    model_id = str(
        getattr(root, "model_id", getattr(root, "model_name", "unknown"))
    )
    dim = int(
        getattr(
            root,
            "dimension",
            getattr(root, "_dim", getattr(root, "_embedding_dim", 0)),
        )
    )
    pooling = str(getattr(root, "pooling", "mean"))
    norm = bool(getattr(root, "normalization", True))

    fastembed_ver = getattr(root, "fastembed_version", None)
    if not fastembed_ver:
        try:
            import fastembed
            fastembed_ver = fastembed.__version__
        except ImportError:
            fastembed_ver = "unavailable"

    onnxruntime_ver = getattr(root, "onnxruntime_version", None)
    if not onnxruntime_ver:
        try:
            import onnxruntime
            onnxruntime_ver = onnxruntime.__version__
        except ImportError:
            onnxruntime_ver = "unavailable"

    # Canonical source: provision manifest (attested offline)
    cache_sha = getattr(root, "cache_tree_sha256", None)
    if not cache_sha or cache_sha == "UNRESOLVED":
        if manifest:
            cache_sha = manifest.get("cache_tree_sha256", "UNRESOLVED")
        else:
            try:
                m_data = load_provision_manifest()
                cache_sha = m_data.get("cache_tree_sha256", "UNRESOLVED")
            except ValueError:
                cache_sha = "UNRESOLVED"

    return {
        "embedding_model_id": str(model_id),
        "fastembed_version": str(fastembed_ver),
        "onnxruntime_version": str(onnxruntime_ver),
        "pooling": str(pooling),
        "dimension": int(dim),
        "normalization": bool(norm),
        "cache_tree_sha256": str(cache_sha),
    }


def verify_embedding_parity(
    retrievers: dict[str, object],
    logger: logging.Logger | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Verify that all strategy retrievers share identical embedding fingerprints.

    Returns dict mapping strategy_label -> fingerprint dict.
    Raises ValueError if any strategy diverges.
    """
    fingerprints: dict[str, dict[str, Any]] = {}
    for label, retriever in retrievers.items():
        fingerprints[label] = get_embedding_fingerprint(retriever, manifest)

    if not fingerprints:
        raise ValueError("No retrievers provided for embedding parity check")

    labels = list(fingerprints.keys())
    ref_label = labels[0]
    ref_fp = fingerprints[ref_label]

    divergent = [lbl for lbl in labels[1:] if fingerprints[lbl] != ref_fp]

    if divergent:
        msg = (
            f"EMBEDDING_PARITY_FAILED: Strategy {divergent} diverges from {ref_label}. "
            f"Fingerprints: {json.dumps(fingerprints, indent=2)}"
        )
        if logger:
            logger.error(msg)
        print(f"EMBEDDING_PARITY_FAILED: {divergent}", file=sys.stderr)
        raise ValueError(msg)

    msg_ok = (
        f"EMBEDDING_PARITY_OK: All {len(fingerprints)} strategies share identical "
        f"embedding fingerprint: {ref_fp['embedding_model_id']} (dim={ref_fp['dimension']})"
    )
    if logger:
        logger.info(msg_ok)
    print("EMBEDDING_PARITY_OK")

    return fingerprints


# ─── Typed Evaluation Metrics ────────────────────────────────────

def make_metric_entry(
    name: str,
    status: str,
    score: float | None = None,
    reason: str = "",
    evaluator_model: str = "",
    attempts: int = 0,
) -> dict[str, Any]:
    """Create a typed metric entry with explicit status.

    Rules:
    - score must be in [0,1] when status=COMPUTED
    - score must be None for other statuses
    - status must be one of COMPUTED, NOT_APPLICABLE, FAILED, NOT_EXECUTED
    """
    if status not in _VALID_METRIC_STATUSES:
        raise ValueError(f"Invalid metric status: {status!r}")
    if status == METRIC_COMPUTED:
        if score is None:
            raise ValueError(f"Metric {name}: COMPUTED requires a score")
        if not math.isfinite(score) or not (0.0 <= score <= 1.0):
            raise ValueError(
                f"Metric {name}: score {score} out of [0,1] or non-finite"
            )
    elif score is not None:
        raise ValueError(
            f"Metric {name}: status={status} must have score=None, got {score}"
        )
    return {
        "name": name,
        "status": status,
        "score": score,
        "reason": reason,
        "evaluator_model": evaluator_model,
        "attempts": attempts,
    }


def compute_abstention_correctness(
    is_abstention_question: bool,
    abstained: bool,
) -> dict[str, Any]:
    """Deterministic abstention correctness metric.

    Definition:
      is_abstention_question=True  AND abstained=True  → 1.0
      is_abstention_question=False AND abstained=True  → 0.0
      is_abstention_question=True  AND abstained=False → 0.0
      is_abstention_question=False AND abstained=False → N/A (computed by Triad)
    """
    if not is_abstention_question and not abstained:
        return make_metric_entry(
            "abstention_correctness",
            METRIC_NOT_APPLICABLE,
            reason="SUBSTANTIVE_ANSWER_NOT_ABSTENTION_QUESTION",
        )
    score = 1.0 if is_abstention_question and abstained else 0.0
    return make_metric_entry(
        "abstention_correctness",
        METRIC_COMPUTED,
        score=score,
        reason=(
            "CORRECT_ABSTENTION" if score == 1.0
            else (
                "INCORRECT_ABSTENTION_ON_ANSWERABLE"
                if not is_abstention_question
                else "FAILED_TO_ABSTAIN_ON_UNANSWERABLE"
            )
        ),
        evaluator_model="deterministic",
        attempts=1,
    )


# ─── Retrieval Evidence Serialization ────────────────────────────

def _extract_page_from_chunk_id(chunk_id_val: str) -> int | None:
    """Extract page number from chunk_id like 'doc_p92_c0'. Returns None on failure."""
    try:
        parts = chunk_id_val.split("_p")
        if len(parts) >= 2:
            page_part = parts[-1].split("_")[0]
            return int(page_part)
    except (ValueError, IndexError):
        pass
    return None


def serialize_retrieval_evidence(
    evidence: list,
    relevant_pages: list[int],
    text_preview_limit: int = 80,
) -> dict[str, Any]:
    """Serialize retrieval evidence for auditable output.

    Produces per-candidate records with no secrets.
    Distinguishes score=None (absent) from score=0.
    """
    candidates: list[dict[str, Any]] = []
    pages_found: list[int] = []

    for i, ev in enumerate(evidence):
        chunk_id_val = (
            ev.chunk_id.value
            if hasattr(ev.chunk_id, "value")
            else str(ev.chunk_id)
        )
        page_num = _extract_page_from_chunk_id(chunk_id_val)
        if page_num and page_num in relevant_pages:
            pages_found.append(page_num)

        text_raw = ev.text if hasattr(ev, "text") else ""
        text_preview = text_raw[:text_preview_limit].replace("\n", " ")
        text_sha = hashlib.sha256(text_raw.encode("utf-8")).hexdigest()

        # Sanitize: no secrets in preview
        for pattern in _SECRET_PATTERNS:
            if pattern.lower() in text_preview.lower():
                text_preview = "[REDACTED]"
                break

        entry: dict[str, Any] = {
            "chunk_id": chunk_id_val,
            "page_number": page_num,
            "retrieval_rank": i + 1,
            "retrieval_score": ev.score if hasattr(ev, "score") else None,
            "rerank_rank": None,  # set by reranker if applicable
            "rerank_score": None,
            "parent_node_id": getattr(ev, "parent_node_id", None),
            "text_sha256": text_sha,
            "text_preview": text_preview,
        }
        candidates.append(entry)

    retrieval_hit = bool(pages_found)
    missing_pages = [p for p in relevant_pages if p not in pages_found]

    return {
        "candidate_count": len(candidates),
        "candidates": candidates,
        "relevant_pages_expected": relevant_pages,
        "relevant_pages_found": sorted(set(pages_found)),
        "relevant_pages_missing": missing_pages,
        "retrieval_hit": retrieval_hit,
    }


# ─── Core runner (shared by smoke and full) ───────────────────────

def run_benchmark(
    run_id: str,
    questions: list[dict],
    strategy_labels: tuple[str, ...],
    logger: logging.Logger,
    pdf_path: Path,
    manifest: dict[str, Any] | None = None,
) -> Path:
    """Execute generation + evaluation for requested questions × strategies.

    Evaluation contract (schema=slice4_v2):
    - Each metric has typed status: COMPUTED, NOT_APPLICABLE, FAILED, NOT_EXECUTED
    - Abstained answers get: context_relevance (COMPUTED if evidence exists),
      groundedness (NOT_APPLICABLE), abstention_correctness (COMPUTED)
    - Substantive answers get: context_relevance, groundedness, answer_relevance
    - evaluation is never null; it always has typed metrics
    """
    from raglab.domain.enums import PipelineStrategy
    from raglab.domain.quota import QuotaManager
    from raglab.domain.retry import RetryPolicy
    from raglab.infrastructure.gemini.gemini_generator_adapter import (
        GeminiGeneratorAdapter,
        sanitize_answer_for_artifact,
    )
    from raglab.infrastructure.gemini.gemini_judge_adapter import (
        GeminiJudgeAdapter,
    )
    from raglab.infrastructure.persistence.generation_checkpoint_store import (
        GenerationCheckpointStore,
    )

    # ── Load & validate manifest before any Gemini call ──────────
    if manifest is None:
        manifest = load_provision_manifest()
    logger.info(
        "Manifest validated: model=%s cache_sha=%s",
        manifest["model_id"],
        manifest["cache_tree_sha256"][:16] + "...",
    )

    pages = load_pdf_pages(pdf_path, logger)
    embed_model = load_embedding_model(logger)
    retrievers = build_retrievers(pages, embed_model, strategies=strategy_labels)
    embedding_fps = verify_embedding_parity(retrievers, logger, manifest)

    # ── Reject UNRESOLVED fingerprints ───────────────────────────
    for label, fp in embedding_fps.items():
        if fp.get("cache_tree_sha256") == "UNRESOLVED":
            raise ValueError(
                f"EMBEDDING_ATTESTATION_FAILED: {label} has UNRESOLVED "
                "cache_tree_sha256. Run: python scripts/provision_embedding_model.py "
                "--attest-existing"
            )

    shared_quota = QuotaManager()
    shared_retry = RetryPolicy()
    generator = GeminiGeneratorAdapter(
        model_id=GEMINI_MODEL,
        quota_manager=shared_quota,
        retry_policy=shared_retry,
        temperature=0.0,
    )
    logger.info("Generator initialized: %s", generator.model_id)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = GenerationCheckpointStore(run_id=run_id, store_dir=CHECKPOINT_DIR)
    logger.info("Checkpoint: %d entries already completed", ckpt.completed_count())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[dict]] = {}
    run_times: dict[str, float] = {}

    for strategy_label in strategy_labels:
        retriever = retrievers[strategy_label]
        pipeline_strategy = PipelineStrategy.from_label(strategy_label)
        logger.info("=== Strategy: %s ===", strategy_label)
        strategy_results: list[dict] = []
        t0 = time.monotonic()

        judge = GeminiJudgeAdapter(
            judge_model_id=GEMINI_MODEL,
            strategy=pipeline_strategy,
            quota_manager=shared_quota,
            retry_policy=shared_retry,
            temperature=0.0,
        )

        for q in questions:
            qid = q["qid"]
            query = q["query"]
            is_abstention = q.get("abstention_expected", False)
            relevant_pages = q.get("relevant_pages", [])
            query_id = f"{strategy_label}::{qid}"

            if ckpt.is_completed(qid, strategy_label):
                logger.info("  SKIP (already done): %s", query_id)
                continue

            # Defensive holdout guard — never run holdout questions
            if "holdout" in qid:
                logger.error(
                    "HOLDOUT GUARD: qid=%s rejected — holdout is SEALED", qid
                )
                sys.exit(2)

            logger.info("  Processing: %s (abstention=%s)", query_id, is_abstention)

            quota_before = int(shared_quota.stats["total_requests"])

            evidence = retriever.retrieve(query, top_k=TOP_K)

            # ── Serialize retrieval evidence for audit ───────────
            evidence_record = serialize_retrieval_evidence(
                evidence, relevant_pages,
            )

            answer = generator.generate(
                query_id=query_id,
                query=query,
                evidence=evidence,
            )
            gen_calls = 1
            cr_calls = 0
            gr_calls = 0
            ar_calls = 0

            sanitized_answer = sanitize_answer_for_artifact(answer)

            # ── Build typed evaluation with short-circuiting ──────
            evaluation_metrics: list[dict[str, Any]] = []

            # Abstention correctness (deterministic, always computed)
            evaluation_metrics.append(
                compute_abstention_correctness(is_abstention, answer.abstained)
            )

            if answer.abstained:
                # Abstained answer: short-circuit judge calls for GR and AR
                if evidence:
                    try:
                        cr_score = judge.evaluate_context_relevance(
                            query_id=query_id,
                            query=query,
                            evidence=evidence,
                        )
                        cr_calls = 1
                        evaluation_metrics.append(
                            make_metric_entry(
                                "context_relevance",
                                METRIC_COMPUTED,
                                score=cr_score,
                                evaluator_model=GEMINI_MODEL,
                                attempts=1,
                            )
                        )
                    except Exception as exc:
                        logger.warning(
                            "  CR evaluation failed for %s: %s",
                            query_id, exc,
                        )
                        evaluation_metrics.append(
                            make_metric_entry(
                                "context_relevance",
                                METRIC_FAILED,
                                reason=str(exc)[:200],
                                evaluator_model=GEMINI_MODEL,
                                attempts=1,
                            )
                        )
                else:
                    evaluation_metrics.append(
                        make_metric_entry(
                            "context_relevance",
                            METRIC_NOT_APPLICABLE,
                            reason="NO_RETRIEVAL_CONTEXT",
                        )
                    )

                # Groundedness: NOT_APPLICABLE for ABSTAIN (0 calls)
                evaluation_metrics.append(
                    make_metric_entry(
                        "groundedness",
                        METRIC_NOT_APPLICABLE,
                        reason="ABSTAINED",
                    )
                )

                # Answer Relevance: NOT_APPLICABLE for ABSTAIN (0 calls)
                evaluation_metrics.append(
                    make_metric_entry(
                        "answer_relevance",
                        METRIC_NOT_APPLICABLE,
                        reason="ABSTAINED",
                    )
                )
            else:
                # Substantive answer: evaluate dimensions
                if evidence:
                    # 1. Context Relevance
                    try:
                        cr_score = judge.evaluate_context_relevance(
                            query_id=query_id, query=query, evidence=evidence
                        )
                        cr_calls = 1
                        evaluation_metrics.append(
                            make_metric_entry(
                                "context_relevance",
                                METRIC_COMPUTED,
                                score=cr_score,
                                evaluator_model=GEMINI_MODEL,
                                attempts=1,
                            )
                        )
                    except Exception as exc:
                        logger.warning("  CR evaluation failed for %s: %s", query_id, exc)
                        evaluation_metrics.append(
                            make_metric_entry(
                                "context_relevance",
                                METRIC_FAILED,
                                reason=str(exc)[:200],
                                evaluator_model=GEMINI_MODEL,
                                attempts=1,
                            )
                        )

                    # 2. Groundedness
                    try:
                        gr_score = judge.evaluate_groundedness(
                            query_id=query_id, query=query, answer=answer, evidence=evidence
                        )
                        gr_calls = 1
                        evaluation_metrics.append(
                            make_metric_entry(
                                "groundedness",
                                METRIC_COMPUTED,
                                score=gr_score,
                                evaluator_model=GEMINI_MODEL,
                                attempts=1,
                            )
                        )
                    except Exception as exc:
                        logger.warning("  GR evaluation failed for %s: %s", query_id, exc)
                        evaluation_metrics.append(
                            make_metric_entry(
                                "groundedness",
                                METRIC_FAILED,
                                reason=str(exc)[:200],
                                evaluator_model=GEMINI_MODEL,
                                attempts=1,
                            )
                        )
                else:
                    evaluation_metrics.append(
                        make_metric_entry(
                            "context_relevance",
                            METRIC_NOT_APPLICABLE,
                            reason="NO_RETRIEVAL_CONTEXT",
                        )
                    )
                    evaluation_metrics.append(
                        make_metric_entry(
                            "groundedness",
                            METRIC_NOT_APPLICABLE,
                            reason="NO_RETRIEVAL_CONTEXT",
                        )
                    )

                # 3. Answer Relevance
                try:
                    ar_score = judge.evaluate_answer_relevance(
                        query_id=query_id, query=query, answer=answer
                    )
                    ar_calls = 1
                    evaluation_metrics.append(
                        make_metric_entry(
                            "answer_relevance",
                            METRIC_COMPUTED,
                            score=ar_score,
                            evaluator_model=GEMINI_MODEL,
                            attempts=1,
                        )
                    )
                except Exception as exc:
                    logger.warning("  AR evaluation failed for %s: %s", query_id, exc)
                    evaluation_metrics.append(
                        make_metric_entry(
                            "answer_relevance",
                            METRIC_FAILED,
                            reason=str(exc)[:200],
                            evaluator_model=GEMINI_MODEL,
                            attempts=1,
                        )
                    )

            # ── Call Accounting & Ledger ─────────────────────────
            total_ext_calls = gen_calls + cr_calls + gr_calls + ar_calls
            call_ledger = {
                "generation_calls": gen_calls,
                "context_relevance_calls": cr_calls,
                "groundedness_calls": gr_calls,
                "answer_relevance_calls": ar_calls,
                "total_external_requests": total_ext_calls,
            }

            quota_after = int(shared_quota.stats["total_requests"])
            quota_delta = quota_after - quota_before
            if quota_delta != total_ext_calls:
                raise ValueError(
                    f"EXTERNAL_CALL_ACCOUNTING_MISMATCH: query_id={query_id} "
                    f"ledger={total_ext_calls}, quota_delta={quota_delta}"
                )

            # ── Strategy Provenance Verification ────────────────
            query_id_prefix = query_id.split("::")[0]
            judge_strat = getattr(judge, "strategy", None)
            if (
                query_id_prefix != strategy_label
                or (judge_strat is not None and judge_strat != pipeline_strategy and judge_strat != strategy_label)
            ):
                raise ValueError(
                    f"STRATEGY_PROVENANCE_MISMATCH: requested={strategy_label}, "
                    f"judge={judge_strat}, query_id_prefix={query_id_prefix}"
                )

            # ── Citation pages used by generator ────────────────
            citation_pages = [c.page_number for c in answer.citations]

            evaluation_record = {
                "protocol_version": PROTOCOL_VERSION,
                "artifact_schema_version": _EVAL_SCHEMA_VERSION,
                "schema_version": _EVAL_SCHEMA_VERSION,
                "metrics": evaluation_metrics,
            }

            result_entry = {
                "qid": qid,
                "split": q["split"],
                "strategy": strategy_label,
                "relevant_pages": relevant_pages,
                "abstained": answer.abstained,
                "is_abstention_question": is_abstention,
                "answer": sanitized_answer,
                "evaluation": evaluation_record,
                "retrieval_evidence": evidence_record,
                "citation_pages": citation_pages,
                "call_ledger": call_ledger,
                "quota_stats": shared_quota.stats,
            }
            strategy_results.append(result_entry)

            ckpt.mark_completed(
                qid,
                strategy_label,
                abstained=answer.abstained,
                citation_count=len(answer.citations),
            )

        elapsed_ms = (time.monotonic() - t0) * 1000
        run_times[strategy_label] = round(elapsed_ms, 1)
        all_results[strategy_label] = strategy_results
        logger.info("  %s complete in %.1f ms", strategy_label, elapsed_ms)

    # Write sanitized results
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = RESULTS_DIR / f"slice4_results_{run_id}_{ts}.json"
    output = {
        "experiment_id": run_id,
        "protocol_version": PROTOCOL_VERSION,
        "artifact_schema_version": _EVAL_SCHEMA_VERSION,
        "schema": _EVAL_SCHEMA_VERSION,
        "gemini_model": GEMINI_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_fingerprints": embedding_fps,
        "manifest_fingerprint": manifest.get("cache_tree_sha256", "UNRESOLVED"),
        "reranker_class": "bi_encoder_rescoring",
        "run_time_ms": run_times,
        "quota_final_stats": shared_quota.stats,
        "rag_triad_dimensions": [
            "context_relevance",
            "groundedness",
            "answer_relevance",
        ],
        "holdout_status": "SEALED",
        "results": all_results,
    }

    # Final secret scan on serialized output
    output_json = json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False)
    for pattern in _SECRET_PATTERNS:
        if pattern in output_json:
            raise RuntimeError(
                f"SECRET DETECTED in output: pattern={pattern!r}. Aborting."
            )

    output_path.write_text(output_json, encoding="utf-8")
    logger.info("Results written to: %s", output_path)
    logger.info("Quota stats: %s", shared_quota.stats)
    return output_path


# ─── Smoke Validation ─────────────────────────────────────────────

def validate_smoke_result(
    data: dict[str, Any],
    strategy: str,
    qid: str,
    is_abstention_question: bool,
    logger: logging.Logger,
) -> str:
    """Validate smoke result and return SMOKE_POSITIVE_OK, SMOKE_ABSTENTION_OK, or SMOKE_FAILED.

    Fail-closed: any missing/invalid field returns SMOKE_FAILED.
    """
    failures: list[str] = []

    # 1. Secret scan
    serialized = json.dumps(data)
    for pattern in _SECRET_PATTERNS:
        if pattern in serialized:
            failures.append(f"SECRET_DETECTED: {pattern}")

    # 2. Fingerprint resolved
    fps = data.get("embedding_fingerprints", {})
    for label, fp in fps.items():
        sha = fp.get("cache_tree_sha256", "UNRESOLVED")
        if sha == "UNRESOLVED" or not sha or len(sha) != 64:
            failures.append(f"UNRESOLVED_FINGERPRINT: {label}")

    manifest_fp = data.get("manifest_fingerprint", "UNRESOLVED")
    if manifest_fp == "UNRESOLVED" or not manifest_fp:
        failures.append("UNRESOLVED_MANIFEST_FINGERPRINT")

    # 3. Results structure
    results = data.get("results", {})
    strategy_results = results.get(strategy, [])
    if not strategy_results:
        failures.append(f"NO_RESULTS_FOR_STRATEGY: {strategy}")
        return _emit_smoke_result(failures, is_abstention_question, logger)

    result = strategy_results[0]

    # 4. Holdout guard
    if "holdout" in result.get("qid", ""):
        failures.append("HOLDOUT_ACCESSED")

    # 5. Evaluation must not be null
    evaluation = result.get("evaluation")
    if evaluation is None:
        failures.append("EVALUATION_NULL")
        return _emit_smoke_result(failures, is_abstention_question, logger)

    metrics = evaluation.get("metrics", [])
    if not metrics:
        failures.append("EVALUATION_NO_METRICS")
        return _emit_smoke_result(failures, is_abstention_question, logger)

    # 6. Validate each metric
    metrics_by_name = {m["name"]: m for m in metrics}
    for m in metrics:
        status = m.get("status")
        score = m.get("score")
        name = m.get("name")

        if status == METRIC_FAILED:
            failures.append(f"METRIC_FAILED: {name}")
        elif status == METRIC_NOT_EXECUTED:
            failures.append(f"METRIC_NOT_EXECUTED: {name}")
        elif status == METRIC_COMPUTED:
            if score is None or not isinstance(score, (int, float)):
                failures.append(f"METRIC_NO_SCORE: {name}")
            elif not math.isfinite(score) or not (0.0 <= score <= 1.0):
                failures.append(f"METRIC_SCORE_INVALID: {name}={score}")

    abstained = result.get("abstained", False)

    if is_abstention_question:
        # ── Smoke de abstenção ───────────────────────────────
        if not abstained:
            failures.append("ABSTENTION_EXPECTED_BUT_ANSWERED")

        # abstention_correctness must be 1.0
        ac = metrics_by_name.get("abstention_correctness", {})
        if ac.get("status") != METRIC_COMPUTED or ac.get("score") != 1.0:
            failures.append(f"ABSTENTION_CORRECTNESS_NOT_1: {ac}")

        # groundedness must be NOT_APPLICABLE
        gr = metrics_by_name.get("groundedness", {})
        if gr.get("status") != METRIC_NOT_APPLICABLE:
            failures.append(f"GROUNDEDNESS_NOT_NA_FOR_ABSTAIN: {gr.get('status')}")

        # no invented citations
        citation_pages = result.get("citation_pages", [])
        if citation_pages:
            failures.append(f"CITATIONS_ON_ABSTAIN: {citation_pages}")

    else:
        # ── Smoke positivo ───────────────────────────────────
        if abstained:
            failures.append("ANSWERABLE_QUESTION_GOT_ABSTAIN")

        # All RAG Triad dimensions must be COMPUTED
        for dim in ("context_relevance", "groundedness", "answer_relevance"):
            dm = metrics_by_name.get(dim, {})
            if dm.get("status") != METRIC_COMPUTED:
                failures.append(f"TRIAD_NOT_COMPUTED: {dim}={dm.get('status')}")

        # Must have at least one citation
        citation_pages = result.get("citation_pages", [])
        if not citation_pages:
            failures.append("NO_CITATIONS_ON_SUBSTANTIVE")

        # Retrieval hit on relevant pages
        evidence_rec = result.get("retrieval_evidence", {})
        if not evidence_rec.get("retrieval_hit"):
            failures.append("NO_RETRIEVAL_HIT_ON_RELEVANT_PAGES")

        # Citations must be from retrieved evidence
        evidence_pages = {
            c.get("page_number")
            for c in evidence_rec.get("candidates", [])
            if c.get("page_number") is not None
        }
        for cp in citation_pages:
            if cp not in evidence_pages and cp != 0:
                failures.append(f"CITATION_NOT_IN_EVIDENCE: page={cp}")

    # 7. Retrieval evidence must exist
    if "retrieval_evidence" not in result:
        failures.append("MISSING_RETRIEVAL_EVIDENCE")

    # 8. Result and checkpoint valid
    if evaluation.get("schema_version") != _EVAL_SCHEMA_VERSION:
        failures.append(
            f"WRONG_EVAL_SCHEMA: {evaluation.get('schema_version')} != {_EVAL_SCHEMA_VERSION}"
        )

    return _emit_smoke_result(failures, is_abstention_question, logger)


def _emit_smoke_result(
    failures: list[str],
    is_abstention_question: bool,
    logger: logging.Logger,
) -> str:
    """Emit the final smoke verdict."""
    if failures:
        for f in failures:
            logger.error("  SMOKE FAILURE: %s", f)
        print("SMOKE_FAILED", file=sys.stderr)
        return "SMOKE_FAILED"

    if is_abstention_question:
        logger.info("SMOKE_ABSTENTION_OK")
        print("SMOKE_ABSTENTION_OK")
        return "SMOKE_ABSTENTION_OK"
    else:
        logger.info("SMOKE_POSITIVE_OK")
        print("SMOKE_POSITIVE_OK")
        return "SMOKE_POSITIVE_OK"


# ─── Mode handlers ────────────────────────────────────────────────

def cmd_smoke(args: argparse.Namespace, pdf_path: Path, logger: logging.Logger) -> None:
    strategy = args.smoke_strategy
    qid = args.smoke_question

    # Holdout guard at CLI level
    if "holdout" in qid:
        logger.error("Smoke test refuses holdout question: %s", qid)
        sys.exit(2)
    if strategy not in VALID_STRATEGIES:
        logger.error("Invalid strategy: %s. Valid: %s", strategy, VALID_STRATEGIES)
        sys.exit(2)

    question_map = {q["qid"]: q for q in ACTIVE_QUESTIONS}
    if qid not in question_map:
        logger.error("Question ID not found: %s", qid)
        sys.exit(2)

    question = question_map[qid]
    is_abstention = question.get("abstention_expected", False)

    smoke_run_id = f"smoke_{EXPERIMENT_ID}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    logger.info("=== SMOKE TEST: strategy=%s question=%s run_id=%s ===",
                strategy, qid, smoke_run_id)

    output_path = run_benchmark(
        run_id=smoke_run_id,
        questions=[question],
        strategy_labels=(strategy,),
        logger=logger,
        pdf_path=pdf_path,
    )

    # Post-smoke validation (fail-closed)
    data = json.loads(output_path.read_text(encoding="utf-8"))
    verdict = validate_smoke_result(data, strategy, qid, is_abstention, logger)
    if verdict == "SMOKE_FAILED":
        logger.error(
            "SMOKE_FAILED: result at %s did not pass validation. "
            "Fix issues before proceeding.",
            output_path,
        )
        sys.exit(4)


def cmd_full(args: argparse.Namespace, pdf_path: Path, logger: logging.Logger) -> None:
    if not args.confirm_full_benchmark:
        logger.error(
            "Full benchmark requires --confirm-full-benchmark flag. "
            "Failing closed to prevent accidental execution."
        )
        sys.exit(3)

    logger.info("=== FULL BENCHMARK AUTHORIZED === run_id=%s", EXPERIMENT_ID)
    output_path = run_benchmark(
        run_id=EXPERIMENT_ID,
        questions=ACTIVE_QUESTIONS,
        strategy_labels=VALID_STRATEGIES,
        logger=logger,
        pdf_path=pdf_path,
    )
    logger.info("=== Slice 4 Full Benchmark Complete: %s ===", output_path)


def cmd_resume(args: argparse.Namespace, pdf_path: Path, logger: logging.Logger) -> None:
    run_id = args.run_id
    if not run_id:
        logger.error("--mode resume requires --run-id. Example: --run-id %s", EXPERIMENT_ID)
        sys.exit(3)

    # Validate checkpoint exists
    ckpt_files = list(CHECKPOINT_DIR.glob(f"*{run_id}*.json"))
    if not ckpt_files:
        logger.error(
            "No checkpoint found for run_id=%s in %s. "
            "Available: %s",
            run_id,
            CHECKPOINT_DIR,
            list(CHECKPOINT_DIR.glob("*.json")),
        )
        sys.exit(3)

    logger.info("=== RESUME: run_id=%s (checkpoint: %s) ===", run_id, ckpt_files[0])
    output_path = run_benchmark(
        run_id=run_id,
        questions=ACTIVE_QUESTIONS,
        strategy_labels=VALID_STRATEGIES,
        logger=logger,
        pdf_path=pdf_path,
    )
    logger.info("=== Slice 4 Resume Complete: %s ===", output_path)


# ─── Preflight: validate embedding cache offline ─────────────────

def cmd_preflight(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Validate embedding cache offline — no Gemini key needed."""
    logger.info("=== PREFLIGHT: Embedding Cache Validation ===")

    _validate_no_credentials_for_preflight(logger)

    # Resolve and validate cache
    from raglab.infrastructure.embeddings.fastembed_adapter import resolve_cache_dir

    try:
        cache_dir = resolve_cache_dir()
    except ValueError as exc:
        logger.error("PREFLIGHT_FAILED: %s", exc)
        sys.exit(1)

    if not cache_dir.exists():
        logger.error(
            "PREFLIGHT_FAILED: Cache directory does not exist: %s. "
            "Run provisioning first: .venv/bin/python scripts/provision_embedding_model.py",
            cache_dir,
        )
        sys.exit(1)

    # PDF verification (optional in preflight but validates corpus integrity)
    pdf_path_str = os.environ.get("RAGLAB_PDF_PATH")
    if pdf_path_str:
        pdf_path = Path(pdf_path_str)
        verify_pdf(pdf_path, logger)
    else:
        logger.info("PREFLIGHT: RAGLAB_PDF_PATH not set — skipping PDF validation")

    # Load embedding model offline
    logger.info("PREFLIGHT: Loading embedding model from cache (local_files_only=True)")
    try:
        embed_model = load_embedding_model(logger, local_files_only=True)
    except Exception as exc:
        logger.error(
            "PREFLIGHT_FAILED: Could not load embedding from cache: %s. "
            "Run provisioning first: .venv/bin/python scripts/provision_embedding_model.py",
            exc,
        )
        sys.exit(1)

    # Canary embedding
    canary_text = "Este é um texto canário para validação do embedding."
    logger.info("PREFLIGHT: Generating canary embedding...")
    try:
        from raglab.infrastructure.embeddings.fastembed_adapter import (
            FastEmbedEmbeddingAdapter,
        )
        assert isinstance(embed_model, FastEmbedEmbeddingAdapter)
        canary_vec = embed_model._embed(canary_text)  # noqa: SLF001
    except Exception as exc:
        logger.error("PREFLIGHT_FAILED: Canary embedding failed: %s", exc)
        sys.exit(1)

    # Validate dimension
    if len(canary_vec) != 384:
        logger.error(
            "PREFLIGHT_FAILED: Dimension mismatch: expected 384, got %d",
            len(canary_vec),
        )
        sys.exit(1)

    # Validate finite
    if not all(math.isfinite(v) for v in canary_vec):
        logger.error("PREFLIGHT_FAILED: Canary contains non-finite values")
        sys.exit(1)

    logger.info("PREFLIGHT: Canary OK (dim=%d, all finite)", len(canary_vec))
    logger.info("PREFLIGHT: model_id=%s", EMBEDDING_MODEL)
    logger.info("PREFLIGHT: cache_dir=%s", cache_dir)
    logger.info("")
    logger.info("EMBEDDING_OFFLINE_READY")


# ─── Preflight: structural validation of all 7 retriever builders ─

def cmd_preflight_retrievers(
    args: argparse.Namespace, logger: logging.Logger,
) -> None:
    """Validate all 7 retriever builders structurally — no Gemini, no real corpus.

    Uses fake pages and a deterministic embedding to prove every builder
    imports, constructs, indexes, and retrieves without error.
    """
    logger.info("=== PREFLIGHT-RETRIEVERS: Structural validation of all 7 builders ===")

    _validate_no_credentials_for_preflight(logger)

    from raglab.domain.value_objects import DocumentPage
    from raglab.infrastructure.retrieval.baseline_adapter import DeterministicEmbedding

    # Fake embedding that implements the FastEmbedEmbeddingAdapter interface subset
    class _FakeEmbeddingAdapter:
        """Minimal fake that satisfies embed_texts / _embed / _get_query_embedding."""
        def __init__(self, dim: int = 64) -> None:
            self._det = DeterministicEmbedding(dimension=dim)
            self.dimension = dim
            self.model_id = "fake-deterministic"

        def _embed(self, text: str) -> list[float]:
            return self._det.embed(text)

        def _get_query_embedding(self, text: str) -> list[float]:
            return self._det.embed(text)

        def _get_text_embedding(self, text: str) -> list[float]:
            return self._det.embed(text)

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [self._det.embed(t) for t in texts]

    fake_embed = _FakeEmbeddingAdapter()
    fake_pages = [
        DocumentPage(
            document_id="fake_doc",
            page_number=1,
            text="Este é um parágrafo fake para validação estrutural dos retrievers. " * 20,
        ),
        DocumentPage(
            document_id="fake_doc",
            page_number=2,
            text="Segundo parágrafo fake com conteúdo diferente para testar indexação. " * 20,
        ),
    ]

    errors: list[str] = []
    passed: list[str] = []

    for label in VALID_STRATEGIES:
        logger.info("  Building: %s", label)
        try:
            retrievers = build_retrievers(
                pages=fake_pages,
                embed_model=fake_embed,
                strategies=(label,),
            )
            assert label in retrievers, f"{label} not in result dict"
            r = retrievers[label]

            # Verify retrieve method exists
            assert hasattr(r, "retrieve"), f"{label} has no retrieve() method"

            # Call retrieve with fake query
            results = r.retrieve("teste de validação estrutural", top_k=2)
            assert isinstance(results, (list, tuple)), (
                f"{label} retrieve returned {type(results)}, not list"
            )

            logger.info("    %s: OK (%d results)", label, len(results))
            passed.append(label)

        except Exception as exc:
            logger.error("    %s: FAILED — %s", label, exc)
            errors.append(f"{label}: {exc}")

    logger.info("")
    logger.info("PREFLIGHT-RETRIEVERS: %d/7 passed, %d/7 failed", len(passed), len(errors))

    if errors:
        for e in errors:
            logger.error("  FAIL: %s", e)
        sys.exit(1)

    # Verify parity across all 7 builders together
    try:
        all_built = build_retrievers(
            pages=fake_pages,
            embed_model=fake_embed,
            strategies=VALID_STRATEGIES,
        )
        verify_embedding_parity(all_built, logger)
    except Exception as exc:
        logger.error("PREFLIGHT-RETRIEVERS: Embedding parity check failed — %s", exc)
        sys.exit(1)

    logger.info("PREFLIGHT-RETRIEVERS: ALL 7 BUILDERS & PARITY VALIDATED")
    logger.info("RETRIEVER_BUILDERS_OK")


# ─── Entry point ─────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # --help is handled by argparse before reaching here.
    # --mode is required; argparse enforces this, so execution always has a mode.

    run_id_for_log = args.run_id or args.mode
    logger = _configure_logging(run_id_for_log)

    logger.info("=== RAGLab v7 Slice 4 Benchmark — mode=%s ===", args.mode)
    logger.info("Embedding model: %s", EMBEDDING_MODEL)

    # ── Preflight mode: no Gemini key needed ──────────────────────
    if args.mode == "preflight":
        cmd_preflight(args, logger)
        return

    # ── Preflight-retrievers: no Gemini, no real corpus ──────────
    if args.mode == "preflight-retrievers":
        cmd_preflight_retrievers(args, logger)
        return

    # ── For smoke/full/resume: validate embedding cache BEFORE reading key ──
    logger.info("Gemini model: %s", GEMINI_MODEL)

    from raglab.infrastructure.embeddings.fastembed_adapter import resolve_cache_dir

    try:
        cache_dir = resolve_cache_dir()
    except ValueError as exc:
        logger.error(
            "Embedding cache not configured: %s. "
            "Run provisioning first.",
            exc,
        )
        sys.exit(1)

    if not cache_dir.exists():
        logger.error(
            "Embedding cache missing: %s. "
            "Run: .venv/bin/python scripts/provision_embedding_model.py --execute "
            "then: .venv/bin/python benchmarks/run_slice4_benchmark.py --mode preflight",
            cache_dir,
        )
        sys.exit(1)

    # Security check: GEMINI_API_KEY must be present for execution modes
    check_credential(logger)

    # PDF verification
    pdf_path_str = os.environ.get("RAGLAB_PDF_PATH")
    if not pdf_path_str:
        logger.error("RAGLAB_PDF_PATH not set")
        sys.exit(1)
    pdf_path = Path(pdf_path_str)
    verify_pdf(pdf_path, logger)

    # Dispatch
    if args.mode == "smoke":
        cmd_smoke(args, pdf_path, logger)
    elif args.mode == "full":
        cmd_full(args, pdf_path, logger)
    elif args.mode == "resume":
        cmd_resume(args, pdf_path, logger)
    else:
        # Unreachable (argparse choices enforced), but defensive
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
