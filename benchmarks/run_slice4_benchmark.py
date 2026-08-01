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

USAGE:

  # Show help (no key, no PDF, no model loaded)
  .venv/bin/python benchmarks/run_slice4_benchmark.py --help

  # Preflight: validate embedding cache offline (no Gemini key needed)
  .venv/bin/python benchmarks/run_slice4_benchmark.py --mode preflight

  # Smoke test: 1 strategy x 1 question — mandatory before full run
  .venv/bin/python benchmarks/run_slice4_benchmark.py \\
      --mode smoke \\
      --smoke-strategy F0_baseline \\
      --smoke-question q_dev_01

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

# ─── Path setup ───────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ─── Constants ───────────────────────────────────────────────────
EXPERIMENT_ID = "raglab_v7_slice4_v1_20260731T1230UTC"
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
        choices=["preflight", "smoke", "full", "resume"],
        required=True,
        help=(
            "preflight: validate embedding cache offline (no Gemini key). "
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
    # but .dimension is what BaselineRetrieverAdapter uses internally.
    logger.info("Model loaded (dim=%d)", adapter.dimension)
    return adapter


def build_retrievers(
    pages: list,
    embed_model: object,
    strategies: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Build retrievers for requested strategies only."""
    from raglab.infrastructure.retrieval.auto_merging_adapter import AutoMergingAdapter
    from raglab.infrastructure.retrieval.baseline_adapter import (
        BaselineRetrieverAdapter,
    )
    from raglab.infrastructure.retrieval.reranker_adapter import LocalRerankerAdapter
    from raglab.infrastructure.retrieval.sentence_anchor_adapter import (
        SentenceAnchorAdapter,
    )
    from raglab.infrastructure.retrieval.sentence_window_adapter import (
        SentenceWindowAdapter,
    )

    all_retrievers: dict[str, object] = {}

    def _want(label: str) -> bool:
        return strategies is None or label in strategies

    if _want("F0_baseline"):
        all_retrievers["F0_baseline"] = BaselineRetrieverAdapter(
            pages=pages,
            embed_model=embed_model,
            chunk_size=CHUNK_SIZE,
            top_k=TOP_K,
        )
    if _want("S0_sentence_anchor"):
        all_retrievers["S0_sentence_anchor"] = SentenceAnchorAdapter(
            pages=pages, embed_model=embed_model, top_k=TOP_K
        )
    if _want("W0_sentence_window"):
        all_retrievers["W0_sentence_window"] = SentenceWindowAdapter(
            pages=pages,
            embed_model=embed_model,
            window_size=WINDOW_SIZE,
            top_k=TOP_K,
        )
    if _want("W1_sentence_window_rerank"):
        w1_base = SentenceWindowAdapter(
            pages=pages,
            embed_model=embed_model,
            window_size=WINDOW_SIZE,
            top_k=CANDIDATE_K,
        )
        all_retrievers["W1_sentence_window_rerank"] = LocalRerankerAdapter(
            base_retriever=w1_base, embed_model=embed_model, top_k=TOP_K
        )
    if _want("H0_hierarchical_leaf"):
        all_retrievers["H0_hierarchical_leaf"] = AutoMergingAdapter(
            pages=pages,
            embed_model=embed_model,
            top_k=TOP_K,
            merge_threshold=MERGE_THRESHOLD,
            enable_auto_merging=False,
            enable_reranking=False,
        )
    if _want("H1_auto_merging"):
        all_retrievers["H1_auto_merging"] = AutoMergingAdapter(
            pages=pages,
            embed_model=embed_model,
            top_k=TOP_K,
            merge_threshold=MERGE_THRESHOLD,
            enable_auto_merging=True,
            enable_reranking=False,
        )
    if _want("H2_auto_merging_rerank"):
        all_retrievers["H2_auto_merging_rerank"] = AutoMergingAdapter(
            pages=pages,
            embed_model=embed_model,
            top_k=TOP_K,
            merge_threshold=MERGE_THRESHOLD,
            enable_auto_merging=True,
            enable_reranking=True,
        )

    return all_retrievers


# ─── Core runner (shared by smoke and full) ───────────────────────

def run_benchmark(
    run_id: str,
    questions: list[dict],
    strategy_labels: tuple[str, ...],
    logger: logging.Logger,
    pdf_path: Path,
) -> Path:
    """Execute generation + evaluation for requested questions × strategies."""
    from raglab.domain.enums import PipelineStrategy
    from raglab.domain.quota import QuotaManager
    from raglab.domain.retry import RetryPolicy
    from raglab.infrastructure.gemini.gemini_generator_adapter import (
        GeminiGeneratorAdapter,
        sanitize_answer_for_artifact,
    )
    from raglab.infrastructure.gemini.gemini_judge_adapter import (
        GeminiJudgeAdapter,
        sanitize_evaluation_for_artifact,
    )
    from raglab.infrastructure.persistence.generation_checkpoint_store import (
        GenerationCheckpointStore,
    )

    pages = load_pdf_pages(pdf_path, logger)
    embed_model = load_embedding_model(logger)
    retrievers = build_retrievers(pages, embed_model, strategies=strategy_labels)

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
        logger.info("=== Strategy: %s ===", strategy_label)
        strategy_results: list[dict] = []
        t0 = time.monotonic()

        judge = GeminiJudgeAdapter(
            judge_model_id=GEMINI_MODEL,
            strategy=PipelineStrategy.BASELINE,
            quota_manager=shared_quota,
            retry_policy=shared_retry,
            temperature=0.0,
        )

        for q in questions:
            qid = q["qid"]
            query = q["query"]
            is_abstention = q.get("abstention_expected", False)

            if ckpt.is_completed(qid, strategy_label):
                logger.info("  SKIP (already done): %s::%s", qid, strategy_label)
                continue

            # Defensive holdout guard — never run holdout questions
            if "holdout" in qid:
                logger.error(
                    "HOLDOUT GUARD: qid=%s rejected — holdout is SEALED", qid
                )
                sys.exit(2)

            logger.info("  Processing: %s (abstention=%s)", qid, is_abstention)
            evidence = retriever.retrieve(query)

            answer = generator.generate(
                query_id=f"{strategy_label}::{qid}",
                query=query,
                evidence=evidence,
            )

            eval_result = None
            if not answer.abstained and not is_abstention and evidence:
                eval_result = judge.evaluate(
                    query_id=f"{strategy_label}::{qid}",
                    query=query,
                    answer=answer,
                    evidence=evidence,
                )

            sanitized_answer = sanitize_answer_for_artifact(answer)
            sanitized_eval = (
                sanitize_evaluation_for_artifact(eval_result) if eval_result else None
            )

            result_entry = {
                "qid": qid,
                "split": q["split"],
                "strategy": strategy_label,
                "relevant_pages": q.get("relevant_pages", []),
                "abstained": answer.abstained,
                "is_abstention_question": is_abstention,
                "answer": sanitized_answer,
                "evaluation": sanitized_eval,
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
        "schema": "slice4_v1",
        "gemini_model": GEMINI_MODEL,
        "embedding_model": EMBEDDING_MODEL,
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
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Results written to: %s", output_path)
    logger.info("Quota stats: %s", shared_quota.stats)
    return output_path


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

    smoke_run_id = f"smoke_{EXPERIMENT_ID}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    logger.info("=== SMOKE TEST: strategy=%s question=%s run_id=%s ===",
                strategy, qid, smoke_run_id)

    output_path = run_benchmark(
        run_id=smoke_run_id,
        questions=[question_map[qid]],
        strategy_labels=(strategy,),
        logger=logger,
        pdf_path=pdf_path,
    )

    # Post-smoke validation
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert "results" in data, "SMOKE: missing 'results' key in output"
    assert "GEMINI_API_KEY" not in json.dumps(data), "SMOKE: CREDENTIAL LEAKED"
    logger.info("SMOKE_OK: sanitized result validated — proceed to full run if correct.")


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

    # Must NOT require Gemini key
    for key_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(key_name):
            logger.warning(
                "PREFLIGHT: %s is set but will NOT be used. "
                "Preflight validates embedding only.",
                key_name,
            )

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
            "Run: .venv/bin/python scripts/provision_embedding_model.py "
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
