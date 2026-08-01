#!/usr/bin/env python3
"""RAGLab v7 Slice 4 Benchmark — RAG Generation + RAG Triad Evaluation.

EXECUTION ENVIRONMENT: Ambiente B (human terminal only).

This script:
1. Loads the PDF corpus (same 25 pages as Slice 3)
2. Builds retrievers for each strategy (F0, S0, W0, W1, H0, H1, H2)
3. Generates answers using GeminiGeneratorAdapter
4. Evaluates answers using GeminiJudgeAdapter (RAG Triad + optional factual)
5. Checkpoints every (query_id, strategy) pair atomically
6. Writes sanitized results to benchmarks/results/

SECURITY:
- GEMINI_API_KEY must be exported before running
- No key is logged, stored, or printed
- All artifacts are sanitized via sanitize_*_for_artifact()

PRE-CONDITIONS:
- Gate 3 passed (GATE_3_PASSED_WITH_METHODOLOGICAL_DEBT)
- Slice 4 authorized
- RAGLAB_PDF_PATH set to the corpus PDF
- GEMINI_API_KEY set (via human secure injection)
- HF_HUB_OFFLINE=1 (embeddings are local)
- Holdout remains sealed

USAGE (human terminal):
    export RAGLAB_PDF_PATH="/path/to/gersting.pdf"
    export GEMINI_API_KEY="$(decrypt-my-key)"
    HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \\
        .venv/bin/python benchmarks/run_slice4_benchmark.py
    unset GEMINI_API_KEY

HOLDOUT: SEALED. Do not add q_holdout_01 or q_holdout_02 to ACTIVE_QUESTIONS.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# ─── Path setup ───────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("slice4_benchmark")

# ─── Constants ───────────────────────────────────────────────────
EXPERIMENT_ID = "raglab_v7_slice4_v1_20260731T1230UTC"
PDF_SHA256_EXPECTED = "33e2e9f1e190158b3e99c19fced1acd050720247c7556780bad82b2f93bf1254"
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

# ─── Active questions (holdout sealed) ───────────────────────────
# HOLDOUT: q_holdout_01, q_holdout_02 — DO NOT ADD HERE
ACTIVE_QUESTIONS = [
    {"qid": "q_dev_01", "split": "development",
     "query": "O que é demonstração por exaustão e quando é aplicável?",
     "relevant_pages": [92]},
    {"qid": "q_dev_02", "split": "development",
     "query": "Como o método de prova por contradição funciona em matemática discreta?",
     "relevant_pages": [95]},
    {"qid": "q_dev_03", "split": "development",
     "query": "Quais são as etapas do princípio da indução matemática?",
     "relevant_pages": [97]},
    {"qid": "q_dev_04", "split": "development",
     "query": "Como funciona a indução forte comparada à indução fraca?",
     "relevant_pages": [101, 102]},
    {"qid": "q_test_01", "split": "test",
     "query": "Qual é a diferença entre indução fraca e indução forte?",
     "relevant_pages": [101, 102]},
    {"qid": "q_test_02", "split": "test",
     "query": "Como se define a base e o passo indutivo em demonstração por indução?",
     "relevant_pages": [95]},
    {"qid": "q_test_03", "split": "test",
     "query": "Quais são os passos para provar uma afirmação usando indução completa?",
     "relevant_pages": [101, 102, 103]},
    {"qid": "q_test_04", "split": "test",
     "query": "Qual é a capital da França?",
     "relevant_pages": [],
     "abstention_expected": True},
]


# ─── Helpers ─────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_no_credential_in_env() -> None:
    """Log a warning (not error) if GEMINI_API_KEY is missing."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        logger.error(
            "GEMINI_API_KEY not found. "
            "Export it before running: export GEMINI_API_KEY=... "
            "See docs/security/credential_boundary.md"
        )
        sys.exit(1)
    # NEVER log the key value — only log that it was found
    logger.info("GEMINI_API_KEY detected (value not logged)")


def load_pdf_pages(pdf_path: Path) -> list:
    from raglab.infrastructure.pdf_parsers.pdf_parser_adapter import (
        PyPdfExtractorAdapter,
    )
    adapter = PyPdfExtractorAdapter()
    pages = adapter.read_document(
        str(pdf_path), page_start=PAGES_START, page_end=PAGES_END
    )
    logger.info("Extracted %d pages (pages %d–%d) from PDF", len(pages), PAGES_START, PAGES_END)
    return pages


def load_embedding_model() -> object:
    from raglab.infrastructure.embeddings.fastembed_adapter import FastEmbedAdapter
    logger.info("Loading FastEmbed model: %s", EMBEDDING_MODEL)
    adapter = FastEmbedAdapter(model_name=EMBEDDING_MODEL)
    logger.info("Model loaded (dim=%d)", adapter.embedding_dim)
    return adapter


# ─── Retriever builders ──────────────────────────────────────────

def build_retrievers(pages: list, embed_model: object) -> dict[str, object]:
    """Build one retriever per strategy. Returns dict keyed by strategy label."""
    from raglab.infrastructure.retrieval.baseline_adapter import BaselineRetrieverAdapter
    from raglab.infrastructure.retrieval.sentence_anchor_adapter import SentenceAnchorAdapter
    from raglab.infrastructure.retrieval.sentence_window_adapter import SentenceWindowAdapter
    from raglab.infrastructure.retrieval.reranker_adapter import LocalRerankerAdapter
    from raglab.infrastructure.retrieval.auto_merging_adapter import AutoMergingAdapter

    logger.info("Building retrievers...")

    f0 = BaselineRetrieverAdapter(
        pages=pages, embed_model=embed_model,
        chunk_size=CHUNK_SIZE, top_k=TOP_K,
    )
    s0 = SentenceAnchorAdapter(
        pages=pages, embed_model=embed_model, top_k=TOP_K,
    )
    w0 = SentenceWindowAdapter(
        pages=pages, embed_model=embed_model,
        window_size=WINDOW_SIZE, top_k=TOP_K,
    )
    w1_base = SentenceWindowAdapter(
        pages=pages, embed_model=embed_model,
        window_size=WINDOW_SIZE, top_k=CANDIDATE_K,
    )
    w1 = LocalRerankerAdapter(
        base_retriever=w1_base, embed_model=embed_model, top_k=TOP_K,
    )
    h0 = AutoMergingAdapter(
        pages=pages, embed_model=embed_model,
        top_k=TOP_K, merge_threshold=MERGE_THRESHOLD,
        enable_auto_merging=False, enable_reranking=False,
    )
    h1 = AutoMergingAdapter(
        pages=pages, embed_model=embed_model,
        top_k=TOP_K, merge_threshold=MERGE_THRESHOLD,
        enable_auto_merging=True, enable_reranking=False,
    )
    h2 = AutoMergingAdapter(
        pages=pages, embed_model=embed_model,
        top_k=TOP_K, merge_threshold=MERGE_THRESHOLD,
        enable_auto_merging=True, enable_reranking=True,
    )

    return {
        "F0_baseline": f0,
        "S0_sentence_anchor": s0,
        "W0_sentence_window": w0,
        "W1_sentence_window_rerank": w1,
        "H0_hierarchical_leaf": h0,
        "H1_auto_merging": h1,
        "H2_auto_merging_rerank": h2,
    }


# ─── Main ────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== RAGLab v7 Slice 4 Benchmark — Starting ===")
    logger.info("Experiment ID: %s", EXPERIMENT_ID)
    logger.info("Gemini model: %s", GEMINI_MODEL)
    logger.info("Reranker: bi_encoder_rescoring (NOT a cross-encoder)")

    # Step 1: Security check
    verify_no_credential_in_env()

    # Step 2: Locate and verify PDF
    pdf_path_str = os.environ.get("RAGLAB_PDF_PATH")
    if not pdf_path_str:
        logger.error("RAGLAB_PDF_PATH not set")
        sys.exit(1)
    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        logger.error("PDF not found: %s", pdf_path)
        sys.exit(1)

    actual_sha = sha256_file(pdf_path)
    if actual_sha != PDF_SHA256_EXPECTED:
        logger.error(
            "PDF SHA-256 mismatch. Expected %s, got %s",
            PDF_SHA256_EXPECTED, actual_sha,
        )
        sys.exit(1)
    logger.info("PDF SHA-256 verified: %s", actual_sha)

    # Step 3: Load corpus
    pages = load_pdf_pages(pdf_path)
    embed_model = load_embedding_model()

    # Step 4: Load questions
    active_q = [q for q in ACTIVE_QUESTIONS if not q.get("abstention_expected", False)]
    abstention_q = [q for q in ACTIVE_QUESTIONS if q.get("abstention_expected", False)]
    logger.info("Active questions: %d, Abstention: %d (holdout: sealed)",
                len(active_q), len(abstention_q))

    # Step 5: Build retrievers
    retrievers = build_retrievers(pages, embed_model)

    # Step 6: Init Gemini adapters (reads GEMINI_API_KEY)
    from raglab.infrastructure.gemini.gemini_generator_adapter import (
        GeminiGeneratorAdapter,
        sanitize_answer_for_artifact,
    )
    from raglab.infrastructure.gemini.gemini_judge_adapter import (
        GeminiJudgeAdapter,
        sanitize_evaluation_for_artifact,
    )
    from raglab.domain.enums import PipelineStrategy
    from raglab.domain.quota import QuotaManager
    from raglab.domain.retry import RetryPolicy
    from raglab.infrastructure.persistence.generation_checkpoint_store import (
        GenerationCheckpointStore,
    )

    # Shared quota manager — all adapters share the same API key budget
    shared_quota = QuotaManager()
    shared_retry = RetryPolicy()

    generator = GeminiGeneratorAdapter(
        model_id=GEMINI_MODEL,
        quota_manager=shared_quota,
        retry_policy=shared_retry,
        temperature=0.0,
    )
    logger.info("Generator initialized: %s", generator.model_id)

    # Step 7: Init checkpoint store
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = GenerationCheckpointStore(
        run_id=EXPERIMENT_ID, store_dir=CHECKPOINT_DIR
    )
    logger.info("Checkpoint: %d entries already completed", ckpt.completed_count())

    # Step 8: Run generation + evaluation per strategy × question
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[dict]] = {}
    run_times: dict[str, float] = {}

    all_questions = ACTIVE_QUESTIONS  # includes abstention questions

    for strategy_label, retriever in retrievers.items():
        logger.info("=== Strategy: %s ===", strategy_label)
        strategy_results = []
        t0 = time.monotonic()

        # Strategy-specific judge (independently configured)
        judge = GeminiJudgeAdapter(
            judge_model_id=GEMINI_MODEL,
            strategy=PipelineStrategy.BASELINE,  # best-effort mapping
            quota_manager=shared_quota,
            retry_policy=shared_retry,
            temperature=0.0,
        )

        for q in all_questions:
            qid = q["qid"]
            query = q["query"]
            is_abstention = q.get("abstention_expected", False)

            # Idempotent skip
            if ckpt.is_completed(qid, strategy_label):
                logger.info("  SKIP (already done): %s::%s", qid, strategy_label)
                continue

            logger.info("  Processing: %s (abstention=%s)", qid, is_abstention)

            # Retrieve
            evidence = retriever.retrieve(query)

            # Generate
            answer = generator.generate(
                query_id=f"{strategy_label}::{qid}",
                query=query,
                evidence=evidence,
            )

            # Evaluate (skip for abstention questions)
            eval_result = None
            if not answer.abstained and not is_abstention and evidence:
                eval_result = judge.evaluate(
                    query_id=f"{strategy_label}::{qid}",
                    query=query,
                    answer=answer,
                    evidence=evidence,
                )

            # Build sanitized result
            sanitized_answer = sanitize_answer_for_artifact(answer)
            sanitized_eval = (
                sanitize_evaluation_for_artifact(eval_result)
                if eval_result else None
            )

            result_entry = {
                "qid": qid,
                "split": q["split"],
                "strategy": strategy_label,
                "relevant_pages": q.get("relevant_pages", []),
                "abstented": answer.abstained,
                "is_abstention_question": is_abstention,
                "answer": sanitized_answer,
                "evaluation": sanitized_eval,
                "quota_stats": shared_quota.stats,
            }
            strategy_results.append(result_entry)

            # Checkpoint
            ckpt.mark_completed(
                qid, strategy_label,
                abstained=answer.abstained,
                citation_count=len(answer.citations),
            )

        elapsed_ms = (time.monotonic() - t0) * 1000
        run_times[strategy_label] = round(elapsed_ms, 1)
        all_results[strategy_label] = strategy_results
        logger.info("  %s complete in %.1f ms", strategy_label, elapsed_ms)

    # Step 9: Write results
    output_path = RESULTS_DIR / f"slice4_results_{EXPERIMENT_ID}.json"
    output = {
        "experiment_id": EXPERIMENT_ID,
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
    logger.info("=== Slice 4 Benchmark Complete ===")
    logger.info("Quota stats: %s", shared_quota.stats)


if __name__ == "__main__":
    main()
