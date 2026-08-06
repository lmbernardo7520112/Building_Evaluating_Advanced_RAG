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
import re
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from raglab.evaluation.contracts.ground_truth_v2 import (
    UnanswerableReason,
)
from raglab.evaluation.contracts.human_qrels_v2 import (
    load_human_qrels_set,
)
from raglab.evaluation.metrics.deterministic_v2 import (
    compute_legacy_page_metrics,
)
from raglab.evaluation.metrics.human_qrels_metrics import (
    compute_human_qrels_metrics_for_question,
)
from raglab.evaluation.migration.legacy_to_gt_v2 import (
    migrate_legacy_qrel_item,
)
from raglab.evaluation.pooling.canonical_passage_mapper import (
    CanonicalPassageMapper,
)

# ─── Path setup ───────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ─── Constants ───────────────────────────────────────────────────
PROTOCOL_VERSION = "raglab_v7_slice4_v3"
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

DEFAULT_QRELS_PATH: Final[Path] = (
    _REPO_ROOT
    / "benchmarks"
    / "ground_truth"
    / "v2"
    / "hybrid"
    / "qrels"
    / "human_qrels_final.jsonl"
)
DEFAULT_QRELS_MANIFEST_PATH: Final[Path] = (
    _REPO_ROOT
    / "benchmarks"
    / "ground_truth"
    / "v2"
    / "hybrid"
    / "qrels"
    / "human_qrels_manifest.json"
)

_EVAL_SCHEMA_VERSION = "slice4_v5"


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
        choices=["preflight", "preflight-retrievers", "preflight-human-qrels", "smoke", "full", "resume"],
        required=True,
        help=(
            "preflight: validate embedding cache offline (no Gemini key). "
            "preflight-retrievers: validate all 7 retriever builders structurally "
            "preflight-human-qrels: validate human qrels dataset & metrics offline (no Gemini key). "
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
        "--qrels-path",
        default=str(DEFAULT_QRELS_PATH),
        help="Path to final human qrels JSONL file.",
    )
    parser.add_argument(
        "--qrels-manifest",
        default=str(DEFAULT_QRELS_MANIFEST_PATH),
        help="Path to final human qrels manifest JSON file.",
    )
    parser.add_argument(
        "--smoke-question",
        default="q_dev_01",
        choices=sorted(_NON_HOLDOUT_QIDS),
        help="Question ID for smoke test (default: q_dev_01, must not be holdout).",
    )
    parser.add_argument(
        "--pdf-path",
        default=None,
        help="Path to textbook PDF file (defaults to RAGLAB_PDF_PATH environment variable).",
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


# ─── Strategy retrieval configuration helpers (OBJETIVO 2) ──────────

def build_retrieval_configuration(strategy_label: str) -> dict[str, Any]:
    """Return the canonical retrieval_configuration dict for a given strategy.

    Rules:
    - F0: reranker_enabled=False, reranker_class=None, reranker_top_n=None
    - S0: False/None
    - W0: False/None
    - W1: True/"bi_encoder_rescoring", reranker_top_n=TOP_K
    - H0: False/None
    - H1: False/None
    - H2: True/"bi_encoder_rescoring", reranker_top_n=TOP_K
    """
    if strategy_label == "F0_baseline":
        return {
            "strategy": "F0_baseline",
            "retrieval_family": "fixed_chunk",
            "candidate_k": None,
            "final_k": TOP_K,
            "window_size": None,
            "hierarchy_chunk_sizes": None,
            "auto_merge_threshold": None,
            "reranker_enabled": False,
            "reranker_class": None,
            "reranker_top_n": None,
        }
    elif strategy_label == "S0_sentence_anchor":
        return {
            "strategy": "S0_sentence_anchor",
            "retrieval_family": "sentence_anchor",
            "candidate_k": None,
            "final_k": TOP_K,
            "window_size": None,
            "hierarchy_chunk_sizes": None,
            "auto_merge_threshold": None,
            "reranker_enabled": False,
            "reranker_class": None,
            "reranker_top_n": None,
        }
    elif strategy_label == "W0_sentence_window":
        return {
            "strategy": "W0_sentence_window",
            "retrieval_family": "sentence_window",
            "candidate_k": None,
            "final_k": TOP_K,
            "window_size": WINDOW_SIZE,
            "hierarchy_chunk_sizes": None,
            "auto_merge_threshold": None,
            "reranker_enabled": False,
            "reranker_class": None,
            "reranker_top_n": None,
        }
    elif strategy_label == "W1_sentence_window_rerank":
        return {
            "strategy": "W1_sentence_window_rerank",
            "retrieval_family": "sentence_window",
            "candidate_k": CANDIDATE_K,
            "final_k": TOP_K,
            "window_size": WINDOW_SIZE,
            "hierarchy_chunk_sizes": None,
            "auto_merge_threshold": None,
            "reranker_enabled": True,
            "reranker_class": "bi_encoder_rescoring",
            "reranker_top_n": TOP_K,
        }
    elif strategy_label == "H0_hierarchical_leaf":
        return {
            "strategy": "H0_hierarchical_leaf",
            "retrieval_family": "hierarchical",
            "candidate_k": None,
            "final_k": TOP_K,
            "window_size": None,
            "hierarchy_chunk_sizes": [1024, 512, 256],
            "auto_merge_threshold": None,
            "reranker_enabled": False,
            "reranker_class": None,
            "reranker_top_n": None,
        }
    elif strategy_label == "H1_auto_merging":
        return {
            "strategy": "H1_auto_merging",
            "retrieval_family": "hierarchical",
            "candidate_k": None,
            "final_k": TOP_K,
            "window_size": None,
            "hierarchy_chunk_sizes": [1024, 512, 256],
            "auto_merge_threshold": MERGE_THRESHOLD,
            "reranker_enabled": False,
            "reranker_class": None,
            "reranker_top_n": None,
        }
    elif strategy_label == "H2_auto_merging_rerank":
        return {
            "strategy": "H2_auto_merging_rerank",
            "retrieval_family": "hierarchical",
            "candidate_k": CANDIDATE_K,
            "final_k": TOP_K,
            "window_size": None,
            "hierarchy_chunk_sizes": [1024, 512, 256],
            "auto_merge_threshold": MERGE_THRESHOLD,
            "reranker_enabled": True,
            "reranker_class": "bi_encoder_rescoring",
            "reranker_top_n": TOP_K,
        }
    else:
        raise ValueError(f"Unknown strategy for retrieval_configuration: {strategy_label}")


def compute_retrieval_configuration_sha256(config: dict[str, Any]) -> str:
    """Compute canonical SHA-256 hash of a retrieval configuration dict."""
    canonical_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _extract_page_from_chunk_id(chunk_id_val: str) -> int | None:
    """Extract page number integer from chunk_id or document_id like 'doc_p92_c0' or 'page_0092'. Returns None on failure."""
    if not chunk_id_val:
        return None
    try:
        m = re.search(r"(?:_p|page_|^p|page)(\d+)", str(chunk_id_val), re.IGNORECASE)
        if m:
            val = int(m.group(1))
            return val if val >= 1 else None
        parts = str(chunk_id_val).split("_p")
        if len(parts) >= 2:
            page_part = parts[-1].split("_")[0]
            val = int(page_part)
            return val if val >= 1 else None
    except (ValueError, IndexError):
        pass
    return None



def resolve_candidate_page_number(cand: Any) -> int | None:
    """Resolve physical page number integer (>= 1) for a candidate record or object.

    Aborts with ValueError if explicit page_number attribute conflicts with chunk_id page.
    Returns None if page_number is missing, None, boolean, non-integer, zero or negative.
    """
    if cand is None:
        return None

    raw_page = None
    chunk_id_str = ""
    doc_id_str = ""

    if isinstance(cand, dict):
        raw_page = cand.get("page_number")
        chunk_id_str = str(cand.get("chunk_id", ""))
        doc_id_str = str(cand.get("document_id", ""))
    else:
        raw_page = getattr(cand, "page_number", None)
        chunk_id_val = getattr(cand, "chunk_id", "")
        if hasattr(chunk_id_val, "value"):
            chunk_id_val = chunk_id_val.value
        chunk_id_str = str(chunk_id_val or "")
        doc_id_str = str(getattr(cand, "document_id", "") or "")

    # Reject booleans and mocks explicitly
    if isinstance(raw_page, bool) or (raw_page is not None and "Mock" in type(raw_page).__name__):
        raw_page = None


    extracted = None
    if chunk_id_str:
        extracted = _extract_page_from_chunk_id(chunk_id_str)
    if extracted is None and doc_id_str:
        extracted = _extract_page_from_chunk_id(doc_id_str)

    if (
        raw_page is not None
        and isinstance(raw_page, int)
        and raw_page >= 1
        and extracted is not None
        and extracted >= 1
        and raw_page != extracted
    ):
        raise ValueError(
            f"CITATION_PROVENANCE_MISMATCH: Page number mismatch between explicit page {raw_page} "
            f"and page from chunk_id ({extracted})"
        )

    if isinstance(raw_page, int) and raw_page >= 1:
        return raw_page

    if extracted is not None and extracted >= 1:
        return extracted

    return None


# ─── Citation mapping helper (OBJETIVO 3) ───────────────────────────

def build_citation_map_and_status(
    answer_text: str,
    abstained: bool,
    evidence: list[Any],
    query_id: str,
    citations: Sequence[Any] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Map citation markers ([E1], [E2], ...) or typed citations to evidence candidates.

    Protocol V2 (Authoritative):
        - Uses evidence_id markers ([E1], [E2], etc.) or citations sequence with evidence_id.
        - Resolves exclusively against PromptEvidence / RetrievedEvidence candidates.
        - Returns status "AVAILABLE".

    Protocol Legacy (Isolated):
        - Uses numeric markers ([1], [2], etc.) without evidence_id.
        - Maps strictly to 1-indexed evidence rank.
        - Returns status "LEGACY".
    """
    if abstained:
        return ("NOT_APPLICABLE", [])

    # 1. Extract V2 evidence_id markers like [E1], [E2]
    v2_matches = re.findall(r"\[(E\d+)\]", answer_text)
    v2_markers: list[str] = []
    for m in v2_matches:
        marker_str = f"[{m}]"
        if marker_str not in v2_markers:
            v2_markers.append(marker_str)

    # Add any structured citations with evidence_id passed in citations parameter
    if citations:
        for c in citations:
            ev_id = getattr(c, "evidence_id", None) or (
                c.get("evidence_id") if isinstance(c, dict) else None
            )
            if ev_id and str(ev_id).startswith("E") and str(ev_id)[1:].isdigit():
                m_str = f"[{ev_id}]"
                if m_str not in v2_markers:
                    v2_markers.append(m_str)

    if v2_markers:
        # V2 Protocol Execution — Status is "AVAILABLE"
        citation_map: list[dict[str, Any]] = []
        for marker_str in v2_markers:
            m_content = marker_str.strip("[]")  # e.g. "E1"
            rank_val = int(m_content[1:])       # e.g. 1

            matched_cand = None
            if 1 <= rank_val <= len(evidence):
                matched_cand = evidence[rank_val - 1]
            else:
                for cand in evidence:
                    cand_ev_id = (
                        cand.get("evidence_id")
                        if isinstance(cand, dict)
                        else getattr(cand, "evidence_id", None)
                    )
                    if cand_ev_id == m_content:
                        matched_cand = cand
                        break

            if matched_cand is None:
                raise ValueError(
                    f"CITATION_PROVENANCE_MISMATCH: Citation marker {marker_str} "
                    f"in query_id={query_id} does not map to any retrieved evidence candidate"
                )

            # Derive chunk_id
            if isinstance(matched_cand, dict):
                cid = matched_cand.get("chunk_id")
            else:
                cid = getattr(matched_cand, "chunk_id", None)
            if hasattr(cid, "value"):
                cid = cid.value
            cand_chunk_id = str(cid) if cid is not None else ""
            if not cand_chunk_id or not cand_chunk_id.strip():
                raise ValueError(
                    f"CITATION_PROVENANCE_MISMATCH: Candidate for {marker_str} has empty chunk_id"
                )

            # Derive page_number (exclusively from evidence)
            cand_page = resolve_candidate_page_number(matched_cand)
            if (
                cand_page is None
                or isinstance(cand_page, bool)
                or not isinstance(cand_page, int)
                or cand_page < 1
            ):
                raise ValueError(
                    f"CITATION_PROVENANCE_MISMATCH: Candidate for {marker_str} has invalid page_number {cand_page!r}"
                )

            # Derive content_sha256 (exclusively from evidence)
            if isinstance(matched_cand, dict):
                c_sha = matched_cand.get("content_sha256") or matched_cand.get("text_sha256")
                c_text = matched_cand.get("text")
            else:
                c_sha = getattr(matched_cand, "content_sha256", None) or getattr(matched_cand, "text_sha256", None)
                c_text = getattr(matched_cand, "text", None)

            if isinstance(c_sha, str) and len(c_sha) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in c_sha):
                cand_sha = c_sha
            elif isinstance(c_text, str) and c_text.strip():
                cand_sha = hashlib.sha256(c_text.encode("utf-8")).hexdigest()
            else:
                raise ValueError(
                    f"CITATION_PROVENANCE_MISMATCH: Candidate for {marker_str} has no valid content_sha256 or text"
                )

            # Derive passage_id and rank
            if isinstance(matched_cand, dict):
                cand_rank = matched_cand.get("retrieval_rank", matched_cand.get("rank", rank_val))
                cand_passage_id = matched_cand.get("passage_id")
                cand_doc = matched_cand.get("document_id", "doc")
            else:
                cand_rank = getattr(matched_cand, "rank", rank_val)
                cand_passage_id = getattr(matched_cand, "passage_id", None)
                cand_doc = getattr(matched_cand, "document_id", "doc")

            cand_rank = int(cand_rank) if isinstance(cand_rank, int) else rank_val
            if not cand_passage_id or not isinstance(cand_passage_id, str):
                cand_passage_id = f"{cand_doc}_p{cand_page}_rank{cand_rank}"

            citation_map.append({
                "marker": marker_str,
                "evidence_id": m_content,
                "passage_id": cand_passage_id,
                "page_number": cand_page,
                "content_sha256": cand_sha,
                "retrieval_rank": cand_rank,
                "chunk_id": cand_chunk_id,
                "text_sha256": cand_sha,
            })

        return ("AVAILABLE", citation_map)

    # 2. Legacy Check: numeric markers like [1], [2] (isolated as LEGACY)
    legacy_matches = re.findall(r"\[(\d+)\]", answer_text)
    if legacy_matches:
        citation_map = []
        for m in legacy_matches:
            marker_str = f"[{m}]"
            n = int(m)

            # Legacy matching strictly by 1-indexed rank (not model page)
            if 1 <= n <= len(evidence):
                matched_cand = evidence[n - 1]
            else:
                raise ValueError(
                    f"CITATION_PROVENANCE_MISMATCH: Legacy marker {marker_str} in query_id={query_id} "
                    f"exceeds evidence list length {len(evidence)}"
                )

            if isinstance(matched_cand, dict):
                cid = matched_cand.get("chunk_id")
            else:
                cid = getattr(matched_cand, "chunk_id", None)
            if hasattr(cid, "value"):
                cid = cid.value
            cand_chunk_id = str(cid) if cid is not None else ""
            if not cand_chunk_id or not cand_chunk_id.strip():
                raise ValueError(
                    f"CITATION_PROVENANCE_MISMATCH: Candidate for {marker_str} has empty chunk_id"
                )

            cand_page = resolve_candidate_page_number(matched_cand)
            if (
                cand_page is None
                or isinstance(cand_page, bool)
                or not isinstance(cand_page, int)
                or cand_page < 1
            ):
                raise ValueError(
                    f"CITATION_PROVENANCE_MISMATCH: Candidate for {marker_str} has invalid page_number {cand_page!r}"
                )

            if isinstance(matched_cand, dict):
                c_sha = matched_cand.get("content_sha256") or matched_cand.get("text_sha256")
                c_text = matched_cand.get("text")
            else:
                c_sha = getattr(matched_cand, "content_sha256", None) or getattr(matched_cand, "text_sha256", None)
                c_text = getattr(matched_cand, "text", None)

            if isinstance(c_sha, str) and len(c_sha) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in c_sha):
                cand_sha = c_sha
            elif isinstance(c_text, str) and c_text.strip():
                cand_sha = hashlib.sha256(c_text.encode("utf-8")).hexdigest()
            else:
                raise ValueError(
                    f"CITATION_PROVENANCE_MISMATCH: Candidate for {marker_str} has no valid content_sha256 or text"
                )

            citation_map.append({
                "marker": marker_str,
                "evidence_id": f"E{n}",
                "passage_id": f"doc_p{cand_page}_rank{n}",
                "page_number": cand_page,
                "content_sha256": cand_sha,
                "retrieval_rank": n,
                "chunk_id": cand_chunk_id,
                "text_sha256": cand_sha,
            })

        return ("LEGACY", citation_map)

    return ("UNAVAILABLE", [])


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
            self, base_retriever: Any, reranker: Any,
            candidate_k: int, top_n: int,
        ) -> None:
            self._base = base_retriever
            self._reranker = reranker
            self._candidate_k = candidate_k
            self._top_n = top_n
            self.last_pre_rerank_candidates: list[Any] = []

        @property
        def pre_rerank_candidates(self) -> list[Any]:
            return self.last_pre_rerank_candidates

        def retrieve(self, query: str, top_k: int = 3) -> list[Any]:
            candidates: list[Any] = self._base.retrieve(query, top_k=self._candidate_k)
            self.last_pre_rerank_candidates = list(candidates)
            reranked, _ = self._reranker.rerank(query, candidates, top_n=self._top_n)
            return list(reranked)




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


# ─── Retrieval Evidence & Canonical Candidate Mapping (TAREFA 1) ───

def _extract_candidate_raw_info(c: Any) -> tuple[str, str, int, str, float | None]:
    """Extract raw_id, doc_id, page_num, text, score from candidate (dict or object)."""
    raw_id = ""
    doc_id = "gersting_discrete_math"
    page_num = 1
    text = ""
    score = None

    if isinstance(c, dict):
        raw_id = c.get("raw_candidate_id", c.get("chunk_id", c.get("id", "")))
        doc_id = c.get("document_id", "gersting_discrete_math")
        page_num = resolve_candidate_page_number(c) or 1
        text = c.get("text", "")
        score = c.get("retrieval_score", c.get("score"))
    else:
        score = getattr(c, "score", getattr(c, "retrieval_score", None))

        is_mock = "Mock" in type(c).__name__
        node = getattr(c, "node", None) if not is_mock else None

        if node is not None and "Mock" not in type(node).__name__:
            raw_id = getattr(node, "node_id", getattr(node, "id_", ""))
            text = getattr(node, "text", "")
            if not text and hasattr(node, "get_content"):
                text = node.get_content()
            meta = getattr(node, "metadata", {}) or getattr(node, "extra_info", {}) or {}
            doc_id = meta.get("document_id", getattr(c, "document_id", "gersting_discrete_math"))
        else:
            raw_id = getattr(c, "chunk_id", getattr(c, "id", ""))
            doc_id = getattr(c, "document_id", "gersting_discrete_math")
            text = getattr(c, "text", "")

        page_num = resolve_candidate_page_number(c) or 1

    if hasattr(raw_id, "value"):
        raw_id = raw_id.value
    raw_id_str = str(raw_id) if raw_id is not None and "Mock" not in type(raw_id).__name__ else ""
    doc_id_str = (
        str(doc_id)
        if doc_id is not None and str(doc_id).strip() and "Mock" not in type(doc_id).__name__
        else "gersting_discrete_math"
    )
    text_str = str(text) if text is not None and "Mock" not in type(text).__name__ else ""
    score_val = (
        float(score)
        if score is not None and isinstance(score, (int, float)) and not isinstance(score, bool)
        else None
    )

    return raw_id_str, doc_id_str, page_num, text_str, score_val




def map_candidate_to_canonical(
    c: Any,
    mapper: CanonicalPassageMapper,
    rank: int,
    qrels_set: Any | None = None,
    question_id: str = "",
) -> dict[str, Any]:
    """Canonical candidate mapping function shared by preflight and real runner (TAREFA 1)."""
    raw_id, doc_id, page_num, text, score = _extract_candidate_raw_info(c)

    c_res = mapper.map_chunk({
        "chunk_id": raw_id,
        "document_id": doc_id,
        "page_number": page_num,
        "text": text,
    })

    canon_pid = c_res.mapped_passage_id
    is_valid_canonical = (
        canon_pid is not None
        and isinstance(canon_pid, str)
        and canon_pid.startswith("ps_")
        and not canon_pid.endswith("_rank1")
        and not canon_pid.endswith("_rank2")
        and not canon_pid.endswith("_rank3")
    )

    if is_valid_canonical:
        canonical_passage_id = canon_pid
        mapping_status = (
            c_res.mapping_status.value
            if hasattr(c_res.mapping_status, "value")
            else str(c_res.mapping_status)
        )

        confidence = float(c_res.confidence)
    else:
        canonical_passage_id = "UNMAPPED_NEEDS_REVIEW"
        mapping_status = "UNMAPPED"
        confidence = 0.0

    judged_status = "UNJUDGED"
    relevance_grade: int | None = None
    if (
        qrels_set is not None
        and question_id
        and is_valid_canonical
        and "holdout" not in question_id.lower()
    ):
        qrel = qrels_set.get_qrel(question_id, canonical_passage_id)
        if qrel is not None:
            judged_status = "JUDGED"
            relevance_grade = qrel.relevance_grade


    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""
    text_preview = text[:80].replace("\n", " ") if text else ""
    for pattern in _SECRET_PATTERNS:
        if pattern.lower() in text_preview.lower():
            text_preview = "[REDACTED]"
            break

    return {
        "raw_candidate_id": raw_id,
        "chunk_id": raw_id,
        "legacy_display_passage_id": f"{doc_id}_p{page_num}_rank{rank}",
        "canonical_passage_id": canonical_passage_id,
        "passage_id": canonical_passage_id,
        "page_number": page_num,
        "mapping_status": mapping_status,
        "confidence": confidence,
        "retrieval_rank": rank,
        "retrieval_score": score,
        "judged_status": judged_status,
        "relevance_grade": relevance_grade,
        "text": text,
        "text_sha256": text_sha,
        "content_sha256": text_sha,
        "text_preview": text_preview,
        "evidence_id": f"E{rank}",
    }


def serialize_retrieval_evidence(
    evidence: list,
    relevant_pages: list[int],
    mapper: CanonicalPassageMapper | None = None,
    qrels_set: Any | None = None,
    question_id: str = "",
    text_preview_limit: int = 80,
) -> dict[str, Any]:
    """Serialize retrieval evidence for auditable output with canonical passage_ids."""
    candidates: list[dict[str, Any]] = []
    pages_found: list[int] = []

    if mapper is None:
        mapper = CanonicalPassageMapper()

    for i, ev in enumerate(evidence):
        rec = map_candidate_to_canonical(
            ev, mapper, rank=i + 1, qrels_set=qrels_set, question_id=question_id
        )
        page_num = rec["page_number"]
        if page_num and page_num in relevant_pages:
            pages_found.append(page_num)

        candidates.append(rec)

    retrieval_hit = bool(pages_found)
    missing_pages = [p for p in relevant_pages if p not in pages_found]
    mapped_count = sum(1 for c in candidates if str(c["canonical_passage_id"]).startswith("ps_"))
    unresolved_count = sum(1 for c in candidates if not str(c["canonical_passage_id"]).startswith("ps_"))

    return {
        "candidate_count": len(candidates),
        "mapped_count": mapped_count,
        "unresolved_mapping_count": unresolved_count,
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
    qrels_path: Path | str | None = None,
    qrels_manifest: Path | str | None = None,
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
    qrels_p = Path(qrels_path) if qrels_path else DEFAULT_QRELS_PATH
    manifest_p = (
        Path(qrels_manifest) if qrels_manifest else DEFAULT_QRELS_MANIFEST_PATH
    )

    try:
        qrels_set = load_human_qrels_set(qrels_p, manifest_p)
    except ValueError as exc:
        logger.error("HUMAN_QRELS_REQUIRED_OR_INVALID: %s", exc)
        print("HUMAN_QRELS_REQUIRED_OR_INVALID")
        raise ValueError(f"HUMAN_QRELS_REQUIRED_OR_INVALID: {exc}") from exc

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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # In resume mode (or if partial artifact exists), merge partial artifact into checkpoint store
    partial_files = sorted(RESULTS_DIR.glob(f"slice4_results_{run_id}_*.json"))
    for pfile in partial_files:
        merged = ckpt.merge_partial_artifact(pfile)
        if merged > 0:
            logger.info(
                "Merged %d complete result rows from partial artifact %s into checkpoint",
                merged, pfile.name,
            )

    logger.info(
        "Checkpoint: %d total entries (%d complete result rows)",
        ckpt.completed_count(), ckpt.complete_rows_count(),
    )

    all_results: dict[str, list[dict]] = {
        label: [] for label in strategy_labels
    }
    run_times: dict[str, float] = {}

    # Rehydrate existing complete result rows from checkpoint
    rehydrated = ckpt.rehydrate_complete_rows()
    for strat, rows in rehydrated.items():
        if strat in all_results:
            all_results[strat].extend(rows)
            logger.info(
                "Rehydrated %d complete result rows for strategy %s from checkpoint",
                len(rows), strat,
            )

    mapper = CanonicalPassageMapper()

    for strategy_label in strategy_labels:
        retriever = retrievers[strategy_label]
        pipeline_strategy = PipelineStrategy.from_label(strategy_label)
        logger.info("=== Strategy: %s ===", strategy_label)
        strategy_results: list[dict] = all_results.get(strategy_label, [])
        t0 = time.monotonic()

        judge = GeminiJudgeAdapter(
            judge_model_id=GEMINI_MODEL,
            strategy=pipeline_strategy,
            quota_manager=shared_quota,
            retry_policy=shared_retry,
            temperature=0.0,
        )

        retrieval_config = build_retrieval_configuration(strategy_label)
        retrieval_config_sha = compute_retrieval_configuration_sha256(retrieval_config)

        for q in questions:
            qid = q["qid"]
            query = q["query"]
            is_abstention = q.get("abstention_expected", False)
            relevant_pages = q.get("relevant_pages", [])
            query_id = f"{strategy_label}::{qid}"

            if ckpt.has_complete_result_row(qid, strategy_label):
                logger.info("  SKIP (already complete): %s", query_id)
                continue

            # Defensive holdout guard — never run holdout questions
            if "holdout" in qid:
                logger.error(
                    "HOLDOUT GUARD: qid=%s rejected — holdout is SEALED", qid
                )
                sys.exit(2)

            logger.info("  Processing: %s (abstention=%s)", query_id, is_abstention)

            quota_before = int(shared_quota.stats["total_requests"])
            retries_before = int(shared_quota.stats.get("total_retries", 0))
            r429_before = int(shared_quota.stats.get("rate_limit_429_count", 0))
            r5xx_before = int(shared_quota.stats.get("server_5xx_retry_count", 0))
            rother_before = int(shared_quota.stats.get("other_retryable_error_count", 0))

            evidence = retriever.retrieve(query, top_k=TOP_K)

            # ── Serialize retrieval evidence for audit with Canonical Mapper ───
            evidence_record = serialize_retrieval_evidence(
                evidence,
                relevant_pages,
                mapper=mapper,
                qrels_set=qrels_set,
                question_id=qid,
            )

            # ── TAREFA 2: Fail-Closed Check on Production Mapping ───
            if "holdout" not in qid.lower():
                cands_list = evidence_record.get("candidates", [])
                retrieved_cnt = len(cands_list)
                mapped_cnt = evidence_record.get("mapped_count", 0)
                unresolved_cnt = evidence_record.get("unresolved_mapping_count", 0)

                has_synthetic_pid = any(
                    not str(c.get("canonical_passage_id", "")).startswith("ps_")
                    or str(c.get("canonical_passage_id", "")).endswith("_rank1")
                    or str(c.get("canonical_passage_id", "")).endswith("_rank2")
                    or str(c.get("canonical_passage_id", "")).endswith("_rank3")
                    for c in cands_list
                )

                is_real_qrels = (
                    qrels_set is not None
                    and "Mock" not in type(qrels_set).__name__
                    and getattr(qrels_set, "qrels_sha256", None) == "9c83aa9dc75924f5d9942cc2d6fb518368f2ab34f95306f080dbb111b4138d3e"
                )
                is_production_run = (
                    is_real_qrels
                    and pdf_path.exists()
                    and sha256_file(pdf_path) == PDF_SHA256_EXPECTED
                )

                if is_production_run and retrieved_cnt > 0 and (
                    mapped_cnt != retrieved_cnt
                    or unresolved_cnt > 0
                    or has_synthetic_pid
                ):

                    logger.error(
                        "PRODUCTION_CANONICAL_MAPPING_FAILED: Strategy %s QID %s produced %d unresolved mapping(s)",
                        strategy_label,
                        qid,
                        unresolved_cnt,
                    )
                    print("PRODUCTION_CANONICAL_MAPPING_FAILED")
                    raise ValueError(
                        f"PRODUCTION_CANONICAL_MAPPING_FAILED: Strategy {strategy_label} QID {qid}"
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

            # ── Lossless Answer Integrity Verification ────────────
            evaluated_text = answer.text
            persisted_text = str(sanitized_answer.get("text", ""))
            persisted_sha = str(sanitized_answer.get("text_sha256", ""))
            computed_sha = hashlib.sha256(persisted_text.encode("utf-8")).hexdigest()
            if (
                evaluated_text != persisted_text
                or persisted_sha != computed_sha
                or sanitized_answer.get("text_length_chars") != len(evaluated_text)
                or sanitized_answer.get("truncated") is not False
            ):
                raise ValueError(
                    f"EVALUATED_ANSWER_ARTIFACT_MISMATCH: query_id={query_id} "
                    f"evaluated text, persisted text, or SHA-256 diverged"
                )

            # ── Citation Provenance Auditability ─────────────────
            citation_mapping_status, citation_map = build_citation_map_and_status(
                answer_text=evaluated_text,
                abstained=answer.abstained,
                evidence=evidence_record["candidates"],
                query_id=query_id,
                citations=answer.citations,
            )

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
            logical_external_requests = gen_calls + cr_calls + gr_calls + ar_calls

            quota_after = int(shared_quota.stats["total_requests"])
            retries_after = int(shared_quota.stats.get("total_retries", 0))
            r429_after = int(shared_quota.stats.get("rate_limit_429_count", 0))
            r5xx_after = int(shared_quota.stats.get("server_5xx_retry_count", 0))
            rother_after = int(shared_quota.stats.get("other_retryable_error_count", 0))

            physical_http_attempts = quota_after - quota_before
            retry_attempts = retries_after - retries_before
            rate_limit_429_count = r429_after - r429_before
            server_5xx_retry_count = r5xx_after - r5xx_before
            other_retryable_error_count = rother_after - rother_before

            successful_http_responses = logical_external_requests
            failed_http_attempts = retry_attempts

            call_ledger = {
                "generation_calls": gen_calls,
                "context_relevance_calls": cr_calls,
                "groundedness_calls": gr_calls,
                "answer_relevance_calls": ar_calls,
                "total_external_requests": logical_external_requests,
                "physical_http_attempts": physical_http_attempts,
                "successful_http_responses": successful_http_responses,
                "failed_http_attempts": failed_http_attempts,
                "retry_attempts": retry_attempts,
                "rate_limit_429_count": rate_limit_429_count,
                "server_5xx_retry_count": server_5xx_retry_count,
                "other_retryable_error_count": other_retryable_error_count,
            }

            if (
                retry_attempts < 0
                or rate_limit_429_count < 0
                or server_5xx_retry_count < 0
                or other_retryable_error_count < 0
            ):
                raise ValueError(
                    f"EXTERNAL_CALL_ACCOUNTING_MISMATCH: query_id={query_id} "
                    f"negative counters (retries={retry_attempts}, 429={rate_limit_429_count}, "
                    f"5xx={server_5xx_retry_count}, other={other_retryable_error_count})"
                )

            causal_sum = rate_limit_429_count + server_5xx_retry_count + other_retryable_error_count
            if retry_attempts != causal_sum:
                raise ValueError(
                    f"EXTERNAL_CALL_ACCOUNTING_MISMATCH: query_id={query_id} "
                    f"causal sum mismatch: retry_attempts={retry_attempts} != causal_sum={causal_sum} "
                    f"(429={rate_limit_429_count}, 5xx={server_5xx_retry_count}, other={other_retryable_error_count})"
                )

            if physical_http_attempts < logical_external_requests:
                raise ValueError(
                    f"EXTERNAL_CALL_ACCOUNTING_MISMATCH: query_id={query_id} "
                    f"physical_http_attempts={physical_http_attempts} < logical_external_requests={logical_external_requests}"
                )

            if physical_http_attempts != logical_external_requests + retry_attempts:
                raise ValueError(
                    f"EXTERNAL_CALL_ACCOUNTING_MISMATCH: query_id={query_id} "
                    f"physical={physical_http_attempts}, logical={logical_external_requests}, retries={retry_attempts}"
                )

            if physical_http_attempts != successful_http_responses + failed_http_attempts:
                raise ValueError(
                    f"EXTERNAL_CALL_ACCOUNTING_MISMATCH: query_id={query_id} "
                    f"physical={physical_http_attempts}, successful={successful_http_responses}, failed={failed_http_attempts}"
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
            citation_pages = [c["page_number"] for c in citation_map] if citation_map else []

            # ── Ground Truth v2 & Legacy Page Metrics Adaptation (Gate A) ──────
            gt_item = migrate_legacy_qrel_item(q)

            retrieved_pages_raw = [
                cand.get("page_number") if isinstance(cand, dict) else getattr(cand, "page_number", None)
                for cand in evidence_record.get("candidates", [])
            ]
            retrieved_pages_valid = [p for p in retrieved_pages_raw if p is not None and isinstance(p, int)]

            legacy_page_metrics = compute_legacy_page_metrics(
                retrieved_pages=retrieved_pages_valid,
                relevant_pages=relevant_pages,
                cited_pages=citation_pages,
            )

            # ── TAREFA 4: Pre-Rerank Candidates & Reranker Damage ───
            pre_candidates = getattr(retriever, "pre_rerank_candidates", None)
            if pre_candidates is not None:
                pre_mapped_records = [
                    map_candidate_to_canonical(
                        pc, mapper, rank=idx + 1, qrels_set=qrels_set, question_id=qid
                    )
                    for idx, pc in enumerate(pre_candidates)
                ]
                pre_rerank_pids: list[str | None] | None = [
                    r["canonical_passage_id"] for r in pre_mapped_records
                ]
            else:
                pre_rerank_pids = None

            post_rerank_pids = [
                c["canonical_passage_id"]
                for c in evidence_record.get("candidates", [])
                if c.get("canonical_passage_id")
            ]

            human_qrels_metrics = compute_human_qrels_metrics_for_question(
                qrels_set=qrels_set,
                question_id=qid,
                retrieved_passage_ids=post_rerank_pids,
                k=TOP_K,
                candidate_passage_ids_pre_rerank=pre_rerank_pids,
            )

            # ── TAREFA 6: Annotation Completeness Alignment ───
            ground_truth_record = {
                "contract_version": "v2",
                "source_schema": "human_qrels_v2",
                "provenance_status": "HUMAN_ADJUDICATED_AND_CONSENSUS",
                "annotation_completeness": {
                    "passage_qrels_present": True,
                    "graded_qrels_present": True,
                    "gold_answer_present": False,
                    "nuggets_present": False,
                    "adjudication_present": qrels_set.adjudicated_count > 0,
                },
                "answerable": not is_abstention,
                "unanswerable_reason": (
                    UnanswerableReason.EXPLICIT_ABSTENTION_REQUIRED.value
                    if is_abstention
                    else None
                ),
                "legacy_relevant_pages": list(gt_item.legacy_relevant_pages),
                "passage_qrels_status": "HUMAN_ANNOTATED_GRADED",
                "graded_qrels_status": "HUMAN_ANNOTATED_GRADED",
                "gold_answer_status": "LEGACY_EXPERT_REFERENCE_SUMMARY",
                "reference_answer_provenance": "LEGACY_EXPERT_REFERENCE_SUMMARY",
            }

            qrels_rel_path = (
                str(qrels_p.relative_to(_REPO_ROOT))
                if qrels_p.is_relative_to(_REPO_ROOT)
                else str(qrels_p)
            )

            # ── TAREFA 5 & 8: Generation Evaluation & Schema v5 Record ───
            evaluation_record = {
                "protocol_version": PROTOCOL_VERSION,
                "artifact_schema_version": _EVAL_SCHEMA_VERSION,
                "schema_version": _EVAL_SCHEMA_VERSION,
                "canonical_mapping_status": "PASSED",
                "mapped_count": evidence_record.get(
                    "mapped_count", len(evidence_record.get("candidates", []))
                ),
                "unresolved_mapping_count": 0,
                "judged_coverage_rate": human_qrels_metrics.get("retrieval_accounting", {}).get("judged_coverage_rate", 0.0),

                "qrels_path": qrels_rel_path,
                "qrels_sha256": qrels_set.qrels_sha256,
                "qrels_manifest_sha256": qrels_set.manifest_sha256,
                "qrels_schema_version": qrels_set.schema_version,
                "qrels_authority": "HUMAN_VALIDATED_GRADED_PASSAGE_RELEVANCE",
                "relevance_threshold": 1,
                "unjudged_policy": "EXPLICIT_UNJUDGED_DISTINCT_FROM_ZERO",
                "canonical_evaluation_unit": "PASSAGE_LEVEL",
                "retrieval_evaluation": human_qrels_metrics,
                "generation_evaluation": {
                    "groundedness": next(
                        (m for m in evaluation_metrics if m.get("name") == "groundedness"), None
                    ),
                    "answer_relevance": next(
                        (m for m in evaluation_metrics if m.get("name") == "answer_relevance"), None
                    ),
                    "context_relevance": next(
                        (m for m in evaluation_metrics if m.get("name") == "context_relevance"), None
                    ),
                    "abstention_correctness": next(
                        (m for m in evaluation_metrics if m.get("name") == "abstention_correctness"), None
                    ),
                },
                "reference_answer_provenance": "LEGACY_EXPERT_REFERENCE_SUMMARY",
                "metrics": evaluation_metrics,
                "legacy_page_metrics": legacy_page_metrics,
                "deterministic_v2_metrics": human_qrels_metrics["metrics"],
                "rag_triad": evaluation_metrics,
            }


            result_entry = {
                "qid": qid,
                "split": q["split"],
                "strategy": strategy_label,
                "retrieval_configuration": retrieval_config,
                "retrieval_configuration_sha256": retrieval_config_sha,
                "relevant_pages": relevant_pages,
                "abstained": answer.abstained,
                "is_abstention_question": is_abstention,
                "answer": sanitized_answer,
                "citation_mapping_status": citation_mapping_status,
                "citation_map": citation_map,
                "ground_truth": ground_truth_record,
                "evaluation": evaluation_record,
                "retrieval_evidence": evidence_record,
                "citation_pages": citation_pages,
                "call_ledger": call_ledger,
                "quota_stats": shared_quota.stats,
            }
            strategy_results.append(result_entry)
            ckpt.mark_complete_row(qid, strategy_label, result_entry)

        elapsed_ms = (time.monotonic() - t0) * 1000
        run_times[strategy_label] = round(elapsed_ms, 1)
        all_results[strategy_label] = strategy_results
        logger.info("  %s complete in %.1f ms", strategy_label, elapsed_ms)

    # ── Pre-Materialization Invariant Validation ─────────────────────────
    expected_total_rows = len(strategy_labels) * len(questions)
    all_rows: list[dict[str, Any]] = []

    for strat in strategy_labels:
        strat_rows = all_results.get(strat, [])
        if not strat_rows:
            raise ValueError(f"FINAL_ARTIFACT_INCOMPLETE: strategy '{strat}' has 0 result rows")
        if len(strat_rows) != len(questions):
            raise ValueError(
                f"FINAL_ARTIFACT_INCOMPLETE: strategy '{strat}' has {len(strat_rows)} rows, expected {len(questions)}"
            )
        all_rows.extend(strat_rows)

    if len(all_rows) != expected_total_rows:
        raise ValueError(
            f"FINAL_ARTIFACT_INCOMPLETE: expected {expected_total_rows} complete result rows, "
            f"got {len(all_rows)}. Aborting final artifact materialization."
        )

    # 1. No duplicate (qid, strategy) pairs
    seen_pairs = set()
    for r in all_rows:
        pair_key = f"{r.get('qid')}::{r.get('strategy')}"
        if pair_key in seen_pairs:
            raise ValueError(f"DUPLICATE_RESULT_ROW: pair '{pair_key}' appears multiple times")
        seen_pairs.add(pair_key)

    # 2. No holdout questions
    for r in all_rows:
        if "holdout" in str(r.get("qid", "")):
            raise ValueError(f"HOLDOUT_QUESTION_DETECTED: {r.get('qid')}")

    # 3. Expected QIDs match
    expected_qids = {q["qid"] for q in questions}
    for r in all_rows:
        if r.get("qid") not in expected_qids:
            raise ValueError(f"UNKNOWN_QID_DETECTED: {r.get('qid')}")

    # 4. Hashes and citations valid
    for r in all_rows:
        if r.get("citation_mapping_status") == "AVAILABLE" and not r.get("citation_pages"):
            raise ValueError(f"SUBSTANTIVE_ANSWER_MISSING_CITATIONS: {r.get('qid')}")

    # Write sanitized results
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = RESULTS_DIR / f"slice4_results_{run_id}_{ts}.json"
    all_retrieval_configs = {s: build_retrieval_configuration(s) for s in VALID_STRATEGIES}
    all_retrieval_config_shas = {s: compute_retrieval_configuration_sha256(c) for s, c in all_retrieval_configs.items()}

    qrels_rel_path = (
        str(qrels_p.relative_to(_REPO_ROOT))
        if qrels_p.is_relative_to(_REPO_ROOT)
        else str(qrels_p)
    )
    output = {
        "experiment_id": run_id,
        "protocol_version": PROTOCOL_VERSION,
        "artifact_schema_version": _EVAL_SCHEMA_VERSION,
        "schema": _EVAL_SCHEMA_VERSION,
        "gemini_model": GEMINI_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "qrels_path": qrels_rel_path,
        "qrels_sha256": qrels_set.qrels_sha256,
        "qrels_manifest_sha256": qrels_set.manifest_sha256,
        "qrels_schema_version": qrels_set.schema_version,
        "qrels_authority": "HUMAN_VALIDATED_GRADED_PASSAGE_RELEVANCE",
        "relevance_threshold": 1,
        "unjudged_policy": "EXPLICIT_UNJUDGED_DISTINCT_FROM_ZERO",
        "canonical_evaluation_unit": "PASSAGE_LEVEL",
        "qrels_total_pairs": qrels_set.total_pairs,
        "qrels_consensus_count": qrels_set.consensus_count,
        "qrels_adjudicated_count": qrels_set.adjudicated_count,
        "qrels_grade_distribution": qrels_set.grade_distribution,
        "reference_answer_provenance": "LEGACY_EXPERT_REFERENCE_SUMMARY",
        "embedding_fingerprints": embedding_fps,
        "manifest_fingerprint": manifest.get("cache_tree_sha256", "UNRESOLVED"),
        "retrieval_configurations": all_retrieval_configs,
        "retrieval_configuration_sha256": all_retrieval_config_shas,
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
    logger.info("Resume Complete: all %d complete result rows rehydrated and validated.", len(all_rows))
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

        # Must have citation mapping status AVAILABLE
        cms = result.get("citation_mapping_status")
        if cms != "AVAILABLE":
            failures.append(f"CITATION_MAPPING_STATUS_NOT_AVAILABLE: {cms}")

        citation_map = result.get("citation_map", [])
        if not citation_map:
            failures.append("EMPTY_CITATION_MAP_ON_SUBSTANTIVE")

        cand_map = {
            c.get("chunk_id"): c
            for c in evidence_rec.get("candidates", [])
            if c.get("chunk_id")
        }

        for cit in citation_map:
            cp = cit.get("page_number")
            if cp is None or isinstance(cp, bool) or not isinstance(cp, int) or cp < 1:
                failures.append(f"CITATION_PAGE_INVALID: page_number={cp!r} in marker {cit.get('marker')}")
                continue

            matching_cand = cand_map.get(cit.get("chunk_id"))
            if matching_cand is None:
                failures.append(f"CITATION_CHUNK_NOT_IN_EVIDENCE: {cit.get('chunk_id')}")
            else:
                if cit.get("text_sha256") != matching_cand.get("text_sha256"):
                    failures.append(f"CITATION_SHA_MISMATCH: {cit.get('marker')}")
                if cit.get("page_number") != matching_cand.get("page_number"):
                    failures.append(f"CITATION_PAGE_MISMATCH: {cit.get('marker')}")

        evidence_pages = {
            c.get("page_number")
            for c in evidence_rec.get("candidates", [])
            if c.get("page_number") is not None
        }
        for cp in citation_pages:
            if cp < 1 or isinstance(cp, bool) or not isinstance(cp, int):
                failures.append(f"CITATION_PAGE_INVALID_IN_LIST: page={cp!r}")
            elif cp not in evidence_pages:
                failures.append(f"CITATION_NOT_IN_EVIDENCE: page={cp}")

    # 7. Retrieval evidence must exist
    if "retrieval_evidence" not in result:
        failures.append("MISSING_RETRIEVAL_EVIDENCE")

    # 8. Lossless answer contract validation
    answer_obj = result.get("answer", {})
    if answer_obj.get("truncated") is not False:
        failures.append("ANSWER_TRUNCATED")
    text_val = str(answer_obj.get("text", ""))
    text_sha = str(answer_obj.get("text_sha256", ""))
    if not text_sha or hashlib.sha256(text_val.encode("utf-8")).hexdigest() != text_sha:
        failures.append("ANSWER_TEXT_SHA256_MISMATCH")

    # 9. Result and checkpoint valid
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
        qrels_path=getattr(args, "qrels_path", None),
        qrels_manifest=getattr(args, "qrels_manifest", None),
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
        qrels_path=getattr(args, "qrels_path", None),
        qrels_manifest=getattr(args, "qrels_manifest", None),
    )
    logger.info("=== Slice 4 Full Benchmark Complete: %s ===", output_path)


def cmd_resume(args: argparse.Namespace, pdf_path: Path, logger: logging.Logger) -> None:
    run_id = args.run_id
    if not run_id:
        logger.error("--mode resume requires --run-id. Example: --run-id %s", EXPERIMENT_ID)
        sys.exit(3)

    # Validate exact checkpoint file exists for run_id (no glob matching smoke files)
    exact_ckpt_path = CHECKPOINT_DIR / f"slice4_gen_checkpoint_{run_id}.json"
    if not exact_ckpt_path.exists():
        logger.error(
            "CHECKPOINT_NOT_FOUND: No exact checkpoint file found at %s. "
            "Available: %s",
            exact_ckpt_path,
            list(CHECKPOINT_DIR.glob("*.json")),
        )
        raise ValueError(f"CHECKPOINT_NOT_FOUND: {exact_ckpt_path}")

    logger.info("=== RESUME: run_id=%s (checkpoint: %s) ===", run_id, exact_ckpt_path)
    output_path = run_benchmark(
        run_id=run_id,
        questions=ACTIVE_QUESTIONS,
        strategy_labels=VALID_STRATEGIES,
        logger=logger,
        pdf_path=pdf_path,
        qrels_path=getattr(args, "qrels_path", None),
        qrels_manifest=getattr(args, "qrels_manifest", None),
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

def cmd_preflight_human_qrels(
    args: argparse.Namespace, logger: logging.Logger
) -> None:
    """Validate human-validated graded qrels & real retriever mapping offline — no Gemini key, no network."""
    logger.info("=== PREFLIGHT-HUMAN-QRELS: Human Qrels Integrity & Real Retriever Validation ===")

    _validate_no_credentials_for_preflight(logger)

    qrels_p = Path(args.qrels_path)
    manifest_p = Path(args.qrels_manifest)

    try:
        qrels_set = load_human_qrels_set(qrels_p, manifest_p)
    except ValueError as exc:
        logger.error("HUMAN_QRELS_PREFLIGHT_FAILED: %s", exc)
        print("HUMAN_QRELS_REQUIRED_OR_INVALID")
        sys.exit(1)

    # TAREFA 4 — Check canonical_evaluation_unit
    manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
    unit = manifest_data.get("canonical_evaluation_unit", "PASSAGE_LEVEL")
    if unit != "PASSAGE_LEVEL":
        logger.error("CANONICAL_EVALUATION_UNIT_MISMATCH: manifest unit=%s, expected PASSAGE_LEVEL", unit)
        print("CANONICAL_EVALUATION_UNIT_MISMATCH")
        sys.exit(1)
    canonical_evaluation_unit = "PASSAGE_LEVEL"
    logger.info("  Canonical Evaluation Unit: %s", canonical_evaluation_unit)

    logger.info("Human Qrels dataset loaded successfully:")
    logger.info("  Total pairs: %d", qrels_set.total_pairs)
    logger.info("  Consensus count: %d", qrels_set.consensus_count)
    logger.info("  Adjudicated count: %d", qrels_set.adjudicated_count)
    logger.info("  Grade distribution: %s", qrels_set.grade_distribution)
    logger.info("  SHA-256 digest: %s", qrels_set.qrels_sha256)

    q4_items = qrels_set.get_qrels_for_question("q_test_04")
    if len(q4_items) != 10 or not qrels_set.is_abstention_question("q_test_04"):
        logger.error("HUMAN_QRELS_PREFLIGHT_FAILED: q_test_04 invalid negative control state")
        print("HUMAN_QRELS_REQUIRED_OR_INVALID")
        sys.exit(1)
    logger.info("  q_test_04 negative control audit: 10/10 OK")

    # Resolve PDF Path
    pdf_path_str = getattr(args, "pdf_path", None) or os.environ.get("RAGLAB_PDF_PATH")
    if not pdf_path_str:
        default_pdf = (
            _REPO_ROOT.parent
            / "Fundamentos matemáticos para a ciência da computação Matemática Discreta e Suas Aplicações (Judith L. Gersting).pdf"
        )
        if default_pdf.exists():
            pdf_path_str = str(default_pdf)

    if not pdf_path_str:
        logger.error(
            "HUMAN_QRELS_PREFLIGHT_FAILED: PDF path not specified. Use --pdf-path or set RAGLAB_PDF_PATH."
        )
        sys.exit(1)

    pdf_path = Path(pdf_path_str)
    verify_pdf(pdf_path, logger)

    pages = load_pdf_pages(pdf_path, logger)
    embed_model = load_embedding_model(logger)

    from raglab.evaluation.pooling.canonical_passage_mapper import (
        CanonicalPassageMapper,
    )

    registry_path = _REPO_ROOT / "benchmarks" / "ground_truth" / "v2" / "passage_registry.jsonl"
    mapper = CanonicalPassageMapper.from_registry_file(registry_path)

    question = next(q for q in ACTIVE_QUESTIONS if q["qid"] == "q_dev_01")
    # Holdout safety check
    assert not question["qid"].startswith("q_holdout_"), "HOLDOUT_LEAKAGE: Holdout question detected in preflight"

    def _extract_candidate_info(c: Any) -> tuple[str, int, str]:
        raw_id = getattr(c, "chunk_id", None)
        if raw_id is None and isinstance(c, dict):
            raw_id = c.get("chunk_id") or c.get("passage_id")
        if raw_id is None:
            raw_id = str(c)

        doc_id = getattr(c, "document_id", None)
        if doc_id is None and isinstance(c, dict):
            doc_id = c.get("document_id")
        doc_id_str = str(doc_id) if doc_id else "gersting_discrete_math"

        page_num = getattr(c, "page_number", None) or getattr(c, "start_page", None)
        if page_num is None and isinstance(c, dict):
            page_num = c.get("page_number") or c.get("start_page")
        if page_num is None:
            import re
            for s in (doc_id_str, str(raw_id)):
                m = re.search(r"_p(\d+)", s)
                if m:
                    page_num = int(m.group(1))
                    break

        text = getattr(c, "text", "")
        if not text and isinstance(c, dict):
            text = c.get("text", "")

        return str(raw_id), int(page_num) if page_num is not None else 0, text

    strategy_table_rows: list[dict[str, Any]] = []

    for strategy in VALID_STRATEGIES:
        retrievers = build_retrievers(pages, embed_model, strategies=(strategy,))
        retriever = retrievers[strategy]
        candidates = retriever.retrieve(question["query"], top_k=3)

        if not candidates:
            logger.error("REAL_RETRIEVER_MAPPING_FAILED: Strategy %s returned 0 candidates", strategy)
            print("REAL_RETRIEVER_MAPPING: BLOCKED")
            sys.exit(1)

        mapped_pids: list[str | None] = []
        unresolved_count = 0
        judged_count = 0
        mapped_count = 0

        for idx, c in enumerate(candidates):
            rec = map_candidate_to_canonical(
                c, mapper, rank=idx + 1, qrels_set=qrels_set, question_id=question["qid"]
            )
            canon_pid = rec["canonical_passage_id"]
            if not canon_pid or not canon_pid.startswith("ps_"):
                unresolved_count += 1
                mapped_pids.append("UNMAPPED_NEEDS_REVIEW")
            else:
                mapped_count += 1
                mapped_pids.append(canon_pid)
                if rec["judged_status"] == "JUDGED":
                    judged_count += 1

        if unresolved_count > 0:
            logger.error(
                "REAL_RETRIEVER_MAPPING_FAILED: Strategy %s produced %d unresolved candidate(s)",
                strategy,
                unresolved_count,
            )
            print("REAL_RETRIEVER_MAPPING: BLOCKED")
            sys.exit(1)

        pre_candidates = getattr(retriever, "pre_rerank_candidates", None)
        pre_mapped_pids: list[str | None] | None = None
        if pre_candidates is not None:
            pre_mapped_pids = []
            for pc_idx, pc in enumerate(pre_candidates):
                pc_rec = map_candidate_to_canonical(
                    pc, mapper, rank=pc_idx + 1, qrels_set=qrels_set, question_id=question["qid"]
                )
                pre_mapped_pids.append(pc_rec["canonical_passage_id"])
            pre_count = len(pre_candidates)
        else:
            pre_count = 0



        metric_res = compute_human_qrels_metrics_for_question(
            qrels_set=qrels_set,
            question_id=question["qid"],
            retrieved_passage_ids=mapped_pids,
            k=3,
            candidate_passage_ids_pre_rerank=pre_mapped_pids,
        )

        m_dict = metric_res["metrics"]
        ndcg_score = m_dict["ndcg_at_k"]["score"]
        recall_score = m_dict["recall_at_k"]["score"]
        mrr_score = m_dict["mrr_at_k"]["score"]
        judged_cov = metric_res["retrieval_accounting"]["judged_coverage_rate"]


        if pre_candidates is not None:
            rd = metric_res.get("reranker_damage")
            if rd and rd.get("dropped_relevant_count", 0) > 0:
                rd_status = "DAMAGE_DETECTED"
            else:
                rd_status = "NO_DAMAGE"
        else:
            rd_status = "NOT_APPLICABLE"

        strategy_table_rows.append({
            "strategy": strategy,
            "retrieved_count": len(candidates),
            "mapped_count": mapped_count,
            "unresolved_count": unresolved_count,
            "judged_count": judged_count,
            "judged_coverage_at_3": round(judged_cov, 4),
            "ndcg_at_3": ndcg_score if ndcg_score is not None else "NOT_APPLICABLE",
            "recall_at_3": recall_score if recall_score is not None else "NOT_APPLICABLE",
            "mrr_at_3": mrr_score if mrr_score is not None else "NOT_APPLICABLE",
            "pre_rerank_count": pre_count,
            "post_rerank_count": len(candidates),
            "reranker_damage_status": rd_status,
        })

    # Validate embedding parity across all built retrievers
    all_built = build_retrievers(pages, embed_model, strategies=VALID_STRATEGIES)
    verify_embedding_parity(all_built, logger)

    # Output Summary Table
    table_header = (
        "| strategy | retrieved_count | mapped_count | unresolved_count | judged_count | "
        "judged_coverage_at_3 | ndcg_at_3 | recall_at_3 | mrr_at_3 | pre_rerank_count | "
        "post_rerank_count | reranker_damage_status |"
    )
    table_sep = (
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    logger.info("\n=== Real Retriever Preflight Mapping Results ===")
    logger.info(table_header)
    logger.info(table_sep)
    for row in strategy_table_rows:
        logger.info(
            "| %s | %d | %d | %d | %d | %.4f | %s | %s | %s | %d | %d | %s |",
            row["strategy"],
            row["retrieved_count"],
            row["mapped_count"],
            row["unresolved_count"],
            row["judged_count"],
            row["judged_coverage_at_3"],
            str(row["ndcg_at_3"]),
            str(row["recall_at_3"]),
            str(row["mrr_at_3"]),
            row["pre_rerank_count"],
            row["post_rerank_count"],
            row["reranker_damage_status"],
        )

    preflight_report = {
        "preflight_status": "REAL_RETRIEVER_HUMAN_QRELS_PREFLIGHT_PASSED",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "canonical_evaluation_unit": canonical_evaluation_unit,
        "qrels_path": str(qrels_p),
        "qrels_sha256": qrels_set.qrels_sha256,
        "qrels_manifest_sha256": qrels_set.manifest_sha256,
        "total_pairs": qrels_set.total_pairs,
        "consensus_pairs_count": qrels_set.consensus_count,
        "adjudicated_pairs_count": qrels_set.adjudicated_count,
        "grade_distribution": qrels_set.grade_distribution,
        "q_test_04_negative_control_count": len(q4_items),
        "authoritative_for_evaluation": True,
        "silver_used_as_ground_truth": False,
        "holdout_sealed": True,
        "strategies_evaluated": strategy_table_rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / f"preflight_human_qrels_{int(time.time())}.json"
    report_path.write_text(json.dumps(preflight_report, indent=2), encoding="utf-8")
    logger.info("Saved preflight report artifact: %s", report_path)

    logger.info("")
    logger.info("REAL_RETRIEVER_HUMAN_QRELS_PREFLIGHT_PASSED")




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
    if args.mode == "preflight-human-qrels":
        cmd_preflight_human_qrels(args, logger)
        return

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
