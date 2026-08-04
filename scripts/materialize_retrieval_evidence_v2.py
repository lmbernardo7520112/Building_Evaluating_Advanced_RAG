#!/usr/bin/env python3
"""Materialize full retrieval evidence v2 offline.

Runs all 7 retriever strategies against all 8 active questions using:
- the same PDF (verified by SHA-256);
- the same page range (91–115);
- the same embedding model from local cache;
- the same parser, chunker, and retriever configurations;
- NO LLM generator, NO judge, NO API calls, NO credentials.

Output:  benchmarks/results/retrieval_evidence_v2.json
Schema:  retrieval_evidence_v2  (one record per candidate)

Each candidate record contains the FULL text from node.get_content() /
node["window_text"] / node["anchor_text"] — never a truncated preview.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ─── Path setup ─────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

# ─── Constants (identical to run_slice4_benchmark.py) ───────────
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

VALID_STRATEGIES = (
    "F0_baseline",
    "S0_sentence_anchor",
    "W0_sentence_window",
    "W1_sentence_window_rerank",
    "H0_hierarchical_leaf",
    "H1_auto_merging",
    "H2_auto_merging_rerank",
)

ACTIVE_QUESTIONS: list[dict[str, Any]] = [
    {"qid": "q_dev_01", "split": "development",
     "query": "O que é demonstração por exaustão e quando é aplicável?"},
    {"qid": "q_dev_02", "split": "development",
     "query": "Como o método de prova por contradição funciona em matemática discreta?"},
    {"qid": "q_dev_03", "split": "development",
     "query": "Quais são as etapas do princípio da indução matemática?"},
    {"qid": "q_dev_04", "split": "development",
     "query": "Como funciona a indução forte comparada à indução fraca?"},
    {"qid": "q_test_01", "split": "test",
     "query": "Qual é a diferença entre indução fraca e indução forte?"},
    {"qid": "q_test_02", "split": "test",
     "query": "Como se define a base e o passo indutivo em demonstração por indução?"},
    {"qid": "q_test_03", "split": "test",
     "query": "Quais são os passos para provar uma afirmação usando indução completa?"},
    {"qid": "q_test_04", "split": "test",
     "query": "Qual é a capital da França?",
     "abstention_expected": True},
]

HOLDOUT_QIDS = frozenset({"q_holdout_01", "q_holdout_02"})

RESULTS_DIR = _REPO_ROOT / "benchmarks" / "results"
OUTPUT_PATH = RESULTS_DIR / "retrieval_evidence_v2.json"


# ─── Helpers ────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_dict(d: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _classify_retrieval_family(strategy: str) -> str:
    families = {
        "F0_baseline": "fixed_chunk",
        "S0_sentence_anchor": "sentence_anchor",
        "W0_sentence_window": "sentence_window",
        "W1_sentence_window_rerank": "sentence_window_rerank",
        "H0_hierarchical_leaf": "hierarchical_leaf",
        "H1_auto_merging": "auto_merging",
        "H2_auto_merging_rerank": "auto_merging_rerank",
    }
    return families.get(strategy, "unknown")


def _classify_node_type(strategy: str, is_reranked: bool = False) -> str:
    """Classify the raw node type that the retriever returns."""
    if strategy in ("F0_baseline",):
        return "fixed_chunk"
    if strategy in ("S0_sentence_anchor",):
        return "sentence_anchor"
    if strategy in ("W0_sentence_window", "W1_sentence_window_rerank"):
        return "sentence_window"
    if strategy in ("H0_hierarchical_leaf",):
        return "hierarchical_leaf"
    if strategy in ("H1_auto_merging", "H2_auto_merging_rerank"):
        return "auto_merged_or_leaf"
    return "unknown"


def _resolve_page_number(evidence: object) -> int | None:
    """Extract page number from evidence.document_id or chunk_id."""
    import re
    doc_id = getattr(evidence, "document_id", "")
    if "_p" in doc_id:
        try:
            part = doc_id.rsplit("_p", 1)[1]
            # Remove any trailing suffixes after the number
            num = re.match(r"(\d+)", part)
            if num:
                return int(num.group(1))
        except (ValueError, IndexError):
            pass
    # Fallback: try chunk_id
    chunk_id = getattr(evidence, "chunk_id", None)
    cid_val = chunk_id.value if hasattr(chunk_id, "value") else str(chunk_id or "")
    m = re.search(r"_p(\d+)_", cid_val)
    if m:
        return int(m.group(1))
    return None


# ─── Main ───────────────────────────────────────────────────────

def main() -> None:
    # ── Security: reject credentials ─────────────────────────────
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(var):
            print(f"ERROR: {var} is set. This script must run without credentials.")
            sys.exit(1)

    # ── Reject holdout ───────────────────────────────────────────
    for q in ACTIVE_QUESTIONS:
        if q["qid"] in HOLDOUT_QIDS:
            print(f"ERROR: Holdout question {q['qid']} found in ACTIVE_QUESTIONS")
            sys.exit(1)

    # ── Resolve PDF ──────────────────────────────────────────────
    pdf_path_str = os.environ.get("RAGLAB_PDF_PATH")
    if not pdf_path_str:
        # Try common location
        candidate = (
            _REPO_ROOT.parent
            / "Fundamentos matemáticos para a ciência da computação "
              "Matemática Discreta e Suas Aplicações (Judith L. Gersting).pdf"
        )
        if candidate.exists():
            pdf_path_str = str(candidate)
        else:
            print("ERROR: RAGLAB_PDF_PATH not set and PDF not found at default location")
            sys.exit(1)
    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)
    actual_sha = sha256_file(pdf_path)
    if actual_sha != PDF_SHA256_EXPECTED:
        print(f"ERROR: PDF SHA-256 mismatch: {actual_sha} != {PDF_SHA256_EXPECTED}")
        sys.exit(1)
    print(f"PDF verified: {pdf_path.name} SHA-256={actual_sha[:16]}...")

    # ── Load pages ───────────────────────────────────────────────
    from raglab.infrastructure.pdf_parsers.pdf_parser_adapter import (
        PyPdfExtractorAdapter,
    )
    adapter = PyPdfExtractorAdapter()
    pages = adapter.read_document(str(pdf_path), page_start=PAGES_START, page_end=PAGES_END)
    print(f"Extracted {len(pages)} pages (pages {PAGES_START}–{PAGES_END})")
    corpus_sha = sha256_text(
        "".join(p.text for p in sorted(pages, key=lambda p: p.page_number))
    )

    # ── Load embedding model (from local cache only) ─────────────
    from raglab.infrastructure.embeddings.fastembed_adapter import (
        FastEmbedEmbeddingAdapter,
        resolve_cache_dir,
    )
    cache_dir = str(resolve_cache_dir())
    embed_model = FastEmbedEmbeddingAdapter(
        model_name=EMBEDDING_MODEL,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    print(f"Embedding model loaded: {EMBEDDING_MODEL}")
    embed_fingerprint = {
        "model_id": EMBEDDING_MODEL,
        "dimension": embed_model.dimension,
    }

    # ── Build retrievers (same code as run_slice4_benchmark) ─────
    from benchmarks.run_slice4_benchmark import (
        build_retrieval_configuration,
        build_retrievers,
        compute_retrieval_configuration_sha256,
    )

    retrievers = build_retrievers(pages, embed_model, strategies=VALID_STRATEGIES)
    print(f"Built {len(retrievers)} retrievers")

    # ── Materialize evidence ─────────────────────────────────────
    all_records: list[dict[str, Any]] = []
    experiment_id = "raglab_v7_slice4_v2_20260731T1230UTC"
    t0 = time.monotonic()

    for strategy_label in VALID_STRATEGIES:
        retriever = retrievers[strategy_label]
        retrieval_family = _classify_retrieval_family(strategy_label)
        node_type = _classify_node_type(strategy_label)
        retrieval_config = build_retrieval_configuration(strategy_label)
        retriever_config_sha = compute_retrieval_configuration_sha256(retrieval_config)
        is_reranked = strategy_label in ("W1_sentence_window_rerank", "H2_auto_merging_rerank")

        for q in ACTIVE_QUESTIONS:
            qid = q["qid"]
            query = q["query"]

            # For reranked strategies, get both pre- and post-rerank
            if is_reranked and hasattr(retriever, "_base"):
                # Get pre-rerank candidates
                pre_rerank_candidates = retriever._base.retrieve(query, top_k=CANDIDATE_K)
                # Get post-rerank results
                post_rerank_candidates = retriever.retrieve(query, top_k=TOP_K)

                # Build pre-rerank lookup by chunk_id
                pre_rank_map = {
                    c.chunk_id.value: (c.rank, c.score)
                    for c in pre_rerank_candidates
                }

                for rank_idx, ev in enumerate(post_rerank_candidates):
                    cid = ev.chunk_id.value
                    pre_info = pre_rank_map.get(cid, (None, None))
                    full_text = ev.text
                    page_num = _resolve_page_number(ev)

                    record = {
                        "schema": "retrieval_evidence_v2",
                        "experiment_id": experiment_id,
                        "qid": qid,
                        "strategy": strategy_label,
                        "retrieval_family": retrieval_family,
                        "raw_candidate_id": cid,
                        "node_type": node_type,
                        "full_text": full_text,
                        "full_text_sha256": sha256_text(full_text),
                        "text_length": len(full_text),
                        "is_truncated": False,
                        "document_id": ev.document_id,
                        "page_numbers": [page_num] if page_num else [],
                        "start_char": None,
                        "end_char": None,
                        "retrieval_rank": rank_idx + 1,
                        "retrieval_score": round(ev.score, 6) if ev.score is not None else None,
                        "pre_rerank_rank": pre_info[0],
                        "pre_rerank_score": round(pre_info[1], 6) if pre_info[1] is not None else None,
                        "post_rerank_rank": rank_idx + 1,
                        "post_rerank_score": round(ev.score, 6) if ev.score is not None else None,
                        "parent_node_id": None,
                        "child_node_ids": [],
                        "window_metadata": None,
                        "parser_config_sha256": sha256_dict({
                            "pages_start": PAGES_START,
                            "pages_end": PAGES_END,
                            "chunk_size": CHUNK_SIZE,
                        }),
                        "retriever_config_sha256": retriever_config_sha,
                        "embedding_fingerprint": embed_fingerprint,
                        "corpus_sha256": corpus_sha,
                        "source_artifact_sha256": PDF_SHA256_EXPECTED,
                    }
                    all_records.append(record)

                # Emit dropped pre-rerank candidates (ONCE per question)
                post_ids = {c.chunk_id.value for c in post_rerank_candidates}
                for pre_ev in pre_rerank_candidates:
                    if pre_ev.chunk_id.value not in post_ids:
                        pre_full = pre_ev.text
                        pre_page = _resolve_page_number(pre_ev)
                        dropped_record = {
                            "schema": "retrieval_evidence_v2",
                            "experiment_id": experiment_id,
                            "qid": qid,
                            "strategy": strategy_label,
                            "retrieval_family": retrieval_family,
                            "raw_candidate_id": pre_ev.chunk_id.value,
                            "node_type": node_type,
                            "full_text": pre_full,
                            "full_text_sha256": sha256_text(pre_full),
                            "text_length": len(pre_full),
                            "is_truncated": False,
                            "document_id": pre_ev.document_id,
                            "page_numbers": [pre_page] if pre_page else [],
                            "start_char": None,
                            "end_char": None,
                            "retrieval_rank": pre_ev.rank,
                            "retrieval_score": round(pre_ev.score, 6) if pre_ev.score is not None else None,
                            "pre_rerank_rank": pre_ev.rank,
                            "pre_rerank_score": round(pre_ev.score, 6) if pre_ev.score is not None else None,
                            "post_rerank_rank": None,  # dropped by reranker
                            "post_rerank_score": None,
                            "parent_node_id": None,
                            "child_node_ids": [],
                            "window_metadata": None,
                            "parser_config_sha256": sha256_dict({
                                "pages_start": PAGES_START,
                                "pages_end": PAGES_END,
                                "chunk_size": CHUNK_SIZE,
                            }),
                            "retriever_config_sha256": retriever_config_sha,
                            "embedding_fingerprint": embed_fingerprint,
                            "corpus_sha256": corpus_sha,
                            "source_artifact_sha256": PDF_SHA256_EXPECTED,
                        }
                        all_records.append(dropped_record)
            else:
                # Non-reranked strategies
                candidates = retriever.retrieve(query, top_k=TOP_K)
                for ev in candidates:
                    cid = ev.chunk_id.value
                    full_text = ev.text
                    page_num = _resolve_page_number(ev)

                    # For sentence window: the text IS the window, not just anchor
                    # Detect this and add window metadata
                    window_meta = None
                    if strategy_label in ("W0_sentence_window",):
                        window_meta = {
                            "window_size": WINDOW_SIZE,
                            "note": "text is expanded window, not anchor sentence",
                        }

                    record = {
                        "schema": "retrieval_evidence_v2",
                        "experiment_id": experiment_id,
                        "qid": qid,
                        "strategy": strategy_label,
                        "retrieval_family": retrieval_family,
                        "raw_candidate_id": cid,
                        "node_type": node_type,
                        "full_text": full_text,
                        "full_text_sha256": sha256_text(full_text),
                        "text_length": len(full_text),
                        "is_truncated": False,
                        "document_id": ev.document_id,
                        "page_numbers": [page_num] if page_num else [],
                        "start_char": None,
                        "end_char": None,
                        "retrieval_rank": ev.rank,
                        "retrieval_score": round(ev.score, 6) if ev.score is not None else None,
                        "pre_rerank_rank": None,
                        "pre_rerank_score": None,
                        "post_rerank_rank": None,
                        "post_rerank_score": None,
                        "parent_node_id": None,
                        "child_node_ids": [],
                        "window_metadata": window_meta,
                        "parser_config_sha256": sha256_dict({
                            "pages_start": PAGES_START,
                            "pages_end": PAGES_END,
                            "chunk_size": CHUNK_SIZE,
                        }),
                        "retriever_config_sha256": retriever_config_sha,
                        "embedding_fingerprint": embed_fingerprint,
                        "corpus_sha256": corpus_sha,
                        "source_artifact_sha256": PDF_SHA256_EXPECTED,
                    }
                    all_records.append(record)

    elapsed = time.monotonic() - t0

    # ── Build output artifact ────────────────────────────────────
    artifact = {
        "schema": "retrieval_evidence_v2_collection",
        "experiment_id": experiment_id,
        "materialized_utc": datetime.now(UTC).isoformat(),
        "materialization_time_seconds": round(elapsed, 2),
        "strategies": list(VALID_STRATEGIES),
        "questions": [q["qid"] for q in ACTIVE_QUESTIONS],
        "holdout_sealed": True,
        "holdout_qids": sorted(HOLDOUT_QIDS),
        "total_records": len(all_records),
        "corpus_sha256": corpus_sha,
        "source_pdf_sha256": PDF_SHA256_EXPECTED,
        "embedding_fingerprint": embed_fingerprint,
        "passage_registry_note": (
            "The passage_registry.jsonl has 25 entries at PAGE-LEVEL granularity. "
            "Retrievers produce sub-page chunks/sentences/windows/leaves. "
            "This evidence v2 captures the actual retrieval unit, not the page."
        ),
        "records": all_records,
    }

    # Compute artifact-level SHA
    artifact_sha = sha256_text(
        json.dumps(all_records, sort_keys=True, ensure_ascii=False)
    )
    artifact["records_sha256"] = artifact_sha

    # ── Write atomically ─────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = OUTPUT_PATH.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    tmp_path.rename(OUTPUT_PATH)

    # ── Verify written file ──────────────────────────────────────
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        verify = json.load(f)
    assert verify["records_sha256"] == artifact_sha, "SHA-256 verification failed"

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Retrieval Evidence v2 materialized successfully")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Total records: {len(all_records)}")
    print(f"  Records SHA-256: {artifact_sha[:16]}...")
    print(f"  Time: {elapsed:.2f}s")

    # Per-strategy summary
    from collections import Counter
    strat_counts = Counter(r["strategy"] for r in all_records)
    for s in VALID_STRATEGIES:
        c = strat_counts.get(s, 0)
        print(f"  {s}: {c} records")

    # Verify no truncated text
    truncated = [r for r in all_records if r["is_truncated"]]
    preview_only = [r for r in all_records if len(r["full_text"]) <= 80]
    print(f"\n  Truncated records: {len(truncated)}")
    print(f"  Records with text ≤ 80 chars: {len(preview_only)}")
    if preview_only:
        for r in preview_only:
            print(f"    {r['strategy']} {r['qid']} rank={r['retrieval_rank']} len={r['text_length']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
