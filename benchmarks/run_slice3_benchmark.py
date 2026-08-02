"""Controlled benchmark script for RAGLab v7 Slice 3.

Executes all retrieval variants in the causal matrix:
  F0  — Baseline fixed chunks (from Slice 2, re-run for completeness)
  S0  — Sentence anchor (no expansion) — NEW causal control
  W0  — Sentence-window (expansion)
  W1  — Sentence-window + bi-encoder rescoring
  H0  — Hierarchical leaf retrieval (auto-merge disabled)
  H1  — Auto-merging retrieval
  H2  — Auto-merging + bi-encoder rescoring

Usage:
  python benchmarks/run_slice3_benchmark.py --pdf-path /path/to/gersting.pdf

Restrictions enforced:
  - Holdout NOT executed (q_holdout_* filtered out)
  - No API calls
  - No external credentials
  - No reranker presented as cross-encoder
  - Results reported separately by split
  - No combined result used to hide test=0

Output:
  - checkpoints/slice3_<experiment_id>.checkpoint.json
  - benchmarks/results/slice3_results_<experiment_id>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow running from project root without pip install
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from raglab.domain.entities import Chunk, RetrievedEvidence
from raglab.domain.enums import RerankerClass
from raglab.domain.value_objects import ChunkId, DocumentPage
from raglab.infrastructure.embeddings.fastembed_adapter import FastEmbedEmbeddingAdapter
from raglab.infrastructure.retrieval.auto_merging_adapter import (
    HierarchicalRetrievalAdapter,
)
from raglab.infrastructure.retrieval.reranker_adapter import LocalRerankerAdapter
from raglab.infrastructure.retrieval.sentence_anchor_adapter import (
    SentenceAnchorAdapter,
)
from raglab.infrastructure.retrieval.sentence_window_adapter import (
    SentenceWindowAdapter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("slice3_benchmark")

# ---------------------------------------------------------------------------
# Pre-registered manifest path
# ---------------------------------------------------------------------------
_MANIFEST_PATH = _PROJECT_ROOT / "benchmarks" / "slice3_experiment_manifest.json"
_QUESTIONS_PATH = _PROJECT_ROOT / "benchmarks" / "questions" / "controlled_chapter2.json"
_RESULTS_DIR = _PROJECT_ROOT / "benchmarks" / "results"
_CHECKPOINTS_DIR = _PROJECT_ROOT / "checkpoints"

# ---------------------------------------------------------------------------
# Pre-registered constants from manifest
# ---------------------------------------------------------------------------
EXPECTED_PDF_SHA256 = "33e2e9f1e190158b3e99c19fced1acd050720247c7556780bad82b2f93bf1254"
PAGES_START = 91
PAGES_END = 115
CHUNK_SIZE = 512      # F0 fixed chunk chars
WINDOW_SIZE = 2       # W0/W1
CANDIDATE_K = 6       # W1/H0/H1/H2 candidate pool
TOP_K = 3             # final top-k for F0/S0/W0/W1
TOP_K_HIER = 6        # leaf candidate-k for H0/H1/H2
TOP_N = 3             # reranker output
MERGE_THRESHOLD = 0.5 # H1/H2 auto-merge ratio threshold
SEED = 42
CHUNK_SIZES_TOKENS = [1024, 512, 256]   # hierarchy levels

# Acceptable relevant pages per question (from qrel audit)
QREL_PAGES: dict[str, set[int]] = {
    "q_dev_01": {92},
    "q_dev_02": {95},
    "q_dev_03": {97},
    "q_dev_04": {101, 102},
    "q_test_01": {101, 102},
    "q_test_02": {95},
    "q_test_03": {101, 102, 103},
    "q_test_04": set(),  # abstention
}

# Reranker classification — must NOT be presented as cross-encoder
RERANKER_CLASS = RerankerClass.BI_ENCODER_RESCORING
RERANKER_NOTE = (
    "LocalRerankerAdapter uses same FastEmbed model for rescoring. "
    "This is bi_encoder_rescoring — NOT a cross-encoder."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pdf_pages(pdf_path: Path) -> list[DocumentPage]:
    """Extract pages 91-115 from PDF using PyPdfExtractorAdapter."""
    from raglab.infrastructure.pdf_parsers.pdf_parser_adapter import (  # type: ignore[import]
        PyPdfExtractorAdapter,
    )
    adapter = PyPdfExtractorAdapter()
    # read_document accepts page_start/page_end (1-indexed, inclusive)
    selected = adapter.read_document(
        str(pdf_path), page_start=PAGES_START, page_end=PAGES_END
    )
    logger.info(
        "Extracted %d pages (pages %d–%d) from PDF",
        len(selected), PAGES_START, PAGES_END,
    )
    return selected


def pages_to_fixed_chunks(pages: list[DocumentPage], chunk_size: int = CHUNK_SIZE) -> list[Chunk]:
    """Split pages into fixed-size chunks for F0 baseline."""
    chunks: list[Chunk] = []
    for page in pages:
        text = page.text.strip()
        if not text:
            continue
        for i in range(0, len(text), chunk_size):
            segment = text[i:i + chunk_size]
            cid = f"{page.document_id}_p{page.page_number}_c{i}"
            chunks.append(
                Chunk(
                    chunk_id=ChunkId(cid),
                    document_id=page.document_id,
                    text=segment,
                    start_page=page.page_number,
                    end_page=page.page_number,
                )
            )
    return chunks


def load_questions(path: Path) -> list[dict]:
    """Load only development and test questions (NOT holdout)."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    active_splits = {"development", "test"}
    return [
        q for q in data["questions"]
        if q["split"] in active_splits
    ]


def recall_at_k(
    evidence: list[RetrievedEvidence],
    relevant_pages: set[int],
    k: int = TOP_K,
) -> float:
    """Page-level Recall@k — checks if any retrieved evidence has a relevant page."""
    if not relevant_pages:
        return 0.0
    hits = 0
    for ev in evidence[:k]:
        try:
            page = int(ev.document_id.split("_p")[-1])
            if page in relevant_pages:
                hits += 1
        except (ValueError, IndexError):
            pass
    return min(1.0, hits / len(relevant_pages))


def hit_at_k(evidence: list[RetrievedEvidence], relevant_pages: set[int], k: int = TOP_K) -> bool:
    return recall_at_k(evidence, relevant_pages, k) > 0.0


def mrr(evidence: list[RetrievedEvidence], relevant_pages: set[int]) -> float:
    if not relevant_pages:
        return 0.0
    for ev in evidence:
        try:
            page = int(ev.document_id.split("_p")[-1])
            if page in relevant_pages:
                return 1.0 / ev.rank
        except (ValueError, IndexError):
            pass
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Per-strategy runners
# ---------------------------------------------------------------------------

def run_f0(questions: list[dict], pages: list[DocumentPage], embed: FastEmbedEmbeddingAdapter) -> dict:
    """F0 — Fixed chunks, semantic embedding."""
    # Use a simple baseline with FastEmbed rather than DeterministicEmbedding
    # (consistent with Slice 2's real corpus benchmark)
    # We reuse SentenceWindowAdapter with window=0 as a chunk-level baseline.
    # Actually we build chunks and use a direct cosine similarity retriever.
    chunks = pages_to_fixed_chunks(pages, CHUNK_SIZE)
    texts = [c.text for c in chunks]
    embeddings = embed.embed_texts(texts)
    emb_map = {chunks[i].chunk_id.value: list(embeddings[i]) for i in range(len(chunks))}

    results = []
    for q in questions:
        qid = q["qid"]
        is_abs = q.get("is_abstention", False)
        relevant_pages = QREL_PAGES.get(qid, set())

        if is_abs:
            results.append({"qid": qid, "split": q["split"], "is_abstention": True,
                            "recall": 0.0, "mrr": 0.0, "hit": False, "evidence": []})
            continue

        q_emb = embed._get_query_embedding(q["question"])
        scored = []
        for c in chunks:
            vec = emb_map[c.chunk_id.value]
            dot = sum(a * b for a, b in zip(q_emb, vec))
            qn = sum(a * a for a in q_emb) ** 0.5
            vn = sum(b * b for b in vec) ** 0.5
            sim = (dot / (qn * vn)) if (qn > 0 and vn > 0) else 0.0
            scored.append((sim, c))

        scored.sort(key=lambda x: (-x[0], x[1].chunk_id.value))
        top = scored[:TOP_K]
        evidence = [
            RetrievedEvidence(
                chunk_id=c.chunk_id,
                document_id=f"{c.document_id}_p{c.start_page}",
                text=c.text,
                rank=rank + 1,
                score=round(max(0.0, min(1.0, (s + 1.0) / 2.0)), 4),
            )
            for rank, (s, c) in enumerate(top)
        ]

        r = recall_at_k(evidence, relevant_pages)
        results.append({
            "qid": qid,
            "split": q["split"],
            "is_abstention": False,
            "relevant_pages": list(relevant_pages),
            "recall": r,
            "mrr": mrr(evidence, relevant_pages),
            "hit": hit_at_k(evidence, relevant_pages),
            "evidence": [{"doc_id": e.document_id, "rank": e.rank, "score": e.score} for e in evidence],
        })
    return {"strategy": "F0_baseline", "results": results}


def run_s0(questions: list[dict], pages: list[DocumentPage], embed: FastEmbedEmbeddingAdapter) -> dict:
    """S0 — Sentence anchor, no expansion."""
    adapter = SentenceAnchorAdapter(embedding_adapter=embed)
    adapter.index_pages(pages)
    logger.info("S0: indexed %d sentence nodes", len(adapter._sentence_nodes))

    results = []
    for q in questions:
        qid = q["qid"]
        is_abs = q.get("is_abstention", False)
        relevant_pages = QREL_PAGES.get(qid, set())

        if is_abs:
            results.append({"qid": qid, "split": q["split"], "is_abstention": True,
                            "recall": 0.0, "mrr": 0.0, "hit": False, "evidence": []})
            continue

        evidence = adapter.retrieve(q["question"], top_k=TOP_K)
        r = recall_at_k(evidence, relevant_pages)
        results.append({
            "qid": qid, "split": q["split"], "is_abstention": False,
            "relevant_pages": list(relevant_pages),
            "recall": r, "mrr": mrr(evidence, relevant_pages),
            "hit": hit_at_k(evidence, relevant_pages),
            "evidence": [{"doc_id": e.document_id, "rank": e.rank, "score": e.score} for e in evidence],
        })
    return {"strategy": "S0_sentence_anchor", "results": results}


def run_w0(questions: list[dict], pages: list[DocumentPage], embed: FastEmbedEmbeddingAdapter) -> dict:
    """W0 — Sentence-window, no reranker."""
    adapter = SentenceWindowAdapter(embedding_adapter=embed, window_size=WINDOW_SIZE)
    adapter.index_pages(pages)
    logger.info("W0: indexed %d sentence nodes", len(adapter._sentence_nodes))

    results = []
    for q in questions:
        qid = q["qid"]
        is_abs = q.get("is_abstention", False)
        relevant_pages = QREL_PAGES.get(qid, set())

        if is_abs:
            results.append({"qid": qid, "split": q["split"], "is_abstention": True,
                            "recall": 0.0, "mrr": 0.0, "hit": False, "evidence": []})
            continue

        evidence = adapter.retrieve(q["question"], top_k=TOP_K)
        r = recall_at_k(evidence, relevant_pages)
        results.append({
            "qid": qid, "split": q["split"], "is_abstention": False,
            "relevant_pages": list(relevant_pages),
            "recall": r, "mrr": mrr(evidence, relevant_pages),
            "hit": hit_at_k(evidence, relevant_pages),
            "evidence": [{"doc_id": e.document_id, "rank": e.rank, "score": e.score} for e in evidence],
        })
    return {"strategy": "W0_sentence_window", "results": results}


def run_w1(questions: list[dict], pages: list[DocumentPage], embed: FastEmbedEmbeddingAdapter) -> dict:
    """W1 — Sentence-window + bi-encoder rescoring reranker."""
    adapter = SentenceWindowAdapter(embedding_adapter=embed, window_size=WINDOW_SIZE)
    adapter.index_pages(pages)
    reranker = LocalRerankerAdapter(embedding_adapter=embed)

    results = []
    for q in questions:
        qid = q["qid"]
        is_abs = q.get("is_abstention", False)
        relevant_pages = QREL_PAGES.get(qid, set())

        if is_abs:
            results.append({"qid": qid, "split": q["split"], "is_abstention": True,
                            "recall": 0.0, "mrr": 0.0, "hit": False, "evidence": []})
            continue

        candidates = adapter.retrieve(q["question"], top_k=CANDIDATE_K)
        reranked, dropped = reranker.rerank(q["question"], candidates, top_n=TOP_N)

        rel_ids = set()  # not needed for page-level recall
        recall_pre = recall_at_k(candidates[:TOP_N], relevant_pages, TOP_N)
        recall_post = recall_at_k(reranked, relevant_pages, TOP_N)

        results.append({
            "qid": qid, "split": q["split"], "is_abstention": False,
            "relevant_pages": list(relevant_pages),
            "recall": recall_at_k(reranked, relevant_pages),
            "recall_pre_reranker": recall_pre,
            "recall_post_reranker": recall_post,
            "delta_recall": round(recall_post - recall_pre, 4),
            "mrr": mrr(reranked, relevant_pages),
            "hit": hit_at_k(reranked, relevant_pages),
            "relevant_passage_dropped": any(
                int(d.document_id.split("_p")[-1]) in relevant_pages
                for d in dropped
                if "_p" in d.document_id
            ),
            "evidence": [{"doc_id": e.document_id, "rank": e.rank, "score": e.score} for e in reranked],
        })
    return {
        "strategy": "W1_sentence_window_rerank",
        "reranker_class": RERANKER_CLASS.value,
        "reranker_note": RERANKER_NOTE,
        "results": results,
    }


def run_h0_h1(
    questions: list[dict],
    pages: list[DocumentPage],
    embed: FastEmbedEmbeddingAdapter,
    auto_merge: bool,
) -> dict:
    """H0 (auto_merge=False) or H1 (auto_merge=True) — hierarchical retrieval."""
    strategy_name = "H1_auto_merging" if auto_merge else "H0_hierarchical_leaf"
    # Re-create to avoid stale state
    adapter = HierarchicalRetrievalAdapter(
        chunk_sizes=CHUNK_SIZES_TOKENS,
        merge_threshold=MERGE_THRESHOLD,
        auto_merge=auto_merge,
        top_k=TOP_K_HIER,
    )
    stats = adapter.index_pages(pages)
    logger.info(
        "%s: hierarchy built — %d leaves, %d middle, %d parents",
        strategy_name, stats.leaf_count, stats.middle_count, stats.parent_count,
    )

    results = []
    traces_summary = []
    for q in questions:
        qid = q["qid"]
        is_abs = q.get("is_abstention", False)
        relevant_pages = QREL_PAGES.get(qid, set())

        if is_abs:
            results.append({"qid": qid, "split": q["split"], "is_abstention": True,
                            "recall": 0.0, "mrr": 0.0, "hit": False, "evidence": []})
            continue

        evidence, trace = adapter.retrieve_with_trace(
            q["question"], query_id=qid, relevant_page_numbers=relevant_pages
        )
        r = recall_at_k(evidence, relevant_pages)
        results.append({
            "qid": qid, "split": q["split"], "is_abstention": False,
            "relevant_pages": list(relevant_pages),
            "recall": r, "mrr": mrr(evidence, relevant_pages),
            "hit": hit_at_k(evidence, relevant_pages),
            "evidence": [{"doc_id": e.document_id, "rank": e.rank, "score": e.score} for e in evidence],
        })
        traces_summary.append({
            "qid": qid,
            "leaves_retrieved": trace.leaves_retrieved,
            "parent_candidates": trace.parent_candidates,
            "merges_performed": trace.merges_performed,
            "merges_refused": trace.merges_refused,
            "merge_rate": round(trace.merge_rate, 4),
            "parent_promotion_rate": round(trace.parent_promotion_rate, 4),
            "tokens_before": trace.tokens_before,
            "tokens_after": trace.tokens_after,
            "context_expansion_ratio": round(trace.context_expansion_ratio, 4),
            "relevant_evidence_preservation": round(trace.relevant_evidence_preservation, 4),
            "latency_ms": trace.latency_ms,
        })

    return {
        "strategy": strategy_name,
        "hierarchy_stats": {
            "total_nodes": stats.total_nodes,
            "leaf_count": stats.leaf_count,
            "middle_count": stats.middle_count,
            "parent_count": stats.parent_count,
            "avg_leaf_tokens": stats.avg_leaf_tokens,
            "avg_middle_tokens": stats.avg_middle_tokens,
            "avg_parent_tokens": stats.avg_parent_tokens,
        },
        "results": results,
        "merge_traces": traces_summary,
    }


def run_h2(
    questions: list[dict],
    pages: list[DocumentPage],
    embed: FastEmbedEmbeddingAdapter,
) -> dict:
    """H2 — Auto-merging + bi-encoder rescoring."""
    adapter = HierarchicalRetrievalAdapter(
        chunk_sizes=CHUNK_SIZES_TOKENS,
        merge_threshold=MERGE_THRESHOLD,
        auto_merge=True,
        top_k=TOP_K_HIER,
    )
    stats = adapter.index_pages(pages)
    reranker = LocalRerankerAdapter(embedding_adapter=embed)

    results = []
    for q in questions:
        qid = q["qid"]
        is_abs = q.get("is_abstention", False)
        relevant_pages = QREL_PAGES.get(qid, set())

        if is_abs:
            results.append({"qid": qid, "split": q["split"], "is_abstention": True,
                            "recall": 0.0, "mrr": 0.0, "hit": False, "evidence": []})
            continue

        candidates, trace = adapter.retrieve_with_trace(
            q["question"], query_id=qid, relevant_page_numbers=relevant_pages
        )
        reranked, dropped = reranker.rerank(q["question"], candidates, top_n=TOP_N)

        recall_pre = recall_at_k(candidates[:TOP_N], relevant_pages, TOP_N)
        recall_post = recall_at_k(reranked, relevant_pages, TOP_N)

        results.append({
            "qid": qid, "split": q["split"], "is_abstention": False,
            "relevant_pages": list(relevant_pages),
            "recall": recall_at_k(reranked, relevant_pages),
            "recall_pre_reranker": recall_pre,
            "recall_post_reranker": recall_post,
            "delta_recall": round(recall_post - recall_pre, 4),
            "mrr": mrr(reranked, relevant_pages),
            "hit": hit_at_k(reranked, relevant_pages),
            "relevant_passage_dropped": any(
                int(d.document_id.split("_p")[-1]) in relevant_pages
                for d in dropped
                if "_p" in d.document_id
            ),
            "evidence": [{"doc_id": e.document_id, "rank": e.rank, "score": e.score} for e in reranked],
        })

    return {
        "strategy": "H2_auto_merging_rerank",
        "reranker_class": RERANKER_CLASS.value,
        "reranker_note": RERANKER_NOTE,
        "results": results,
    }


def aggregate_by_split(results: list[dict]) -> dict[str, dict]:
    """Aggregate metrics by split. Never combine to hide test=0."""
    splits: dict[str, list[float]] = {"development": [], "test": []}
    for r in results:
        if r.get("is_abstention"):
            continue
        s = r.get("split", "unknown")
        if s in splits:
            splits[s].append(r.get("recall", 0.0))
    return {
        split: {
            "mean_recall": round(mean(vals), 4),
            "n_questions": len(vals),
        }
        for split, vals in splits.items()
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RAGLab v7 Slice 3 Benchmark")
    parser.add_argument("--pdf-path", default=os.environ.get("RAGLAB_PDF_PATH", ""))
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip PDF loading, run with dummy pages (CI mode)")
    args = parser.parse_args()

    logger.info("=== RAGLab v7 Slice 3 Benchmark — Starting ===")
    logger.info("Reranker classification: %s", RERANKER_CLASS.value)
    logger.info("Reranker note: %s", RERANKER_NOTE)

    # --- Load manifest ---
    with _MANIFEST_PATH.open(encoding="utf-8") as f:
        manifest = json.load(f)
    experiment_id = manifest["experiment_id"]
    logger.info("Experiment ID: %s", experiment_id)

    # --- Verify PDF (skip in dry-run) ---
    if args.dry_run:
        logger.warning("DRY-RUN mode — using synthetic pages, not real PDF")
        pages = [
            DocumentPage(
                document_id="gersting_dry",
                page_number=p,
                text=f"Página {p} — Técnicas de demonstração matemática. "
                     "Por exaustão, contraposição, contradição e indução matemática. " * 3,
            )
            for p in range(PAGES_START, PAGES_END + 1)
        ]
    else:
        pdf_path = Path(args.pdf_path)
        if not pdf_path.exists():
            logger.error("PDF not found: %s. Use --pdf-path or RAGLAB_PDF_PATH.", pdf_path)
            sys.exit(1)
        actual_sha = sha256_file(pdf_path)
        if actual_sha != EXPECTED_PDF_SHA256:
            logger.error(
                "SHA-256 mismatch. Expected %s, got %s", EXPECTED_PDF_SHA256, actual_sha
            )
            sys.exit(1)
        logger.info("PDF SHA-256 verified: %s", actual_sha)
        pages = load_pdf_pages(pdf_path)

    if not pages:
        logger.error("No pages extracted from corpus. Aborting.")
        sys.exit(1)

    # --- Load questions (holdout excluded) ---
    questions = load_questions(_QUESTIONS_PATH)
    logger.info("Loaded %d active questions (holdout sealed)", len(questions))

    # --- Embedding model (shared across all variants for fair comparison) ---
    logger.info("Loading FastEmbed model...")
    embed = FastEmbedEmbeddingAdapter()

    # --- Run all strategies ---
    run_time = {}
    all_variant_results = {}

    for name, runner in [
        ("F0", lambda: run_f0(questions, pages, embed)),
        ("S0", lambda: run_s0(questions, pages, embed)),
        ("W0", lambda: run_w0(questions, pages, embed)),
        ("W1", lambda: run_w1(questions, pages, embed)),
        ("H0", lambda: run_h0_h1(questions, pages, embed, auto_merge=False)),
        ("H1", lambda: run_h0_h1(questions, pages, embed, auto_merge=True)),
        ("H2", lambda: run_h2(questions, pages, embed)),
    ]:
        logger.info("Running %s...", name)
        t0 = time.perf_counter()
        all_variant_results[name] = runner()
        run_time[name] = round((time.perf_counter() - t0) * 1000, 1)
        logger.info("%s complete in %.1f ms", name, run_time[name])

    # --- Aggregate by split ---
    aggregated = {}
    for variant_name, variant_data in all_variant_results.items():
        aggregated[variant_name] = aggregate_by_split(variant_data["results"])

    # --- Print summary (split-separated — never combined to hide test=0) ---
    print("\n=== SLICE 3 RESULTS — BY SPLIT ===")
    print(f"{'Variant':<10} {'Split':<15} {'Mean Recall':<15} {'N Questions'}")
    print("-" * 55)
    for variant_name in ["F0", "S0", "W0", "W1", "H0", "H1", "H2"]:
        if variant_name not in aggregated:
            continue
        for split_name, agg in aggregated[variant_name].items():
            print(
                f"{variant_name:<10} {split_name:<15} "
                f"{agg['mean_recall']:<15.4f} {agg['n_questions']}"
            )

    print("\n=== CAUSAL COMPARISONS (development split) ===")
    def _dev_recall(vname: str) -> float:
        return aggregated.get(vname, {}).get("development", {}).get("mean_recall", 0.0)

    pairs = [
        ("F0 × S0", "F0", "S0", "granularity effect"),
        ("S0 × W0", "S0", "W0", "expansion effect"),
        ("W0 × W1", "W0", "W1", "reranking on sentence-window"),
        ("H0 × H1", "H0", "H1", "auto-merging effect"),
        ("H1 × H2", "H1", "H2", "reranking on auto-merging"),
    ]
    for label, a, b, effect in pairs:
        da, db = _dev_recall(a), _dev_recall(b)
        delta = db - da
        print(f"  {label:10} ({effect}): {a}={da:.4f}, {b}={db:.4f}, Δ={delta:+.4f}")

    print(f"\nReranker class: {RERANKER_CLASS.value}")
    print(f"Note: {RERANKER_NOTE}")

    # --- Save results ---
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = _RESULTS_DIR / f"slice3_results_{experiment_id}.json"
    output = {
        "experiment_id": experiment_id,
        "run_time_ms": run_time,
        "aggregated_by_split": aggregated,
        "variants": all_variant_results,
        "reranker_class": RERANKER_CLASS.value,
        "reranker_note": RERANKER_NOTE,
        "holdout_status": "SEALED — not executed",
        "second_annotator_status": "GROUND_TRUTH_SINGLE_ANNOTATOR",
    }
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Results saved to %s", results_path)

    logger.info("=== Slice 3 Benchmark Complete ===")


if __name__ == "__main__":
    main()
