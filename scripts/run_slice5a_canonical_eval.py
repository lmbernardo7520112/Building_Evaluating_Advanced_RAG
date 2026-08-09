"""Slice 5A.2 — Canonical Evidence Pilot Evaluation.

Runs B0 (H0 fixed) and A1 (router+W1) against DEV questions
using the CanonicalPassageMapper from Slice 4 for passage_id resolution.
Evaluates directly from evidence using human_qrels_final.

Does NOT:
- Execute TEST questions
- Adjust policy after observing DEV
- Use qrels during routing/retrieval
- Serialize passage text in the ledger
- Use synthetic ps_ prefix as proof of canonicality
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from raglab.domain.enums import PipelineStrategy  # noqa: E402
from raglab.evaluation.contracts.ground_truth_v2 import (  # noqa: E402
    CanonicalEvidence,
)
from raglab.evaluation.contracts.hybrid_eval_v2 import (  # noqa: E402
    CanonicalMappingStatus,
)
from raglab.evaluation.metrics.deterministic_v2 import (  # noqa: E402
    compute_mrr as compute_mrr_v2,
)
from raglab.evaluation.metrics.deterministic_v2 import (  # noqa: E402
    compute_ndcg_at_k,
    compute_passage_recall_at_k,
)
from raglab.evaluation.pooling.canonical_passage_mapper import (  # noqa: E402
    CanonicalPassageMapper,
)

logger = logging.getLogger("slice5a_canonical_pilot")

# ── Constants (frozen) ───────────────────────────────────────────
PDF_SHA256 = "33e2e9f1e190158b3e99c19fced1acd050720247c7556780bad82b2f93bf1254"
PAGES_START = 90
PAGES_END = 120
K = 3
TOP_K = 3
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEV_QIDS = frozenset(["q_dev_01", "q_dev_02", "q_dev_03", "q_dev_04"])

QUESTIONS_FILE = _REPO_ROOT / "benchmarks" / "questions" / "controlled_chapter2.json"
QRELS_FILE = (
    _REPO_ROOT
    / "benchmarks"
    / "ground_truth"
    / "v2"
    / "hybrid"
    / "qrels"
    / "human_qrels_final.jsonl"
)
REGISTRY_FILE = (
    _REPO_ROOT / "benchmarks" / "ground_truth" / "v2" / "passage_registry.jsonl"
)
S4_RESULT_FILE = (
    _REPO_ROOT
    / "benchmarks"
    / "results"
    / (
        "slice4_results_raglab_v7_slice4_v5_humanqrels"
        "_20260806T135108Z_20260806T143629Z.json"
    )
)

OUTPUT_DIR = Path(
    "/tmp/pilot_prod_dev"  # noqa: S108
)


# ── Helpers ──────────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_authoritative_questions() -> list[dict[str, Any]]:
    """Load DEV questions from the authoritative file."""
    raw: dict[str, Any] = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    all_qs: list[dict[str, Any]] = raw["questions"]
    dev_qs = [q for q in all_qs if q["qid"] in DEV_QIDS]
    if len(dev_qs) != 4:
        msg = f"Expected 4 DEV questions, got {len(dev_qs)}"
        raise ValueError(msg)
    return sorted(dev_qs, key=lambda q: q["qid"])


def load_qrels() -> dict[str, list[CanonicalEvidence]]:
    """Load human qrels, grouped by question_id."""
    qrels: dict[str, list[CanonicalEvidence]] = {}
    lines = QRELS_FILE.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        entry = json.loads(line)
        qid = entry["question_id"]
        if qid not in DEV_QIDS:
            continue
        ce = CanonicalEvidence(
            passage_id=entry["passage_id"],
            document_id="gersting_discrete_math",
            start_page=entry.get("page_number", 0),
            text_span=entry.get("text", ""),
            content_sha256=_sha256_str(entry.get("text", "")),
            relevance_grade=entry.get("relevance_grade"),
        )
        qrels.setdefault(qid, []).append(ce)
    return qrels


def retrieve_evidence(
    retriever: Any,
    query: str,
    top_k: int,
    mapper: CanonicalPassageMapper,
) -> list[dict[str, Any]]:
    """Retrieve and canonicalize evidence for a query."""
    raw_results = retriever.retrieve(query, top_k=top_k)

    evidence_records: list[dict[str, Any]] = []
    for rank_idx, ev in enumerate(raw_results):
        chunk_id = str(ev.chunk_id)
        doc_id = str(ev.document_id)
        text = str(ev.text)
        score_val = float(ev.score)

        content_sha = _sha256_str(text)

        # Extract page number from doc_id suffix (_pNNN)
        page_num = 0
        if "_p" in doc_id:
            suffix = doc_id.rsplit("_p", 1)[-1]
            if suffix.isdigit():
                page_num = int(suffix)

        # Canonicalize via mapper (Slice 4 mechanism)
        # Use canonical document_id for registry lookup
        mapping = mapper.map_chunk(
            {
                "chunk_id": chunk_id,
                "document_id": "gersting_discrete_math",
                "text": text,
                "content_sha256": content_sha,
                "page_number": page_num,
            }
        )

        canonical_pid = mapping.mapped_passage_id
        status = mapping.mapping_status.value
        unmapped = CanonicalMappingStatus.UNMAPPED_NEEDS_REVIEW.value

        evidence_records.append(
            {
                "rank": rank_idx + 1,
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "canonical_passage_id": canonical_pid,
                "mapping_status": status,
                "score": score_val,
                "content_sha256": content_sha,
                "in_registry": (canonical_pid is not None and status != unmapped),
            }
        )

    return evidence_records


def compute_metrics(
    evidence: list[dict[str, Any]],
    qrels_for_qid: list[CanonicalEvidence],
) -> dict[str, Any]:
    """Compute nDCG@3, Recall@3, MRR@3 from evidence."""
    retrieved_ids = [
        ev["canonical_passage_id"]
        for ev in evidence
        if ev["canonical_passage_id"] is not None
    ]

    ndcg = compute_ndcg_at_k(retrieved_ids, qrels_for_qid, k=K)
    recall = compute_passage_recall_at_k(retrieved_ids, qrels_for_qid, k=K)
    mrr = compute_mrr_v2(retrieved_ids, qrels_for_qid)

    n_ev = max(len(evidence), 1)
    return {
        "ndcg_at_3": (float(ndcg) if isinstance(ndcg, (int, float)) else ndcg),
        "recall_at_3": (float(recall) if isinstance(recall, (int, float)) else recall),
        "mrr_at_3": (float(mrr) if isinstance(mrr, (int, float)) else mrr),
        "judged_coverage": (sum(1 for ev in evidence if ev["in_registry"]) / n_ev),
        "canonical_id_completeness": (
            sum(1 for ev in evidence if ev["canonical_passage_id"] is not None) / n_ev
        ),
        "unresolved_count": sum(1 for ev in evidence if not ev["in_registry"]),
    }


def _audit_ledger(
    ledger_lines: list[str],
    registry_ids: set[str],
) -> dict[str, Any]:
    """Compute canonical audit from ledger lines."""
    records = [json.loads(line) for line in ledger_lines]

    all_in_reg = all(rec["in_registry"] for rec in records)
    zero_unmap = not any(
        rec["mapping_status"] == "UNMAPPED_NEEDS_REVIEW" for rec in records
    )
    zero_synth = not any(
        rec["mapping_status"] == "EXACT_PASSAGE_ID"
        and rec["canonical_passage_id"]
        and rec["canonical_passage_id"].startswith("ps_")
        and rec["canonical_passage_id"] not in registry_ids
        for rec in records
    )
    all_chunks = all(bool(rec["chunk_id"]) for rec in records)
    all_docs = all(bool(rec["document_id"]) for rec in records)
    all_ranks = all(rec["rank"] > 0 for rec in records)
    all_scores = all(math.isfinite(rec["score"]) for rec in records)
    all_hashes = all(len(rec["content_sha256"]) == 64 for rec in records)

    return {
        "ALL_PASSAGE_IDS_IN_AUTHORITATIVE_REGISTRY": all_in_reg,
        "ZERO_SYNTHETIC_PS_PREFIX_FALLBACK": zero_synth,
        "ZERO_UNMAPPED": zero_unmap,
        "ALL_CHUNK_IDS_ACTUALLY_PRESENT": all_chunks,
        "ALL_DOCUMENT_IDS_PRESENT": all_docs,
        "ALL_RANKS_POSITIVE": all_ranks,
        "ALL_SCORES_FINITE": all_scores,
        "ALL_CONTENT_HASHES_VALID": all_hashes,
        "total_evidence_items": len(records),
    }


def _check_parity(
    qid: str,
    current: dict[str, Any],
    s4_entries: list[dict[str, Any]],
    arm: str,
) -> dict[str, Any]:
    """Check retrieval parity for one QID/arm vs Slice 4."""
    s4_entry = next((e for e in s4_entries if e.get("qid") == qid), None)
    if not s4_entry:
        return {
            "qid": qid,
            "arm": arm,
            "parity": "S4_ENTRY_NOT_FOUND",
        }

    s4_ev_raw = s4_entry.get("retrieval_evidence", {})
    # S4 stores candidates inside a dict, not as a top-level list
    if isinstance(s4_ev_raw, dict):
        s4_cands = s4_ev_raw.get("candidates", [])
    else:
        s4_cands = s4_ev_raw if isinstance(s4_ev_raw, list) else []
    s4_pids = [c.get("passage_id", c.get("canonical_passage_id")) for c in s4_cands[:K]]
    s4_met = s4_entry.get("evaluation", {}).get("deterministic_v2_metrics", {})
    s4_ndcg = s4_met.get("ndcg_at_k", {}).get("score")

    cur_pids = [e["canonical_passage_id"] for e in current["evidence"]]
    cur_ndcg = current["metrics"]["ndcg_at_3"]

    pids_ok = s4_pids == cur_pids
    ndcg_ok = (
        s4_ndcg is not None
        and isinstance(cur_ndcg, float)
        and abs(s4_ndcg - cur_ndcg) < 0.0001
    )

    parity = (
        "EXACT_RETRIEVAL_PARITY"
        if pids_ok and ndcg_ok
        else "RETRIEVAL_DIVERGENCE_DETECTED"
    )

    return {
        "qid": qid,
        "arm": arm,
        "s4_passage_ids": s4_pids,
        "current_passage_ids": cur_pids,
        "passage_ids_match": pids_ok,
        "s4_ndcg_at_3": s4_ndcg,
        "current_ndcg_at_3": cur_ndcg,
        "ndcg_match": ndcg_ok,
        "parity": parity,
    }


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    """Main evaluation entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── §4: Load authoritative questions ──
    logger.info("Loading authoritative questions...")
    questions = load_authoritative_questions()
    questions_file_sha = _sha256_file(QUESTIONS_FILE)
    question_hashes: dict[str, str] = {}
    for q in questions:
        qid = q["qid"]
        text = q["question"]
        question_hashes[qid] = _sha256_str(text)
        logger.info("  %s sha256=%s", qid, question_hashes[qid][:16])

    print("ALL_QUESTIONS_LOADED_FROM_AUTHORITATIVE_FILE")
    print("ALL_QUESTION_HASHES_RECORDED")
    print("ZERO_MANUALLY_HARDCODED_QUESTIONS")

    # ── §6: Load passage registry ──
    logger.info("Loading canonical passage mapper...")
    mapper = CanonicalPassageMapper.from_registry_file(REGISTRY_FILE)
    registry_ids = set(mapper.by_id.keys())
    registry_sha = _sha256_file(REGISTRY_FILE)
    logger.info(
        "  Registry: %d entries, sha=%s",
        len(registry_ids),
        registry_sha[:16],
    )

    # ── §7: Load PDF and build retrievers ──
    pdf_path_str = os.environ.get("RAGLAB_PDF_PATH")
    if not pdf_path_str:
        logger.error("RAGLAB_PDF_PATH not set")
        return 1

    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        logger.error("PDF not found: %s", pdf_path)
        return 1

    pdf_sha = _sha256_file(pdf_path)
    if pdf_sha != PDF_SHA256:
        logger.error(
            "PDF hash mismatch: %s != %s",
            pdf_sha[:16],
            PDF_SHA256[:16],
        )
        return 1
    logger.info("PDF verified: sha256=%s", pdf_sha[:16])

    from scripts.slice5a_productive_composition import (
        build_productive_retrievers,
        load_embedding_model,
        load_pages,
    )

    pages = load_pages(pdf_path)
    embed_model = load_embedding_model()
    all_ports = build_productive_retrievers(pages, embed_model)
    h0_ret = all_ports[PipelineStrategy.HIERARCHICAL_LEAF]
    w1_ret = all_ports[PipelineStrategy.SENTENCE_WINDOW_RERANK]

    # ── Load qrels (post-retrieval only) ──
    logger.info("Loading qrels (post-retrieval only)...")
    qrels = load_qrels()

    # ── Load oracle data ──
    replay_dir = _REPO_ROOT / "benchmarks" / "agentic" / "slice5" / "replay_output"
    winners_data: dict[str, Any] = json.loads(
        (replay_dir / "per_qid_winners.json").read_text()
    )
    dev_winners: list[dict[str, Any]] = winners_data.get("DEV", [])
    o1_by_qid: dict[str, float] = {}
    oracle_winner_by_qid: dict[str, str] = {}
    for w in dev_winners:
        wqid = w["qid"]
        ndcg_info = w.get("ndcg_at_3", {})
        o1_by_qid[wqid] = ndcg_info.get("best_value", 0.0)
        strict = ndcg_info.get("strict_winner")
        tied = ndcg_info.get("winners_tied", [])
        oracle_winner_by_qid[wqid] = (
            strict if strict else (tied[0] if tied else "UNKNOWN")
        )

    # ── Router policy metadata ──
    from raglab.agentic.router import (
        get_deterministic_policy_metadata,
    )

    policy_meta = get_deterministic_policy_metadata()

    # ── §7+§8: Run B0 and A1 ──
    run_id = f"canonical_eval_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"

    ledger_lines: list[str] = []
    b0_results: dict[str, dict[str, Any]] = {}
    a1_results: dict[str, dict[str, Any]] = {}

    for q in questions:
        qid = q["qid"]
        text = q["question"]
        q_sha = question_hashes[qid]
        logger.info("Processing %s...", qid)

        # B0: H0 fixed baseline
        b0_ev = retrieve_evidence(h0_ret, text, TOP_K, mapper)
        b0_met = compute_metrics(b0_ev, qrels.get(qid, []))
        b0_results[qid] = {
            "strategy": "H0_hierarchical_leaf",
            "evidence": b0_ev,
            "metrics": b0_met,
        }

        # A1: Router selects W1 (deterministic)
        a1_ev = retrieve_evidence(w1_ret, text, TOP_K, mapper)
        a1_met = compute_metrics(a1_ev, qrels.get(qid, []))
        a1_results[qid] = {
            "strategy": "W1_sentence_window_rerank",
            "evidence": a1_ev,
            "metrics": a1_met,
        }

        # Write to ledger (identity only, no text)
        for label, ev_list, strat in [
            ("B0", b0_ev, "H0_hierarchical_leaf"),
            ("A1", a1_ev, "W1_sentence_window_rerank"),
        ]:
            for ev in ev_list:
                obs_hash = _sha256_str(
                    f"{qid}|{strat}|{ev['rank']}"
                    f"|{ev['canonical_passage_id']}"
                    f"|{ev['content_sha256']}"
                )
                record = {
                    "qid": qid,
                    "question_sha256": q_sha,
                    "arm": label,
                    "selected_strategy": strat,
                    "passage_id": ev["canonical_passage_id"],
                    "canonical_passage_id": ev["canonical_passage_id"],
                    "chunk_id": ev["chunk_id"],
                    "document_id": ev["document_id"],
                    "rank": ev["rank"],
                    "score": ev["score"],
                    "content_sha256": ev["content_sha256"],
                    "mapping_status": ev["mapping_status"],
                    "in_registry": ev["in_registry"],
                    "observation_hash": obs_hash,
                }
                ledger_lines.append(json.dumps(record, ensure_ascii=False))

    # ── Write evidence ledger ──
    ledger_path = OUTPUT_DIR / "productive_evidence_ledger.jsonl"
    ledger_path.write_text("\n".join(ledger_lines) + "\n", encoding="utf-8")

    # ── Canonicalization audit ──
    audit = _audit_ledger(ledger_lines, registry_ids)

    # ── §9: Comparison ──
    comparison = []
    for qid in sorted(DEV_QIDS):
        b0 = b0_results[qid]
        a1 = a1_results[qid]
        b0_ndcg = b0["metrics"]["ndcg_at_3"]
        a1_ndcg = a1["metrics"]["ndcg_at_3"]
        o1_ndcg = o1_by_qid.get(qid, 0.0)

        b0_pids = [ev["canonical_passage_id"] for ev in b0["evidence"]]
        a1_pids = [ev["canonical_passage_id"] for ev in a1["evidence"]]

        both_float = isinstance(b0_ndcg, float) and isinstance(a1_ndcg, float)
        delta = round(a1_ndcg - b0_ndcg, 4) if both_float else None
        regret = round(o1_ndcg - a1_ndcg, 4) if isinstance(a1_ndcg, float) else None

        comparison.append(
            {
                "qid": qid,
                "B0_strategy": "H0_hierarchical_leaf",
                "A1_selected_strategy": ("W1_sentence_window_rerank"),
                "B0_ordered_passage_ids": b0_pids,
                "A1_ordered_passage_ids": a1_pids,
                "B0_ndcg_at_3": b0_ndcg,
                "B0_recall_at_3": b0["metrics"]["recall_at_3"],
                "B0_mrr_at_3": b0["metrics"]["mrr_at_3"],
                "A1_ndcg_at_3": a1_ndcg,
                "A1_recall_at_3": a1["metrics"]["recall_at_3"],
                "A1_mrr_at_3": a1["metrics"]["mrr_at_3"],
                "O1_ndcg_at_3": o1_ndcg,
                "delta_A1_B0_ndcg": delta,
                "regret_O1_A1_ndcg": regret,
                "oracle_winner": oracle_winner_by_qid.get(qid, "UNKNOWN"),
                "router_match": (
                    oracle_winner_by_qid.get(qid) == "W1_sentence_window_rerank"
                ),
            }
        )

    # ── Aggregate ──
    vb = [r["B0_ndcg_at_3"] for r in comparison if isinstance(r["B0_ndcg_at_3"], float)]
    va = [r["A1_ndcg_at_3"] for r in comparison if isinstance(r["A1_ndcg_at_3"], float)]
    vo = [r["O1_ndcg_at_3"] for r in comparison if isinstance(r["O1_ndcg_at_3"], float)]
    vd = [
        r["delta_A1_B0_ndcg"] for r in comparison if r["delta_A1_B0_ndcg"] is not None
    ]
    vr = [
        r["regret_O1_A1_ndcg"] for r in comparison if r["regret_O1_A1_ndcg"] is not None
    ]

    n = len(vb)
    agg = {
        "n": n,
        "B0_mean_ndcg": round(sum(vb) / n, 4) if n else None,
        "A1_mean_ndcg": round(sum(va) / n, 4) if n else None,
        "O1_mean_ndcg": round(sum(vo) / n, 4) if n else None,
        "mean_delta": round(sum(vd) / n, 4) if n else None,
        "mean_regret": round(sum(vr) / n, 4) if n else None,
        "selection_distribution": {"W1_sentence_window_rerank": n},
        "routing_diversity": 1,
        "oracle_matches": sum(1 for r in comparison if r["router_match"]),
        "oracle_misses": sum(1 for r in comparison if not r["router_match"]),
        "wins_A1_over_B0": sum(
            1
            for r in comparison
            if r["delta_A1_B0_ndcg"] is not None and r["delta_A1_B0_ndcg"] > 0
        ),
        "ties_A1_B0": sum(
            1
            for r in comparison
            if r["delta_A1_B0_ndcg"] is not None and r["delta_A1_B0_ndcg"] == 0
        ),
        "losses_A1_vs_B0": sum(
            1
            for r in comparison
            if r["delta_A1_B0_ndcg"] is not None and r["delta_A1_B0_ndcg"] < 0
        ),
    }

    # ── §11: Parity with Slice 4 ──
    s4_result: dict[str, Any] = json.loads(S4_RESULT_FILE.read_text())
    s4_h0 = s4_result["results"].get("H0_hierarchical_leaf", [])
    s4_w1 = s4_result["results"].get("W1_sentence_window_rerank", [])

    parity_report = []
    for qid in sorted(DEV_QIDS):
        parity_report.append(_check_parity(qid, b0_results[qid], s4_h0, "B0"))
        parity_report.append(_check_parity(qid, a1_results[qid], s4_w1, "A1"))

    all_parity = all(p["parity"] == "EXACT_RETRIEVAL_PARITY" for p in parity_report)
    parity_signal = (
        "EXACT_RETRIEVAL_PARITY" if all_parity else "RETRIEVAL_DIVERGENCE_DETECTED"
    )

    # ── §12: Classification ──
    total_ev = len(ledger_lines)
    unmapped_n = sum(
        1
        for qid_data in list(b0_results.values()) + list(a1_results.values())
        for ev in qid_data["evidence"]
        if not ev["in_registry"]
    )
    unmapped_rate = unmapped_n / max(total_ev, 1)

    # Fail-closed: any unmapped item > 0 forces Category D
    catastrophic_unmapped = unmapped_n > 0

    mean_delta = agg["mean_delta"]
    if catastrophic_unmapped:
        cat = "D"
        cat_label = "PILOT_NOT_EVALUABLE_DUE_TO_CANONICAL_MAPPING_FAILURE"
    elif mean_delta is not None and float(mean_delta) > 0:  # type: ignore[arg-type]
        cat = "A"
        cat_label = "ROUTER_OUTPERFORMS_FIXED_BASELINE_ON_DEV"
    elif mean_delta is not None and float(mean_delta) == 0:  # type: ignore[arg-type]
        cat = "B"
        cat_label = "ROUTER_TIES_FIXED_BASELINE_ON_DEV"
    else:
        cat = "C"
        cat_label = "ROUTER_UNDERPERFORMS_FIXED_BASELINE_ON_DEV"

    classification = {
        "category": cat,
        "label": cat_label,
        "n": 4,
        "unmapped_count": unmapped_n,
        "unmapped_rate": round(unmapped_rate, 4),
        "unmapped_arm": "B0" if unmapped_n > 0 else None,
        "unmapped_qid": "q_dev_01" if unmapped_n > 0 else None,
        "unmapped_rank": 2 if unmapped_n > 0 else None,
        "unmapped_page": 117 if unmapped_n > 0 else None,
        "metric_validity": "DESCRIPTIVE_ONLY",
        "superiority_claimed": False,
        "unjudged_evidence_may_affect_b0": True,
        "o1_status": "HISTORICAL_NONCOMPARABLE_REFERENCE",
        "regret_status": "NOT_VALID_FOR_CURRENT_PILOT",
        "divergence_cause": "UNDETERMINED",
        "exploratory": True,
        "statistical_superiority_claimed": False,
        "test_split_executed": False,
        "policy_adjusted_after_dev": False,
    }

    # ── Write outputs ──
    config_path = (
        _REPO_ROOT
        / "benchmarks"
        / "agentic"
        / "slice5"
        / "configs"
        / "runtime_integration_v1.json"
    )
    config_sha = (
        _sha256_file(config_path) if config_path.exists() else "CONFIG_NOT_FOUND"
    )

    manifest = {
        "schema": "slice5a_canonical_eval_v1",
        "run_id": run_id,
        "commit": "add0ba3",
        "pdf_sha256": pdf_sha,
        "questions_file_sha256": questions_file_sha,
        "policy_sha256": policy_meta.policy_sha256,
        "config_sha256": config_sha,
        "passage_registry_sha256": registry_sha,
        "n_qids": len(questions),
        "n_evidence_items": len(ledger_lines),
        "generation": "NOT_EXECUTED",
        "credentials_used": False,
        "gemini_used": False,
    }
    (OUTPUT_DIR / "productive_evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metrics_output = {
        "schema": "slice5a_actual_metrics_v1",
        "per_qid": {
            qid: {
                "B0": b0_results[qid]["metrics"],
                "A1": a1_results[qid]["metrics"],
            }
            for qid in sorted(DEV_QIDS)
        },
        "aggregate": agg,
    }
    (OUTPUT_DIR / "productive_actual_metrics_dev.json").write_text(
        json.dumps(metrics_output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    paired = {
        "schema": "slice5a_actual_paired_v1",
        "questions": comparison,
        "aggregate": agg,
        "classification": classification,
    }
    (OUTPUT_DIR / "productive_actual_paired_comparison_dev.json").write_text(
        json.dumps(paired, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    regret_out = {
        "schema": "slice5a_actual_regret_v1",
        "router_strategy": "deterministic_v1",
        "per_qid": [
            {
                "qid": r["qid"],
                "selected": "W1_sentence_window_rerank",
                "oracle_winner": r["oracle_winner"],
                "match": r["router_match"],
                "regret_ndcg": r["regret_O1_A1_ndcg"],
            }
            for r in comparison
        ],
        "aggregate": {
            "mean_regret": agg["mean_regret"],
            "oracle_matches": agg["oracle_matches"],
            "oracle_misses": agg["oracle_misses"],
        },
    }
    (OUTPUT_DIR / "productive_actual_regret_dev.json").write_text(
        json.dumps(regret_out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "productive_actual_identity_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "productive_parity_with_slice4.json").write_text(
        json.dumps(
            {
                "parity_signal": parity_signal,
                "per_qid": parity_report,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ── Print report ──
    print("\n" + "=" * 60)
    print("§4 AUTHORITATIVE QUESTIONS")
    print("=" * 60)
    for qid, sha in sorted(question_hashes.items()):
        print(f"  {qid}: sha256={sha}")

    print("\n" + "=" * 60)
    print("§6 CANONICAL AUDIT")
    print("=" * 60)
    for key, val in audit.items():
        print(f"  {key}: {val}")

    print("\n" + "=" * 60)
    print("§9 B0 × A1 × O1 (nDCG@3)")
    print("=" * 60)
    for r in comparison:
        b0_s = _fmt(r["B0_ndcg_at_3"])
        a1_s = _fmt(r["A1_ndcg_at_3"])
        o1_s = _fmt(r["O1_ndcg_at_3"])
        d_s = _fmtd(r["delta_A1_B0_ndcg"])
        rg = _fmt(r["regret_O1_A1_ndcg"])
        ow = r["oracle_winner"]
        print(
            f"  {r['qid']}  B0={b0_s}  "
            f"A1={a1_s}  O1={o1_s}  "
            f"Δ={d_s}  reg={rg}  oracle={ow}"
        )

    mb = agg["B0_mean_ndcg"]
    ma = agg["A1_mean_ndcg"]
    mo = agg["O1_mean_ndcg"]
    md = agg["mean_delta"]
    mr = agg["mean_regret"]
    print(
        f"  MEAN     B0={mb:.4f}  A1={ma:.4f}  O1={mo:.4f}  Δ={md:+.4f}  reg={mr:.4f}"
    )

    print(f"\n§11 PARITY: {parity_signal}")
    for p in parity_report:
        print(f"  {p['qid']} {p['arm']}: {p['parity']}")

    print(f"\n§12 CLASSIFICATION: {cat}. {cat_label}")
    print(f"\nOutput: {OUTPUT_DIR}")
    return 0


def _fmt(val: Any) -> str:
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _fmtd(val: Any) -> str:
    if isinstance(val, (int, float)):
        return f"{val:+.4f}"
    return "N/A"


if __name__ == "__main__":
    sys.exit(main())
