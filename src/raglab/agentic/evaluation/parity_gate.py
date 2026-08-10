"""Slice 5A.3 Parity Gate execution and human coverage evaluation engine."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from raglab.agentic.evaluation.canonical_corpus import (
    load_canonical_corpus_snapshot,
    passages_to_chunks,
    passages_to_document_pages,
)
from raglab.agentic.evaluation.parity_contracts import (
    ArmRun,
    CanonicalRetrievedItem,
    JudgmentStatus,
    MappingStatus,
    ParityGateResult,
    ParityOutcomeCategory,
)
from raglab.agentic.evaluation.reproducibility import (
    compare_arm_runs,
)
from raglab.infrastructure.retrieval.auto_merging_adapter import (
    HierarchicalRetrievalAdapter,
)
from raglab.infrastructure.retrieval.baseline_adapter import InMemoryBaselineAdapter
from raglab.infrastructure.retrieval.reranker_adapter import LocalRerankerAdapter
from raglab.infrastructure.retrieval.sentence_anchor_adapter import (
    SentenceAnchorAdapter,
)
from raglab.infrastructure.retrieval.sentence_window_adapter import (
    SentenceWindowAdapter,
)


class ParityGateError(Exception):
    """Exception raised when Parity Gate execution encounters fatal errors."""


def load_human_qrels(qrels_path: str | Path) -> dict[tuple[str, str], float]:
    """Load human qrels mapping (qid, passage_id) -> grade."""
    p = Path(qrels_path)
    if not p.exists():
        raise ParityGateError(f"Human qrels file missing: {p}")

    qrels: dict[tuple[str, str], float] = {}
    lines = p.read_text(encoding="utf-8").splitlines()
    for line_idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as err:
            raise ParityGateError(
                f"Invalid JSON at line {line_idx} in qrels: {err}"
            ) from err

        qid = data.get("question_id") or data.get("qid")
        pid = data.get("passage_id")
        grade = data.get("relevance_grade")
        if grade is None:
            grade = data.get("grade")

        if qid and pid and grade is not None:
            qrels[(qid, pid)] = float(grade)
    return qrels


def execute_arm_retrieval(
    arm_id: str,
    query: str,
    qid: str,
    chunks: list[Any],
    pages: list[Any],
    top_k: int = 3,
) -> tuple[list[tuple[str, float]], float]:
    """Execute retrieval arm in-memory. Return list of (passage_id, score)."""
    start_t = time.perf_counter()

    if arm_id == "F0":
        adapter = InMemoryBaselineAdapter()
        adapter.index_chunks(chunks)
        evidences = adapter.retrieve(query, top_k=top_k)
        results = [(e.passage_id or e.chunk_id.value, e.score) for e in evidences]

    elif arm_id == "S0":
        s_adapter = SentenceAnchorAdapter()
        s_adapter.index_pages(pages)
        evidences = s_adapter.retrieve(query, top_k=top_k)
        results = [(e.passage_id or e.chunk_id.value, e.score) for e in evidences]

    elif arm_id == "S1":
        base_s = SentenceAnchorAdapter()
        base_s.index_pages(pages)
        candidates = base_s.retrieve(query, top_k=top_k * 3)
        reranker = LocalRerankerAdapter()
        reranked, _ = reranker.rerank(query, candidates, top_n=top_k)
        results = [(e.passage_id or e.chunk_id.value, e.score) for e in reranked]

    elif arm_id == "W0":
        w_adapter = SentenceWindowAdapter(window_size=3)
        w_adapter.index_pages(pages)
        evidences = w_adapter.retrieve(query, top_k=top_k)
        results = [(e.passage_id or e.chunk_id.value, e.score) for e in evidences]

    elif arm_id == "W1":
        base_w = SentenceWindowAdapter(window_size=3)
        base_w.index_pages(pages)
        candidates = base_w.retrieve(query, top_k=top_k * 3)
        reranker = LocalRerankerAdapter()
        reranked, _ = reranker.rerank(query, candidates, top_n=top_k)
        results = [(e.passage_id or e.chunk_id.value, e.score) for e in reranked]

    elif arm_id == "C0":
        c_adapter = HierarchicalRetrievalAdapter()
        c_adapter.index_pages(pages)
        evidences = c_adapter.retrieve(query, top_k=top_k)
        results = [(e.passage_id or e.chunk_id.value, e.score) for e in evidences]

    elif arm_id == "C1":
        base_c = HierarchicalRetrievalAdapter()
        base_c.index_pages(pages)
        candidates = base_c.retrieve(query, top_k=top_k * 3)
        reranker = LocalRerankerAdapter()
        reranked, _ = reranker.rerank(query, candidates, top_n=top_k)
        results = [(e.passage_id or e.chunk_id.value, e.score) for e in reranked]

    else:
        raise ParityGateError(f"Unknown arm_id: {arm_id}")

    elapsed = time.perf_counter() - start_t
    return results[:top_k], elapsed


def run_parity_gate_evaluation(
    protocol: dict[str, Any],
    passage_registry_path: str | Path,
    passage_registry_manifest_path: str | Path,
    human_qrels_path: str | Path,
    questions_path: str | Path,
    top_k: int = 3,
) -> tuple[ParityGateResult, list[ArmRun], list[dict[str, Any]]]:
    """Run full Parity Gate evaluation across all DEV QIDs and 7 fixed arms.

    Returns:
        Tuple of (ParityGateResult, list[ArmRun], human_review_queue).
    """
    # 1. Load canonical corpus snapshot
    expected_reg_sha = protocol.get("passage_registry_sha256")
    corpus_snapshot, raw_passages = load_canonical_corpus_snapshot(
        passage_registry_path,
        passage_registry_manifest_path,
        expected_registry_sha256=expected_reg_sha,
    )

    canonical_set = set(corpus_snapshot.ordered_canonical_passage_ids)
    passage_text_map = {p["passage_id"]: p["text"] for p in raw_passages}
    passage_page_map = {p["passage_id"]: p["page_number"] for p in raw_passages}

    chunks = passages_to_chunks(raw_passages)
    pages = passages_to_document_pages(raw_passages)

    # 2. Load human qrels
    qrels = load_human_qrels(human_qrels_path)

    # 3. Load DEV questions
    q_path = Path(questions_path)
    if not q_path.exists():
        raise ParityGateError(f"Questions file missing: {q_path}")

    q_data = json.loads(q_path.read_text(encoding="utf-8"))
    questions_raw = q_data.get("questions", [])

    dev_qids = protocol.get(
        "dev_qids", ["q_dev_01", "q_dev_02", "q_dev_03", "q_dev_04"]
    )
    dev_questions = [
        q for q in questions_raw
        if q.get("qid") in dev_qids or q.get("question_id") in dev_qids
    ]

    fixed_arms = protocol.get("fixed_arms", ["F0", "S0", "S1", "W0", "W1", "C0", "C1"])

    effective_top_k = protocol.get("top_k", top_k)

    # First Execution Run
    arm_runs_1: list[ArmRun] = []
    unmapped_count = 0
    unjudged_count = 0
    total_retrieved_items = 0
    canonical_mapped_items = 0
    explicit_judged_items = 0
    synthetic_fallback_count = 0

    unjudged_queue_map: dict[tuple[str, str], dict[str, Any]] = {}

    for q_item in dev_questions:
        qid = q_item.get("qid") or q_item.get("question_id")
        q_text = q_item.get("question_text") or q_item.get("text", "")

        if not qid or not q_text:
            continue

        for arm_id in fixed_arms:
            raw_results, latency = execute_arm_retrieval(
                arm_id=arm_id,
                query=q_text,
                qid=qid,
                chunks=chunks,
                pages=pages,
                top_k=effective_top_k,
            )

            retrieved_items: list[CanonicalRetrievedItem] = []
            res_content_hashes: list[str] = []

            for rank, (pid, score) in enumerate(raw_results, start=1):
                page_num = passage_page_map.get(pid, 0)
                total_retrieved_items += 1

                # Mapping check
                if pid in canonical_set:
                    mapping_status = MappingStatus.EXACT_SUBSTRING
                    canonical_mapped_items += 1
                else:
                    mapping_status = MappingStatus.UNMAPPED
                    unmapped_count += 1

                # Judgment check
                grade = qrels.get((qid, pid))
                if grade is not None:
                    judgment_status = JudgmentStatus.JUDGED
                    explicit_judged_items += 1
                else:
                    judgment_status = JudgmentStatus.UNJUDGED
                    unjudged_count += 1

                    # Queue item for unjudged human review
                    key = (qid, pid)
                    if key not in unjudged_queue_map:
                        unjudged_queue_map[key] = {
                            "qid": qid,
                            "passage_id": pid,
                            "page_number": page_num,
                            "text": passage_text_map.get(pid, ""),
                            "arms_recovering": [arm_id],
                            "ranks_recovering": [rank],
                        }
                    else:
                        unjudged_queue_map[key]["arms_recovering"].append(arm_id)
                        unjudged_queue_map[key]["ranks_recovering"].append(rank)

                c_text = passage_text_map.get(pid, f"text_{pid}")
                c_hash = hashlib.sha256(c_text.encode("utf-8")).hexdigest()
                res_content_hashes.append(c_hash)

                retrieved_items.append(
                    CanonicalRetrievedItem(
                        qid=qid,
                        arm_id=arm_id,
                        rank=rank,
                        passage_id=pid,
                        page_number=page_num,
                        content_sha256=c_hash,
                        score=score,
                        mapping_status=mapping_status,
                        judgment_status=judgment_status,
                        human_grade=grade,
                    )
                )

            deterministic_content_hash = hashlib.sha256(
                json.dumps(res_content_hashes).encode("utf-8")
            ).hexdigest()

            arm_runs_1.append(
                ArmRun(
                    qid=qid,
                    arm_id=arm_id,
                    retrieved_items=retrieved_items,
                    latency=latency,
                    configuration_hashes={
                        "arm_id": arm_id,
                        "corpus_sha256": corpus_snapshot.corpus_sha256,
                    },
                    corpus_hash=corpus_snapshot.corpus_sha256,
                    deterministic_content_hash=deterministic_content_hash,
                )
            )

    # Second Execution Run for Repeatability Check
    arm_runs_2: list[ArmRun] = []
    for q_item in dev_questions:
        qid = q_item.get("qid") or q_item.get("question_id")
        q_text = q_item.get("question_text") or q_item.get("text", "")
        if not qid or not q_text:
            continue

        for arm_id in fixed_arms:
            raw_results, latency = execute_arm_retrieval(
                arm_id=arm_id,
                query=q_text,
                qid=qid,
                chunks=chunks,
                pages=pages,
                top_k=effective_top_k,
            )
            items_2: list[CanonicalRetrievedItem] = []
            hashes_2: list[str] = []

            for rank, (pid, score) in enumerate(raw_results, start=1):
                page_num = passage_page_map.get(pid, 0)
                c_text = passage_text_map.get(pid, f"text_{pid}")
                c_hash = hashlib.sha256(c_text.encode("utf-8")).hexdigest()
                hashes_2.append(c_hash)
                grade = qrels.get((qid, pid))
                is_mapped = pid in canonical_set
                m_status = (
                    MappingStatus.EXACT_SUBSTRING
                    if is_mapped
                    else MappingStatus.UNMAPPED
                )
                j_status = (
                    JudgmentStatus.JUDGED
                    if grade is not None
                    else JudgmentStatus.UNJUDGED
                )
                items_2.append(
                    CanonicalRetrievedItem(
                        qid=qid,
                        arm_id=arm_id,
                        rank=rank,
                        passage_id=pid,
                        page_number=page_num,
                        content_sha256=c_hash,
                        score=score,
                        mapping_status=m_status,
                        judgment_status=j_status,
                        human_grade=grade,
                    )
                )

            deterministic_content_hash = hashlib.sha256(
                json.dumps(hashes_2).encode("utf-8")
            ).hexdigest()

            arm_runs_2.append(
                ArmRun(
                    qid=qid,
                    arm_id=arm_id,
                    retrieved_items=items_2,
                    latency=latency,
                    configuration_hashes={
                        "arm_id": arm_id,
                        "corpus_sha256": corpus_snapshot.corpus_sha256,
                    },
                    corpus_hash=corpus_snapshot.corpus_sha256,
                    deterministic_content_hash=deterministic_content_hash,
                )
            )

    # Compare runs for repeatability
    comparisons, topk_identity, rank_match = compare_arm_runs(arm_runs_1, arm_runs_2)

    # Compute coverages
    if total_retrieved_items > 0:
        canonical_coverage = canonical_mapped_items / total_retrieved_items
        judged_coverage = explicit_judged_items / total_retrieved_items
    else:
        canonical_coverage = 0.0
        judged_coverage = 0.0

    # Same-run arm completeness
    expected_run_count = len(dev_questions) * len(fixed_arms)
    same_run_arm_completeness = (len(arm_runs_1) == expected_run_count)

    failure_reasons: list[str] = []
    if canonical_coverage < 1.0 or unmapped_count > 0:
        failure_reasons.append(
            f"Canonical coverage incomplete: {canonical_mapped_items}/"
            f"{total_retrieved_items} mapped ({unmapped_count} unmapped)"
        )
    if judged_coverage < 1.0 or unjudged_count > 0:
        failure_reasons.append(
            f"Judged coverage incomplete: {explicit_judged_items}/"
            f"{total_retrieved_items} judged ({unjudged_count} unjudged)"
        )
    if topk_identity < 1.0 or not rank_match:
        failure_reasons.append(
            f"Retrieval non-determinism detected: "
            f"topk_identity={topk_identity}, rank_match={rank_match}"
        )
    if not same_run_arm_completeness:
        failure_reasons.append("Same-run arm completeness failed")

    # Determine status
    if unmapped_count > 0 or canonical_coverage < 1.0:
        status = ParityOutcomeCategory.NOT_EVALUABLE_CANONICAL_COVERAGE
        metrics_status = "NOT_APPLICABLE"
        metrics = None
    elif unjudged_count > 0 or judged_coverage < 1.0:
        status = ParityOutcomeCategory.NOT_EVALUABLE_JUDGED_COVERAGE
        metrics_status = "NOT_APPLICABLE"
        metrics = None
    elif topk_identity < 1.0 or not rank_match:
        status = ParityOutcomeCategory.NOT_EVALUABLE_RETRIEVAL_NONDETERMINISM
        metrics_status = "NOT_APPLICABLE"
        metrics = None
    else:
        status = ParityOutcomeCategory.SCIENTIFICALLY_EVALUABLE
        metrics_status = "APPLICABLE"
        metrics = {"status": "EVALUABLE_PLACEHOLDER"}

    protocol_str = json.dumps(protocol, sort_keys=True)
    protocol_hash = hashlib.sha256(protocol_str.encode("utf-8")).hexdigest()

    result = ParityGateResult(
        protocol_hash=protocol_hash,
        input_hashes={
            "passage_registry_sha256": corpus_snapshot.corpus_sha256,
            "human_qrels_sha256": protocol.get("human_qrels_sha256", ""),
        },
        canonical_coverage=canonical_coverage,
        judged_coverage=judged_coverage,
        unmapped_count=unmapped_count,
        unjudged_count=unjudged_count,
        synthetic_fallback_count=synthetic_fallback_count,
        repeatability_topk_identity=topk_identity,
        repeatability_rank_match=rank_match,
        same_run_arm_completeness=same_run_arm_completeness,
        status=status,
        failure_reasons=failure_reasons,
        metrics_status=metrics_status,
        metrics=metrics,
    )

    unjudged_queue = list(unjudged_queue_map.values())
    return result, arm_runs_1, unjudged_queue
