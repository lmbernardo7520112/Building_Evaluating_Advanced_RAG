"""Slice 5A.3.1 Parity Gate V3 execution and human coverage evaluation engine."""

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
from raglab.evaluation.contracts.human_annotation_v2 import PassageRegistryEntry
from raglab.evaluation.contracts.hybrid_eval_v2 import CanonicalMappingStatus
from raglab.evaluation.pooling.canonical_passage_mapper import CanonicalPassageMapper
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
    """Load human qrels mapping (qid, passage_id) -> grade strictly from human_qrels_final.jsonl.

    Enforces strict parser rules:
    - Must have question_id or qid
    - Must have passage_id
    - Must have relevance_grade (integer or float)
    - No fallback chaining!
    """
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

        if grade is None and "relevance_grade" not in data:
            raise ParityGateError(
                f"Strict Qrels Parser Error: Line {line_idx} lacks required 'relevance_grade' field"
            )

        if qid and pid and grade is not None:
            qrels[(qid, pid)] = float(grade)
    return qrels


class _RerankedRetrieverShim:

    def __init__(self, base_retriever: Any, reranker: Any, candidate_k: int) -> None:
        self._base = base_retriever
        self._reranker = reranker
        self._candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int = 3) -> list[Any]:
        candidates = self._base.retrieve(query, top_k=self._candidate_k)
        reranked, _ = self._reranker.rerank(query, candidates, top_n=top_k)
        return list(reranked)


def build_all_arm_retrievers(
    chunks: list[Any],
    pages: list[Any],
    embed_model: Any | None = None,
) -> dict[str, Any]:
    """Build and index all 7 consolidated retrieval arms ONCE for efficient query execution."""
    # F0
    adapter_f0 = InMemoryBaselineAdapter()
    adapter_f0.index_chunks(chunks)

    # S0
    adapter_s0 = SentenceAnchorAdapter(embedding_adapter=embed_model)
    adapter_s0.index_pages(pages)

    # W0
    adapter_w0 = SentenceWindowAdapter(
        embedding_adapter=embed_model, window_size=3
    )
    adapter_w0.index_pages(pages)

    # W1
    base_w1 = SentenceWindowAdapter(
        embedding_adapter=embed_model, window_size=3
    )
    base_w1.index_pages(pages)
    reranker_w1 = LocalRerankerAdapter(embedding_adapter=embed_model)
    adapter_w1 = _RerankedRetrieverShim(base_w1, reranker_w1, candidate_k=9)

    # H0
    adapter_h0 = HierarchicalRetrievalAdapter(auto_merge=False, top_k=3)
    adapter_h0.index_pages(pages)

    # H1
    adapter_h1 = HierarchicalRetrievalAdapter(auto_merge=True, top_k=3)
    adapter_h1.index_pages(pages)

    # H2
    base_h2 = HierarchicalRetrievalAdapter(auto_merge=True, top_k=9)
    base_h2.index_pages(pages)
    reranker_h2 = LocalRerankerAdapter(embedding_adapter=embed_model)
    adapter_h2 = _RerankedRetrieverShim(base_h2, reranker_h2, candidate_k=9)

    return {
        "F0": adapter_f0,
        "F0_baseline": adapter_f0,
        "baseline": adapter_f0,
        "S0": adapter_s0,
        "S0_sentence_anchor": adapter_s0,
        "sentence_anchor": adapter_s0,
        "W0": adapter_w0,
        "W0_sentence_window": adapter_w0,
        "sentence_window": adapter_w0,
        "W1": adapter_w1,
        "W1_sentence_window_rerank": adapter_w1,
        "sentence_window_rerank": adapter_w1,
        "H0": adapter_h0,
        "H0_hierarchical_leaf": adapter_h0,
        "hierarchical_leaf": adapter_h0,
        "C0": adapter_h0,
        "H1": adapter_h1,
        "H1_auto_merging": adapter_h1,
        "auto_merging": adapter_h1,
        "H2": adapter_h2,
        "H2_auto_merging_rerank": adapter_h2,
        "auto_merging_rerank": adapter_h2,
        "C1": adapter_h2,
    }


def run_parity_gate_evaluation(
    protocol: dict[str, Any],
    passage_registry_path: str | Path,
    passage_registry_manifest_path: str | Path,
    human_qrels_path: str | Path,
    questions_path: str | Path,
    top_k: int = 3,
) -> tuple[ParityGateResult, list[ArmRun], list[dict[str, Any]], dict[str, Any]]:
    """Run full Parity Gate V3 evaluation across DEV QIDs and 7 fixed arms.

    Returns:
        Tuple of (ParityGateResult, list[ArmRun], human_review_queue, execution_metadata).
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
    passage_start_map = {p["passage_id"]: p.get("start_char", 0) for p in raw_passages}
    passage_end_map = {p["passage_id"]: p.get("end_char", len(p["text"])) for p in raw_passages}

    chunks = passages_to_chunks(raw_passages)
    pages = passages_to_document_pages(raw_passages)

    mapper = CanonicalPassageMapper(
        [
            PassageRegistryEntry(
                passage_id=p["passage_id"],
                document_id=p.get("document_id", "gersting_discrete_math"),
                page_number=p["page_number"],
                start_char=p.get("start_char", 0),
                end_char=p.get("end_char", len(p["text"])),
                content_sha256=p.get(
                    "content_sha256",
                    hashlib.sha256(p["text"].encode("utf-8")).hexdigest(),
                ),
                text=p["text"],
            )
            for p in raw_passages
        ]
    )

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
        q
        for q in questions_raw
        if q.get("qid") in dev_qids or q.get("question_id") in dev_qids
    ]

    fixed_arms = protocol.get(
        "fixed_arms", ["F0", "H0", "H1", "H2", "S0", "W0", "W1"]
    )

    effective_top_k = protocol.get("top_k", top_k)

    retrievers_1 = build_all_arm_retrievers(chunks, pages)

    # First Execution Run
    arm_runs_1: list[ArmRun] = []
    unmapped_count = 0
    ambiguous_count = 0
    unjudged_count = 0
    total_retrieved_items = 0
    canonical_mapped_items = 0
    explicit_judged_items = 0
    synthetic_fallback_count = 0

    unique_canonical_pairs: set[tuple[str, str]] = set()

    unjudged_queue_map: dict[tuple[str, str], dict[str, Any]] = {}
    retrieval_matrix_rows: list[dict[str, Any]] = []

    for q_item in dev_questions:
        qid = q_item.get("qid") or q_item.get("question_id")
        q_text = (
            q_item.get("question")
            or q_item.get("question_text")
            or q_item.get("text", "")
        )

        if not qid or not q_text:
            continue

        for arm_id in fixed_arms:
            retriever = retrievers_1.get(arm_id)
            if not retriever:
                raise ParityGateError(f"Unknown arm_id: {arm_id}")

            start_t = time.perf_counter()
            raw_evidences = retriever.retrieve(q_text, top_k=effective_top_k)
            latency = time.perf_counter() - start_t

            retrieved_count = len(raw_evidences)
            missing_count = effective_top_k - retrieved_count
            missing_ranks = list(range(retrieved_count + 1, effective_top_k + 1))
            reason = (
                "Full top_k returned"
                if missing_count == 0
                else "Candidate node consolidation during hierarchical auto-merging"
            )

            retrieval_matrix_rows.append(
                {
                    "qid": qid,
                    "arm": arm_id,
                    "requested_top_k": effective_top_k,
                    "returned_count": retrieved_count,
                    "missing_count": missing_count,
                    "missing_ranks": missing_ranks,
                    "reason": reason,
                }
            )

            retrieved_items: list[CanonicalRetrievedItem] = []
            res_content_hashes: list[str] = []

            for rank, ev in enumerate(raw_evidences[:effective_top_k], start=1):
                total_retrieved_items += 1

                chunk_id_str = str(
                    getattr(ev.chunk_id, "value", ev.chunk_id)
                    if hasattr(ev, "chunk_id")
                    else getattr(ev, "passage_id", "unknown")
                )
                node_id_str = getattr(ev, "node_id", None)
                doc_id_raw = str(
                    getattr(ev, "document_id", "gersting_discrete_math")
                )
                text_raw = str(getattr(ev, "text", "")).strip()
                score = float(getattr(ev, "score", 0.0))

                page_num = getattr(ev, "page_number", 0) or getattr(
                    ev, "start_page", 0
                )
                if page_num == 0 and "_p" in doc_id_raw:
                    suf = doc_id_raw.rsplit("_p", 1)[-1]
                    if suf.isdigit():
                        page_num = int(suf)

                content_sha = hashlib.sha256(text_raw.encode("utf-8")).hexdigest()

                chunk_dict = {
                    "chunk_id": chunk_id_str,
                    "document_id": "gersting_discrete_math",
                    "text": text_raw,
                    "content_sha256": content_sha,
                    "page_number": page_num,
                }
                map_res = mapper.map_chunk(chunk_dict)

                canonical_pid = map_res.mapped_passage_id
                proj_method = map_res.mapping_status.value

                source_offsets = (
                    (passage_start_map[canonical_pid], passage_end_map[canonical_pid])
                    if canonical_pid in passage_start_map
                    else None
                )

                if map_res.mapping_status == CanonicalMappingStatus.AMBIGUOUS_NEEDS_REVIEW:
                    ambiguous_count += 1
                    mapping_status = MappingStatus.AMBIGUOUS
                elif canonical_pid is not None and canonical_pid in canonical_set:
                    canonical_mapped_items += 1
                    unique_canonical_pairs.add((qid, canonical_pid))
                    if (
                        map_res.mapping_status
                        == CanonicalMappingStatus.EXACT_PASSAGE_ID
                    ):
                        mapping_status = MappingStatus.DIRECT_MATCH
                    else:
                        mapping_status = MappingStatus.EXACT_SUBSTRING
                else:
                    mapping_status = MappingStatus.UNMAPPED
                    unmapped_count += 1

                # Judgment check
                if canonical_pid is not None:
                    grade = qrels.get((qid, canonical_pid))
                    if grade is not None:
                        judgment_status = JudgmentStatus.JUDGED
                        explicit_judged_items += 1
                    else:
                        judgment_status = JudgmentStatus.UNJUDGED
                        unjudged_count += 1
                        key = (qid, canonical_pid)
                        if key not in unjudged_queue_map:
                            unjudged_queue_map[key] = {
                                "qid": qid,
                                "passage_id": canonical_pid,
                                "page_number": page_num,
                                "text": passage_text_map.get(
                                    canonical_pid, text_raw
                                ),
                                "arms_recovering": [arm_id],
                                "ranks_recovering": [rank],
                            }
                        else:
                            unjudged_queue_map[key]["arms_recovering"].append(
                                arm_id
                            )
                            unjudged_queue_map[key]["ranks_recovering"].append(
                                rank
                            )
                else:
                    grade = None
                    judgment_status = JudgmentStatus.UNJUDGED
                    unjudged_count += 1

                c_hash = content_sha
                res_content_hashes.append(c_hash)

                retrieved_items.append(
                    CanonicalRetrievedItem(
                        qid=qid,
                        arm_id=arm_id,
                        rank=rank,
                        technical_chunk_id=chunk_id_str,
                        technical_node_id=node_id_str,
                        anchor_passage_id=canonical_pid,
                        supporting_passage_ids=[],
                        source_offsets=source_offsets,
                        projection_method=proj_method,
                        page_number=page_num
                        or passage_page_map.get(canonical_pid or "", 0),
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
    retrievers_2 = build_all_arm_retrievers(chunks, pages)
    arm_runs_2: list[ArmRun] = []

    for q_item in dev_questions:
        qid = q_item.get("qid") or q_item.get("question_id")
        q_text = (
            q_item.get("question")
            or q_item.get("question_text")
            or q_item.get("text", "")
        )
        if not qid or not q_text:
            continue

        for arm_id in fixed_arms:
            retriever = retrievers_2.get(arm_id)
            if not retriever:
                raise ParityGateError(f"Unknown arm_id: {arm_id}")

            start_t = time.perf_counter()
            raw_evidences = retriever.retrieve(q_text, top_k=effective_top_k)
            latency = time.perf_counter() - start_t

            items_2: list[CanonicalRetrievedItem] = []
            hashes_2: list[str] = []

            for rank, ev in enumerate(raw_evidences[:effective_top_k], start=1):
                chunk_id_str = str(
                    getattr(ev.chunk_id, "value", ev.chunk_id)
                    if hasattr(ev, "chunk_id")
                    else getattr(ev, "passage_id", "unknown")
                )
                node_id_str = getattr(ev, "node_id", None)
                doc_id_raw = str(
                    getattr(ev, "document_id", "gersting_discrete_math")
                )
                text_raw = str(getattr(ev, "text", "")).strip()
                score = float(getattr(ev, "score", 0.0))

                page_num = getattr(ev, "page_number", 0) or getattr(
                    ev, "start_page", 0
                )
                if page_num == 0 and "_p" in doc_id_raw:
                    suf = doc_id_raw.rsplit("_p", 1)[-1]
                    if suf.isdigit():
                        page_num = int(suf)

                c_hash = hashlib.sha256(text_raw.encode("utf-8")).hexdigest()
                hashes_2.append(c_hash)

                chunk_dict = {
                    "chunk_id": chunk_id_str,
                    "document_id": "gersting_discrete_math",
                    "text": text_raw,
                    "content_sha256": c_hash,
                    "page_number": page_num,
                }
                map_res = mapper.map_chunk(chunk_dict)
                canonical_pid = map_res.mapped_passage_id
                proj_method = map_res.mapping_status.value

                source_offsets = (
                    (passage_start_map[canonical_pid], passage_end_map[canonical_pid])
                    if canonical_pid in passage_start_map
                    else None
                )

                is_mapped = canonical_pid is not None and canonical_pid in canonical_set
                m_status = (
                    MappingStatus.DIRECT_MATCH
                    if (
                        is_mapped
                        and map_res.mapping_status
                        == CanonicalMappingStatus.EXACT_PASSAGE_ID
                    )
                    else MappingStatus.EXACT_SUBSTRING
                    if is_mapped
                    else MappingStatus.UNMAPPED
                )

                grade = (
                    qrels.get((qid, canonical_pid))
                    if canonical_pid is not None
                    else None
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
                        technical_chunk_id=chunk_id_str,
                        technical_node_id=node_id_str,
                        anchor_passage_id=canonical_pid,
                        supporting_passage_ids=[],
                        source_offsets=source_offsets,
                        projection_method=proj_method,
                        page_number=page_num
                        or passage_page_map.get(canonical_pid or "", 0),
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

    expected_technical_slots = len(dev_questions) * len(fixed_arms) * effective_top_k
    missing_technical_slots = expected_technical_slots - total_retrieved_items

    same_run_arm_completeness = (
        len(arm_runs_1) == len(dev_questions) * len(fixed_arms)
    )

    failure_reasons: list[str] = []
    if ambiguous_count > 0:
        failure_reasons.append(
            f"Ambiguous canonical projection detected: {ambiguous_count} ambiguous items"
        )
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

    # Fail-closed outcome order
    if ambiguous_count > 0:
        status = ParityOutcomeCategory.NOT_EVALUABLE_AMBIGUOUS_PROJECTION
        metrics_status = "NOT_APPLICABLE"
        metrics = None
    elif unmapped_count > 0 or canonical_coverage < 1.0:
        status = ParityOutcomeCategory.NOT_EVALUABLE_CANONICAL_COVERAGE
        metrics_status = "NOT_APPLICABLE"
        metrics = None
    elif topk_identity < 1.0 or not rank_match:
        status = ParityOutcomeCategory.NOT_EVALUABLE_REPEATABILITY
        metrics_status = "NOT_APPLICABLE"
        metrics = None
    elif unjudged_count > 0 or judged_coverage < 1.0:
        status = ParityOutcomeCategory.NOT_EVALUABLE_JUDGED_COVERAGE
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

    execution_metadata = {
        "expected_technical_slots": expected_technical_slots,
        "technical_items_returned": total_retrieved_items,
        "missing_technical_slots": missing_technical_slots,
        "canonical_occurrences_mapped": canonical_mapped_items,
        "unique_canonical_pairs_count": len(unique_canonical_pairs),
        "duplicate_projection_count": canonical_mapped_items - len(unique_canonical_pairs),
        "unmapped_count": unmapped_count,
        "ambiguous_count": ambiguous_count,
        "retrieval_matrix": retrieval_matrix_rows,
    }

    return result, arm_runs_1, unjudged_queue, execution_metadata
