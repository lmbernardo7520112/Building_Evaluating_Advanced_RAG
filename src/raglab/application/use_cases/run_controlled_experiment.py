"""Run Controlled Experiment Use Case for RAGLab v7 (Slice 2).

Executes pre-registered controlled evaluation across F0, W0, and W1
over the audited textbook sub-corpus (pages 91-115).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from raglab.domain.entities import Checkpoint, Chunk, RetrievedEvidence
from raglab.domain.value_objects import ChunkId, IntegrityDigest, RunId
from raglab.infrastructure.embeddings.fastembed_adapter import FastEmbedEmbeddingAdapter
from raglab.infrastructure.pdf_parsers.pdf_parser_adapter import PyPdfExtractorAdapter
from raglab.infrastructure.persistence.checkpoint_store import FilesystemCheckpointStore
from raglab.infrastructure.retrieval.baseline_adapter import InMemoryBaselineAdapter
from raglab.infrastructure.retrieval.reranker_adapter import (
    LocalRerankerAdapter,
    RerankerDamageMetrics,
)
from raglab.infrastructure.retrieval.sentence_window_adapter import (
    SentenceWindowAdapter,
)


@dataclass(frozen=True, slots=True)
class QueryEvaluationResult:
    qid: str
    split: str
    question: str
    is_abstention: bool
    relevant_page: int | None
    retrieved_pages: list[int]
    recall: float | None
    mrr: float | None
    hit: bool | None
    precision: float | None


def compute_bootstrap_ci(
    diffs: Sequence[float], num_resamples: int = 1000, seed: int = 42
) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval for paired differences."""
    if not diffs:
        return (0.0, 0.0)

    rng = random.Random(seed)  # noqa: S311
    n = len(diffs)
    means: list[float] = []

    for _ in range(num_resamples):
        sample = [rng.choice(diffs) for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    lower_idx = int(0.025 * num_resamples)
    upper_idx = int(0.975 * num_resamples)
    return (round(means[lower_idx], 4), round(means[upper_idx], 4))


def compute_cohens_d(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Cohen's d effect size for paired samples."""
    if not x or not y or len(x) != len(y):
        return 0.0

    diffs = [a - b for a, b in zip(x, y, strict=False)]
    mean_diff = sum(diffs) / len(diffs)

    val = (
        sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)
        if len(diffs) > 1
        else 0.0
    )
    std_dev = math.sqrt(val)

    if std_dev == 0.0:
        return 0.0

    return round(mean_diff / std_dev, 4)


class RunControlledExperimentUseCase:
    """Orchestrates Slice 2 controlled experiment evaluation."""

    def __init__(
        self,
        pdf_path: str,
        questions_path: str,
        checkpoint_dir: str,
        page_start: int = 91,
        page_end: int = 115,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        self.questions_path = Path(questions_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.page_start = page_start
        self.page_end = page_end

        self.pdf_extractor = PyPdfExtractorAdapter()
        self.embedding_adapter = FastEmbedEmbeddingAdapter()
        self.reranker_adapter = LocalRerankerAdapter(self.embedding_adapter)
        self.checkpoint_store = FilesystemCheckpointStore(self.checkpoint_dir)

    def run(self, run_id: str = "run_controlled_slice2") -> dict[str, Any]:
        print(f"=== Starting Controlled Experiment ({run_id}) ===")
        print(
            f"Sub-corpus PDF: {self.pdf_path.name} "
            f"(pages {self.page_start}-{self.page_end})"
        )

        # 1. Extract sub-corpus pages with audit
        pages, audit_report = self.pdf_extractor.extract_pages_with_audit(
            str(self.pdf_path), page_start=self.page_start, page_end=self.page_end
        )
        print(f"Extracted {len(pages)} physical pages.")

        # 2. Load questions
        q_data = json.loads(self.questions_path.read_text())
        all_questions = q_data.get("questions", [])

        # Filter out sealed holdout questions
        active_questions = [
            q for q in all_questions if q.get("split") in ("development", "test")
        ]
        print(
            f"Loaded {len(active_questions)} active questions "
            "(development + test splits). Holdout sealed."
        )

        # 3. Setup Pipelines
        # F0 — Baseline (fixed chunks: 512 chars)
        f0_chunks: list[Chunk] = []
        for p in pages:
            text = p.text
            step = 448
            for i in range(0, len(text), step):
                chunk_str = text[i : i + 512]
                if chunk_str.strip():
                    cid = f"{p.document_id}_p{p.page_number}_c{i}"
                    f0_chunks.append(
                        Chunk(
                            chunk_id=ChunkId(cid),
                            document_id=p.document_id,
                            start_page=p.page_number,
                            end_page=p.page_number,
                            text=chunk_str,
                        )
                    )

        f0_adapter = InMemoryBaselineAdapter()
        f0_adapter.index_chunks(f0_chunks)

        # W0 & W1 — Sentence Window Adapter
        sw_adapter = SentenceWindowAdapter(
            embedding_adapter=self.embedding_adapter, window_size=2
        )
        sw_adapter.index_pages(pages)

        # 4. Execute Evaluations
        f0_results = self._eval_pipeline_f0(f0_adapter, active_questions, top_k=3)
        w0_results = self._eval_pipeline_w0(sw_adapter, active_questions, top_k=3)
        w1_results, damage_metrics_list = self._eval_pipeline_w1(
            sw_adapter, self.reranker_adapter, active_questions, candidate_k=6, top_n=3
        )

        # 5. Statistical Comparisons
        f0_vs_w0 = self._compare_pipelines(
            "F0_Baseline", "W0_SentenceWindow", f0_results, w0_results
        )
        w0_vs_w1 = self._compare_pipelines(
            "W0_SentenceWindow", "W1_RerankedWindow", w0_results, w1_results
        )

        # Aggregate Damage Metrics for W1
        avg_dropped_rate = (
            sum(d.relevant_passage_dropped_rate for d in damage_metrics_list)
            / len(damage_metrics_list)
            if damage_metrics_list
            else 0.0
        )
        avg_delta_recall = (
            sum(d.delta_recall for d in damage_metrics_list)
            / len(damage_metrics_list)
            if damage_metrics_list
            else 0.0
        )

        experiment_summary = {
            "run_id": run_id,
            "corpus_audit": {
                "filename": self.pdf_path.name,
                "doc_fingerprint": audit_report["doc_fingerprint"],
                "pages_range": [self.page_start, self.page_end],
                "pages_extracted": len(pages),
            },
            "embedding_model": {
                "model_id": self.embedding_adapter.model_id,
                "dimension": self.embedding_adapter.dimension,
                "execution_mode": "local_onnx_cpu",
            },
            "evaluations": {
                "F0_Baseline": self._summarize_results(f0_results),
                "W0_SentenceWindow": self._summarize_results(w0_results),
                "W1_RerankedWindow": self._summarize_results(w1_results),
            },
            "comparisons": {
                "F0_vs_W0": f0_vs_w0,
                "W0_vs_W1": w0_vs_w1,
            },
            "reranker_damage_summary": {
                "candidate_k": 6,
                "top_n": 3,
                "mean_delta_recall": round(avg_delta_recall, 4),
                "mean_relevant_passage_dropped_rate": round(avg_dropped_rate, 4),
                "total_queries_evaluated": len(damage_metrics_list),
            },
            "detailed_per_question": {
                "F0": [asdict(r) for r in f0_results],
                "W0": [asdict(r) for r in w0_results],
                "W1": [asdict(r) for r in w1_results],
            },
        }

        # 6. Save Checkpoint
        ckp = Checkpoint(
            run_id=RunId(run_id),
            corpus_fingerprint=IntegrityDigest(audit_report["doc_fingerprint"]),
            config_fingerprint=IntegrityDigest(hashlib.sha256(b"slice2_controlled_config").hexdigest()),
            completed_query_ids=frozenset(q["qid"] for q in active_questions),
        )
        self.checkpoint_store.save(ckp)
        print("Checkpoint saved successfully.")

        return experiment_summary

    @staticmethod
    def _parse_page_number(e: RetrievedEvidence) -> int:
        match = re.search(r"_p(\d+)", e.chunk_id.value)
        if match:
            return int(match.group(1))
        match_doc = re.search(r"_p(\d+)", e.document_id)
        if match_doc:
            return int(match_doc.group(1))
        return 1

    def _eval_pipeline_f0(
        self,
        adapter: InMemoryBaselineAdapter,
        questions: list[dict[str, Any]],
        top_k: int = 3,
    ) -> list[QueryEvaluationResult]:
        results: list[QueryEvaluationResult] = []
        for q in questions:
            qid = q["qid"]
            split = q["split"]
            text = q["question"]
            is_abs = q["is_abstention"]
            rel_page = q["relevant_page"]

            evidences = adapter.retrieve(text, top_k=top_k)
            ret_pages = [self._parse_page_number(e) for e in evidences]

            if is_abs or rel_page is None:
                results.append(
                    QueryEvaluationResult(
                        qid=qid,
                        split=split,
                        question=text,
                        is_abstention=True,
                        relevant_page=None,
                        retrieved_pages=ret_pages,
                        recall=None,
                        mrr=None,
                        hit=None,
                        precision=None,
                    )
                )
            else:
                hit = rel_page in ret_pages
                recall = 1.0 if hit else 0.0
                rank_idx = ret_pages.index(rel_page) + 1 if hit else 0
                mrr = (1.0 / rank_idx) if rank_idx > 0 else 0.0
                precision = (1.0 / top_k) if hit else 0.0

                results.append(
                    QueryEvaluationResult(
                        qid=qid,
                        split=split,
                        question=text,
                        is_abstention=False,
                        relevant_page=rel_page,
                        retrieved_pages=ret_pages,
                        recall=recall,
                        mrr=mrr,
                        hit=hit,
                        precision=precision,
                    )
                )
        return results

    def _eval_pipeline_w0(
        self,
        adapter: SentenceWindowAdapter,
        questions: list[dict[str, Any]],
        top_k: int = 3,
    ) -> list[QueryEvaluationResult]:
        results: list[QueryEvaluationResult] = []
        for q in questions:
            qid = q["qid"]
            split = q["split"]
            text = q["question"]
            is_abs = q["is_abstention"]
            rel_page = q["relevant_page"]

            evidences = adapter.retrieve(text, top_k=top_k)
            ret_pages = [self._parse_page_number(e) for e in evidences]

            if is_abs or rel_page is None:
                results.append(
                    QueryEvaluationResult(
                        qid=qid,
                        split=split,
                        question=text,
                        is_abstention=True,
                        relevant_page=None,
                        retrieved_pages=ret_pages,
                        recall=None,
                        mrr=None,
                        hit=None,
                        precision=None,
                    )
                )
            else:
                hit = rel_page in ret_pages
                recall = 1.0 if hit else 0.0
                rank_idx = ret_pages.index(rel_page) + 1 if hit else 0
                mrr = (1.0 / rank_idx) if rank_idx > 0 else 0.0
                precision = (1.0 / top_k) if hit else 0.0

                results.append(
                    QueryEvaluationResult(
                        qid=qid,
                        split=split,
                        question=text,
                        is_abstention=False,
                        relevant_page=rel_page,
                        retrieved_pages=ret_pages,
                        recall=recall,
                        mrr=mrr,
                        hit=hit,
                        precision=precision,
                    )
                )
        return results

    def _eval_pipeline_w1(
        self,
        sw_adapter: SentenceWindowAdapter,
        reranker: LocalRerankerAdapter,
        questions: list[dict[str, Any]],
        candidate_k: int = 6,
        top_n: int = 3,
    ) -> tuple[list[QueryEvaluationResult], list[RerankerDamageMetrics]]:
        results: list[QueryEvaluationResult] = []
        damage_list: list[RerankerDamageMetrics] = []

        for q in questions:
            qid = q["qid"]
            split = q["split"]
            text = q["question"]
            is_abs = q["is_abstention"]
            rel_page = q["relevant_page"]

            candidates_pre = sw_adapter.retrieve(text, top_k=candidate_k)
            candidates_post, _ = reranker.rerank(text, candidates_pre, top_n=top_n)

            ret_pages = [self._parse_page_number(e) for e in candidates_post]

            if is_abs or rel_page is None:
                results.append(
                    QueryEvaluationResult(
                        qid=qid,
                        split=split,
                        question=text,
                        is_abstention=True,
                        relevant_page=None,
                        retrieved_pages=ret_pages,
                        recall=None,
                        mrr=None,
                        hit=None,
                        precision=None,
                    )
                )
            else:
                hit = rel_page in ret_pages
                recall = 1.0 if hit else 0.0
                rank_idx = ret_pages.index(rel_page) + 1 if hit else 0
                mrr = (1.0 / rank_idx) if rank_idx > 0 else 0.0
                precision = (1.0 / top_n) if hit else 0.0

                results.append(
                    QueryEvaluationResult(
                        qid=qid,
                        split=split,
                        question=text,
                        is_abstention=False,
                        relevant_page=rel_page,
                        retrieved_pages=ret_pages,
                        recall=recall,
                        mrr=mrr,
                        hit=hit,
                        precision=precision,
                    )
                )

                # Damage metric calculation
                fn = q.get("corpus_target", {}).get("filename", "Gersting")
                rel_set = {f"{fn}_p{rel_page}_s0"}
                damage = reranker.calculate_damage_metrics(
                    candidates_pre, candidates_post, rel_set, candidate_k, top_n
                )
                damage_list.append(damage)

        return results, damage_list

    def _summarize_results(
        self, results: list[QueryEvaluationResult]
    ) -> dict[str, Any]:
        valid_results = [
            r for r in results if not r.is_abstention and r.recall is not None
        ]
        if not valid_results:
            return {"mean_recall": 0.0, "mean_mrr": 0.0, "hit_rate": 0.0, "count": 0}

        recalls = [r.recall for r in valid_results if r.recall is not None]
        mrrs = [r.mrr for r in valid_results if r.mrr is not None]
        hits = [1.0 if r.hit else 0.0 for r in valid_results if r.hit is not None]

        mean_recall = sum(recalls) / len(recalls)
        mean_mrr = sum(mrrs) / len(mrrs)
        hit_rate = sum(hits) / len(hits)

        return {
            "mean_recall": round(mean_recall, 4),
            "mean_mrr": round(mean_mrr, 4),
            "hit_rate": round(hit_rate, 4),
            "total_questions_evaluated": len(valid_results),
            "abstention_questions_count": len(results) - len(valid_results),
        }

    def _compare_pipelines(
        self,
        name_a: str,
        name_b: str,
        results_a: list[QueryEvaluationResult],
        results_b: list[QueryEvaluationResult],
    ) -> dict[str, Any]:
        val_a = [r for r in results_a if not r.is_abstention and r.recall is not None]
        val_b = [r for r in results_b if not r.is_abstention and r.recall is not None]

        rec_a = [r.recall for r in val_a if r.recall is not None]
        rec_b = [r.recall for r in val_b if r.recall is not None]

        diffs = [b - a for a, b in zip(rec_a, rec_b, strict=False)]
        wins = sum(1 for d in diffs if d > 0)
        ties = sum(1 for d in diffs if d == 0)
        losses = sum(1 for d in diffs if d < 0)

        ci_lower, ci_upper = compute_bootstrap_ci(diffs, num_resamples=1000, seed=42)
        cohens_d = compute_cohens_d(rec_b, rec_a)

        conclusion = (
            "exploratory_gain"
            if wins > losses
            else ("inconclusive" if wins == losses else "exploratory_loss")
        )

        return {
            "comparison": f"{name_b} vs {name_a}",
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "mean_diff_recall": round(sum(diffs) / len(diffs) if diffs else 0.0, 4),
            "bootstrap_95ci_recall": [ci_lower, ci_upper],
            "cohens_d": cohens_d,
            "conclusion": conclusion,
        }
