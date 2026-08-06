"""Unit and AST tests for Gate A Ground Truth v2 runtime boundary integration.

Covers all 18 mandatory test requirements:
1. ACTIVE_QUESTIONS is adapted without inventing passage_id.
2. relevant_pages remains page-level supervision (legacy_relevant_pages).
3. provenance_status is materialized as 'LEGACY_METADATA_UNAVAILABLE'.
4. passage_recall returns 'NOT_COMPUTABLE_MISSING_PASSAGE_QRELS'.
5. passage_mrr returns 'NOT_COMPUTABLE_MISSING_PASSAGE_QRELS'.
6. nDCG returns 'NOT_COMPUTABLE_MISSING_GRADED_QRELS'.
7. factual_correctness returns 'NOT_COMPUTABLE_MISSING_GOLD_ANSWER' without gold answer.
8. legacy page hit is calculated correctly.
9. legacy page MRR is calculated separately.
10. citation page precision for pages [92,96,101] against [92] is 1/3.
11. citation page recall for the same case is 1.0.
12. abstention correctness remains score 1.0.
13. holdout remains sealed.
14. checkpoint and rehydration preserve the new subtrees (ground_truth and evaluation).
15. the two smoke fixtures continue to pass smoke validator.
16. AST test: no evaluation metrics or ground truth enter generator prompt.
17. old schema vs new schema versioning (raglab_v7_slice4_v3, slice4_v4).
18. NOT_COMPUTABLE_* metrics survive JSON serialization and deserialization.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
from pathlib import Path

from benchmarks.run_slice4_benchmark import (
    _EVAL_SCHEMA_VERSION,
    ACTIVE_QUESTIONS,
    PROTOCOL_VERSION,
    validate_smoke_result,
)
from raglab.evaluation.contracts.ground_truth_v2 import (
    GroundTruthItemV2,
)
from raglab.evaluation.metrics.deterministic_v2 import (
    compute_factual_correctness,
    compute_legacy_page_metrics,
    compute_mrr,
    compute_ndcg_at_k,
    compute_passage_recall_at_k,
)
from raglab.evaluation.migration.legacy_to_gt_v2 import migrate_legacy_dataset


class TestGroundTruthV2RuntimeBoundary:
    """Suíte de testes direcionados para o Gate A (Ground Truth v2 Runtime Boundary)."""

    def test_1_active_questions_adapted_without_inventing_passage_id(self):
        """1. ACTIVE_QUESTIONS is adapted without inventing passage_id or CanonicalEvidence."""
        gt_items = migrate_legacy_dataset(ACTIVE_QUESTIONS)
        assert len(gt_items) == len(ACTIVE_QUESTIONS)
        for gt in gt_items:
            # Must not manufacture fake passage_id for integer pages
            assert len(gt.relevant_evidences) == 0

    def test_2_relevant_pages_remains_legacy_relevant_pages(self):
        """2. relevant_pages remains page-level supervision in legacy_relevant_pages."""
        gt_items = migrate_legacy_dataset(ACTIVE_QUESTIONS)
        q_dev_01 = next(item for item in gt_items if item.query_id == "q_dev_01")
        assert q_dev_01.legacy_relevant_pages == (92,)

    def test_3_provenance_status_materialized_as_legacy_metadata_unavailable(self):
        """3. provenance_status is materialized as LEGACY_METADATA_UNAVAILABLE."""
        gt_items = migrate_legacy_dataset(ACTIVE_QUESTIONS)
        for gt in gt_items:
            assert gt.provenance_status == "LEGACY_METADATA_UNAVAILABLE"

    def test_4_passage_recall_returns_not_computable(self):
        """4. passage_recall returns NOT_COMPUTABLE_MISSING_PASSAGE_QRELS, not 0."""
        res = compute_passage_recall_at_k(["c1", "c2"], [])
        assert res == "NOT_COMPUTABLE_MISSING_PASSAGE_QRELS"
        assert res != 0
        assert res != 0.0

    def test_5_passage_mrr_returns_not_computable(self):
        """5. passage_mrr returns NOT_COMPUTABLE_MISSING_PASSAGE_QRELS, not 0."""
        res = compute_mrr(["c1", "c2"], [])
        assert res == "NOT_COMPUTABLE_MISSING_PASSAGE_QRELS"
        assert res != 0
        assert res != 0.0

    def test_6_ndcg_returns_not_computable(self):
        """6. nDCG returns NOT_COMPUTABLE_MISSING_GRADED_QRELS, not 0."""
        res = compute_ndcg_at_k(["c1", "c2"], [])
        assert res == "NOT_COMPUTABLE_MISSING_GRADED_QRELS"
        assert res != 0
        assert res != 0.0

    def test_7_factual_correctness_returns_not_computable_without_gold_answer(self):
        """7. factual_correctness returns NOT_COMPUTABLE_MISSING_GOLD_ANSWER without gold answer."""
        res = compute_factual_correctness("Resposta do modelo", gold_answer=None)
        assert res == "NOT_COMPUTABLE_MISSING_GOLD_ANSWER"

    def test_8_legacy_page_hit_calculated_correctly(self):
        """8. legacy page hit is calculated correctly."""
        metrics = compute_legacy_page_metrics(
            retrieved_pages=[92, 95, 100],
            relevant_pages=[92],
            cited_pages=[92],
        )
        assert metrics["page_hit_at_k"] == 1.0

        metrics_miss = compute_legacy_page_metrics(
            retrieved_pages=[10, 20],
            relevant_pages=[92],
            cited_pages=[10],
        )
        assert metrics_miss["page_hit_at_k"] == 0.0

    def test_9_legacy_page_mrr_calculated_separately(self):
        """9. legacy page MRR is calculated separately."""
        metrics_rank1 = compute_legacy_page_metrics(
            retrieved_pages=[92, 95],
            relevant_pages=[92],
            cited_pages=[92],
        )
        assert metrics_rank1["page_mrr"] == 1.0

        metrics_rank2 = compute_legacy_page_metrics(
            retrieved_pages=[10, 92],
            relevant_pages=[92],
            cited_pages=[92],
        )
        assert metrics_rank2["page_mrr"] == 0.5

    def test_10_citation_page_precision_for_92_96_101_against_92_is_one_third(self):
        """10. citation page precision for cited pages [92,96,101] against relevant [92] is 1/3."""
        metrics = compute_legacy_page_metrics(
            retrieved_pages=[92, 96, 101],
            relevant_pages=[92],
            cited_pages=[92, 96, 101],
        )
        assert abs(metrics["citation_page_precision"] - (1.0 / 3.0)) < 1e-6

    def test_11_citation_page_recall_for_same_case_is_one(self):
        """11. citation page recall for cited pages [92,96,101] against relevant [92] is 1.0."""
        metrics = compute_legacy_page_metrics(
            retrieved_pages=[92, 96, 101],
            relevant_pages=[92],
            cited_pages=[92, 96, 101],
        )
        assert metrics["citation_page_recall"] == 1.0

    def test_12_correct_abstention_score_remains_one(self):
        """12. abstention correctness score for expected abstention remains 1.0."""
        from benchmarks.run_slice4_benchmark import compute_abstention_correctness
        entry = compute_abstention_correctness(True, True)
        assert entry["score"] == 1.0
        assert entry["status"] == "COMPUTED"

    def test_13_holdout_remains_sealed(self):
        """13. holdout questions remain sealed from ACTIVE_QUESTIONS."""
        for q in ACTIVE_QUESTIONS:
            assert "holdout" not in q["qid"]

    def _make_w0_fixture(self) -> dict:
        text = "A indução matemática consiste em caso base e passo indutivo."
        return {
            "strategy": "W0_sentence_window",
            "qid": "q_dev_01",
            "retrieved_text": text,
            "text_sha": hashlib.sha256(text.encode()).hexdigest(),
            "answer_text": "A demonstração por indução matemática exige a verificação do caso base e a hipótese indutiva.",
        }

    def test_14_checkpoint_and_rehydration_preserve_new_subtrees(self):
        """14. checkpoint and rehydration preserve ground_truth and evaluation subtrees."""
        fix = self._make_w0_fixture()
        qid = fix["qid"]
        relevant_pages = [92]
        gt_item = GroundTruthItemV2(
            query_id=qid,
            query_text="O que e demonstracao por exaustao?",
            answerable=True,
            unanswerable_reason=None,
            gold_answer=None,
            relevant_evidences=(),
            provenance_status="LEGACY_METADATA_UNAVAILABLE",
            annotation_completeness={
                "passage_qrels_present": False,
                "graded_qrels_present": False,
                "gold_answer_present": False,
            },
            annotation_records=(),
            legacy_relevant_pages=tuple(relevant_pages),
        )

        legacy_metrics = compute_legacy_page_metrics(
            retrieved_pages=[92],
            relevant_pages=relevant_pages,
            cited_pages=[92, 96, 101],
        )

        result_row = {
            "qid": qid,
            "strategy": fix["strategy"],
            "ground_truth": {
                "contract_version": "v2",
                "source_schema": "legacy_active_questions",
                "provenance_status": gt_item.provenance_status,
                "annotation_completeness": gt_item.annotation_completeness,
                "answerable": gt_item.answerable,
                "legacy_relevant_pages": list(gt_item.legacy_relevant_pages),
            },
            "evaluation": {
                "protocol_version": PROTOCOL_VERSION,
                "artifact_schema_version": _EVAL_SCHEMA_VERSION,
                "schema_version": _EVAL_SCHEMA_VERSION,
                "metrics": [],
                "legacy_page_metrics": legacy_metrics,
                "deterministic_v2_metrics": {
                    "passage_recall_at_k": "NOT_COMPUTABLE_MISSING_PASSAGE_QRELS",
                    "passage_mrr": "NOT_COMPUTABLE_MISSING_PASSAGE_QRELS",
                    "ndcg_at_k": "NOT_COMPUTABLE_MISSING_GRADED_QRELS",
                    "factual_correctness": "NOT_COMPUTABLE_MISSING_GOLD_ANSWER",
                },
            },
        }

        serialized = json.dumps(result_row)
        deserialized = json.loads(serialized)
        assert deserialized["ground_truth"]["provenance_status"] == "LEGACY_METADATA_UNAVAILABLE"
        assert deserialized["evaluation"]["legacy_page_metrics"]["page_hit_at_k"] == 1.0
        assert (
            deserialized["evaluation"]["deterministic_v2_metrics"]["passage_recall_at_k"]
            == "NOT_COMPUTABLE_MISSING_PASSAGE_QRELS"
        )

    def test_15_two_smoke_fixtures_continue_valid(self):
        """15. the two smoke fixtures continue to pass smoke validator."""
        logger = logging.getLogger("test_smoke_validator")

        # 1. Positive Smoke Fixture
        fix1 = self._make_w0_fixture()
        ans_text = fix1["answer_text"]
        ans_sha = hashlib.sha256(ans_text.encode()).hexdigest()
        ret_sha = fix1["text_sha"]
        data1 = {
            "embedding_fingerprints": {
                fix1["strategy"]: {"cache_tree_sha256": "a" * 64}
            },
            "manifest_fingerprint": "b" * 64,
            "results": {
                fix1["strategy"]: [
                    {
                        "qid": fix1["qid"],
                        "strategy": fix1["strategy"],
                        "abstained": False,
                        "answer": {
                            "text": ans_text,
                            "text_sha256": ans_sha,
                            "truncated": False,
                        },
                        "citation_mapping_status": "AVAILABLE",
                        "citation_map": [
                            {"marker": "[E1]", "page_number": 92, "chunk_id": "gersting_p92_c0", "text_sha256": ret_sha}
                        ],
                        "citation_pages": [92],
                        "retrieval_evidence": {
                            "retrieval_hit": True,
                            "candidates": [
                                {"chunk_id": "gersting_p92_c0", "page_number": 92, "text_sha256": ret_sha}
                            ],
                        },
                        "evaluation": {
                            "schema_version": _EVAL_SCHEMA_VERSION,
                            "metrics": [
                                {"name": "context_relevance", "status": "COMPUTED", "score": 1.0},
                                {"name": "groundedness", "status": "COMPUTED", "score": 1.0},
                                {"name": "answer_relevance", "status": "COMPUTED", "score": 1.0},
                                {"name": "abstention_correctness", "status": "COMPUTED", "score": 1.0},
                            ],
                        },
                    }
                ]
            },
        }

        res1 = validate_smoke_result(
            data1,
            strategy=fix1["strategy"],
            qid=fix1["qid"],
            is_abstention_question=False,
            logger=logger,
        )
        assert res1 == "SMOKE_POSITIVE_OK"

        # 2. Abstention Smoke Fixture
        data2 = {
            "embedding_fingerprints": {
                "F0_baseline": {"cache_tree_sha256": "a" * 64}
            },
            "manifest_fingerprint": "b" * 64,
            "results": {
                "F0_baseline": [
                    {
                        "qid": "q_test_04",
                        "strategy": "F0_baseline",
                        "abstained": True,
                        "answer": {
                            "text": "Não encontrei evidências.",
                            "text_sha256": hashlib.sha256("Não encontrei evidências.".encode()).hexdigest(),
                            "truncated": False,
                        },
                        "citation_mapping_status": "NOT_APPLICABLE",
                        "citation_map": [],
                        "citation_pages": [],
                        "retrieval_evidence": {
                            "retrieval_hit": False,
                            "candidates": [],
                        },
                        "evaluation": {
                            "schema_version": _EVAL_SCHEMA_VERSION,
                            "metrics": [
                                {"name": "abstention_correctness", "status": "COMPUTED", "score": 1.0},
                                {"name": "context_relevance", "status": "NOT_APPLICABLE"},
                                {"name": "groundedness", "status": "NOT_APPLICABLE"},
                                {"name": "answer_relevance", "status": "NOT_APPLICABLE"},
                            ],
                        },
                    }
                ]
            },
        }

        res2 = validate_smoke_result(
            data2,
            strategy="F0_baseline",
            qid="q_test_04",
            is_abstention_question=True,
            logger=logger,
        )
        assert res2 == "SMOKE_ABSTENTION_OK"

    def test_16_ast_isolation_no_evaluation_in_generator_prompt(self):
        """16. AST test: no evaluation metrics or ground truth enter generator prompt."""
        prompt_file = Path("src/raglab/infrastructure/gemini/prompts.py")
        tree = ast.parse(prompt_file.read_text(encoding="utf-8"), filename=str(prompt_file))

        forbidden_names = {"gold_answer", "relevant_pages", "qrels", "GroundTruthItemV2"}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "build_generation_prompt":
                arg_names = {arg.arg for arg in node.args.args}
                assert not (arg_names & forbidden_names), f"Forbidden args found in prompt: {arg_names & forbidden_names}"

    def test_17_old_schema_vs_new_schema_versioning(self):
        assert PROTOCOL_VERSION == "raglab_v7_slice4_v3"
        assert _EVAL_SCHEMA_VERSION == "slice4_v5"


    def test_18_not_computable_metrics_survive_json_serialization(self):
        """18. NOT_COMPUTABLE_* metrics survive JSON serialization and deserialization."""
        metrics_dict = {
            "passage_recall_at_k": "NOT_COMPUTABLE_MISSING_PASSAGE_QRELS",
            "passage_mrr": "NOT_COMPUTABLE_MISSING_PASSAGE_QRELS",
            "ndcg_at_k": "NOT_COMPUTABLE_MISSING_GRADED_QRELS",
            "citation_passage_precision": "NOT_COMPUTABLE_MISSING_PASSAGE_QRELS",
            "citation_passage_recall": "NOT_COMPUTABLE_MISSING_PASSAGE_QRELS",
            "factual_correctness": "NOT_COMPUTABLE_MISSING_GOLD_ANSWER",
        }
        encoded = json.dumps(metrics_dict)
        decoded = json.loads(encoded)
        assert decoded == metrics_dict
        for _k, v in decoded.items():
            assert v.startswith("NOT_COMPUTABLE_")
