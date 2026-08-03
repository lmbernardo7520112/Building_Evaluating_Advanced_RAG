"""Unit tests for Ground Truth v2 contract, legacy migration, and deterministic metrics."""

from __future__ import annotations

import pytest

from raglab.evaluation.contracts.ground_truth_v2 import (
    CanonicalEvidence,
    GroundTruthItemV2,
)
from raglab.evaluation.metrics.deterministic_v2 import (
    compute_abstention_confusion_matrix,
    compute_citation_precision_recall,
    compute_mrr,
    compute_ndcg_at_k,
    compute_nugget_and_contradiction_metrics,
    compute_passage_recall_at_k,
)
from raglab.evaluation.migration.legacy_to_gt_v2 import (
    migrate_legacy_dataset,
    migrate_legacy_qrel_item,
)


class TestGroundTruthV2Contract:
    def test_valid_item_creation(self):
        ev = CanonicalEvidence(
            passage_id="p1",
            document_id="doc1",
            start_page=10,
            text_span="Some span",
            content_sha256="abc123sha",
            relevance_grade=2,
        )
        item = GroundTruthItemV2(
            query_id="q1",
            query_text="What is X?",
            answerable=True,
            unanswerable_reason=None,
            gold_answer="X is Y.",
            relevant_evidences=(ev,),
            provenance_status="SINGLE_ANNOTATOR",
        )
        assert item.query_id == "q1"
        assert item.answerable is True
        assert len(item.relevant_evidences) == 1

    def test_unanswerable_item_requires_reason(self):
        with pytest.raises(ValueError, match="unanswerable_reason is required"):
            GroundTruthItemV2(
                query_id="q1",
                query_text="What is X?",
                answerable=False,
                unanswerable_reason=None,
                gold_answer=None,
                relevant_evidences=(),
                provenance_status="SINGLE_ANNOTATOR",
            )

    def test_adjudicated_item_requires_two_records(self):
        with pytest.raises(ValueError, match="requires at least 2 annotation records"):
            GroundTruthItemV2(
                query_id="q1",
                query_text="What is X?",
                answerable=True,
                unanswerable_reason=None,
                gold_answer="X is Y.",
                relevant_evidences=(),
                provenance_status="ADJUDICATED",
                annotation_records=({"annotator": "A1"},),
            )


class TestLegacyMigration:
    def test_migration_sets_legacy_metadata_unavailable(self):
        legacy = {
            "query_id": "q1",
            "query": "Sample question",
            "relevant_pages": [10, 12],
            "answerable": True,
            "gold_answer": "Sample answer",
        }
        item = migrate_legacy_qrel_item(legacy)
        assert item.provenance_status == "LEGACY_METADATA_UNAVAILABLE"
        assert len(item.relevant_evidences) == 2
        # EXECUTION GUARD 2: relevance_grade is None for binary legacy qrels
        assert item.relevant_evidences[0].relevance_grade is None

    def test_migrate_dataset_batch(self):
        items = migrate_legacy_dataset([{"query_id": "q1"}, {"query_id": "q2"}])
        assert len(items) == 2
        assert items[0].query_id == "q1"
        assert items[1].query_id == "q2"


class TestDeterministicMetrics:
    def test_passage_recall_at_k(self):
        ev1 = CanonicalEvidence("p1", "doc1", 1, "text", "sha")
        ev2 = CanonicalEvidence("p2", "doc1", 2, "text", "sha")
        recall = compute_passage_recall_at_k(["p1", "p3", "p4"], [ev1, ev2], k=3)
        assert recall == 0.5

    def test_mrr(self):
        ev1 = CanonicalEvidence("p1", "doc1", 1, "text", "sha")
        mrr = compute_mrr(["p3", "p1", "p2"], [ev1])
        assert mrr == 0.5

    def test_ndcg_returns_not_computable_for_missing_grades(self):
        """EXECUTION GUARD 2: returns string NOT_COMPUTABLE_MISSING_GRADED_QRELS when grades are None."""
        ev1 = CanonicalEvidence("p1", "doc1", 1, "text", "sha", relevance_grade=None)
        res = compute_ndcg_at_k(["p1"], [ev1], k=3)
        assert res == "NOT_COMPUTABLE_MISSING_GRADED_QRELS"

    def test_ndcg_with_grades(self):
        ev1 = CanonicalEvidence("p1", "doc1", 1, "text", "sha", relevance_grade=3)
        ev2 = CanonicalEvidence("p2", "doc1", 2, "text", "sha", relevance_grade=1)
        res = compute_ndcg_at_k(["p1", "p2"], [ev1, ev2], k=3)
        assert isinstance(res, float)
        assert res == 1.0

    def test_citation_precision_recall(self):
        ev1 = CanonicalEvidence("p1", "doc1", 1, "text", "sha")
        res = compute_citation_precision_recall(
            cited_passage_ids=["p1"],
            retrieved_passage_ids=["p1", "p2"],
            gold_evidences=[ev1],
        )
        assert res["citation_precision"] == 1.0
        assert res["citation_recall"] == 1.0

    def test_abstention_matrix(self):
        res = compute_abstention_confusion_matrix(is_abstained=True, is_unanswerable=True)
        assert res["matrix_category"] == "TRUE_POSITIVE_ABSTENTION"
        assert res["abstention_correct"] is True

    def test_nugget_matcher_not_configured(self):
        res = compute_nugget_and_contradiction_metrics("answer", ["nugget1"])
        assert res == "NOT_COMPUTABLE_MATCHER_NOT_CONFIGURED"
