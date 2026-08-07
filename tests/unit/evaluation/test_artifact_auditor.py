"""Unit tests for offline artifact auditor (validate_slice4_v4_artifacts.py).

Verifies data-driven recalculation of citation page precision/recall and contract enforcement:
- cited [92, 96, 101] vs relevant [92] -> precision 1/3, recall 1
- cited [92] vs relevant [92] -> precision 1, recall 1
- cited [96] vs relevant [92] -> precision 0, recall 0
- serialized value mismatch -> validation error
- invalid protocol/schema -> validation error
- synthetic passage_id -> validation error
"""

from __future__ import annotations

import math
from typing import Any

from raglab.evaluation.metrics.deterministic_v2 import compute_legacy_page_metrics
from scripts.validate_slice4_v4_artifacts import validate_positive_json


def _make_valid_positive_payload(
    cited_pages: list[int],
    relevant_pages: list[int],
    override_gt: dict[str, Any] | None = None,
    override_eval: dict[str, Any] | None = None,
    override_protocol: str = "raglab_v7_slice4_v3",
    override_schema: str = "slice4_v4",
) -> dict[str, Any]:
    legacy_metrics = compute_legacy_page_metrics(
        retrieved_pages=[92, 96, 101],
        relevant_pages=relevant_pages,
        cited_pages=cited_pages,
    )

    if override_eval:
        legacy_metrics.update(override_eval)

    gt = {
        "contract_version": "v2",
        "source_schema": "legacy_active_questions",
        "provenance_status": "LEGACY_METADATA_UNAVAILABLE",
        "legacy_relevant_pages": relevant_pages,
        "passage_qrels_status": "NOT_ANNOTATED",
        "graded_qrels_status": "NOT_ANNOTATED",
        "gold_answer_status": "NOT_ANNOTATED",
    }
    if override_gt:
        gt.update(override_gt)

    return {
        "protocol_version": override_protocol,
        "artifact_schema_version": override_schema,
        "results": {
            "W0_sentence_window": [
                {
                    "qid": "q_dev_01",
                    "strategy": "W0_sentence_window",
                    "abstained": False,
                    "citation_pages": cited_pages,
                    "retrieval_evidence": {
                        "candidates": [
                            {"chunk_id": "c1", "page_number": 92},
                            {"chunk_id": "c2", "page_number": 96},
                            {"chunk_id": "c3", "page_number": 101},
                        ]
                    },
                    "ground_truth": gt,
                    "evaluation": {
                        "protocol_version": override_protocol,
                        "artifact_schema_version": override_schema,
                        "legacy_page_metrics": legacy_metrics,
                        "deterministic_v2_metrics": {
                            "passage_recall_at_k": "NOT_COMPUTABLE_MISSING_PASSAGE_QRELS",
                            "passage_mrr": "NOT_COMPUTABLE_MISSING_PASSAGE_QRELS",
                            "ndcg_at_k": "NOT_COMPUTABLE_MISSING_GRADED_QRELS",
                            "factual_correctness": "NOT_COMPUTABLE_MISSING_GOLD_ANSWER",
                        },
                    },
                }
            ]
        },
    }


class TestArtifactAuditorDataDriven:
    """Suíte de testes direcionados para o Offline Artifact Auditor."""

    def test_citation_set_92_96_101_against_92(self):
        """[92, 96, 101] against [92] -> precision 1/3, recall 1."""
        metrics = compute_legacy_page_metrics(
            retrieved_pages=[92, 96, 101],
            relevant_pages=[92],
            cited_pages=[92, 96, 101],
        )
        assert math.isclose(metrics["citation_page_precision"], 1.0 / 3.0, abs_tol=1e-5)
        assert math.isclose(metrics["citation_page_recall"], 1.0, abs_tol=1e-5)

        payload = _make_valid_positive_payload(cited_pages=[92, 96, 101], relevant_pages=[92])
        errors = validate_positive_json(payload)
        assert not errors, f"Unexpected validation errors: {errors}"

    def test_citation_set_92_against_92(self):
        """[92] against [92] -> precision 1, recall 1."""
        metrics = compute_legacy_page_metrics(
            retrieved_pages=[92, 96],
            relevant_pages=[92],
            cited_pages=[92],
        )
        assert math.isclose(metrics["citation_page_precision"], 1.0, abs_tol=1e-5)
        assert math.isclose(metrics["citation_page_recall"], 1.0, abs_tol=1e-5)

        payload = _make_valid_positive_payload(cited_pages=[92], relevant_pages=[92])
        errors = validate_positive_json(payload)
        assert not errors, f"Unexpected validation errors: {errors}"

    def test_citation_set_96_against_92(self):
        """[96] against [92] -> precision 0, recall 0."""
        metrics = compute_legacy_page_metrics(
            retrieved_pages=[92, 96],
            relevant_pages=[92],
            cited_pages=[96],
        )
        assert math.isclose(metrics["citation_page_precision"], 0.0, abs_tol=1e-5)
        assert math.isclose(metrics["citation_page_recall"], 0.0, abs_tol=1e-5)

        payload = _make_valid_positive_payload(cited_pages=[96], relevant_pages=[92])
        errors = validate_positive_json(payload)
        assert not errors, f"Unexpected validation errors: {errors}"

    def test_mismatched_serialized_value_triggers_error(self):
        """Divergent serialized value triggers a validation error."""
        payload = _make_valid_positive_payload(
            cited_pages=[92, 96, 101],
            relevant_pages=[92],
            override_eval={"citation_page_precision": 0.999},  # Mismatch with 0.333
        )
        errors = validate_positive_json(payload)
        assert any("Mismatch in citation_page_precision" in e for e in errors)

    def test_invalid_protocol_or_schema_triggers_error(self):
        """Incorrect protocol/schema triggers a validation error."""
        payload = _make_valid_positive_payload(
            cited_pages=[92],
            relevant_pages=[92],
            override_protocol="raglab_v7_slice4_v2",
        )
        errors = validate_positive_json(payload)
        assert any("Invalid protocol_version" in e for e in errors)

    def test_synthetic_passage_id_triggers_error(self):
        """Synthetic passage_id (e.g. p92) triggers a validation error."""
        payload = _make_valid_positive_payload(
            cited_pages=[92],
            relevant_pages=[92],
            override_gt={"relevant_evidences": [{"passage_id": "p92"}]},
        )
        errors = validate_positive_json(payload)
        assert any("SYNTHETIC PASSAGE_ID DETECTED" in e for e in errors)
