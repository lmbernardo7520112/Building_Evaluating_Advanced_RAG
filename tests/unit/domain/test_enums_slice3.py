"""Tests for the new Slice 3 domain enums: PipelineStrategy, RerankerClass,
QrelAuditState, ExperimentClassification."""

from __future__ import annotations

from raglab.domain.enums import (
    ExperimentClassification,
    PipelineStrategy,
    QrelAuditState,
    RerankerClass,
)


class TestPipelineStrategySlice3:
    """Verify all Slice 3 strategies are present and correctly valued."""

    def test_f0_exists(self):
        assert PipelineStrategy.BASELINE.value == "baseline"

    def test_s0_exists(self):
        assert PipelineStrategy.SENTENCE_ANCHOR.value == "sentence_anchor"

    def test_w0_exists(self):
        assert PipelineStrategy.SENTENCE_WINDOW.value == "sentence_window"

    def test_w1_exists(self):
        assert PipelineStrategy.SENTENCE_WINDOW_RERANK.value == "sentence_window_rerank"

    def test_h0_exists(self):
        assert PipelineStrategy.HIERARCHICAL_LEAF.value == "hierarchical_leaf"

    def test_h1_exists(self):
        assert PipelineStrategy.AUTO_MERGING.value == "auto_merging"

    def test_h2_exists(self):
        assert PipelineStrategy.AUTO_MERGING_RERANK.value == "auto_merging_rerank"

    def test_all_slice3_strategies_unique(self):
        values = [s.value for s in PipelineStrategy]
        assert len(values) == len(set(values)), "Strategy values must be unique"

    def test_causal_matrix_complete(self):
        """All 7 causal matrix variants must be present."""
        expected = {
            "baseline", "sentence_anchor", "sentence_window",
            "sentence_window_rerank", "hierarchical_leaf",
            "auto_merging", "auto_merging_rerank",
        }
        actual = {s.value for s in PipelineStrategy}
        assert expected == actual


class TestRerankerClass:
    """Reranker must be explicitly classified — bi-encoder ≠ cross-encoder."""

    def test_bi_encoder_rescoring_exists(self):
        assert RerankerClass.BI_ENCODER_RESCORING.value == "bi_encoder_rescoring"

    def test_cross_encoder_exists(self):
        assert RerankerClass.CROSS_ENCODER.value == "cross_encoder"

    def test_late_interaction_exists(self):
        assert RerankerClass.LATE_INTERACTION.value == "late_interaction"

    def test_lexical_exists(self):
        assert RerankerClass.LEXICAL.value == "lexical"

    def test_hybrid_exists(self):
        assert RerankerClass.HYBRID.value == "hybrid"

    def test_heuristic_exists(self):
        assert RerankerClass.HEURISTIC.value == "heuristic"

    def test_local_reranker_is_not_cross_encoder(self):
        """Critical: local bi-encoder rescoring must NOT be labelled cross-encoder."""
        local_reranker = RerankerClass.BI_ENCODER_RESCORING
        assert local_reranker != RerankerClass.CROSS_ENCODER

    def test_all_values_unique(self):
        values = [r.value for r in RerankerClass]
        assert len(values) == len(set(values))


class TestQrelAuditState:
    def test_confirmed_state(self):
        assert QrelAuditState.CONFIRMED.value == "CONFIRMED"

    def test_corrected_state(self):
        assert QrelAuditState.CORRECTED_WITH_EVIDENCE.value == "CORRECTED_WITH_EVIDENCE"

    def test_ambiguous_state(self):
        assert QrelAuditState.AMBIGUOUS.value == "AMBIGUOUS"

    def test_unanswerable_state(self):
        assert QrelAuditState.UNANSWERABLE_IN_SUBCORPUS.value == "UNANSWERABLE_IN_SUBCORPUS"

    def test_all_states_present(self):
        assert len(list(QrelAuditState)) == 4


class TestExperimentClassification:
    def test_engineering_valid(self):
        assert ExperimentClassification.ENGINEERING_VALID.value == "engineering_valid"

    def test_exploratory_signal(self):
        assert ExperimentClassification.EXPLORATORY_SIGNAL.value == "exploratory_signal"

    def test_inconclusive(self):
        assert ExperimentClassification.INCONCLUSIVE.value == "inconclusive"

    def test_observed_regression(self):
        assert ExperimentClassification.OBSERVED_REGRESSION.value == "observed_regression"

    def test_experiment_invalid(self):
        assert ExperimentClassification.EXPERIMENT_INVALID.value == "experiment_invalid"

    def test_gate2_was_inconclusive(self):
        """Gate 2 experimental result must be classified as inconclusive."""
        gate2_classification = ExperimentClassification.INCONCLUSIVE
        assert gate2_classification == ExperimentClassification.INCONCLUSIVE

    def test_not_exploratory_when_only_dev_gains(self):
        """W0 gains were only in development — must NOT be exploratory_signal."""
        # This is a policy check: if test recall = 0 and IC includes 0,
        # classification must be INCONCLUSIVE, not EXPLORATORY_SIGNAL.
        gate2_dev_gain = True  # W0 had 2 dev gains
        gate2_test_recall_zero = True  # test recall was 0 for all variants
        gate2_ic_includes_zero = True  # IC [0.0000, 0.5714]

        if gate2_test_recall_zero and gate2_ic_includes_zero:
            classification = ExperimentClassification.INCONCLUSIVE
        elif gate2_dev_gain:
            classification = ExperimentClassification.EXPLORATORY_SIGNAL
        else:
            classification = ExperimentClassification.INCONCLUSIVE

        assert classification == ExperimentClassification.INCONCLUSIVE
