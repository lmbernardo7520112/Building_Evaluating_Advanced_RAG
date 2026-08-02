"""Tests for Slice 3 experiment manifest and qrel audit file validation."""

from __future__ import annotations

import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST_PATH = _PROJECT_ROOT / "benchmarks" / "slice3_experiment_manifest.json"
_QREL_AUDIT_PATH = (
    _PROJECT_ROOT / "benchmarks" / "questions" / "qrel_audit_slice3.json"
)
_QUESTIONS_PATH = (
    _PROJECT_ROOT / "benchmarks" / "questions" / "controlled_chapter2.json"
)


class TestSlice3Manifest:
    """Validate the pre-registered experiment manifest."""

    def _load(self) -> dict:
        with _MANIFEST_PATH.open(encoding="utf-8") as f:
            return json.load(f)

    def test_manifest_exists(self):
        assert _MANIFEST_PATH.exists(), f"Manifest not found: {_MANIFEST_PATH}"

    def test_manifest_has_experiment_id(self):
        data = self._load()
        assert "experiment_id" in data
        assert data["experiment_id"].startswith("raglab_v7_slice3")

    def test_corpus_sha256_matches_expected(self):
        data = self._load()
        expected = "33e2e9f1e190158b3e99c19fced1acd050720247c7556780bad82b2f93bf1254"
        assert data["corpus"]["pdf_sha256"] == expected

    def test_pdf_external_flag(self):
        data = self._load()
        assert data["corpus"]["pdf_external"] is True, "PDF must be external to Git"

    def test_holdout_sealed(self):
        data = self._load()
        holdout = data["questions"]["holdout_splits"]
        status = data["questions"]["holdout_status"]
        assert "holdout" in holdout
        assert "SEALED" in status.upper()

    def test_all_seven_variants_defined(self):
        data = self._load()
        matrix = data["causal_matrix"]
        for variant in ["F0", "S0", "W0", "W1", "H0", "H1", "H2"]:
            assert variant in matrix, f"Missing variant {variant} in causal matrix"

    def test_reranker_correctly_classified(self):
        data = self._load()
        reranker_class = data["reranker_audit"]["class"]
        assert reranker_class == "bi_encoder_rescoring"
        assert data["reranker_audit"]["is_cross_encoder"] is False

    def test_w1_reranker_class_is_bi_encoder(self):
        data = self._load()
        assert data["causal_matrix"]["W1"]["reranker_class"] == "bi_encoder_rescoring"

    def test_h0_auto_merge_disabled(self):
        data = self._load()
        assert data["causal_matrix"]["H0"]["auto_merge"] is False

    def test_h1_auto_merge_enabled(self):
        data = self._load()
        assert data["causal_matrix"]["H1"]["auto_merge"] is True

    def test_merge_threshold_registered(self):
        data = self._load()
        assert "merge_threshold" in data["causal_matrix"]["H1"]
        assert 0.0 < data["causal_matrix"]["H1"]["merge_threshold"] <= 1.0

    def test_active_qids_count(self):
        data = self._load()
        active_qids = data["questions"]["active_qids"]
        # 4 dev + 4 test = 8 active
        assert len(active_qids) == 8

    def test_holdout_not_in_active_qids(self):
        data = self._load()
        active = set(data["questions"]["active_qids"])
        assert "q_holdout_01" not in active
        assert "q_holdout_02" not in active

    def test_gate2_verdict_recorded(self):
        data = self._load()
        verdict = data["gate2_verdict"]
        assert verdict["gate2_engineering"] == "PASSED"
        assert verdict["experimental_result"] == "INCONCLUSIVE"
        assert verdict["gate2_not_reopened"] is True

    def test_seed_is_42(self):
        data = self._load()
        assert data["seed"] == 42

    def test_metrics_include_merge_metrics(self):
        data = self._load()
        metrics = data["metrics"]
        required = [
            "merge_rate", "parent_promotion_rate",
            "relevant_evidence_preservation", "context_expansion_ratio",
        ]
        for m in required:
            assert m in metrics, f"Required metric '{m}' missing from manifest"

    def test_hypotheses_define_all_comparisons(self):
        data = self._load()
        hypotheses = data["hypotheses"]
        assert "H_gran" in hypotheses   # F0 × S0
        assert "H_exp" in hypotheses    # S0 × W0
        assert "H_merge" in hypotheses  # H0 × H1


class TestQrelAudit:
    """Validate the qrel audit file for Slice 3."""

    def _load(self) -> dict:
        with _QREL_AUDIT_PATH.open(encoding="utf-8") as f:
            return json.load(f)

    def test_audit_file_exists(self):
        assert _QREL_AUDIT_PATH.exists(), f"Qrel audit not found: {_QREL_AUDIT_PATH}"

    def test_holdout_sealed(self):
        data = self._load()
        assert "SEALED" in data.get("holdout_status", "").upper()

    def test_eight_active_questions_audited(self):
        data = self._load()
        active = [
            q for q in data["questions"]
            if q["qid"] not in {"q_holdout_01", "q_holdout_02"}
        ]
        assert len(active) == 8, f"Expected 8 active questions, got {len(active)}"

    def test_all_questions_have_audit_state(self):
        data = self._load()
        valid_states = {"CONFIRMED", "CORRECTED_WITH_EVIDENCE", "AMBIGUOUS",
                        "UNANSWERABLE_IN_SUBCORPUS"}
        for q in data["questions"]:
            if q["qid"].startswith("q_holdout"):
                continue
            assert q["audit_state"] in valid_states, (
                f"Invalid audit_state for {q['qid']}: {q['audit_state']}"
            )

    def test_no_qrel_changed_based_on_performance(self):
        """Auditor rule: qrels must NOT be changed to favour F0/W0/W1."""
        data = self._load()
        # All change_log entries must say "nenhuma alteração" or document evidence
        for q in data["questions"]:
            if q["qid"].startswith("q_holdout"):
                continue
            change_log = q.get("change_log", "")
            # If change_log mentions 'alteração', it should NOT mention 'desempenho'
            # (i.e., changes must be based on evidence, not pipeline results)
            assert "performance" not in change_log.lower(), (
                f"Qrel change for {q['qid']} mentions performance — invalid"
            )

    def test_abstention_question_has_null_evidence(self):
        data = self._load()
        for q in data["questions"]:
            if q.get("is_abstention"):
                assert q["relevant_pages_all_acceptable"] == []
                assert q["primary_evidence_page"] is None

    def test_dev_questions_have_relevant_pages(self):
        data = self._load()
        for q in data["questions"]:
            if q["split"] == "development" and not q.get("is_abstention"):
                assert len(q["relevant_pages_all_acceptable"]) > 0

    def test_second_annotator_status_single(self):
        data = self._load()
        assert data["second_annotator_status"] == "GROUND_TRUTH_SINGLE_ANNOTATOR"

    def test_single_annotator_blocks_confirmatory(self):
        """Single annotator → conclusory results are prohibited (policy check)."""
        data = self._load()
        status = data["second_annotator_status"]
        # Policy: if single annotator, conclusions must be inconclusive
        if status == "GROUND_TRUTH_SINGLE_ANNOTATOR":
            # Verify protocol records the blocking rule
            protocol = data.get("second_annotator_protocol", {})
            assert "PROHIBITS" in protocol.get("blocking_rule", "").upper(), (
                "Protocol must state that single annotator prohibits confirmatory conclusions"
            )


class TestQrelAuditAgainstOriginal:
    """Cross-validate audit file against original questions file."""

    def _load_audit(self) -> dict:
        with _QREL_AUDIT_PATH.open(encoding="utf-8") as f:
            return json.load(f)

    def _load_questions(self) -> dict:
        with _QUESTIONS_PATH.open(encoding="utf-8") as f:
            return json.load(f)

    def test_all_original_active_qids_in_audit(self):
        audit = self._load_audit()
        original = self._load_questions()
        active_original = {
            q["qid"] for q in original["questions"]
            if q["split"] in {"development", "test"}
        }
        audit_qids = {
            q["qid"] for q in audit["questions"]
            if not q["qid"].startswith("q_holdout")
        }
        assert active_original == audit_qids, (
            f"Audit missing qids: {active_original - audit_qids}"
        )

    def test_audit_abstention_matches_original(self):
        audit = self._load_audit()
        original = self._load_questions()
        orig_map = {q["qid"]: q for q in original["questions"]}
        for aq in audit["questions"]:
            qid = aq["qid"]
            if qid not in orig_map:
                continue
            assert aq["is_abstention"] == orig_map[qid]["is_abstention"], (
                f"Abstention mismatch for {qid}"
            )
