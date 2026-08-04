"""Unit tests for Governed Silver Triage, Human Routing, and Calibration (Gate B2 Reconciliation). # noqa: E501

Covers reconciliation invariants 9 to 13.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from raglab.evaluation.contracts.hybrid_eval_v2 import SilverAnnotationRecord
from raglab.evaluation.contracts.silver_annotation_v2 import (
    validate_human_qrels_exclusion,
    validate_silver_record,
)
from scripts.build_human_review_queues import build_human_queues
from scripts.calibrate_silver_against_human import calibrate_silver
from scripts.run_silver_annotation import run_silver_triage_mock


class TestSilverTriageAndRouting:
    """Testes unitários direcionados para Triagem Silver, Roteamento Humano e Calibração."""

    def test_25_silver_schema_valid(self):
        rec = SilverAnnotationRecord(
            question_id="q_dev_01",
            passage_id="ps_01",
            label_source="MACHINE_SILVER",
            relevance_grade=2,
            confidence=0.90,
        )
        assert rec.label_source == "MACHINE_SILVER"
        assert rec.relevance_grade == 2

    def test_26_supporting_span_literal_validation(self):
        text = "Este é o texto exato da passagem documental."
        rec = SilverAnnotationRecord(
            question_id="q_dev_01",
            passage_id="ps_01",
            supporting_span="texto exato",
        )
        assert validate_silver_record(rec, text) is True

    def test_09_mock_silver_does_not_enter_silver_qrels(self, tmp_path: Path):
        bad_qrels = tmp_path / "mock_qrels.jsonl"
        bad_qrels.write_text(
            json.dumps({
                "question_id": "q1",
                "passage_id": "p1",
                "label_source": "MACHINE_SILVER",
                "authoritative": False,
            })
            + "\n"
        )
        with pytest.raises(ValueError):
            validate_human_qrels_exclusion(bad_qrels)

    def test_10_validate_only_mode_does_not_create_authoritative_labels(
        self, tmp_path: Path
    ):
        pool = tmp_path / "pool.jsonl"
        pool.write_text(
            json.dumps({
                "question_id": "q_dev_01",
                "passage_id": "ps_01",
                "text": "abc",
                "page_number": 92,
            })
            + "\n"
        )
        out_dir = tmp_path / "out"
        run_silver_triage_mock(pool, out_dir, mode="validate-only")
        man_file = out_dir / "silver_manifest.json"
        manifest = json.loads(man_file.read_text())
        assert manifest["authoritative"] is False
        assert manifest["execution_mode"] == "VALIDATION_ONLY"

    def test_11_mock_silver_does_not_participate_in_calibration(self, tmp_path: Path):
        sil = tmp_path / "sil.jsonl"
        hum = tmp_path / "hum.jsonl"
        rep = tmp_path / "rep.json"
        res, _ = calibrate_silver(sil, hum, rep)
        assert res["status"] == "CALIBRATION_NOT_EXECUTED"

    def test_12_queue_b_never_removes_mandatory_risk_cases_to_force_under_25(
        self, tmp_path: Path
    ):
        inp = Path("benchmarks/ground_truth/v2/hybrid")
        out = tmp_path / "out_queues"
        build_human_queues(inp, out, without_silver_execution=True)
        man = json.loads((out / "routing_manifest.json").read_text())
        assert man["overlap_exceeded_due_to_mandatory_risk_cases"] is True
        assert man["planned_overlap_rate"] >= 0.15

    def test_13_without_silver_queue_marked_provisional(self, tmp_path: Path):
        inp = Path("benchmarks/ground_truth/v2/hybrid")
        out = tmp_path / "out_queues"
        build_human_queues(inp, out, without_silver_execution=True)
        man = json.loads((out / "routing_manifest.json").read_text())
        assert man["queue_status"] == "PROVISIONAL_WITHOUT_SILVER"

    def test_36_holdout_rejected_by_silver_runner(self, tmp_path: Path):
        bad_pool = tmp_path / "bad_pool.jsonl"
        bad_pool.write_text(
            json.dumps({
                "question_id": "q_holdout_01",
                "passage_id": "ps_01",
                "text": "abc",
                "page_number": 92,
            })
            + "\n"
        )
        out_dir = tmp_path / "out"
        with pytest.raises(ValueError, match="HOLDOUT VIOLATION"):
            run_silver_triage_mock(bad_pool, out_dir, mode="validate-only")
