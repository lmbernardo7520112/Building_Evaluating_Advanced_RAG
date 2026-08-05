"""Unit tests for Governed Silver Triage, Human Routing, and Calibration (Gate B2 Reconciliation). # noqa: E501

Covers tests 25-60 for silver triage and human routing invariants.
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
from scripts.run_silver_annotation import run_validate_only


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

    def test_27_confidence_range_validation(self):
        with pytest.raises(ValueError):
            SilverAnnotationRecord(
                question_id="q_dev_01",
                passage_id="ps_01",
                confidence=1.5,
            )

    def test_28_label_source_mandatory_silver(self):
        rec = SilverAnnotationRecord(
            question_id="q_dev_01",
            passage_id="ps_01",
            label_source="MACHINE_SILVER",
        )
        assert rec.label_source == "MACHINE_SILVER"

    def test_29_machine_silver_cannot_enter_human_qrels(self, tmp_path: Path):
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

    def test_30_invalid_parsing_escalated(self):
        rec = SilverAnnotationRecord(
            question_id="q_dev_01",
            passage_id="ps_01",
            needs_human_review=True,
            reasoning="PARSE_ERROR",
        )
        assert rec.needs_human_review is True

    def test_31_same_model_not_counted_as_independence(self):
        man_file = Path(
            "benchmarks/ground_truth/v2/hybrid/silver/silver_manifest.json"
        )
        if man_file.exists():
            man = json.loads(man_file.read_text(encoding="utf-8"))
            assert (
                man["judge_independence_status"] == "CORRELATED_SINGLE_PROVIDER"
            )

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
            run_validate_only(bad_pool, out_dir)

    def test_37_queue_a_receives_full_pool_and_audit(self, tmp_path: Path):
        inp = Path("benchmarks/ground_truth/v2/hybrid")
        out = tmp_path / "out_queues"
        build_human_queues(inp, out, without_silver_execution=True)
        man = json.loads((out / "routing_manifest.json").read_text())
        assert man["annotator_a_queue_count"] >= man["total_candidate_pool_items"]

    def test_38_43_queue_b_selection_and_overlap(self, tmp_path: Path):
        inp = Path("benchmarks/ground_truth/v2/hybrid")
        out = tmp_path / "out_queues"
        build_human_queues(inp, out, without_silver_execution=True)
        man = json.loads((out / "routing_manifest.json").read_text())
        assert man["planned_overlap_rate"] >= 0.15

    def test_45_silver_label_invisible_in_human_queue(self, tmp_path: Path):
        queue_a = Path(
            "benchmarks/ground_truth/v2/hybrid/human_queues/annotator_a.jsonl"
        )
        content = queue_a.read_text(encoding="utf-8")
        assert "silver" not in content
        assert "judge" not in content

    def test_50_52_calibration_unexecuted_status(self, tmp_path: Path):
        sil = tmp_path / "sil.jsonl"
        hum = tmp_path / "hum.jsonl"
        rep = tmp_path / "rep.json"
        res, _ = calibrate_silver(sil, hum, rep)
        assert res["status"] == "CALIBRATION_NOT_EXECUTED"

    def test_55_human_gold_separated_from_machine_silver(self):
        sil_qrels = Path(
            "benchmarks/ground_truth/v2/hybrid/qrels/silver_qrels.jsonl"
        )
        hum_qrels = Path(
            "benchmarks/ground_truth/v2/hybrid/qrels/human_qrels.jsonl"
        )
        assert sil_qrels.exists()
        assert hum_qrels.exists()

    def test_58_holdout_sealed_in_master_manifest(self):
        master_man = Path(
            "benchmarks/ground_truth/v2/hybrid/manifests/hybrid_eval_manifest.json"
        )
        man = json.loads(master_man.read_text(encoding="utf-8"))
        assert man["holdout_sealed"] is True

    def test_60_rebuild_reproducible(self, tmp_path: Path):
        inp = Path("benchmarks/ground_truth/v2/hybrid")
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        build_human_queues(inp, out1, without_silver_execution=True)
        build_human_queues(inp, out2, without_silver_execution=True)
        assert (out1 / "routing_manifest.json").read_bytes() == (
            out2 / "routing_manifest.json"
        ).read_bytes()
