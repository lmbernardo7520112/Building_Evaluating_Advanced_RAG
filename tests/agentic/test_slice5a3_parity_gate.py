"""Hermetic unit tests for Parity Gate execution and fail-closed governance rules."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from raglab.agentic.evaluation.parity_contracts import (
    CanonicalRetrievedItem,
    JudgmentStatus,
    MappingStatus,
    ParityOutcomeCategory,
)
from raglab.agentic.evaluation.parity_gate import (
    ParityGateError,
    load_human_qrels,
    run_parity_gate_evaluation,
)


class TestParityGateExecution(unittest.TestCase):
    """Test suite for Parity Gate governance rules and fail-closed logic."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

        # Create canonical passage registry with 3 passages
        self.reg_path = self.dir_path / "passage_registry.jsonl"
        self.man_path = self.dir_path / "passage_registry_manifest.json"

        passages = [
            {"passage_id": "ps_001", "document_id": "doc1", "page_number": 91, "text": "Informal proof technique and theorem."},
            {"passage_id": "ps_002", "document_id": "doc1", "page_number": 92, "text": "Counterexample finding in logic."},
            {"passage_id": "ps_003", "document_id": "doc1", "page_number": 93, "text": "Mathematical induction principles."},
        ]
        with self.reg_path.open("w", encoding="utf-8") as f:
            for p in passages:
                f.write(json.dumps(p) + "\n")

        self.man_path.write_text(json.dumps({"version": "2.0.0"}), encoding="utf-8")

        # Create questions file with 1 DEV question and 1 TEST question
        self.q_path = self.dir_path / "controlled_chapter2.json"
        q_data = {
            "questions": [
                {
                    "qid": "q_dev_01",
                    "question_id": "q_dev_01",
                    "question_text": "What is proof by exhaustion?",
                    "split": "development",
                },
                {
                    "qid": "q_test_01",
                    "question_id": "q_test_01",
                    "question_text": "Sealed test question text",
                    "split": "test",
                },
            ]
        }
        self.q_path.write_text(json.dumps(q_data), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_01_unjudged_item_fails_closed_to_not_evaluable_judged(self) -> None:
        """Items without explicit human qrels remain UNJUDGED and trigger NOT_EVALUABLE_JUDGED_COVERAGE."""
        qrels_path = self.dir_path / "human_qrels_final.jsonl"
        qrels_path.write_text("", encoding="utf-8")

        protocol = {
            "protocol_id": "test_proto",
            "dev_qids": ["q_dev_01"],
            "fixed_arms": ["F0"],
            "top_k": 1,
        }

        result, arm_runs, unjudged_queue, exec_metadata = run_parity_gate_evaluation(
            protocol=protocol,
            passage_registry_path=self.reg_path,
            passage_registry_manifest_path=self.man_path,
            human_qrels_path=qrels_path,
            questions_path=self.q_path,
            top_k=1,
        )

        self.assertEqual(result.status, ParityOutcomeCategory.NOT_EVALUABLE_JUDGED_COVERAGE)
        self.assertEqual(result.metrics_status, "NOT_APPLICABLE")
        self.assertGreater(result.unjudged_count, 0)
        self.assertGreater(len(unjudged_queue), 0)

        item = unjudged_queue[0]
        self.assertNotIn("suggested_grade", item)
        self.assertNotIn("gold", item)
        self.assertNotIn("silver", item)
        self.assertIn("qid", item)
        self.assertIn("passage_id", item)

    def test_02_full_coverage_passes_evaluable(self) -> None:
        """When all retrieved items have human qrels, status is SCIENTIFICALLY_EVALUABLE."""
        qrels_path = self.dir_path / "human_qrels_final.jsonl"
        qrels_lines = [
            json.dumps({"question_id": "q_dev_01", "passage_id": "ps_001", "relevance_grade": 2.0}),
            json.dumps({"question_id": "q_dev_01", "passage_id": "ps_002", "relevance_grade": 1.0}),
            json.dumps({"question_id": "q_dev_01", "passage_id": "ps_003", "relevance_grade": 0.0}),
        ]
        qrels_path.write_text("\n".join(qrels_lines), encoding="utf-8")

        protocol = {
            "protocol_id": "test_proto",
            "dev_qids": ["q_dev_01"],
            "fixed_arms": ["F0"],
            "top_k": 1,
        }

        result, arm_runs, unjudged_queue, exec_metadata = run_parity_gate_evaluation(
            protocol=protocol,
            passage_registry_path=self.reg_path,
            passage_registry_manifest_path=self.man_path,
            human_qrels_path=qrels_path,
            questions_path=self.q_path,
            top_k=1,
        )

        self.assertEqual(result.status, ParityOutcomeCategory.SCIENTIFICALLY_EVALUABLE)
        self.assertEqual(result.metrics_status, "APPLICABLE")
        self.assertEqual(result.unjudged_count, 0)
        self.assertEqual(len(unjudged_queue), 0)

    def test_03_technical_id_never_occupies_canonical_passage_id(self) -> None:
        """Prohibit technical chunk ID in canonical passage_id field when unmapped."""
        unmapped_item = CanonicalRetrievedItem(
            qid="q_dev_01",
            arm_id="S0",
            rank=1,
            technical_chunk_id="doc_p99_s0",
            technical_node_id="node_123",
            anchor_passage_id=None,
            supporting_passage_ids=[],
            source_offsets=None,
            projection_method="UNMAPPED",
            page_number=99,
            content_sha256="abc",
            score=0.9,
            mapping_status=MappingStatus.UNMAPPED,
            judgment_status=JudgmentStatus.UNMAPPED,
            human_grade=None,
        )
        self.assertEqual(unmapped_item.passage_id, "")
        self.assertIsNone(unmapped_item.canonical_passage_id)
        self.assertNotEqual(unmapped_item.passage_id, "doc_p99_s0")

    def test_04_strict_qrels_parser_rejects_missing_relevance_grade(self) -> None:
        """Strict Qrels Parser must reject lines lacking relevance_grade."""
        qrels_path = self.dir_path / "bad_qrels.jsonl"
        qrels_path.write_text(json.dumps({"question_id": "q1", "passage_id": "ps_001", "legacy_grade": 2}), encoding="utf-8")
        with self.assertRaises(ParityGateError):
            load_human_qrels(qrels_path)

    def test_05_unknown_arm_id_rejected(self) -> None:
        """Unknown arm ID must raise ParityGateError."""
        qrels_path = self.dir_path / "human_qrels_final.jsonl"
        qrels_path.write_text("", encoding="utf-8")
        protocol = {
            "protocol_id": "test_proto",
            "dev_qids": ["q_dev_01"],
            "fixed_arms": ["INVALID_ARM"],
            "top_k": 1,
        }
        with self.assertRaises(ParityGateError):
            run_parity_gate_evaluation(
                protocol=protocol,
                passage_registry_path=self.reg_path,
                passage_registry_manifest_path=self.man_path,
                human_qrels_path=qrels_path,
                questions_path=self.q_path,
            )


if __name__ == "__main__":
    unittest.main()
