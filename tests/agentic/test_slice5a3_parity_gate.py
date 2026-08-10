"""Hermetic unit tests for Parity Gate execution and fail-closed governance rules."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from raglab.agentic.evaluation.parity_contracts import ParityOutcomeCategory
from raglab.agentic.evaluation.parity_gate import (
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
        # Empty human qrels (no ratings for retrieved passages)
        qrels_path = self.dir_path / "human_qrels.jsonl"
        qrels_path.write_text("", encoding="utf-8")

        protocol = {
            "protocol_id": "test_proto",
            "dev_qids": ["q_dev_01"],
            "fixed_arms": ["F0"],
            "top_k": 1,
        }

        result, arm_runs, unjudged_queue = run_parity_gate_evaluation(
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

        # Confirm queue item does NOT contain suggested grade or gold/silver labels
        item = unjudged_queue[0]
        self.assertNotIn("suggested_grade", item)
        self.assertNotIn("gold", item)
        self.assertNotIn("silver", item)
        self.assertIn("qid", item)
        self.assertIn("passage_id", item)

    def test_02_full_coverage_passes_evaluable(self) -> None:
        """When all retrieved items have human qrels, status is SCIENTIFICALLY_EVALUABLE."""
        qrels_path = self.dir_path / "human_qrels.jsonl"
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

        result, arm_runs, unjudged_queue = run_parity_gate_evaluation(
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


if __name__ == "__main__":
    unittest.main()
