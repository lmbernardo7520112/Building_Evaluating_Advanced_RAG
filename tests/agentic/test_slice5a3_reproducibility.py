"""Hermetic unit tests for reproducibility and byte-by-byte comparison module."""

from __future__ import annotations

import unittest

from raglab.agentic.evaluation.parity_contracts import (
    ArmRun,
    CanonicalRetrievedItem,
    JudgmentStatus,
    MappingStatus,
)
from raglab.agentic.evaluation.reproducibility import (
    compare_arm_runs,
    compute_scientific_payload_hash,
)


class TestReproducibilityModule(unittest.TestCase):
    """Test suite for reproducibility and scientific payload hash calculation."""

    def test_01_payload_hash_ignores_timestamps(self) -> None:
        """compute_scientific_payload_hash ignores timestamp and ephemeral fields."""
        data_1 = {
            "protocol_id": "slice5a3_parity_gate_v1",
            "timestamp": "2026-08-10T12:00:00Z",
            "latency": 0.123,
            "metrics": {"ndcg": 0.8},
        }
        data_2 = {
            "protocol_id": "slice5a3_parity_gate_v1",
            "timestamp": "2026-08-11T15:30:00Z",
            "latency": 0.456,
            "metrics": {"ndcg": 0.8},
        }
        h1 = compute_scientific_payload_hash(data_1)
        h2 = compute_scientific_payload_hash(data_2)
        self.assertEqual(h1, h2)

    def test_02_compare_arm_runs_identical(self) -> None:
        """Identical arm runs yield 1.0 repeatability topk identity."""
        item = CanonicalRetrievedItem(
            qid="q_dev_01",
            arm_id="F0",
            rank=1,
            passage_id="ps_001",
            page_number=91,
            content_sha256="abc",
            score=1.0,
            mapping_status=MappingStatus.EXACT_SUBSTRING,
            judgment_status=JudgmentStatus.JUDGED,
            human_grade=2.0,
        )
        run1 = [
            ArmRun(
                qid="q_dev_01",
                arm_id="F0",
                retrieved_items=[item],
                latency=0.1,
                configuration_hashes={"arm_id": "F0"},
                corpus_hash="corp_hash",
                deterministic_content_hash="det_hash",
            )
        ]
        run2 = [
            ArmRun(
                qid="q_dev_01",
                arm_id="F0",
                retrieved_items=[item],
                latency=0.2,
                configuration_hashes={"arm_id": "F0"},
                corpus_hash="corp_hash",
                deterministic_content_hash="det_hash",
            )
        ]

        comparisons, identity, rank_match = compare_arm_runs(run1, run2)
        self.assertEqual(identity, 1.0)
        self.assertTrue(rank_match)
        self.assertEqual(len(comparisons), 1)
        self.assertTrue(comparisons[0].exact_topk_match)

    def test_03_compare_arm_runs_divergent(self) -> None:
        """Divergent passage IDs yield identity < 1.0."""
        item1 = CanonicalRetrievedItem(
            qid="q_dev_01",
            arm_id="F0",
            rank=1,
            passage_id="ps_001",
            page_number=91,
            content_sha256="abc",
            score=1.0,
            mapping_status=MappingStatus.EXACT_SUBSTRING,
            judgment_status=JudgmentStatus.JUDGED,
        )
        item2 = CanonicalRetrievedItem(
            qid="q_dev_01",
            arm_id="F0",
            rank=1,
            passage_id="ps_002",
            page_number=92,
            content_sha256="def",
            score=1.0,
            mapping_status=MappingStatus.EXACT_SUBSTRING,
            judgment_status=JudgmentStatus.JUDGED,
        )

        run1 = [
            ArmRun(
                qid="q_dev_01",
                arm_id="F0",
                retrieved_items=[item1],
                latency=0.1,
                configuration_hashes={"arm_id": "F0"},
                corpus_hash="corp_hash",
                deterministic_content_hash="det1",
            )
        ]
        run2 = [
            ArmRun(
                qid="q_dev_01",
                arm_id="F0",
                retrieved_items=[item2],
                latency=0.1,
                configuration_hashes={"arm_id": "F0"},
                corpus_hash="corp_hash",
                deterministic_content_hash="det2",
            )
        ]

        comparisons, identity, rank_match = compare_arm_runs(run1, run2)
        self.assertEqual(identity, 0.0)
        self.assertFalse(rank_match)


if __name__ == "__main__":
    unittest.main()
