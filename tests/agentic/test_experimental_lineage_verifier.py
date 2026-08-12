"""Unit tests for LineageVerifier core module — RAGLab V7."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from raglab.agentic.experiments import (
    LineageAuditResult,
    LineageVerifier,
    RunControllerError,
    RunReceipt,
    RunReceiptStore,
    RunState,
    compute_file_sha256,
)


class TestExperimentalLineageVerifier(unittest.TestCase):
    """Test suite for LineageVerifier read-only core functions."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sandbox = Path(self.temp_dir.name).resolve()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_valid_run_dir(
        self, run_id: str = "run_1", slice_id: str = "slice5b"
    ) -> Path:
        run_dir = self.sandbox / slice_id / run_id
        receipts_dir = run_dir / "receipts"
        receipts_dir.mkdir(parents=True, exist_ok=True)

        proto_file = run_dir / "protocol.snapshot.json"
        proto_content = json.dumps({"protocol_id": "p1"}, sort_keys=True)
        proto_file.write_text(proto_content, encoding="utf-8")
        proto_sha = compute_file_sha256(proto_file)

        store = RunReceiptStore(run_dir)
        receipt = RunReceipt(
            schema_version=1,
            run_id=run_id,
            slice_id=slice_id,
            state=RunState.PREPARED,
            artifact_root=str(self.sandbox),
            run_directory=str(run_dir),
            implementation_commit="c1",
            protocol_commit="c2",
            protocol_sha256=proto_sha,
            input_hashes={},
            runner_version="1.0.0",
            created_at_utc="2026-08-11T12:00:00Z",
            previous_receipt_sha256=None,
            receipt_sha256="",
        )
        h = receipt.compute_hash()
        final_receipt = RunReceipt(
            schema_version=receipt.schema_version,
            run_id=receipt.run_id,
            slice_id=receipt.slice_id,
            state=receipt.state,
            artifact_root=receipt.artifact_root,
            run_directory=receipt.run_directory,
            implementation_commit=receipt.implementation_commit,
            protocol_commit=receipt.protocol_commit,
            protocol_sha256=receipt.protocol_sha256,
            input_hashes=receipt.input_hashes,
            runner_version=receipt.runner_version,
            created_at_utc=receipt.created_at_utc,
            previous_receipt_sha256=receipt.previous_receipt_sha256,
            receipt_sha256=h,
        )
        store.append(final_receipt)
        return run_dir

    def test_01_valid_result_minimal(self) -> None:
        run_dir = self._create_valid_run_dir()
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertTrue(audit.is_valid)
        self.assertEqual(len(audit.failure_reasons), 0)

    def test_02_invalid_result_failure_reasons(self) -> None:
        result = LineageAuditResult(
            is_valid=False, failure_reasons=("Reason 1", "Reason 2")
        )
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.failure_reasons), 2)
        self.assertIn("Reason 1", result.failure_reasons)

    def test_03_verify_input_hashes_missing_file(self) -> None:
        verifier = LineageVerifier()
        missing_p = self.sandbox / "non_existent.json"
        with self.assertRaises(RunControllerError):
            verifier.verify_input_hashes(
                actual_inputs={"qrels": missing_p},
                expected_hashes={"qrels": "abc123hash"},
            )

    def test_04_verify_input_hashes_divergent_hash(self) -> None:
        verifier = LineageVerifier()
        input_p = self.sandbox / "input.json"
        input_p.write_text("content", encoding="utf-8")
        with self.assertRaises(RunControllerError):
            verifier.verify_input_hashes(
                actual_inputs={"qrels": input_p},
                expected_hashes={"qrels": "wrong_hash"},
            )

    def test_05_verify_input_hashes_name_mismatch(self) -> None:
        verifier = LineageVerifier()
        input_p = self.sandbox / "input.json"
        input_p.write_text("content", encoding="utf-8")
        with self.assertRaises(RunControllerError):
            verifier.verify_input_hashes(
                actual_inputs={"actual_key": input_p},
                expected_hashes={"expected_key": "hash123"},
            )

    def test_06_verify_lineage_missing_chain(self) -> None:
        empty_dir = self.sandbox / "empty_run"
        empty_dir.mkdir(parents=True, exist_ok=True)
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(empty_dir)
        self.assertFalse(audit.is_valid)
        self.assertTrue(
            any("empty or missing" in r for r in audit.failure_reasons)
        )

    def test_07_verify_lineage_invalid_chain(self) -> None:
        run_dir = self.sandbox / "invalid_chain_run"
        rec_dir = run_dir / "receipts"
        rec_dir.mkdir(parents=True, exist_ok=True)
        (rec_dir / "000_PREPARED.json").write_text(
            "{corrupted json", encoding="utf-8"
        )
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)

    def test_08_verify_lineage_missing_snapshot(self) -> None:
        run_dir = self._create_valid_run_dir(run_id="missing_snap")
        (run_dir / "protocol.snapshot.json").unlink()
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)
        self.assertTrue(
            any("Missing protocol.snapshot.json" in r for r in audit.failure_reasons)
        )

    def test_09_verify_lineage_tampered_snapshot(self) -> None:
        run_dir = self._create_valid_run_dir(run_id="tampered_snap")
        (run_dir / "protocol.snapshot.json").write_text(
            '{"tampered": true}', encoding="utf-8"
        )
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)
        self.assertTrue(
            any("SHA-256 mismatch" in r for r in audit.failure_reasons)
        )

    def test_10_verify_lineage_multiple_failures_accumulated(self) -> None:
        run_dir = self.sandbox / "multi_fail_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        # missing receipts chain and missing protocol.snapshot.json
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)
        self.assertGreaterEqual(len(audit.failure_reasons), 2)

    def test_11_verifier_creates_no_files(self) -> None:
        run_dir = self._create_valid_run_dir(run_id="no_files_run")
        files_before = set(run_dir.rglob("*"))
        verifier = LineageVerifier()
        verifier.verify_lineage(run_dir)
        files_after = set(run_dir.rglob("*"))
        self.assertEqual(files_before, files_after)

    def test_12_verifier_alters_no_bytes(self) -> None:
        run_dir = self._create_valid_run_dir(run_id="no_bytes_run")
        proto = run_dir / "protocol.snapshot.json"
        sha_before = compute_file_sha256(proto)
        verifier = LineageVerifier()
        verifier.verify_lineage(run_dir)
        sha_after = compute_file_sha256(proto)
        self.assertEqual(sha_before, sha_after)

    def test_13_verifier_promotes_no_state(self) -> None:
        run_dir = self._create_valid_run_dir(run_id="no_promote_run")
        store = RunReceiptStore(run_dir)
        rec_before = store.load_latest(run_dir)
        verifier = LineageVerifier()

        with patch.object(
            RunReceiptStore,
            "append",
            side_effect=AssertionError("Mutation attempted!"),
        ):
            audit = verifier.verify_lineage(run_dir)

        rec_after = store.load_latest(run_dir)
        self.assertEqual(rec_before.state, rec_after.state)
        self.assertTrue(audit.is_valid)

    def test_14_is_read_only_returns_true(self) -> None:
        verifier = LineageVerifier(read_only=True)
        self.assertTrue(verifier.is_read_only())

        with self.assertRaises(ValueError):
            LineageVerifier(read_only=False)


if __name__ == "__main__":
    unittest.main()
