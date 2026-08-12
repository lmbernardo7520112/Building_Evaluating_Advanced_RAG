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
        self,
        run_id: str = "run_1",
        slice_id: str = "slice5b",
        artifact_inventory: tuple[str, ...] = (),
        artifact_hashes: dict[str, str] | None = None,
    ) -> Path:
        run_dir = self.sandbox / slice_id / run_id
        receipts_dir = run_dir / "receipts"
        receipts_dir.mkdir(parents=True, exist_ok=True)

        proto_file = run_dir / "protocol.snapshot.json"
        proto_content = json.dumps({"protocol_id": "p1"}, sort_keys=True)
        proto_file.write_text(proto_content, encoding="utf-8")
        proto_sha = compute_file_sha256(proto_file)

        hashes_map = artifact_hashes if artifact_hashes is not None else {}

        hashes_content = (
            "\n".join(f"{h}  {path}" for path, h in hashes_map.items()) + "\n"
            if hashes_map
            else ""
        )
        (run_dir / "hashes.sha256").write_text(hashes_content, encoding="utf-8")

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
            artifact_inventory=artifact_inventory,
            artifact_hashes=hashes_map,
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
            artifact_inventory=receipt.artifact_inventory,
            artifact_hashes=receipt.artifact_hashes,
        )
        store.append(final_receipt)
        return run_dir

    def test_01_valid_result_minimal(self) -> None:
        run_dir = self._create_valid_run_dir()
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertTrue(audit.is_valid)

    def test_02_invalid_result_failure_reasons(self) -> None:
        res = LineageAuditResult(is_valid=False, failure_reasons=("err1",))
        self.assertFalse(res.is_valid)
        self.assertEqual(res.failure_reasons, ("err1",))

    def test_03_verify_input_hashes_missing_file(self) -> None:
        verifier = LineageVerifier()
        with self.assertRaises(RunControllerError):
            verifier.verify_input_hashes(
                actual_inputs={"inp1": self.sandbox / "missing.txt"},
                expected_hashes={"inp1": "abc"},
            )

    def test_04_verify_input_hashes_divergent_hash(self) -> None:
        f = self.sandbox / "inp.txt"
        f.write_text("hello", encoding="utf-8")
        verifier = LineageVerifier()
        with self.assertRaises(RunControllerError):
            verifier.verify_input_hashes(
                actual_inputs={"inp1": f},
                expected_hashes={"inp1": "0000000000000000000000000000000000000000000000000000000000000000"},
            )

    def test_05_verify_input_hashes_name_mismatch(self) -> None:
        f = self.sandbox / "inp.txt"
        f.write_text("hello", encoding="utf-8")
        verifier = LineageVerifier()
        with self.assertRaises(RunControllerError):
            verifier.verify_input_hashes(
                actual_inputs={"inp1": f},
                expected_hashes={"inp2": "abc"},
            )

    def test_06_verify_lineage_missing_chain(self) -> None:
        run_dir = self.sandbox / "missing_chain_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)
        self.assertTrue(len(audit.failure_reasons) > 0)

    def test_07_verify_lineage_invalid_chain(self) -> None:
        run_dir = self._create_valid_run_dir(run_id="invalid_chain_run")
        (run_dir / "receipts" / "000_PREPARED.json").write_text("corrupted", encoding="utf-8")
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)

    def test_08_verify_lineage_missing_snapshot(self) -> None:
        run_dir = self._create_valid_run_dir(run_id="missing_snapshot_run")
        (run_dir / "protocol.snapshot.json").unlink()
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)

    def test_09_verify_lineage_tampered_snapshot(self) -> None:
        run_dir = self._create_valid_run_dir(run_id="tampered_snapshot_run")
        (run_dir / "protocol.snapshot.json").write_text('{"tampered": true}', encoding="utf-8")
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)

    def test_10_verify_lineage_multiple_failures_accumulated(self) -> None:
        run_dir = self._create_valid_run_dir(run_id="multi_fail_run")
        (run_dir / "protocol.snapshot.json").unlink()
        (run_dir / "receipts" / "000_PREPARED.json").write_text("bad", encoding="utf-8")
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

    def test_15_artifact_inventory_valid(self) -> None:
        run_dir = self.sandbox / "slice5b" / "inv_valid"
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        f_path = raw_dir / "data.json"
        f_path.write_text("{}", encoding="utf-8")
        h = compute_file_sha256(f_path)

        self._create_valid_run_dir(
            run_id="inv_valid",
            artifact_inventory=("raw/data.json",),
            artifact_hashes={"raw/data.json": h},
        )
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertTrue(audit.is_valid)

    def test_16_artifact_inventory_missing_file(self) -> None:
        run_dir = self._create_valid_run_dir(
            run_id="inv_missing",
            artifact_inventory=("raw/missing.json",),
        )
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)
        self.assertTrue(
            any("Missing or non-file" in r for r in audit.failure_reasons)
        )

    def test_17_artifact_inventory_absolute_path_rejected(self) -> None:
        run_dir = self._create_valid_run_dir(
            run_id="inv_abs",
            artifact_inventory=("/abs/path/artifact.json",),
        )
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)
        self.assertTrue(
            any("must be relative" in r for r in audit.failure_reasons)
        )

    def test_18_artifact_inventory_path_traversal_rejected(self) -> None:
        run_dir = self._create_valid_run_dir(
            run_id="inv_escape",
            artifact_inventory=("../outside.json",),
        )
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)
        self.assertTrue(
            any("escapes run directory" in r for r in audit.failure_reasons)
        )

    def test_19_artifact_inventory_directory_rejected(self) -> None:
        run_dir = self.sandbox / "slice5b" / "inv_dir"
        sub_dir = run_dir / "raw"
        sub_dir.mkdir(parents=True, exist_ok=True)

        self._create_valid_run_dir(
            run_id="inv_dir",
            artifact_inventory=("raw",),
        )
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)
        self.assertTrue(
            any("Missing or non-file" in r for r in audit.failure_reasons)
        )

    def test_20_artifact_hashes_valid(self) -> None:
        run_dir = self.sandbox / "slice5b" / "hashes_valid"
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        f_path = raw_dir / "data.json"
        f_path.write_text('{"v": 1}', encoding="utf-8")
        h = compute_file_sha256(f_path)

        self._create_valid_run_dir(
            run_id="hashes_valid",
            artifact_inventory=("raw/data.json",),
            artifact_hashes={"raw/data.json": h},
        )
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertTrue(audit.is_valid)

    def test_21_artifact_hashes_tampered_content(self) -> None:
        run_dir = self.sandbox / "slice5b" / "hashes_tampered"
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        f_path = raw_dir / "data.json"
        f_path.write_text('{"v": 1}', encoding="utf-8")

        self._create_valid_run_dir(
            run_id="hashes_tampered",
            artifact_inventory=("raw/data.json",),
            artifact_hashes={"raw/data.json": "0" * 64},
        )
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)
        self.assertTrue(
            any("SHA-256 mismatch" in r for r in audit.failure_reasons)
        )

    def test_22_artifact_hashes_missing_entry(self) -> None:
        run_dir = self.sandbox / "slice5b" / "hashes_missing_entry"
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        f_path = raw_dir / "data.json"
        f_path.write_text('{"v": 1}', encoding="utf-8")

        self._create_valid_run_dir(
            run_id="hashes_missing_entry",
            artifact_inventory=("raw/data.json",),
            artifact_hashes={},
        )
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)
        self.assertTrue(
            any("Missing expected artifact hash entry" in r for r in audit.failure_reasons)
        )

    def test_23_undeclared_artifact_all_declared_valid(self) -> None:
        run_dir = self.sandbox / "slice5b" / "undec_valid"
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        f_path = raw_dir / "data.json"
        f_path.write_text('{"v": 1}', encoding="utf-8")
        h = compute_file_sha256(f_path)

        self._create_valid_run_dir(
            run_id="undec_valid",
            artifact_inventory=("raw/data.json",),
            artifact_hashes={"raw/data.json": h},
        )
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertTrue(audit.is_valid)

    def test_24_undeclared_artifact_extra_in_raw_rejected(self) -> None:
        run_dir = self.sandbox / "slice5b" / "undec_raw"
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        f_path = raw_dir / "data.json"
        f_path.write_text('{"v": 1}', encoding="utf-8")
        h = compute_file_sha256(f_path)

        extra_path = raw_dir / "unexpected.json"
        extra_path.write_text('{"extra": true}', encoding="utf-8")

        self._create_valid_run_dir(
            run_id="undec_raw",
            artifact_inventory=("raw/data.json",),
            artifact_hashes={"raw/data.json": h},
        )
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)
        self.assertTrue(
            any("Undeclared artifact file found" in r for r in audit.failure_reasons)
        )

    def test_25_undeclared_artifact_extra_in_derived_or_logs_rejected(self) -> None:
        run_dir = self.sandbox / "slice5b" / "undec_logs"
        raw_dir = run_dir / "raw"
        logs_dir = run_dir / "logs"
        raw_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        f_path = raw_dir / "data.json"
        f_path.write_text('{"v": 1}', encoding="utf-8")
        h = compute_file_sha256(f_path)

        log_path = logs_dir / "unrecorded.log"
        log_path.write_text("extra log", encoding="utf-8")

        self._create_valid_run_dir(
            run_id="undec_logs",
            artifact_inventory=("raw/data.json",),
            artifact_hashes={"raw/data.json": h},
        )
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)
        self.assertTrue(
            any("Undeclared artifact file found" in r for r in audit.failure_reasons)
        )

    def test_26_hashes_manifest_valid(self) -> None:
        run_dir = self.sandbox / "slice5b" / "manifest_valid"
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        f_path = raw_dir / "data.json"
        f_path.write_text('{"v": 1}', encoding="utf-8")
        h = compute_file_sha256(f_path)

        self._create_valid_run_dir(
            run_id="manifest_valid",
            artifact_inventory=("raw/data.json",),
            artifact_hashes={"raw/data.json": h},
        )
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertTrue(audit.is_valid)

    def test_27_hashes_manifest_digest_mismatch(self) -> None:
        run_dir = self.sandbox / "slice5b" / "manifest_mismatch"
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        f_path = raw_dir / "data.json"
        f_path.write_text('{"v": 1}', encoding="utf-8")
        h = compute_file_sha256(f_path)

        self._create_valid_run_dir(
            run_id="manifest_mismatch",
            artifact_inventory=("raw/data.json",),
            artifact_hashes={"raw/data.json": h},
        )
        (run_dir / "hashes.sha256").write_text(
            f"{'0' * 64}  raw/data.json\n", encoding="utf-8"
        )
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)
        self.assertTrue(
            any("Manifest hash mismatch" in r for r in audit.failure_reasons)
        )

    def test_28_hashes_manifest_malformed_line(self) -> None:
        run_dir = self.sandbox / "slice5b" / "manifest_malformed"
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        f_path = raw_dir / "data.json"
        f_path.write_text('{"v": 1}', encoding="utf-8")
        h = compute_file_sha256(f_path)

        self._create_valid_run_dir(
            run_id="manifest_malformed",
            artifact_inventory=("raw/data.json",),
            artifact_hashes={"raw/data.json": h},
        )
        (run_dir / "hashes.sha256").write_text(
            "this_is_not_a_sha256_line\n", encoding="utf-8"
        )
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)
        self.assertTrue(
            any("Malformed line in hashes.sha256" in r for r in audit.failure_reasons)
        )

    def test_29_hashes_manifest_duplicate_or_missing_path(self) -> None:
        run_dir = self.sandbox / "slice5b" / "manifest_dup"
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        f_path = raw_dir / "data.json"
        f_path.write_text('{"v": 1}', encoding="utf-8")
        h = compute_file_sha256(f_path)

        self._create_valid_run_dir(
            run_id="manifest_dup",
            artifact_inventory=("raw/data.json",),
            artifact_hashes={"raw/data.json": h},
        )
        (run_dir / "hashes.sha256").write_text(
            f"{h}  raw/data.json\n{h}  raw/data.json\n", encoding="utf-8"
        )
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)
        self.assertTrue(
            any("Duplicate path entry" in r for r in audit.failure_reasons)
        )


if __name__ == "__main__":
    unittest.main()
