"""Focal Unit Test Suite — Experimental Readiness Integrity & Receipt Core (GREEN-1).

Tests canonical JSON serialization, SHA-256 computation, immutable receipts,
state machine transitions, receipt chain validation, and artifact integrity checks.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from raglab.agentic.experiments import (
    IntegrityError,
    ReceiptStoreError,
    RunReceipt,
    RunState,
    compute_bytes_sha256,
    compute_canonical_json_bytes,
    compute_canonical_json_sha256,
    compute_file_sha256,
    validate_state_transition,
    verify_artifact_integrity,
    verify_receipt_chain,
)


class TestExperimentalReadinessIntegrityCore(unittest.TestCase):
    """Focal unit tests for experimental integrity and receipt core."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sandbox = Path(self.temp_dir.name).resolve()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # 1. JSON equivalente gera bytes idênticos
    def test_01_canonical_json_equivalent_structures_identical_bytes(self) -> None:
        d1 = {"b": 2, "a": 1, "c": [3, 4]}
        d2 = {"a": 1, "c": [3, 4], "b": 2}
        self.assertEqual(
            compute_canonical_json_bytes(d1), compute_canonical_json_bytes(d2)
        )

    # 2. ordem diferente de chaves não altera digest
    def test_02_canonical_json_different_key_order_same_digest(self) -> None:
        d1 = {"z": "val", "a": "val2"}
        d2 = {"a": "val2", "z": "val"}
        self.assertEqual(
            compute_canonical_json_sha256(d1), compute_canonical_json_sha256(d2)
        )

    # 3. NaN é rejeitado
    def test_03_canonical_json_nan_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_canonical_json_bytes({"invalid": float("nan")})

    # 4. SHA-256 de bytes conhecido
    def test_04_sha256_bytes_known(self) -> None:
        expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        self.assertEqual(compute_bytes_sha256(b"hello"), expected)

    # 5. SHA-256 de arquivo real
    def test_05_sha256_file_real(self) -> None:
        f = self.sandbox / "test.txt"
        f.write_bytes(b"hello")
        expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        self.assertEqual(compute_file_sha256(f), expected)

    # 6. arquivo ausente falha
    def test_06_sha256_file_missing_fails(self) -> None:
        with self.assertRaises(FileNotFoundError):
            compute_file_sha256(self.sandbox / "nonexistent.txt")

    # 7. receipt válido é criado
    def test_07_valid_receipt_created(self) -> None:
        receipt = RunReceipt(
            schema_version=1,
            run_id="run1",
            slice_id="slice5b",
            state=RunState.PREPARED,
            artifact_root=str(self.sandbox),
            run_directory=str(self.sandbox / "run1"),
            implementation_commit="a" * 40,
            protocol_commit="b" * 40,
            protocol_sha256="c" * 64,
            input_hashes={"qrels": "d" * 64},
            runner_version="1.0.0",
            created_at_utc="2026-08-10T12:00:00Z",
        )
        self.assertEqual(receipt.run_id, "run1")
        self.assertEqual(receipt.state, RunState.PREPARED)

    # 8. receipt é imutável
    def test_08_receipt_is_immutable(self) -> None:
        receipt = RunReceipt(
            schema_version=1,
            run_id="run1",
            slice_id="slice5b",
            state=RunState.PREPARED,
            artifact_root=str(self.sandbox),
            run_directory=str(self.sandbox / "run1"),
            implementation_commit="a" * 40,
            protocol_commit="b" * 40,
            protocol_sha256="c" * 64,
            input_hashes={},
            runner_version="1.0.0",
            created_at_utc="2026-08-10T12:00:00Z",
        )
        with self.assertRaises(FrozenInstanceError):
            receipt.state = RunState.RUN_STARTED  # type: ignore[misc]

    # 9. hash do receipt é recomputável
    def test_09_receipt_hash_recomputable(self) -> None:
        receipt = RunReceipt(
            schema_version=1,
            run_id="run1",
            slice_id="slice5b",
            state=RunState.PREPARED,
            artifact_root=str(self.sandbox),
            run_directory=str(self.sandbox / "run1"),
            implementation_commit="a" * 40,
            protocol_commit="b" * 40,
            protocol_sha256="c" * 64,
            input_hashes={},
            runner_version="1.0.0",
            created_at_utc="2026-08-10T12:00:00Z",
        )
        computed_hash = receipt.compute_hash()
        self.assertEqual(len(computed_hash), 64)

    # 10. adulteração invalida receipt
    def test_10_tampered_receipt_invalidated(self) -> None:
        r1 = RunReceipt(
            schema_version=1,
            run_id="run1",
            slice_id="slice5b",
            state=RunState.PREPARED,
            artifact_root=str(self.sandbox),
            run_directory=str(self.sandbox / "run1"),
            implementation_commit="a" * 40,
            protocol_commit="b" * 40,
            protocol_sha256="c" * 64,
            input_hashes={},
            runner_version="1.0.0",
            created_at_utc="2026-08-10T12:00:00Z",
            receipt_sha256="wrong_hash_value",
        )
        with self.assertRaises(ReceiptStoreError):
            verify_receipt_chain([r1])

    # 11. transição válida é aceita
    def test_11_valid_transition_accepted(self) -> None:
        self.assertTrue(
            validate_state_transition(RunState.PREPARED, RunState.RUN_STARTED)
        )
        self.assertTrue(
            validate_state_transition(RunState.RUN_STARTED, RunState.RUN_COMPLETED)
        )
        self.assertTrue(
            validate_state_transition(RunState.RUN_STARTED, RunState.FAILED)
        )
        self.assertTrue(
            validate_state_transition(
                RunState.RUN_COMPLETED, RunState.AUDIT_COMPLETED
            )
        )

    # 12. salto de estado é rejeitado
    def test_12_state_jump_rejected(self) -> None:
        with self.assertRaises(ReceiptStoreError):
            validate_state_transition(RunState.PREPARED, RunState.AUDIT_COMPLETED)

    # 13. regressão de estado é rejeitada
    def test_13_state_regression_rejected(self) -> None:
        with self.assertRaises(ReceiptStoreError):
            validate_state_transition(RunState.RUN_COMPLETED, RunState.RUN_STARTED)

    # 14. cadeia válida é aceita
    def test_14_valid_receipt_chain_accepted(self) -> None:
        r0_data = RunReceipt(
            schema_version=1,
            run_id="run1",
            slice_id="slice5b",
            state=RunState.PREPARED,
            artifact_root=str(self.sandbox),
            run_directory=str(self.sandbox / "run1"),
            implementation_commit="a" * 40,
            protocol_commit="b" * 40,
            protocol_sha256="c" * 64,
            input_hashes={},
            runner_version="1.0.0",
            created_at_utc="2026-08-10T12:00:00Z",
        )
        h0 = r0_data.compute_hash()
        r0 = RunReceipt.from_dict({**r0_data.to_dict(), "receipt_sha256": h0})

        r1_data = RunReceipt(
            schema_version=1,
            run_id="run1",
            slice_id="slice5b",
            state=RunState.RUN_STARTED,
            artifact_root=str(self.sandbox),
            run_directory=str(self.sandbox / "run1"),
            implementation_commit="a" * 40,
            protocol_commit="b" * 40,
            protocol_sha256="c" * 64,
            input_hashes={},
            runner_version="1.0.0",
            created_at_utc="2026-08-10T12:00:00Z",
            started_at_utc="2026-08-10T12:01:00Z",
            previous_receipt_sha256=h0,
        )
        h1 = r1_data.compute_hash()
        r1 = RunReceipt.from_dict({**r1_data.to_dict(), "receipt_sha256": h1})

        self.assertTrue(verify_receipt_chain([r0, r1]))

    # 15. predecessor incorreto é rejeitado
    def test_15_incorrect_predecessor_rejected(self) -> None:
        r0_data = RunReceipt(
            schema_version=1,
            run_id="run1",
            slice_id="slice5b",
            state=RunState.PREPARED,
            artifact_root=str(self.sandbox),
            run_directory=str(self.sandbox / "run1"),
            implementation_commit="a" * 40,
            protocol_commit="b" * 40,
            protocol_sha256="c" * 64,
            input_hashes={},
            runner_version="1.0.0",
            created_at_utc="2026-08-10T12:00:00Z",
        )
        h0 = r0_data.compute_hash()
        r0 = RunReceipt.from_dict({**r0_data.to_dict(), "receipt_sha256": h0})

        r1_data = RunReceipt(
            schema_version=1,
            run_id="run1",
            slice_id="slice5b",
            state=RunState.RUN_STARTED,
            artifact_root=str(self.sandbox),
            run_directory=str(self.sandbox / "run1"),
            implementation_commit="a" * 40,
            protocol_commit="b" * 40,
            protocol_sha256="c" * 64,
            input_hashes={},
            runner_version="1.0.0",
            created_at_utc="2026-08-10T12:00:00Z",
            previous_receipt_sha256="wrong_prev_hash",
        )
        h1 = r1_data.compute_hash()
        r1 = RunReceipt.from_dict({**r1_data.to_dict(), "receipt_sha256": h1})

        with self.assertRaises(ReceiptStoreError):
            verify_receipt_chain([r0, r1])

    # 16. receipt omitido ou duplicado é rejeitado
    def test_16_duplicate_or_omitted_receipt_rejected(self) -> None:
        r0_data = RunReceipt(
            schema_version=1,
            run_id="run1",
            slice_id="slice5b",
            state=RunState.PREPARED,
            artifact_root=str(self.sandbox),
            run_directory=str(self.sandbox / "run1"),
            implementation_commit="a" * 40,
            protocol_commit="b" * 40,
            protocol_sha256="c" * 64,
            input_hashes={},
            runner_version="1.0.0",
            created_at_utc="2026-08-10T12:00:00Z",
        )
        h0 = r0_data.compute_hash()
        r0 = RunReceipt.from_dict({**r0_data.to_dict(), "receipt_sha256": h0})

        with self.assertRaises(ReceiptStoreError):
            verify_receipt_chain([r0, r0])

    # 17. artefato válido é aceito
    def test_17_valid_artifact_accepted(self) -> None:
        raw_dir = self.sandbox / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        art = raw_dir / "data.json"
        art.write_bytes(b'{"key": "val"}')
        art_hash = compute_file_sha256(art)

        self.assertTrue(
            verify_artifact_integrity(
                base_dir=self.sandbox,
                artifact_inventory=["raw/data.json"],
                artifact_hashes={"raw/data.json": art_hash},
                allow_extra_files=True,
            )
        )

    # 18. hash divergente é rejeitado
    def test_18_divergent_hash_rejected(self) -> None:
        raw_dir = self.sandbox / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        art = raw_dir / "data.json"
        art.write_bytes(b'{"key": "val"}')

        with self.assertRaises(IntegrityError):
            verify_artifact_integrity(
                base_dir=self.sandbox,
                artifact_inventory=["raw/data.json"],
                artifact_hashes={"raw/data.json": "0" * 64},
                allow_extra_files=True,
            )

    # 19. arquivo ausente é rejeitado
    def test_19_missing_artifact_rejected(self) -> None:
        with self.assertRaises(IntegrityError):
            verify_artifact_integrity(
                base_dir=self.sandbox,
                artifact_inventory=["raw/missing.json"],
                artifact_hashes={"raw/missing.json": "0" * 64},
                allow_extra_files=True,
            )

    # 20. path traversal é rejeitado
    def test_20_path_traversal_rejected(self) -> None:
        with self.assertRaises(IntegrityError):
            verify_artifact_integrity(
                base_dir=self.sandbox,
                artifact_inventory=["../outside.json"],
                artifact_hashes={"../outside.json": "0" * 64},
                allow_extra_files=True,
            )

    # 21. verificação não modifica os arquivos
    def test_21_verification_does_not_modify_files(self) -> None:
        raw_dir = self.sandbox / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        art = raw_dir / "data.json"
        content = b'{"key": "val"}'
        art.write_bytes(content)
        art_hash = compute_file_sha256(art)

        verify_artifact_integrity(
            base_dir=self.sandbox,
            artifact_inventory=["raw/data.json"],
            artifact_hashes={"raw/data.json": art_hash},
            allow_extra_files=True,
        )

        self.assertEqual(art.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
