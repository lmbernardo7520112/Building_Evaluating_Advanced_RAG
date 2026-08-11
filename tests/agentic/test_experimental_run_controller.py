"""Focal Unit Test Suite — RunController Lifecycle Core (GREEN-2C.2B).

Tests path policy enforcement, lock acquisition, state transitions,
reconciliation after append exceptions, and context manager semantics.
Uses a dedicated cache directory as hermetic test infrastructure only.
"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from raglab.agentic.experiments import (
    ExperimentalRunLock,
    ReceiptStoreError,
    RunController,
    RunControllerError,
    RunReceipt,
    RunReceiptStore,
    RunState,
)


def _make_prepared_receipt(
    run_dir: Path,
    run_id: str = "run_001",
    slice_id: str = "slice_5b",
) -> RunReceipt:
    """Build a minimal valid PREPARED RunReceipt with computed hash."""
    draft = RunReceipt(
        schema_version=1,
        run_id=run_id,
        slice_id=slice_id,
        state=RunState.PREPARED,
        artifact_root=str(run_dir.parent.parent),
        run_directory=str(run_dir),
        implementation_commit="a" * 64,
        protocol_commit="b" * 64,
        protocol_sha256="c" * 64,
        input_hashes={"query": "d" * 64},
        runner_version="raglab-v7-test",
        created_at_utc="2026-08-10T00:00:00Z",
        previous_receipt_sha256=None,
        receipt_sha256="",
    )
    computed_hash = draft.compute_hash()
    return RunReceipt(
        schema_version=draft.schema_version,
        run_id=draft.run_id,
        slice_id=draft.slice_id,
        state=draft.state,
        artifact_root=draft.artifact_root,
        run_directory=draft.run_directory,
        implementation_commit=draft.implementation_commit,
        protocol_commit=draft.protocol_commit,
        protocol_sha256=draft.protocol_sha256,
        input_hashes=dict(draft.input_hashes),
        runner_version=draft.runner_version,
        created_at_utc=draft.created_at_utc,
        previous_receipt_sha256=draft.previous_receipt_sha256,
        receipt_sha256=computed_hash,
    )


class TestRunControllerFocal(unittest.TestCase):
    """Focal unit test suite for RunController lifecycle core."""

    def setUp(self) -> None:
        # Use ~/.cache/raglab_test_sandbox for path policy compliance
        self.sandbox = (
            Path.home() / ".cache" / "raglab_test_sandbox"
        ).resolve()
        if self.sandbox.exists():
            shutil.rmtree(self.sandbox)
        self.sandbox.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.sandbox.exists():
            shutil.rmtree(self.sandbox, ignore_errors=True)

    # 1. Allowlist ausente (None) rejeitada antes de I/O
    def test_01_missing_allowlist_rejected(self) -> None:
        controller = RunController(allowed_roots=None)
        with self.assertRaises(RunControllerError) as ctx:
            controller.acquire_run_lock(self.sandbox / "run1")
        self.assertIn("allowed_roots", str(ctx.exception))

    # 2. Allowlist vazia ([]) rejeitada
    def test_02_empty_allowlist_rejected(self) -> None:
        controller = RunController(allowed_roots=[])
        with self.assertRaises(RunControllerError) as ctx:
            controller.acquire_run_lock(self.sandbox / "run2")
        self.assertIn("allowed_roots", str(ctx.exception))

    # 3. Path não autorizado rejeitado
    def test_03_unauthorized_path_rejected(self) -> None:
        allowed_root = self.sandbox / "srv"
        allowed_root.mkdir(parents=True)
        controller = RunController(allowed_roots=[allowed_root])
        unauthorized = self.sandbox / "other" / "run3"
        with self.assertRaises(RunControllerError) as ctx:
            controller.acquire_run_lock(unauthorized)
        self.assertIn("Path policy validation failed", str(ctx.exception))

    # 4. Lock adquirido antes do append
    def test_04_lock_acquired_before_append(self) -> None:
        run_dir = self.sandbox / "slice_5b" / "run4"
        store = RunReceiptStore(run_dir)
        store.append(_make_prepared_receipt(run_dir))

        controller = RunController(allowed_roots=[self.sandbox])
        lock = controller.acquire_run_lock(run_dir)
        self.assertTrue(lock.is_acquired())
        self.assertTrue(lock.lock_path.exists())

        receipt = controller.start_run(run_dir)
        self.assertEqual(receipt.state, RunState.RUN_STARTED)

    # 5. Segundo controller tentando lock no mesmo run directory rejeitado
    def test_05_second_controller_lock_rejected(self) -> None:
        run_dir = self.sandbox / "slice_5b" / "run5"
        controller1 = RunController(allowed_roots=[self.sandbox])
        _lock1 = controller1.acquire_run_lock(run_dir)

        controller2 = RunController(allowed_roots=[self.sandbox])
        with self.assertRaises(RunControllerError) as ctx:
            controller2.acquire_run_lock(run_dir)
        self.assertIn("already exists", str(ctx.exception))

    # 6. Transição válida persistida
    def test_06_valid_transition_persisted(self) -> None:
        run_dir = self.sandbox / "slice_5b" / "run6"
        store = RunReceiptStore(run_dir)
        r0 = _make_prepared_receipt(run_dir)
        store.append(r0)

        controller = RunController(allowed_roots=[self.sandbox])
        r1 = controller.transition_state(run_dir, RunState.RUN_STARTED)
        self.assertEqual(r1.state, RunState.RUN_STARTED)
        self.assertEqual(r1.previous_receipt_sha256, r0.receipt_sha256)

        chain = store.load_history_chain()
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain[-1].state, RunState.RUN_STARTED)

    # 7. Transição inválida sem escrita
    def test_07_invalid_transition_no_write(self) -> None:
        run_dir = self.sandbox / "slice_5b" / "run7"
        store = RunReceiptStore(run_dir)
        r0 = _make_prepared_receipt(run_dir)
        store.append(r0)

        controller = RunController(allowed_roots=[self.sandbox])
        with self.assertRaises(RunControllerError):
            controller.transition_state(run_dir, RunState.AUDIT_COMPLETED)

        chain = store.load_history_chain()
        self.assertEqual(len(chain), 1)

    # 8. start_run aplica PREPARED -> RUN_STARTED
    def test_08_start_run(self) -> None:
        run_dir = self.sandbox / "slice_5b" / "run8"
        store = RunReceiptStore(run_dir)
        r0 = _make_prepared_receipt(run_dir)
        store.append(r0)

        controller = RunController(allowed_roots=[self.sandbox])
        receipt = controller.start_run(run_dir)
        self.assertEqual(receipt.state, RunState.RUN_STARTED)

    # 9. Append publica e depois lança: releitura reconhece commit
    def test_09_append_publishes_then_raises_reconciled(self) -> None:
        run_dir = self.sandbox / "slice_5b" / "run9"
        store = RunReceiptStore(run_dir)
        r0 = _make_prepared_receipt(run_dir)
        store.append(r0)

        controller = RunController(allowed_roots=[self.sandbox])

        original_append = RunReceiptStore.append

        def append_then_raise(
            self_store: RunReceiptStore, receipt: RunReceipt
        ) -> Path:
            original_append(self_store, receipt)
            raise RuntimeError("Post-write notification failure")

        with patch.object(
            RunReceiptStore, "append", autospec=True, side_effect=append_then_raise
        ):
            receipt = controller.start_run(run_dir)
            self.assertEqual(receipt.state, RunState.RUN_STARTED)

    # 10. Append falha antes da publicação: erro propagado
    def test_10_append_fails_before_write_error_propagated(self) -> None:
        run_dir = self.sandbox / "slice_5b" / "run10"
        store = RunReceiptStore(run_dir)
        r0 = _make_prepared_receipt(run_dir)
        store.append(r0)

        controller = RunController(allowed_roots=[self.sandbox])

        with (
            patch.object(
                RunReceiptStore,
                "append",
                side_effect=OSError("Disk write error"),
            ),
            self.assertRaises(RunControllerError),
        ):
            controller.start_run(run_dir)

        chain = store.load_history_chain()
        self.assertEqual(len(chain), 1)

    # 11. Cadeia divergente durante reconciliação: lock preservado
    def test_11_divergent_chain_preserves_lock(self) -> None:
        run_dir = self.sandbox / "slice_5b" / "run11"
        store = RunReceiptStore(run_dir)
        r0 = _make_prepared_receipt(run_dir)
        store.append(r0)

        controller = RunController(allowed_roots=[self.sandbox])
        lock = controller.acquire_run_lock(run_dir)

        with (
            patch.object(
                RunReceiptStore,
                "load_history_chain",
                side_effect=ReceiptStoreError("Corrupted chain"),
            ),
            patch.object(
                RunReceiptStore,
                "append",
                side_effect=OSError("Append failed"),
            ),
            self.assertRaises(RunControllerError),
        ):
            controller.transition_state(
                run_dir, RunState.RUN_STARTED, lock=lock
            )

        self.assertTrue(lock.is_acquired())

    # 12. Context manager limpo libera lock
    def test_12_clean_context_manager_releases_lock(self) -> None:
        run_dir = self.sandbox / "slice_5b" / "run12"
        controller = RunController(allowed_roots=[self.sandbox])
        with controller.run_context(run_dir) as lock:
            self.assertTrue(lock.is_acquired())
        self.assertFalse(lock.is_acquired())

    # 13. Exceção no context manager preserva lock
    def test_13_context_manager_exception_preserves_lock(self) -> None:
        run_dir = self.sandbox / "slice_5b" / "run13"
        controller = RunController(allowed_roots=[self.sandbox])
        captured_lock: ExperimentalRunLock | None = None
        try:
            with controller.run_context(run_dir) as lock:
                captured_lock = lock
                raise ValueError("Error inside run_context")
        except ValueError:
            pass

        self.assertIsNotNone(captured_lock)
        assert captured_lock is not None
        self.assertTrue(
            captured_lock.is_acquired(),
            "Lock must remain acquired on exception",
        )

    # ------------------------------------------------------------------
    # Lock ↔ run_dir binding (GREEN-2C.2B remediation)
    # ------------------------------------------------------------------

    # 14. Lock de outro diretório rejeitado
    def test_14_cross_directory_lock_rejected(self) -> None:
        run_dir_a = self.sandbox / "slice_5b" / "run14a"
        run_dir_b = self.sandbox / "slice_5b" / "run14b"
        store_a = RunReceiptStore(run_dir_a)
        store_a.append(_make_prepared_receipt(run_dir_a, run_id="run_14a"))
        store_b = RunReceiptStore(run_dir_b)
        store_b.append(_make_prepared_receipt(run_dir_b, run_id="run_14b"))

        lock_a = ExperimentalRunLock.acquire(run_dir_a, run_id="run_14a")
        controller = RunController(allowed_roots=[self.sandbox])

        with self.assertRaises(RunControllerError) as ctx:
            controller.transition_state(
                run_dir_b, RunState.RUN_STARTED, lock=lock_a
            )
        self.assertIn("Supplied lock belongs to", str(ctx.exception))

    # 15. Cadeia alvo inalterada após rejeição de lock estrangeiro
    def test_15_target_chain_preserved_on_rejection(self) -> None:
        run_dir_a = self.sandbox / "slice_5b" / "run15a"
        run_dir_b = self.sandbox / "slice_5b" / "run15b"
        store_a = RunReceiptStore(run_dir_a)
        store_a.append(_make_prepared_receipt(run_dir_a, run_id="run_15a"))
        store_b = RunReceiptStore(run_dir_b)
        r0_b = _make_prepared_receipt(run_dir_b, run_id="run_15b")
        store_b.append(r0_b)

        lock_a = ExperimentalRunLock.acquire(run_dir_a, run_id="run_15a")
        controller = RunController(allowed_roots=[self.sandbox])

        with self.assertRaises(RunControllerError):
            controller.transition_state(
                run_dir_b, RunState.RUN_STARTED, lock=lock_a
            )

        chain_b = store_b.load_history_chain()
        self.assertEqual(len(chain_b), 1)
        self.assertEqual(chain_b[0].receipt_sha256, r0_b.receipt_sha256)

    # 16. Lock alheio permanece adquirido após rejeição
    def test_16_foreign_lock_remains_acquired(self) -> None:
        run_dir_a = self.sandbox / "slice_5b" / "run16a"
        run_dir_b = self.sandbox / "slice_5b" / "run16b"
        store_a = RunReceiptStore(run_dir_a)
        store_a.append(_make_prepared_receipt(run_dir_a, run_id="run_16a"))
        store_b = RunReceiptStore(run_dir_b)
        store_b.append(_make_prepared_receipt(run_dir_b, run_id="run_16b"))

        lock_a = ExperimentalRunLock.acquire(run_dir_a, run_id="run_16a")
        controller = RunController(allowed_roots=[self.sandbox])

        with self.assertRaises(RunControllerError):
            controller.transition_state(
                run_dir_b, RunState.RUN_STARTED, lock=lock_a
            )

        self.assertTrue(
            lock_a.is_acquired(),
            "Foreign lock must remain acquired after rejection",
        )

    # 17. Mesmo run_id em diretórios diferentes rejeitado
    def test_17_same_run_id_different_dirs_rejected(self) -> None:
        run_dir_a = self.sandbox / "slice_5b" / "run17a"
        run_dir_b = self.sandbox / "slice_5b" / "run17b"
        store_a = RunReceiptStore(run_dir_a)
        store_a.append(_make_prepared_receipt(run_dir_a, run_id="shared_id"))
        store_b = RunReceiptStore(run_dir_b)
        store_b.append(_make_prepared_receipt(run_dir_b, run_id="shared_id"))

        lock_a = ExperimentalRunLock.acquire(run_dir_a, run_id="shared_id")
        controller = RunController(allowed_roots=[self.sandbox])

        with self.assertRaises(RunControllerError) as ctx:
            controller.transition_state(
                run_dir_b, RunState.RUN_STARTED, lock=lock_a
            )
        # Must be rejected by directory binding, not run_id
        self.assertIn("Supplied lock belongs to", str(ctx.exception))

    # 18. Lock correto aceito
    def test_18_correct_target_lock_accepted(self) -> None:
        run_dir = self.sandbox / "slice_5b" / "run18"
        store = RunReceiptStore(run_dir)
        r0 = _make_prepared_receipt(run_dir, run_id="run_18")
        store.append(r0)

        lock = ExperimentalRunLock.acquire(run_dir, run_id="run_18")
        controller = RunController(allowed_roots=[self.sandbox])

        receipt = controller.transition_state(
            run_dir, RunState.RUN_STARTED, lock=lock
        )
        self.assertEqual(receipt.state, RunState.RUN_STARTED)
        self.assertEqual(receipt.previous_receipt_sha256, r0.receipt_sha256)

    # 19. start_run herda a proteção de binding
    def test_19_start_run_inherits_lock_binding(self) -> None:
        run_dir_a = self.sandbox / "slice_5b" / "run19a"
        run_dir_b = self.sandbox / "slice_5b" / "run19b"
        store_a = RunReceiptStore(run_dir_a)
        store_a.append(_make_prepared_receipt(run_dir_a, run_id="run_19a"))
        store_b = RunReceiptStore(run_dir_b)
        store_b.append(_make_prepared_receipt(run_dir_b, run_id="run_19b"))

        lock_a = ExperimentalRunLock.acquire(run_dir_a, run_id="run_19a")
        controller = RunController(allowed_roots=[self.sandbox])

        with self.assertRaises(RunControllerError) as ctx:
            controller.start_run(run_dir_b, lock=lock_a)
        self.assertIn("Supplied lock belongs to", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
