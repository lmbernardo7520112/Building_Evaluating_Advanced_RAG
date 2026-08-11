"""Focal Unit Test Suite — RunController Lifecycle Core (GREEN-2C.2B).

Tests path policy enforcement, lock acquisition, state transitions,
reconciliation after append exceptions, and context manager semantics.
Uses a dedicated cache directory as hermetic test infrastructure only.
"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from typing import Any
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


class TestRunControllerPreflight(unittest.TestCase):
    """Focal unit tests for read-only preflight validations (GREEN-2C.2C.1).

    Uses a temporary Git repository for hermetic isolation.
    """

    def setUp(self) -> None:
        import hashlib as _hashlib
        import shutil as _shutil

        self.repo_dir = (
            Path.home() / ".cache" / "raglab_test_git_sandbox"
        ).resolve()
        if self.repo_dir.exists():
            _shutil.rmtree(self.repo_dir)
        self.repo_dir.mkdir(parents=True, exist_ok=True)

        # Initialize hermetic git repo
        self._git("init")
        self._git("config", "user.name", "TestUser")
        self._git("config", "user.email", "test@example.com")

        # Create initial commit (implementation)
        (self.repo_dir / "src_file.py").write_text("# code", encoding="utf-8")
        self._git("add", "src_file.py")
        self._git("commit", "-m", "impl commit")
        self.impl_commit = self._git_output("rev-parse", "HEAD")

        # Create protocol file and commit (protocol)
        self.proto_file = self.repo_dir / "protocol.json"
        self.proto_content = '{"protocol_id":"proto_1","version":"1.0"}'
        self.proto_file.write_text(self.proto_content, encoding="utf-8")
        self.proto_sha = _hashlib.sha256(
            self.proto_content.encode("utf-8")
        ).hexdigest()
        self._git("add", "protocol.json")
        self._git("commit", "-m", "proto commit")
        self.proto_commit = self._git_output("rev-parse", "HEAD")

        # Create input file
        self.input_file = self.repo_dir / "inputs" / "qrels.json"
        self.input_file.parent.mkdir(parents=True, exist_ok=True)
        self.input_file.write_text('{"q1":"d1"}', encoding="utf-8")
        self._git("add", "inputs/qrels.json")
        self._git("commit", "-m", "add inputs")

        self.head_commit = self._git_output("rev-parse", "HEAD")
        self.main_branch = self._git_output("rev-parse", "--abbrev-ref", "HEAD")

    def tearDown(self) -> None:
        import shutil as _shutil

        if self.repo_dir.exists():
            _shutil.rmtree(self.repo_dir, ignore_errors=True)

    def _git(self, *args: str) -> None:
        import shutil as _shutil
        import subprocess as _sp

        git_bin = _shutil.which("git") or "git"
        _sp.run(  # noqa: S603, S607
            [git_bin, *args],
            cwd=str(self.repo_dir),
            check=True,
            capture_output=True,
        )

    def _git_output(self, *args: str) -> str:
        import shutil as _shutil
        import subprocess as _sp

        git_bin = _shutil.which("git") or "git"
        return (
            _sp.run(  # noqa: S603, S607
                [git_bin, *args],
                cwd=str(self.repo_dir),
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )

    def _make_controller(self) -> RunController:
        return RunController(repo_root=self.repo_dir)

    # ------------------------------------------------------------------
    # preflight_git_check
    # ------------------------------------------------------------------

    def test_20_default_no_args_clean_repo_accepted(self) -> None:
        controller = self._make_controller()
        # Default preflight_git_check() with no args on clean repo must pass
        controller.preflight_git_check()

    def test_21_default_no_args_tracked_dirty_rejected(self) -> None:
        (self.repo_dir / "src_file.py").write_text(
            "# modified", encoding="utf-8"
        )
        controller = self._make_controller()
        # Default preflight_git_check() with no args on tracked dirty repo must fail closed
        with self.assertRaises(RunControllerError) as ctx:
            controller.preflight_git_check()
        self.assertIn("dirty", str(ctx.exception).lower())

    def test_22_default_no_args_staged_rejected(self) -> None:
        (self.repo_dir / "new_file.py").write_text(
            "# new", encoding="utf-8"
        )
        self._git("add", "new_file.py")
        controller = self._make_controller()
        # Default preflight_git_check() with no args on non-empty staging must fail closed
        with self.assertRaises(RunControllerError) as ctx:
            controller.preflight_git_check()
        self.assertIn("staged", str(ctx.exception).lower())

    def test_23_explicit_allow_dirty_accepts_tracked_dirty(self) -> None:
        (self.repo_dir / "src_file.py").write_text(
            "# modified", encoding="utf-8"
        )
        controller = self._make_controller()
        # Explicit opt-in allow_dirty=True accepts tracked dirty
        controller.preflight_git_check(allow_dirty=True)

    def test_24_explicit_allow_staged_accepts_staged(self) -> None:
        (self.repo_dir / "new_file.py").write_text(
            "# new", encoding="utf-8"
        )
        self._git("add", "new_file.py")
        controller = self._make_controller()
        # Explicit opt-in allow_staged=True accepts non-empty staging
        controller.preflight_git_check(allow_staged=True)

    def test_24b_opt_in_flags_are_independent(self) -> None:
        # Create both tracked dirty and staged changes
        (self.repo_dir / "src_file.py").write_text(
            "# modified", encoding="utf-8"
        )
        (self.repo_dir / "staged_file.py").write_text(
            "# staged", encoding="utf-8"
        )
        self._git("add", "staged_file.py")
        controller = self._make_controller()

        # allow_dirty=True alone fails because staged is present and allow_staged=False by default
        with self.assertRaises(RunControllerError) as ctx_dirty:
            controller.preflight_git_check(allow_dirty=True)
        self.assertIn("staged", str(ctx_dirty.exception).lower())

        # allow_staged=True alone fails because dirty is present and allow_dirty=False by default
        with self.assertRaises(RunControllerError) as ctx_staged:
            controller.preflight_git_check(allow_staged=True)
        self.assertIn("dirty", str(ctx_staged.exception).lower())

        # Both explicit opt-ins required to pass when both conditions exist
        controller.preflight_git_check(allow_dirty=True, allow_staged=True)

    # ------------------------------------------------------------------
    # validate_commits
    # ------------------------------------------------------------------

    def test_25_valid_commits_accepted(self) -> None:
        controller = self._make_controller()
        controller.validate_commits(
            implementation_commit=self.impl_commit,
            protocol_commit=self.proto_commit,
        )

    def test_26_nonexistent_impl_commit_rejected(self) -> None:
        controller = self._make_controller()
        with self.assertRaises(RunControllerError) as ctx:
            controller.validate_commits(
                implementation_commit="0" * 40,
                protocol_commit=self.proto_commit,
            )
        self.assertIn("implementation_commit", str(ctx.exception))

    def test_27_nonexistent_proto_commit_rejected(self) -> None:
        controller = self._make_controller()
        with self.assertRaises(RunControllerError) as ctx:
            controller.validate_commits(
                implementation_commit=self.impl_commit,
                protocol_commit="0" * 40,
            )
        self.assertIn("protocol_commit", str(ctx.exception))

    # ------------------------------------------------------------------
    # validate_commit_ancestry
    # ------------------------------------------------------------------

    def test_28_valid_ancestry_accepted(self) -> None:
        controller = self._make_controller()
        controller.validate_commit_ancestry(
            impl_commit=self.impl_commit,
            proto_commit=self.proto_commit,
        )

    def test_29_invalid_ancestry_rejected(self) -> None:
        controller = self._make_controller()
        # proto_commit is older than head_commit, so head->proto is not valid
        with self.assertRaises(RunControllerError) as ctx:
            controller.validate_commit_ancestry(
                impl_commit=self.head_commit,
                proto_commit=self.impl_commit,
            )
        self.assertIn("not an ancestor", str(ctx.exception))

    # ------------------------------------------------------------------
    # validate_protocol_in_head_ancestry
    # ------------------------------------------------------------------

    def test_30_protocol_in_head_ancestry_accepted(self) -> None:
        controller = self._make_controller()
        controller.validate_protocol_in_head_ancestry(self.proto_commit)

    def test_31_protocol_not_in_head_ancestry_rejected(self) -> None:
        # Create an orphan branch commit not in HEAD ancestry
        self._git("checkout", "--orphan", "orphan_branch")
        (self.repo_dir / "orphan.txt").write_text("orphan", encoding="utf-8")
        self._git("add", "orphan.txt")
        self._git("commit", "-m", "orphan commit")
        orphan_commit = self._git_output("rev-parse", "HEAD")
        # Go back to main branch
        self._git("checkout", self.main_branch)

        controller = self._make_controller()
        with self.assertRaises(RunControllerError) as ctx:
            controller.validate_protocol_in_head_ancestry(orphan_commit)
        self.assertIn("not in HEAD ancestry", str(ctx.exception))

    # ------------------------------------------------------------------
    # validate_protocol_tracked
    # ------------------------------------------------------------------

    def test_32_tracked_protocol_accepted(self) -> None:
        controller = self._make_controller()
        controller.validate_protocol_tracked(self.proto_file)

    def test_33_untracked_protocol_rejected(self) -> None:
        untracked = self.repo_dir / "untracked_proto.json"
        untracked.write_text("{}", encoding="utf-8")
        controller = self._make_controller()
        with self.assertRaises(RunControllerError) as ctx:
            controller.validate_protocol_tracked(untracked)
        self.assertIn("not tracked", str(ctx.exception))

    def test_34_protocol_outside_repo_rejected(self) -> None:
        outside_dir = self.repo_dir.parent / "outside_dir"
        outside_dir.mkdir(parents=True, exist_ok=True)
        outside_path = outside_dir / "outside_proto.json"
        outside_path.write_text("{}", encoding="utf-8")
        try:
            controller = self._make_controller()
            with self.assertRaises(RunControllerError) as ctx:
                controller.validate_protocol_tracked(outside_path)
            self.assertIn("outside the repository", str(ctx.exception))
        finally:
            import shutil as _shutil
            _shutil.rmtree(outside_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # validate_protocol_unmodified
    # ------------------------------------------------------------------

    def test_35_unmodified_protocol_accepted(self) -> None:
        controller = self._make_controller()
        controller.validate_protocol_unmodified("protocol.json")

    def test_36_worktree_modified_protocol_rejected(self) -> None:
        self.proto_file.write_text('{"modified":true}', encoding="utf-8")
        controller = self._make_controller()
        with self.assertRaises(RunControllerError) as ctx:
            controller.validate_protocol_unmodified("protocol.json")
        self.assertIn("unstaged modifications", str(ctx.exception))

    def test_37_staged_modified_protocol_rejected(self) -> None:
        self.proto_file.write_text('{"staged":true}', encoding="utf-8")
        self._git("add", "protocol.json")
        controller = self._make_controller()
        with self.assertRaises(RunControllerError) as ctx:
            controller.validate_protocol_unmodified("protocol.json")
        self.assertIn("staged modifications", str(ctx.exception))

    # ------------------------------------------------------------------
    # validate_protocol_sha
    # ------------------------------------------------------------------

    def test_38_correct_sha_accepted(self) -> None:
        controller = self._make_controller()
        controller.validate_protocol_sha(
            protocol_path="protocol.json",
            expected_sha=self.proto_sha,
        )

    def test_39_incorrect_sha_rejected(self) -> None:
        controller = self._make_controller()
        with self.assertRaises(RunControllerError) as ctx:
            controller.validate_protocol_sha(
                protocol_path="protocol.json",
                expected_sha="0" * 64,
            )
        self.assertIn("mismatch", str(ctx.exception).lower())

    # ------------------------------------------------------------------
    # validate_inputs
    # ------------------------------------------------------------------

    def test_40_valid_inputs_accepted(self) -> None:
        controller = self._make_controller()
        controller.validate_inputs({"qrels": str(self.input_file)})

    def test_41_missing_input_rejected(self) -> None:
        controller = self._make_controller()
        with self.assertRaises(RunControllerError) as ctx:
            controller.validate_inputs(
                {"missing": "/path/does/not/exist.json"}
            )
        self.assertIn("not found", str(ctx.exception))

    # ------------------------------------------------------------------
    # Zero mutation after preflight
    # ------------------------------------------------------------------

    def test_42_preflight_does_not_mutate_git(self) -> None:
        head_before = self._git_output("rev-parse", "HEAD")
        status_before = self._git_output("status", "--porcelain")

        controller = self._make_controller()
        controller.preflight_git_check(allow_dirty=True, allow_staged=True)
        controller.validate_commits(
            implementation_commit=self.impl_commit,
            protocol_commit=self.proto_commit,
        )
        controller.validate_commit_ancestry(
            impl_commit=self.impl_commit,
            proto_commit=self.proto_commit,
        )
        controller.validate_protocol_in_head_ancestry(self.proto_commit)
        controller.validate_protocol_tracked(self.proto_file)
        controller.validate_protocol_unmodified("protocol.json")
        controller.validate_protocol_sha(
            protocol_path="protocol.json",
            expected_sha=self.proto_sha,
        )
        controller.validate_inputs({"qrels": str(self.input_file)})

        head_after = self._git_output("rev-parse", "HEAD")
        status_after = self._git_output("status", "--porcelain")

        self.assertEqual(head_before, head_after, "HEAD must not change")
        self.assertEqual(
            status_before, status_after, "Git status must not change"
        )

        self.assertEqual(head_before, head_after, "HEAD must not change")
        self.assertEqual(
            status_before, status_after, "Git status must not change"
        )

    # ------------------------------------------------------------------
    # prepare_run remediation focal tests (GREEN-2C.2C.2)
    # ------------------------------------------------------------------

    def test_43_prepare_run_allowed_roots_none_fails_without_io(self) -> None:
        controller = RunController(allowed_roots=None, repo_root=self.repo_dir)
        artifact_root = self.repo_dir / "allowed"
        with self.assertRaises(RunControllerError) as ctx:
            controller.prepare_run(
                run_id="run_none",
                slice_id="slice_5b",
                protocol_path=self.proto_file,
                artifact_root=artifact_root,
            )
        self.assertIn("allowed_roots", str(ctx.exception))
        self.assertFalse((artifact_root / "slice_5b" / "run_none").exists())

    def test_44_prepare_run_allowed_roots_empty_fails_without_io(self) -> None:
        controller = RunController(allowed_roots=[], repo_root=self.repo_dir)
        artifact_root = self.repo_dir / "allowed"
        with self.assertRaises(RunControllerError) as ctx:
            controller.prepare_run(
                run_id="run_empty_roots",
                slice_id="slice_5b",
                protocol_path=self.proto_file,
                artifact_root=artifact_root,
            )
        self.assertIn("allowed_roots", str(ctx.exception))
        self.assertFalse(
            (artifact_root / "slice_5b" / "run_empty_roots").exists()
        )

    def test_45_prepare_run_artifact_root_outside_allowlist_fails_without_io(
        self,
    ) -> None:
        allowed_dir = self.repo_dir / "allowed"
        outside_dir = self.repo_dir / "outside"
        controller = RunController(
            allowed_roots=[allowed_dir], repo_root=self.repo_dir
        )
        with self.assertRaises(RunControllerError) as ctx:
            controller.prepare_run(
                run_id="run_outside",
                slice_id="slice_5b",
                protocol_path=self.proto_file,
                artifact_root=outside_dir,
            )
        self.assertIn("Path policy validation failed", str(ctx.exception))
        self.assertFalse((outside_dir / "slice_5b" / "run_outside").exists())

    def test_46_prepare_run_artifact_root_inside_allowlist_succeeds(
        self,
    ) -> None:
        allowed_dir = self.repo_dir / "allowed"
        allowed_dir.mkdir(parents=True, exist_ok=True)
        controller = RunController(
            allowed_roots=[allowed_dir], repo_root=self.repo_dir
        )
        receipt = controller.prepare_run(
            run_id="run_inside",
            slice_id="slice_5b",
            protocol_path=self.proto_file,
            artifact_root=allowed_dir,
        )
        self.assertEqual(receipt.state, RunState.PREPARED)

    def test_47_prepare_run_preexisting_empty_run_dir_rejected(self) -> None:
        allowed_dir = self.repo_dir / "allowed"
        run_dir = allowed_dir / "slice_5b" / "run_pre_empty"
        run_dir.mkdir(parents=True, exist_ok=True)

        controller = RunController(
            allowed_roots=[allowed_dir], repo_root=self.repo_dir
        )
        with self.assertRaises(RunControllerError) as ctx:
            controller.prepare_run(
                run_id="run_pre_empty",
                slice_id="slice_5b",
                protocol_path=self.proto_file,
                artifact_root=allowed_dir,
            )
        self.assertIn("already exists", str(ctx.exception))
        self.assertEqual(list(run_dir.iterdir()), [])

    def test_48_prepare_run_preexisting_non_empty_dir_rejected_and_preserved(
        self,
    ) -> None:
        allowed_dir = self.repo_dir / "allowed"
        run_dir = allowed_dir / "slice_5b" / "run_pre_nonempty"
        run_dir.mkdir(parents=True, exist_ok=True)
        existing_file = run_dir / "existing.txt"
        existing_file.write_text("preexisting content", encoding="utf-8")
        content_before = existing_file.read_bytes()

        controller = RunController(
            allowed_roots=[allowed_dir], repo_root=self.repo_dir
        )
        with self.assertRaises(RunControllerError) as ctx:
            controller.prepare_run(
                run_id="run_pre_nonempty",
                slice_id="slice_5b",
                protocol_path=self.proto_file,
                artifact_root=allowed_dir,
            )
        self.assertTrue(
            "already exists" in str(ctx.exception)
            or "not empty" in str(ctx.exception)
        )
        self.assertEqual(existing_file.read_bytes(), content_before)
        self.assertEqual(set(run_dir.iterdir()), {existing_file})

    def test_49_prepare_run_confirmed_success_append_releases_lock(
        self,
    ) -> None:
        allowed_dir = self.repo_dir / "allowed"
        allowed_dir.mkdir(parents=True, exist_ok=True)
        controller = RunController(
            allowed_roots=[allowed_dir], repo_root=self.repo_dir
        )
        receipt = controller.prepare_run(
            run_id="run_succ",
            slice_id="slice_5b",
            protocol_path=self.proto_file,
            artifact_root=allowed_dir,
        )
        run_dir = Path(receipt.run_directory)
        self.assertNotIn(run_dir, controller._active_locks)

    def test_50_prepare_run_append_failure_before_publication_releases_lock(
        self,
    ) -> None:
        from unittest.mock import patch

        allowed_dir = self.repo_dir / "allowed"
        allowed_dir.mkdir(parents=True, exist_ok=True)
        controller = RunController(
            allowed_roots=[allowed_dir], repo_root=self.repo_dir
        )

        with patch.object(
            RunReceiptStore, "append", side_effect=OSError("Disk write failed")
        ):
            with self.assertRaises(RunControllerError) as ctx:
                controller.prepare_run(
                    run_id="run_fail_pub",
                    slice_id="slice_5b",
                    protocol_path=self.proto_file,
                    artifact_root=allowed_dir,
                )
            self.assertIn("Disk write failed", str(ctx.exception))

        run_dir = (allowed_dir / "slice_5b" / "run_fail_pub").resolve()
        self.assertNotIn(run_dir, controller._active_locks)

    def test_51_prepare_run_append_publishes_then_throws_re_read_succeeds(
        self,
    ) -> None:
        from unittest.mock import patch

        allowed_dir = self.repo_dir / "allowed"
        allowed_dir.mkdir(parents=True, exist_ok=True)
        controller = RunController(
            allowed_roots=[allowed_dir], repo_root=self.repo_dir
        )

        orig_append = RunReceiptStore.append

        def _append_and_throw(receipt: Any) -> None:
            store_self = RunReceiptStore(Path(receipt.run_directory))
            orig_append(store_self, receipt)
            raise OSError("Post-write notification failed")

        with patch.object(
            RunReceiptStore, "append", side_effect=_append_and_throw
        ):
            receipt = controller.prepare_run(
                run_id="run_pub_throw",
                slice_id="slice_5b",
                protocol_path=self.proto_file,
                artifact_root=allowed_dir,
            )

        self.assertEqual(receipt.state, RunState.PREPARED)
        run_dir = Path(receipt.run_directory)
        self.assertNotIn(run_dir, controller._active_locks)

    def test_52_prepare_run_unreadable_reread_preserves_lock(self) -> None:
        from unittest.mock import patch

        allowed_dir = self.repo_dir / "allowed"
        allowed_dir.mkdir(parents=True, exist_ok=True)
        controller = RunController(
            allowed_roots=[allowed_dir], repo_root=self.repo_dir
        )

        def _append_corrupted(receipt: Any) -> None:
            rec_dir = Path(receipt.run_directory) / "receipts"
            rec_dir.mkdir(parents=True, exist_ok=True)
            rec_file = rec_dir / "000_PREPARED.json"
            rec_file.write_text("{corrupted json", encoding="utf-8")
            raise OSError("Write corrupted file")

        with patch.object(
            RunReceiptStore, "append", side_effect=_append_corrupted
        ):
            with self.assertRaises(RunControllerError) as ctx:
                controller.prepare_run(
                    run_id="run_corrupt",
                    slice_id="slice_5b",
                    protocol_path=self.proto_file,
                    artifact_root=allowed_dir,
                )
            self.assertIn("PREPARE_RUN_STATE_UNCERTAIN", str(ctx.exception))

        run_dir = (allowed_dir / "slice_5b" / "run_corrupt").resolve()
        self.assertIn(run_dir, controller._active_locks)
        self.assertTrue(controller._active_locks[run_dir].is_acquired())

    def test_53_prepare_run_divergent_chain_preserves_lock(self) -> None:
        from unittest.mock import patch

        allowed_dir = self.repo_dir / "allowed"
        allowed_dir.mkdir(parents=True, exist_ok=True)
        controller = RunController(
            allowed_roots=[allowed_dir], repo_root=self.repo_dir
        )

        orig_append = RunReceiptStore.append

        def _append_divergent(receipt: Any) -> None:
            store_self = RunReceiptStore(Path(receipt.run_directory))
            div_receipt = RunReceipt(
                schema_version=1,
                run_id="different_id",
                slice_id="slice_5b",
                state=RunState.PREPARED,
                artifact_root=receipt.artifact_root,
                run_directory=receipt.run_directory,
                implementation_commit=receipt.implementation_commit,
                protocol_commit=receipt.protocol_commit,
                protocol_sha256=receipt.protocol_sha256,
                input_hashes={},
                runner_version=receipt.runner_version,
                created_at_utc=receipt.created_at_utc,
                receipt_sha256="",
            )
            h = div_receipt.compute_hash()
            div_receipt_final = RunReceipt(
                schema_version=div_receipt.schema_version,
                run_id=div_receipt.run_id,
                slice_id=div_receipt.slice_id,
                state=div_receipt.state,
                artifact_root=div_receipt.artifact_root,
                run_directory=div_receipt.run_directory,
                implementation_commit=div_receipt.implementation_commit,
                protocol_commit=div_receipt.protocol_commit,
                protocol_sha256=div_receipt.protocol_sha256,
                input_hashes=div_receipt.input_hashes,
                runner_version=div_receipt.runner_version,
                created_at_utc=div_receipt.created_at_utc,
                receipt_sha256=h,
            )
            orig_append(store_self, div_receipt_final)

        with patch.object(
            RunReceiptStore, "append", side_effect=_append_divergent
        ):
            with self.assertRaises(RunControllerError) as ctx:
                controller.prepare_run(
                    run_id="run_div",
                    slice_id="slice_5b",
                    protocol_path=self.proto_file,
                    artifact_root=allowed_dir,
                )
            self.assertIn("PREPARE_RUN_STATE_UNCERTAIN", str(ctx.exception))

        run_dir = (allowed_dir / "slice_5b" / "run_div").resolve()
        self.assertIn(run_dir, controller._active_locks)

    def test_54_prepare_run_divergent_hash_preserves_lock(self) -> None:
        import json
        from unittest.mock import patch

        allowed_dir = self.repo_dir / "allowed"
        allowed_dir.mkdir(parents=True, exist_ok=True)
        controller = RunController(
            allowed_roots=[allowed_dir], repo_root=self.repo_dir
        )

        def _append_hash_mismatch(receipt: Any) -> None:
            rec_dir = Path(receipt.run_directory) / "receipts"
            rec_dir.mkdir(parents=True, exist_ok=True)
            fake_receipt = RunReceipt(
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
                receipt_sha256="0" * 64,
            )
            rec_file = rec_dir / "000_PREPARED.json"
            rec_file.write_text(
                json.dumps(fake_receipt.to_dict()), encoding="utf-8"
            )
            raise OSError("Write hash mismatch")

        with patch.object(
            RunReceiptStore, "append", side_effect=_append_hash_mismatch
        ):
            with self.assertRaises(RunControllerError) as ctx:
                controller.prepare_run(
                    run_id="run_hash_mismatch",
                    slice_id="slice_5b",
                    protocol_path=self.proto_file,
                    artifact_root=allowed_dir,
                )
            self.assertIn("PREPARE_RUN_STATE_UNCERTAIN", str(ctx.exception))

        run_dir = (allowed_dir / "slice_5b" / "run_hash_mismatch").resolve()
        self.assertIn(run_dir, controller._active_locks)


if __name__ == "__main__":
    unittest.main()
