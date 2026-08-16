"""Hermetic Acceptance Test Suite — RAGLab V7 / Experimental Readiness Contract V1.

Phase: RED EXCLUSIVELY
Imports target modules inside test functions to ensure pytest collects all test cases
individually, failing each test with an explicit ImportError or AssertionError at execution time.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestExperimentalReadinessAcceptance(unittest.TestCase):
    """Frozen acceptance test suite defining required behavior for Experimental Readiness V1."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sandbox = Path(self.temp_dir.name).resolve()

        # Contract JSON path check
        self.contract_json_path = (
            Path(__file__).parent / "readiness_contract_v1.json"
        )
        self.assertTrue(
            self.contract_json_path.exists(),
            "Contract specification JSON readiness_contract_v1.json must exist.",
        )
        self.contract_spec = json.loads(
            self.contract_json_path.read_text(encoding="utf-8")
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _import_target_module(self) -> None:
        """Helper importing target module inside tests to enable individual collection."""
        from raglab.agentic.experiments import (  # noqa: F401
            ExperimentalPathPolicy,
            LineageAuditResult,
            LineageVerifier,
            PathPolicyError,
            ReceiptStoreError,
            RunController,
            RunControllerError,
            RunReceipt,
            RunReceiptStore,
            RunState,
            compute_canonical_json_sha256,
            compute_file_sha256,
        )

    def _create_synthetic_git_repo(self, repo_dir: Path) -> tuple[str, str]:
        """Create synthetic git repository inside sandbox returning (impl_commit, proto_commit)."""
        import shutil

        git_bin = shutil.which("git") or "git"
        repo_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [git_bin, "init"], cwd=repo_dir, check=True, capture_output=True
        )
        subprocess.run(
            [git_bin, "config", "user.name", "TestUser"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [git_bin, "config", "user.email", "test@example.com"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )

        (repo_dir / "src_file.py").write_text("# code", encoding="utf-8")
        subprocess.run(
            [git_bin, "add", "src_file.py"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [git_bin, "commit", "-m", "impl commit"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )
        impl_commit = (
            subprocess.run(
                [git_bin, "rev-parse", "HEAD"],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )

        proto_file = repo_dir / "protocol.json"
        proto_data = {"protocol_id": "proto_1", "version": "1.0"}
        proto_file.write_text(json.dumps(proto_data), encoding="utf-8")
        subprocess.run(
            [git_bin, "add", "protocol.json"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [git_bin, "commit", "-m", "proto commit"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )
        proto_commit = (
            subprocess.run(
                [git_bin, "rev-parse", "HEAD"],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )

        return impl_commit, proto_commit

    def _make_allowed_artifact_root(self, prefix: str = "raglab_acc_") -> Path:
        """Create an isolated, exclusive artifact root under ~/.cache and register cleanup."""
        cache_base = Path.home() / ".cache"
        cache_base.mkdir(parents=True, exist_ok=True)
        temp_dir = tempfile.TemporaryDirectory(prefix=prefix, dir=cache_base)
        self.addCleanup(temp_dir.cleanup)
        return Path(temp_dir.name).resolve()

    def _make_hermetic_run_environment(
        self, test_name: str
    ) -> tuple[object, Path, Path, Path]:
        """Create synthetic git repo, valid protocol, artifact_root under ~/.cache, and RunController."""
        from raglab.agentic.experiments import RunController

        repo_dir = self.sandbox / f"repo_{test_name}"
        self._create_synthetic_git_repo(repo_dir)
        protocol_path = repo_dir / "protocol.json"
        artifact_root = self._make_allowed_artifact_root(
            prefix=f"raglab_acc_{test_name}_"
        )
        controller = RunController(
            allowed_roots=[artifact_root],
            repo_root=repo_dir,
        )
        return controller, repo_dir, protocol_path, artifact_root

    def _create_valid_run_directory(
        self,
        artifact_root: Path,
        slice_id: str = "slice5b",
        run_id: str = "valid_run",
    ) -> Path:
        """Create a synthetic valid run directory layout at <artifact_root>/<slice_id>/<run_id>/."""
        run_dir = artifact_root / slice_id / run_id
        receipts_dir = run_dir / "receipts"
        raw_dir = run_dir / "raw"
        derived_dir = run_dir / "derived"
        logs_dir = run_dir / "logs"

        receipts_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        derived_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "protocol.snapshot.json").write_text(
            json.dumps({"protocol_id": "proto_1"}), encoding="utf-8"
        )

        dummy_artifact = raw_dir / "data.json"
        dummy_artifact.write_text(json.dumps({"result": 1.0}), encoding="utf-8")

        h = hashlib.sha256(dummy_artifact.read_bytes()).hexdigest()
        (run_dir / "hashes.sha256").write_text(
            f"{h}  raw/data.json\n", encoding="utf-8"
        )

        receipt_data = {
            "schema_version": 1,
            "run_id": run_id,
            "slice_id": slice_id,
            "state": "AUDIT_COMPLETED",
            "artifact_root": str(artifact_root),
            "run_directory": str(run_dir),
            "implementation_commit": "commit_1",
            "protocol_commit": "commit_2",
            "protocol_sha256": "proto_sha",
            "input_hashes": {},
            "runner_version": "1.0.0",
            "created_at_utc": "2026-08-10T12:00:00Z",
            "started_at_utc": "2026-08-10T12:01:00Z",
            "finished_at_utc": "2026-08-10T12:02:00Z",
            "exit_code": 0,
            "artifact_inventory": ["raw/data.json"],
            "artifact_hashes": {"raw/data.json": h},
            "previous_receipt_sha256": "sha_prev",
            "receipt_sha256": "sha_curr",
        }
        (run_dir / "run_receipt.json").write_text(
            json.dumps(receipt_data, indent=2), encoding="utf-8"
        )
        (receipts_dir / "000_PREPARED.json").write_text("{}", encoding="utf-8")
        (receipts_dir / "001_RUN_STARTED.json").write_text("{}", encoding="utf-8")
        (receipts_dir / "002_RUN_COMPLETED.json").write_text("{}", encoding="utf-8")
        (receipts_dir / "003_AUDIT_COMPLETED.json").write_text("{}", encoding="utf-8")

        return run_dir

    def _create_valid_inventory_run_directory(
        self,
        artifact_root: Path,
        slice_id: str = "slice5b",
        run_id: str = "run_valid_inventory",
    ) -> Path:
        """Create a valid run directory layout with real artifacts and receipt chain."""
        from raglab.agentic.experiments import (
            RunReceipt,
            RunReceiptStore,
            RunState,
            compute_file_sha256,
        )

        run_dir = artifact_root / slice_id / run_id
        receipts_dir = run_dir / "receipts"
        raw_dir = run_dir / "raw"
        derived_dir = run_dir / "derived"
        logs_dir = run_dir / "logs"

        for d in (receipts_dir, raw_dir, derived_dir, logs_dir):
            d.mkdir(parents=True, exist_ok=True)

        proto_file = run_dir / "protocol.snapshot.json"
        proto_file.write_text(
            json.dumps({"protocol_id": "p1"}, sort_keys=True), encoding="utf-8"
        )
        proto_sha = compute_file_sha256(proto_file)

        raw_file = raw_dir / "data.json"
        raw_file.write_text(
            json.dumps({"raw": True}), encoding="utf-8"
        )
        raw_sha = compute_file_sha256(raw_file)

        derived_file = derived_dir / "metrics.json"
        derived_file.write_text(
            json.dumps({"accuracy": 1.0}), encoding="utf-8"
        )
        derived_sha = compute_file_sha256(derived_file)

        logs_file = logs_dir / "execution.log"
        logs_file.write_text(
            "Execution finished cleanly.\n", encoding="utf-8"
        )
        logs_sha = compute_file_sha256(logs_file)

        inventory = (
            "raw/data.json",
            "derived/metrics.json",
            "logs/execution.log",
        )
        hashes = {
            "raw/data.json": raw_sha,
            "derived/metrics.json": derived_sha,
            "logs/execution.log": logs_sha,
        }

        hashes_content = (
            "\n".join(f"{h}  {path}" for path, h in hashes.items()) + "\n"
        )
        (run_dir / "hashes.sha256").write_text(hashes_content, encoding="utf-8")

        store = RunReceiptStore(run_dir)
        draft = RunReceipt(
            schema_version=1,
            run_id=run_id,
            slice_id=slice_id,
            state=RunState.PREPARED,
            artifact_root=str(artifact_root),
            run_directory=str(run_dir),
            implementation_commit="a" * 64,
            protocol_commit="b" * 64,
            protocol_sha256=proto_sha,
            input_hashes={},
            runner_version="1.0.0",
            created_at_utc="2026-08-11T12:00:00Z",
            previous_receipt_sha256=None,
            receipt_sha256="",
            artifact_inventory=inventory,
            artifact_hashes=hashes,
        )
        computed_hash = draft.compute_hash()
        final_receipt = RunReceipt(
            schema_version=draft.schema_version,
            run_id=draft.run_id,
            slice_id=draft.slice_id,
            state=draft.state,
            artifact_root=draft.artifact_root,
            run_directory=draft.run_directory,
            implementation_commit=draft.implementation_commit,
            protocol_commit=draft.protocol_commit,
            protocol_sha256=draft.protocol_sha256,
            input_hashes=draft.input_hashes,
            runner_version=draft.runner_version,
            created_at_utc=draft.created_at_utc,
            previous_receipt_sha256=None,
            receipt_sha256=computed_hash,
            artifact_inventory=draft.artifact_inventory,
            artifact_hashes=draft.artifact_hashes,
        )
        store.append(final_receipt)

        (run_dir / "run_receipt.json").write_text(
            json.dumps(final_receipt.to_dict(), indent=2), encoding="utf-8"
        )

        return run_dir

    def _get_directory_inventory(
        self, target_dir: Path
    ) -> tuple[dict[str, tuple[int, str]], set[str]]:
        """Get deterministic inventory of files and relative directory paths."""
        files_inv: dict[str, tuple[int, str]] = {}
        dirs_inv: set[str] = set()

        for p in sorted(target_dir.rglob("*")):
            rel_path = str(p.relative_to(target_dir))
            if p.is_dir():
                dirs_inv.add(rel_path)
            elif p.is_file():
                content = p.read_bytes()
                files_inv[rel_path] = (
                    len(content),
                    hashlib.sha256(content).hexdigest(),
                )

        return files_inv, dirs_inv

    # =========================================================================
    # PATHS (Casos 1 a 13)
    # =========================================================================

    def test_01_empty_path_rejected(self) -> None:
        """1. Empty path must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            ExperimentalPathPolicy,
            PathPolicyError,
        )

        policy = ExperimentalPathPolicy()
        with self.assertRaises(PathPolicyError):
            policy.validate_target_directory("")

    def test_02_relative_path_rejected(self) -> None:
        """2. Relative path must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            ExperimentalPathPolicy,
            PathPolicyError,
        )

        policy = ExperimentalPathPolicy()
        with self.assertRaises(PathPolicyError):
            policy.validate_target_directory("relative/path/to/dir")

    def test_03_tmp_path_rejected(self) -> None:
        """3. /tmp root path must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            ExperimentalPathPolicy,
            PathPolicyError,
        )

        policy = ExperimentalPathPolicy()
        with self.assertRaises(PathPolicyError):
            policy.validate_target_directory("/tmp")  # noqa: S108

    def test_04_tmp_descendant_rejected(self) -> None:
        """4. Descendant of /tmp path must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            ExperimentalPathPolicy,
            PathPolicyError,
        )

        policy = ExperimentalPathPolicy()
        with self.assertRaises(PathPolicyError):
            policy.validate_target_directory("/tmp/some_agentic_run_123")  # noqa: S108

    def test_05_var_tmp_rejected(self) -> None:
        """5. /var/tmp path must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            ExperimentalPathPolicy,
            PathPolicyError,
        )

        policy = ExperimentalPathPolicy()
        with self.assertRaises(PathPolicyError):
            policy.validate_target_directory("/var/tmp")  # noqa: S108

    def test_06_var_tmp_descendant_rejected(self) -> None:
        """6. Descendant of /var/tmp path must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            ExperimentalPathPolicy,
            PathPolicyError,
        )

        policy = ExperimentalPathPolicy()
        with self.assertRaises(PathPolicyError):
            policy.validate_target_directory("/var/tmp/some_run")  # noqa: S108

    def test_07_repo_root_rejected(self) -> None:
        """7. Repo root path must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            ExperimentalPathPolicy,
            PathPolicyError,
        )

        policy = ExperimentalPathPolicy()
        repo_root = Path(__file__).parents[2].resolve()
        with self.assertRaises(PathPolicyError):
            policy.validate_target_directory(repo_root)

    def test_08_repo_descendant_rejected(self) -> None:
        """8. Descendant of repo root path must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            ExperimentalPathPolicy,
            PathPolicyError,
        )

        policy = ExperimentalPathPolicy()
        repo_sub = (Path(__file__).parents[2] / "src" / "output").resolve()
        with self.assertRaises(PathPolicyError):
            policy.validate_target_directory(repo_sub)

    def test_09_benchmarks_rejected(self) -> None:
        """9. benchmarks/ path must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            ExperimentalPathPolicy,
            PathPolicyError,
        )

        policy = ExperimentalPathPolicy()
        b_path = (Path(__file__).parents[2] / "benchmarks" / "run1").resolve()
        with self.assertRaises(PathPolicyError):
            policy.validate_target_directory(b_path)

    def test_10_checkpoints_rejected(self) -> None:
        """10. checkpoints/ path must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            ExperimentalPathPolicy,
            PathPolicyError,
        )

        policy = ExperimentalPathPolicy()
        c_path = (Path(__file__).parents[2] / "checkpoints" / "run1").resolve()
        with self.assertRaises(PathPolicyError):
            policy.validate_target_directory(c_path)

    def test_11_symlink_to_forbidden_rejected(self) -> None:
        """11. Symlink pointing to a forbidden path must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            ExperimentalPathPolicy,
            PathPolicyError,
        )

        policy = ExperimentalPathPolicy()
        symlink_path = self.sandbox / "symlink_tmp"
        if not symlink_path.exists():
            with contextlib.suppress(OSError):
                os.symlink("/tmp", symlink_path)  # noqa: S108
        with self.assertRaises(PathPolicyError):
            policy.validate_target_directory(symlink_path)

    def test_12_existing_run_directory_rejected(self) -> None:
        """12. Pre-existing run directory must be rejected during prepare."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            ExperimentalPathPolicy,
            PathPolicyError,
        )

        policy = ExperimentalPathPolicy()
        existing_dir = self.sandbox / "existing_dir"
        existing_dir.mkdir(parents=True, exist_ok=True)
        (existing_dir / "file.txt").write_text("content", encoding="utf-8")
        with self.assertRaises(PathPolicyError):
            policy.validate_target_directory(existing_dir, must_be_empty=True)

    def test_13_used_run_id_rejected(self) -> None:
        """13. Reusing a run_id with existing artifacts must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            RunControllerError,
            RunState,
        )

        controller, repo_dir, protocol_path, artifact_root = (
            self._make_hermetic_run_environment("test_13")
        )
        r1 = controller.prepare_run(
            run_id="duplicate_run_id",
            slice_id="slice5b",
            protocol_path=protocol_path,
            artifact_root=artifact_root,
        )
        self.assertEqual(r1.state, RunState.PREPARED)

        with self.assertRaises(RunControllerError) as ctx:
            controller.prepare_run(
                run_id="duplicate_run_id",
                slice_id="slice5b",
                protocol_path=protocol_path,
                artifact_root=artifact_root,
            )
        self.assertIn("not empty", str(ctx.exception).lower())

    # =========================================================================
    # GIT E PROTOCOLO (Casos 14 a 22)
    # =========================================================================

    def test_14_dirty_git_tree_rejected(self) -> None:
        """14. Dirty tracked git tree must be rejected during prepare."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            RunController,
            RunControllerError,
        )

        repo_dir = self.sandbox / "dirty_repo"
        self._create_synthetic_git_repo(repo_dir)
        (repo_dir / "src_file.py").write_text("# modified", encoding="utf-8")

        controller = RunController(repo_root=repo_dir)
        with self.assertRaises(RunControllerError):
            controller.preflight_git_check(allow_dirty=False)

    def test_15_non_empty_staging_rejected(self) -> None:
        """15. Non-empty git staging must be rejected during prepare."""
        import shutil

        self._import_target_module()
        from raglab.agentic.experiments import (
            RunController,
            RunControllerError,
        )

        repo_dir = self.sandbox / "staged_repo"
        self._create_synthetic_git_repo(repo_dir)
        (repo_dir / "staged_file.py").write_text("# staged", encoding="utf-8")
        git_bin = shutil.which("git") or "git"
        subprocess.run(
            [git_bin, "add", "staged_file.py"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )

        controller = RunController(repo_root=repo_dir)
        with self.assertRaises(RunControllerError):
            controller.preflight_git_check(allow_staged=False)

    def test_16_missing_implementation_commit_rejected(self) -> None:
        """16. Non-existent implementation commit hash must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            RunController,
            RunControllerError,
        )

        controller = RunController()
        with self.assertRaises(RunControllerError):
            controller.validate_commits(
                implementation_commit="0000000000000000000000000000000000000000",
                protocol_commit="245718846707804de171e5e869847b011285b801",
            )

    def test_17_missing_protocol_commit_rejected(self) -> None:
        """17. Non-existent protocol commit hash must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            RunController,
            RunControllerError,
        )

        controller = RunController()
        with self.assertRaises(RunControllerError):
            controller.validate_commits(
                implementation_commit="245718846707804de171e5e869847b011285b801",
                protocol_commit="0000000000000000000000000000000000000000",
            )

    def test_18_incorrect_commit_ancestry_rejected(self) -> None:
        """18. Implementation commit not being an ancestor of protocol commit must fail."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            RunController,
            RunControllerError,
        )

        controller = RunController()
        with self.assertRaises(RunControllerError):
            controller.validate_commit_ancestry(
                impl_commit="head_commit", proto_commit="older_commit"
            )

    def test_19_protocol_commit_outside_head_rejected(self) -> None:
        """19. Protocol commit not being ancestor of HEAD must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            RunController,
            RunControllerError,
        )

        controller = RunController()
        with self.assertRaises(RunControllerError):
            controller.validate_protocol_in_head_ancestry("detached_commit")

    def test_20_untracked_protocol_file_rejected(self) -> None:
        """20. Protocol file not tracked in git must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            RunController,
            RunControllerError,
        )

        controller = RunController()
        untracked_proto = self.sandbox / "untracked_proto.json"
        untracked_proto.write_text("{}", encoding="utf-8")
        with self.assertRaises(RunControllerError):
            controller.validate_protocol_tracked(untracked_proto)

    def test_21_protocol_modified_after_commit_rejected(self) -> None:
        """21. Protocol file modified in workspace after commit must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            RunController,
            RunControllerError,
        )

        repo_dir = self.sandbox / "protocol_modified_repo"
        self._create_synthetic_git_repo(repo_dir)
        (repo_dir / "protocol.json").write_text(
            json.dumps({"protocol_id": "proto_1", "version": "2.0_modified"}),
            encoding="utf-8",
        )

        controller = RunController(repo_root=repo_dir)
        protocol_path = repo_dir / "protocol.json"
        with self.assertRaises(RunControllerError):
            controller.validate_protocol_unmodified(protocol_path)

    def test_22_protocol_sha_mismatch_rejected(self) -> None:
        """22. Protocol SHA-256 hash mismatch must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            RunController,
            RunControllerError,
        )

        controller = RunController()
        with self.assertRaises(RunControllerError):
            controller.validate_protocol_sha(
                protocol_path="proto.json",
                expected_sha="0000000000000000000000000000000000000000000000000000000000000000",
            )

    # =========================================================================
    # INPUTS (Casos 23 a 26)
    # =========================================================================

    def test_23_missing_input_file_rejected(self) -> None:
        """23. Non-existent input file in inputs dict must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            RunController,
            RunControllerError,
        )

        controller = RunController()
        with self.assertRaises(RunControllerError):
            controller.validate_inputs(
                {"missing_input": "/path/does/not/exist.json"}
            )

    def test_24_input_hash_divergence_rejected(self) -> None:
        """24. Input file with hash divergence must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            LineageVerifier,
            RunControllerError,
        )

        verifier = LineageVerifier()
        with self.assertRaises((RunControllerError, ValueError, RuntimeError)):
            verifier.verify_input_hashes(
                actual_inputs={"qrels": self.sandbox / "qrels.json"},
                expected_hashes={"qrels": "wrong_hash"},
            )

    def test_25_input_modified_after_prepare_detected(self) -> None:
        """25. Input file modified after prepare phase must be detected by verifier."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            LineageVerifier,
            RunControllerError,
            compute_file_sha256,
        )

        input_file = self.sandbox / "qrels.json"
        input_file.write_text(
            json.dumps({"qrel_id": 1}), encoding="utf-8"
        )
        initial_hash = compute_file_sha256(input_file)

        # Alter input file after prepare phase
        input_file.write_text(
            json.dumps({"qrel_id": 1, "tampered": True}), encoding="utf-8"
        )

        verifier = LineageVerifier()
        with self.assertRaises((RunControllerError, ValueError)):
            verifier.verify_input_hashes(
                actual_inputs={"qrels": input_file},
                expected_hashes={"qrels": initial_hash},
            )

    def test_26_all_inputs_recorded_in_receipt(self) -> None:
        """26. All passed inputs must appear in input_hashes dictionary of receipt."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            compute_file_sha256,
        )

        controller, repo_dir, protocol_path, artifact_root = (
            self._make_hermetic_run_environment("test_26")
        )
        qrels_file = self.sandbox / "qrels.json"
        qrels_file.write_text('{"q1": ["d1"]}\n', encoding="utf-8")
        expected_hash = compute_file_sha256(qrels_file)

        receipt = controller.prepare_run(
            run_id="test_inputs",
            slice_id="slice5b",
            protocol_path=protocol_path,
            artifact_root=artifact_root,
            inputs={"qrels": str(qrels_file)},
        )
        self.assertEqual(receipt.input_hashes, {"qrels": expected_hash})

    # =========================================================================
    # RECEIPTS (Casos 27 a 39)
    # =========================================================================

    def test_27_all_required_receipt_fields_present(self) -> None:
        """27. RunReceipt must contain all required contract fields."""
        self._import_target_module()
        from raglab.agentic.experiments import RunReceipt, RunState

        receipt = RunReceipt(
            schema_version=1,
            run_id="run_fields",
            slice_id="slice5b",
            state=RunState.PREPARED,
            artifact_root=str(self.sandbox),
            run_directory=str(self.sandbox / "slice5b" / "run_fields"),
            implementation_commit="commit1",
            protocol_commit="commit2",
            protocol_sha256="sha_proto",
            input_hashes={},
            runner_version="1.0.0",
            created_at_utc="2026-08-10T12:00:00Z",
            started_at_utc=None,
            finished_at_utc=None,
            exit_code=None,
            artifact_inventory=[],
            artifact_hashes={},
            previous_receipt_sha256=None,
            receipt_sha256="sha_receipt",
        )
        d = receipt.to_dict()
        required_fields = self.contract_spec["required_receipt_fields"]
        for f in required_fields:
            self.assertIn(f, d)

    def test_28_prepared_created_as_receipt_000(self) -> None:
        """28. PREPARED state must be persisted in receipts/000_PREPARED.json."""
        self._import_target_module()
        from raglab.agentic.experiments import RunState

        controller, repo_dir, protocol_path, artifact_root = (
            self._make_hermetic_run_environment("test_28")
        )
        receipt = controller.prepare_run(
            run_id="run_000",
            slice_id="slice5b",
            protocol_path=protocol_path,
            artifact_root=artifact_root,
        )
        self.assertEqual(receipt.state, RunState.PREPARED)
        rec_000 = Path(receipt.run_directory) / "receipts" / "000_PREPARED.json"
        self.assertTrue(rec_000.exists())

    def test_29_correct_hash_chain(self) -> None:
        """29. Each new receipt in receipts/ must correctly form a cryptographic hash chain."""
        self._import_target_module()
        from raglab.agentic.experiments import RunReceiptStore

        store = RunReceiptStore(self.sandbox / "run_chain")
        self.assertTrue(store.verify_chain_integrity())

    def test_30_previous_receipt_sha256_correct(self) -> None:
        """30. previous_receipt_sha256 must match receipt_sha256 of preceding receipt."""
        self._import_target_module()
        from raglab.agentic.experiments import RunReceiptStore

        store = RunReceiptStore(self.sandbox / "run_chain")
        rec_started = store.get_receipt_by_index(1)
        rec_prepared = store.get_receipt_by_index(0)
        self.assertEqual(
            rec_started.previous_receipt_sha256, rec_prepared.receipt_sha256
        )

    def test_31_receipt_sha256_recalculable(self) -> None:
        """31. receipt_sha256 must be deterministically recalculable from content."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            RunReceiptStore,
            compute_canonical_json_sha256,
        )

        receipt = RunReceiptStore.load_receipt_file(
            self.sandbox / "run_1" / "run_receipt.json"
        )
        data_to_hash = receipt.to_dict()
        data_to_hash.pop("receipt_sha256", None)
        expected = compute_canonical_json_sha256(data_to_hash)
        self.assertEqual(receipt.receipt_sha256, expected)

    def test_32_previous_receipt_byte_intact(self) -> None:
        """32. Previous historical receipt file must remain byte-by-byte intact upon transition."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            RunController,
            compute_file_sha256,
        )

        controller = RunController()
        rec0 = self.sandbox / "receipts" / "000_PREPARED.json"
        sha_before = compute_file_sha256(rec0)
        controller.start_run(self.sandbox)
        sha_after = compute_file_sha256(rec0)
        self.assertEqual(sha_before, sha_after)

    def test_33_allowed_transitions_accepted(self) -> None:
        """33. Allowed transitions (PREPARED->STARTED->COMPLETED->AUDIT_COMPLETED) must succeed."""
        self._import_target_module()
        from raglab.agentic.experiments import RunState

        controller, repo_dir, protocol_path, artifact_root = (
            self._make_hermetic_run_environment("test_33")
        )
        r1 = controller.prepare_run(
            run_id="run_allowed",
            slice_id="slice5b",
            protocol_path=protocol_path,
            artifact_root=artifact_root,
        )
        self.assertEqual(r1.state, RunState.PREPARED)
        r2 = controller.start_run(r1.run_directory)
        self.assertEqual(r2.state, RunState.RUN_STARTED)

    def test_34_forbidden_transitions_rejected(self) -> None:
        """34. Forbidden state transitions (e.g. PREPARED->AUDIT_COMPLETED) must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            RunController,
            RunControllerError,
        )

        controller = RunController()
        with self.assertRaises(RunControllerError):
            controller.transition_state(
                run_dir=self.sandbox / "run_prep",
                target_state="AUDIT_COMPLETED",
            )

    def test_35_audit_completed_terminal(self) -> None:
        """35. AUDIT_COMPLETED is a terminal state and cannot transition further."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            RunController,
            RunControllerError,
        )

        controller = RunController()
        with self.assertRaises(RunControllerError):
            controller.transition_state(
                run_dir=self.sandbox / "run_audited",
                target_state="RUN_STARTED",
            )

    def test_36_failed_terminal(self) -> None:
        """36. FAILED is a terminal state and cannot transition further."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            RunController,
            RunControllerError,
        )

        controller = RunController()
        with self.assertRaises(RunControllerError):
            controller.transition_state(
                run_dir=self.sandbox / "run_failed",
                target_state="PREPARED",
            )

    def test_37_tampered_receipt_detected(self) -> None:
        """37. Tampered receipt in receipts/ must be detected by receipt store auditor."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            ReceiptStoreError,
            RunReceiptStore,
        )

        store = RunReceiptStore(self.sandbox / "run_tampered")
        with self.assertRaises(ReceiptStoreError):
            store.verify_chain_integrity()

    def test_38_incomplete_chain_detected(self) -> None:
        """38. Missing intermediate receipt in receipts/ chain must be detected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            ReceiptStoreError,
            RunReceiptStore,
        )

        store = RunReceiptStore(self.sandbox / "run_incomplete_chain")
        with self.assertRaises(ReceiptStoreError):
            store.verify_chain_integrity()

    def test_39_duplicate_index_detected(self) -> None:
        """39. Duplicate index prefix in receipts/ directory must be detected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            ReceiptStoreError,
            RunReceiptStore,
        )

        store = RunReceiptStore(self.sandbox / "run_dup_index")
        with self.assertRaises(ReceiptStoreError):
            store.verify_chain_integrity()

    # =========================================================================
    # SNAPSHOT E ARTEFATOS (Casos 40 a 48)
    # =========================================================================

    def test_40_protocol_snapshot_byte_exact(self) -> None:
        """40. protocol.snapshot.json must be 100% byte-for-byte identical to source protocol."""
        self._import_target_module()

        controller, repo_dir, protocol_path, artifact_root = (
            self._make_hermetic_run_environment("test_40")
        )
        receipt = controller.prepare_run(
            run_id="run_snapshot",
            slice_id="slice5b",
            protocol_path=protocol_path,
            artifact_root=artifact_root,
        )
        snapshot = Path(receipt.run_directory) / "protocol.snapshot.json"
        self.assertEqual(protocol_path.read_bytes(), snapshot.read_bytes())

    def test_41_tampered_snapshot_detected(self) -> None:
        """41. Modified protocol.snapshot.json must be detected by verifier."""
        self._import_target_module()
        from raglab.agentic.experiments import LineageVerifier

        verifier = LineageVerifier()
        audit = verifier.verify_lineage(self.sandbox / "run_tampered_snapshot")
        self.assertFalse(audit.is_valid)

    def test_42_complete_inventory_accepted(self) -> None:
        """42. Complete artifact inventory matching hashes.sha256 must be accepted."""
        self._import_target_module()
        from raglab.agentic.experiments import LineageVerifier

        run_dir = self._create_valid_inventory_run_directory(
            self.sandbox, run_id="run_valid_inventory"
        )
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertTrue(audit.is_valid)

    def test_43_missing_artifact_detected(self) -> None:
        """43. Missing recorded artifact must trigger audit failure."""
        self._import_target_module()
        from raglab.agentic.experiments import LineageVerifier

        run_dir = self._create_valid_inventory_run_directory(
            self.sandbox, run_id="run_missing_artifact"
        )
        (run_dir / "raw" / "data.json").unlink()

        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)

    def test_44_tampered_artifact_detected(self) -> None:
        """44. Adulterated artifact content must trigger audit failure."""
        self._import_target_module()
        from raglab.agentic.experiments import LineageVerifier

        run_dir = self._create_valid_inventory_run_directory(
            self.sandbox, run_id="run_tampered_artifact"
        )
        (run_dir / "derived" / "metrics.json").write_text(
            json.dumps({"accuracy": 0.0, "tampered": True}), encoding="utf-8"
        )

        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)

    def test_45_unexpected_file_detected(self) -> None:
        """45. Unexpected unrecorded file in run directory must be detected."""
        self._import_target_module()
        from raglab.agentic.experiments import LineageVerifier

        run_dir = self._create_valid_inventory_run_directory(
            self.sandbox, run_id="run_unexpected_file"
        )
        unexpected_path = run_dir / "raw" / "unexpected.json"
        unexpected_path.write_text(
            json.dumps({"unexpected": True}), encoding="utf-8"
        )
        self.assertTrue(unexpected_path.exists())

        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)

    def test_46_hashes_sha256_consistent(self) -> None:
        """46. hashes.sha256 file must be 100% consistent with artifact_hashes."""
        self._import_target_module()
        from raglab.agentic.experiments import LineageVerifier

        run_dir = self._create_valid_inventory_run_directory(
            self.sandbox, run_id="run_inconsistent_hashes"
        )
        hashes_file = run_dir / "hashes.sha256"
        target_file = run_dir / "derived" / "metrics.json"

        self.assertTrue(hashes_file.exists())
        self.assertTrue(target_file.exists())

        lines = hashes_file.read_text(encoding="utf-8").splitlines()
        new_lines = []
        modified_line_found = False
        for line in lines:
            if "derived/metrics.json" in line:
                new_lines.append(f"{'0' * 64}  derived/metrics.json")
                modified_line_found = True
            else:
                new_lines.append(line)

        self.assertTrue(modified_line_found)
        hashes_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

        verifier = LineageVerifier()
        audit = verifier.verify_lineage(run_dir)
        self.assertFalse(audit.is_valid)

    def test_47_run_id_coincides_with_directory(self) -> None:
        """47. run_id in receipt must coincide with directory name."""
        self._import_target_module()
        from raglab.agentic.experiments import LineageVerifier

        original_run_dir = self._create_valid_inventory_run_directory(
            self.sandbox, run_id="run_id_original"
        )
        verifier = LineageVerifier()
        self.assertTrue(verifier.verify_lineage(original_run_dir).is_valid)

        renamed_run_dir = original_run_dir.parent / "run_id_divergent"
        original_run_dir.rename(renamed_run_dir)

        self.assertFalse(original_run_dir.exists())
        self.assertTrue(renamed_run_dir.exists())
        self.assertNotEqual(renamed_run_dir.name, "run_id_original")

        audit = verifier.verify_lineage(renamed_run_dir)
        self.assertFalse(audit.is_valid)

    def test_48_slice_id_coincides_with_directory(self) -> None:
        """48. slice_id in receipt must coincide with parent directory name."""
        self._import_target_module()
        from raglab.agentic.experiments import LineageVerifier

        original_run_dir = self._create_valid_inventory_run_directory(
            self.sandbox, slice_id="slice5b", run_id="run_valid_slice_test"
        )
        verifier = LineageVerifier()
        self.assertTrue(verifier.verify_lineage(original_run_dir).is_valid)

        original_slice_dir = original_run_dir.parent
        renamed_slice_dir = self.sandbox / "slice_divergent"
        original_slice_dir.rename(renamed_slice_dir)

        new_run_dir = renamed_slice_dir / original_run_dir.name

        self.assertFalse(original_slice_dir.exists())
        self.assertTrue(new_run_dir.exists())
        self.assertEqual(new_run_dir.name, "run_valid_slice_test")
        self.assertNotEqual(new_run_dir.parent.name, "slice5b")

        audit = verifier.verify_lineage(new_run_dir)
        self.assertFalse(audit.is_valid)

    # =========================================================================
    # ATOMICIDADE E CONCORRÊNCIA (Casos 49 a 54)
    # =========================================================================

    def test_49_exclusive_lock_acquired(self) -> None:
        """49. Exclusive file lock must be acquired on run directory during execution."""
        self._import_target_module()
        from raglab.agentic.experiments import RunController

        controller = RunController()
        lock = controller.acquire_run_lock(self.sandbox / "run_locked")
        self.assertTrue(lock.is_acquired())

    def test_50_concurrent_run_rejected(self) -> None:
        """50. Second execution trying to acquire lock on active run must be rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            RunController,
            RunControllerError,
        )

        controller = RunController()
        _lock1 = controller.acquire_run_lock(self.sandbox / "run_concurrent")
        with self.assertRaises(RunControllerError):
            controller.acquire_run_lock(self.sandbox / "run_concurrent")

    def test_51_lock_released_on_exit(self) -> None:
        """51. Exclusive lock must be released when controller context closes or finishes."""
        self._import_target_module()
        from raglab.agentic.experiments import RunController

        controller = RunController()
        with controller.run_context(self.sandbox / "run_context"):
            pass
        # Should be able to acquire lock now
        lock = controller.acquire_run_lock(self.sandbox / "run_context")
        self.assertTrue(lock.is_acquired())

    def test_52_fsync_before_os_replace(self) -> None:
        """52. Persistence engine must execute explicit fsync() before os.replace()."""
        self._import_target_module()
        from raglab.agentic.experiments import RunReceiptStore

        store = RunReceiptStore(self.sandbox / "run_fsync")
        self.assertTrue(store.supports_fsync_atomic_write())

    def test_53_failed_write_no_partial_json(self) -> None:
        """53. Failed write during receipt persistence must leave no partial JSON file."""
        self._import_target_module()
        from raglab.agentic.experiments import RunReceiptStore

        target_file = self.sandbox / "run_receipt.json"
        with self.assertRaises((RuntimeError, ValueError, OSError)):
            RunReceiptStore.write_atomic_failing_simulation(target_file)
        self.assertFalse(target_file.exists())

    def test_54_collision_does_not_overwrite(self) -> None:
        """54. Run ID collision must not overwrite existing run directory or receipts."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            RunControllerError,
            RunState,
        )

        controller, repo_dir, protocol_path, artifact_root = (
            self._make_hermetic_run_environment("test_54")
        )
        r1 = controller.prepare_run(
            run_id="existing_run",
            slice_id="slice5b",
            protocol_path=protocol_path,
            artifact_root=artifact_root,
        )
        self.assertEqual(r1.state, RunState.PREPARED)
        rec_000 = Path(r1.run_directory) / "receipts" / "000_PREPARED.json"
        content_before = rec_000.read_bytes()

        with self.assertRaises(RunControllerError) as ctx:
            controller.prepare_run(
                run_id="existing_run",
                slice_id="slice5b",
                protocol_path=protocol_path,
                artifact_root=artifact_root,
            )
        self.assertIn("not empty", str(ctx.exception).lower())
        self.assertEqual(rec_000.read_bytes(), content_before)

    # =========================================================================
    # VERIFICADOR (Casos 55 a 63)
    # =========================================================================

    def test_55_valid_run_returns_valid_status(self) -> None:
        """55. LineageVerifier must return is_valid=True for a completely valid run."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            LineageVerifier,
            RunReceipt,
            RunReceiptStore,
            RunState,
            compute_file_sha256,
        )

        valid_run_dir = self.sandbox / "slice5b" / "valid_run"
        receipts_dir = valid_run_dir / "receipts"
        receipts_dir.mkdir(parents=True, exist_ok=True)

        proto_file = valid_run_dir / "protocol.snapshot.json"
        proto_file.write_text(
            json.dumps({"protocol_id": "p1"}, sort_keys=True), encoding="utf-8"
        )
        proto_sha = compute_file_sha256(proto_file)

        store = RunReceiptStore(valid_run_dir)
        draft = RunReceipt(
            schema_version=1,
            run_id="valid_run",
            slice_id="slice5b",
            state=RunState.PREPARED,
            artifact_root=str(self.sandbox),
            run_directory=str(valid_run_dir),
            implementation_commit="a" * 64,
            protocol_commit="b" * 64,
            protocol_sha256=proto_sha,
            input_hashes={},
            runner_version="1.0.0",
            created_at_utc="2026-08-11T12:00:00Z",
            previous_receipt_sha256=None,
            receipt_sha256="",
        )
        computed_hash = draft.compute_hash()
        final_receipt = RunReceipt(
            schema_version=draft.schema_version,
            run_id=draft.run_id,
            slice_id=draft.slice_id,
            state=draft.state,
            artifact_root=draft.artifact_root,
            run_directory=draft.run_directory,
            implementation_commit=draft.implementation_commit,
            protocol_commit=draft.protocol_commit,
            protocol_sha256=draft.protocol_sha256,
            input_hashes=draft.input_hashes,
            runner_version=draft.runner_version,
            created_at_utc=draft.created_at_utc,
            previous_receipt_sha256=None,
            receipt_sha256=computed_hash,
        )
        store.append(final_receipt)

        verifier = LineageVerifier()
        audit = verifier.verify_lineage(valid_run_dir)
        self.assertTrue(audit.is_valid)

    def test_56_valid_run_returns_exit_code_zero(self) -> None:
        """56. CLI verifier must return exit code 0 for a valid run."""
        script_path = Path("scripts/verify_agentic_run_lineage.py")
        self.assertTrue(
            script_path.exists(), f"CLI script {script_path} must exist."
        )
        artifact_root = self.sandbox / "external_artifacts"
        valid_run_dir = self._create_valid_run_directory(artifact_root)
        files_before, dirs_before = self._get_directory_inventory(valid_run_dir)

        cmd = [
            sys.executable,
            str(script_path),
            "--run-dir",
            str(valid_run_dir),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        diagnostic = f"stdout: {res.stdout}\nstderr: {res.stderr}"
        self.assertEqual(res.returncode, 0, diagnostic)

        files_after, dirs_after = self._get_directory_inventory(valid_run_dir)
        self.assertEqual(files_before, files_after)
        self.assertEqual(dirs_before, dirs_after)

    def test_57_invalid_run_returns_invalid_status(self) -> None:
        """57. LineageVerifier must return is_valid=False for an invalid run."""
        self._import_target_module()
        from raglab.agentic.experiments import LineageVerifier

        verifier = LineageVerifier()
        audit = verifier.verify_lineage(self.sandbox / "invalid_run")
        self.assertFalse(audit.is_valid)

    def test_58_invalid_run_returns_non_zero_exit(self) -> None:
        """58. CLI verifier must return non-zero exit code for an invalid run."""
        script_path = Path("scripts/verify_agentic_run_lineage.py")
        self.assertTrue(
            script_path.exists(), f"CLI script {script_path} must exist."
        )
        artifact_root = self.sandbox / "external_artifacts"
        invalid_run_dir = self._create_valid_run_directory(
            artifact_root, run_id="invalid_run"
        )
        (invalid_run_dir / "raw" / "data.json").write_text(
            "tampered content", encoding="utf-8"
        )
        files_before, dirs_before = self._get_directory_inventory(
            invalid_run_dir
        )

        cmd = [
            sys.executable,
            str(script_path),
            "--run-dir",
            str(invalid_run_dir),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        diagnostic = f"stdout: {res.stdout}\nstderr: {res.stderr}".lower()
        self.assertNotEqual(res.returncode, 0)
        self.assertTrue(
            "hash" in diagnostic
            or "lineage" in diagnostic
            or "corrupted" in diagnostic
            or "invalid" in diagnostic
        )

        files_after, dirs_after = self._get_directory_inventory(
            invalid_run_dir
        )
        self.assertEqual(files_before, files_after)
        self.assertEqual(dirs_before, dirs_after)

    def test_59_verifier_does_not_alter_bytes(self) -> None:
        """59. LineageVerifier must not alter any file bytes in the run directory."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            LineageVerifier,
            RunReceipt,
            RunReceiptStore,
            RunState,
            compute_file_sha256,
        )

        valid_run_dir = self.sandbox / "slice5b" / "valid_run_bytes"
        receipts_dir = valid_run_dir / "receipts"
        receipts_dir.mkdir(parents=True, exist_ok=True)

        proto_file = valid_run_dir / "protocol.snapshot.json"
        proto_file.write_text(
            json.dumps({"protocol_id": "p1"}, sort_keys=True), encoding="utf-8"
        )
        proto_sha = compute_file_sha256(proto_file)

        store = RunReceiptStore(valid_run_dir)
        draft = RunReceipt(
            schema_version=1,
            run_id="valid_run_bytes",
            slice_id="slice5b",
            state=RunState.PREPARED,
            artifact_root=str(self.sandbox),
            run_directory=str(valid_run_dir),
            implementation_commit="a" * 64,
            protocol_commit="b" * 64,
            protocol_sha256=proto_sha,
            input_hashes={},
            runner_version="1.0.0",
            created_at_utc="2026-08-11T12:00:00Z",
            previous_receipt_sha256=None,
            receipt_sha256="",
        )
        computed_hash = draft.compute_hash()
        final_receipt = RunReceipt(
            schema_version=draft.schema_version,
            run_id=draft.run_id,
            slice_id=draft.slice_id,
            state=draft.state,
            artifact_root=draft.artifact_root,
            run_directory=draft.run_directory,
            implementation_commit=draft.implementation_commit,
            protocol_commit=draft.protocol_commit,
            protocol_sha256=draft.protocol_sha256,
            input_hashes=draft.input_hashes,
            runner_version=draft.runner_version,
            created_at_utc=draft.created_at_utc,
            previous_receipt_sha256=None,
            receipt_sha256=computed_hash,
        )
        store.append(final_receipt)

        def get_file_hashes(dir_path: Path) -> dict[str, str]:
            hashes = {}
            for p in sorted(dir_path.rglob("*")):
                if p.is_file():
                    rel = str(p.relative_to(dir_path))
                    hashes[rel] = compute_file_sha256(p)
            return hashes

        hashes_before = get_file_hashes(valid_run_dir)
        verifier = LineageVerifier()
        audit = verifier.verify_lineage(valid_run_dir)
        self.assertTrue(audit.is_valid)

        hashes_after = get_file_hashes(valid_run_dir)
        self.assertEqual(set(hashes_before.keys()), set(hashes_after.keys()))
        self.assertEqual(hashes_before, hashes_after)

    def test_60_verifier_creates_no_files(self) -> None:
        """60. LineageVerifier must create zero new files in the run directory."""
        self._import_target_module()
        from raglab.agentic.experiments import LineageVerifier

        run_dir = self.sandbox / "read_only_run"
        files_before = set(run_dir.rglob("*"))
        verifier = LineageVerifier()
        verifier.verify_lineage(run_dir)
        files_after = set(run_dir.rglob("*"))
        self.assertEqual(files_before, files_after)

    def test_61_verifier_does_not_change_state(self) -> None:
        """61. LineageVerifier is read-only and must NOT promote or mutate receipt state."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            LineageVerifier,
            RunReceipt,
            RunReceiptStore,
            RunState,
            compute_file_sha256,
        )

        valid_run_dir = self.sandbox / "slice5b" / "run_state_check"
        receipts_dir = valid_run_dir / "receipts"
        receipts_dir.mkdir(parents=True, exist_ok=True)

        proto_file = valid_run_dir / "protocol.snapshot.json"
        proto_file.write_text(
            json.dumps({"protocol_id": "p1"}, sort_keys=True), encoding="utf-8"
        )
        proto_sha = compute_file_sha256(proto_file)

        store = RunReceiptStore(valid_run_dir)
        draft = RunReceipt(
            schema_version=1,
            run_id="run_state_check",
            slice_id="slice5b",
            state=RunState.PREPARED,
            artifact_root=str(self.sandbox),
            run_directory=str(valid_run_dir),
            implementation_commit="a" * 64,
            protocol_commit="b" * 64,
            protocol_sha256=proto_sha,
            input_hashes={},
            runner_version="1.0.0",
            created_at_utc="2026-08-11T12:00:00Z",
            previous_receipt_sha256=None,
            receipt_sha256="",
        )
        computed_hash = draft.compute_hash()
        final_receipt = RunReceipt(
            schema_version=draft.schema_version,
            run_id=draft.run_id,
            slice_id=draft.slice_id,
            state=draft.state,
            artifact_root=draft.artifact_root,
            run_directory=draft.run_directory,
            implementation_commit=draft.implementation_commit,
            protocol_commit=draft.protocol_commit,
            protocol_sha256=draft.protocol_sha256,
            input_hashes=draft.input_hashes,
            runner_version=draft.runner_version,
            created_at_utc=draft.created_at_utc,
            previous_receipt_sha256=None,
            receipt_sha256=computed_hash,
        )
        store.append(final_receipt)

        receipt_file = receipts_dir / "000_PREPARED.json"
        receipt_bytes_before = receipt_file.read_bytes()
        receipt_before = RunReceiptStore.load_latest(valid_run_dir)

        verifier = LineageVerifier()
        audit = verifier.verify_lineage(valid_run_dir)
        self.assertTrue(audit.is_valid)

        receipt_after = RunReceiptStore.load_latest(valid_run_dir)
        receipt_bytes_after = receipt_file.read_bytes()

        self.assertEqual(receipt_before.state, receipt_after.state)
        self.assertEqual(
            receipt_before.receipt_sha256, receipt_after.receipt_sha256
        )
        self.assertEqual(receipt_bytes_before, receipt_bytes_after)

    def test_62_verifier_does_not_repair_artifacts(self) -> None:
        """62. LineageVerifier must fail closed and never attempt to repair corrupted files."""
        self._import_target_module()
        from raglab.agentic.experiments import LineageVerifier

        verifier = LineageVerifier()
        audit = verifier.verify_lineage(self.sandbox / "corrupted_run")
        self.assertFalse(audit.is_valid)

    def test_63_verifier_reports_all_relevant_errors(self) -> None:
        """63. LineageVerifier failure_reasons list must contain all encountered errors."""
        self._import_target_module()
        from raglab.agentic.experiments import LineageVerifier

        multi_error_run = self.sandbox / "multi_error_run"
        multi_error_run.mkdir(parents=True, exist_ok=True)

        verifier = LineageVerifier()
        audit = verifier.verify_lineage(multi_error_run)
        self.assertFalse(audit.is_valid)
        self.assertGreaterEqual(len(audit.failure_reasons), 2)

        reasons_text = " ".join(audit.failure_reasons).lower()
        self.assertTrue(
            "receipt" in reasons_text or "chain" in reasons_text
        )
        self.assertTrue(
            "snapshot" in reasons_text or "protocol" in reasons_text
        )

    # =========================================================================
    # SEGURANÇA (Casos 64 a 67)
    # =========================================================================

    def test_64_credentials_not_persisted(self) -> None:
        """64. API credentials or keys must never be persisted in receipts."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            RunControllerError,
        )

        controller, repo_dir, protocol_path, artifact_root = (
            self._make_hermetic_run_environment("test_64")
        )
        with self.assertRaises(RunControllerError) as ctx:
            controller.prepare_run(
                run_id="sec_run",
                slice_id="slice5b",
                protocol_path=protocol_path,
                artifact_root=artifact_root,
                metadata={"GEMINI_API_KEY": "secret_123"},
            )
        self.assertIn("Security validation failed", str(ctx.exception))
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))
        self.assertFalse((artifact_root / "slice5b" / "sec_run").exists())

    def test_65_secret_values_not_persisted(self) -> None:
        """65. Secret value strings must be rejected by receipt sanitizer."""
        self._import_target_module()
        from raglab.agentic.experiments import RunReceipt

        for secret_name in self.contract_spec["secret_fields_forbidden"]:
            with self.assertRaises(ValueError):
                RunReceipt.validate_metadata_security({secret_name: "val"})

    def test_66_sensitive_names_absent_from_receipts(self) -> None:
        """66. Sensitive key names must be absent from all serialized receipt files."""
        self._import_target_module()
        rec_path = self.sandbox / "run_receipt.json"
        content = rec_path.read_text(encoding="utf-8")
        for forbidden_key in self.contract_spec["secret_fields_forbidden"]:
            self.assertNotIn(forbidden_key, content)

    def test_67_logs_contain_no_simulated_tokens(self) -> None:
        """67. Generated logs must contain zero simulated auth tokens or passwords."""
        self._import_target_module()
        from raglab.agentic.experiments import LineageVerifier

        verifier = LineageVerifier()
        audit = verifier.audit_security_logs(self.sandbox / "logs")
        self.assertTrue(audit.is_secure)

    # =========================================================================
    # CLI (Casos 68 a 78)
    # =========================================================================

    def test_68_cli_requires_repo_root(self) -> None:
        """68. prepare CLI must require --repo-root parameter."""
        script_path = Path("scripts/prepare_agentic_run.py")
        self.assertTrue(
            script_path.exists(), f"CLI script {script_path} must exist."
        )
        repo_dir = self.sandbox / "repo"
        impl_commit, proto_commit = self._create_synthetic_git_repo(repo_dir)
        cmd = [
            sys.executable,
            str(script_path),
            "--artifact-root",
            str(self.sandbox / "external_artifacts"),
            "--slice-id",
            "slice5b",
            "--run-id",
            "run1",
            "--protocol",
            str(repo_dir / "protocol.json"),
            "--implementation-commit",
            impl_commit,
            "--protocol-commit",
            proto_commit,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        diagnostic = f"{res.stdout}\n{res.stderr}".lower()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("--repo-root", diagnostic)

    def test_69_cli_requires_artifact_root(self) -> None:
        """69. prepare CLI must require --artifact-root parameter."""
        script_path = Path("scripts/prepare_agentic_run.py")
        self.assertTrue(
            script_path.exists(), f"CLI script {script_path} must exist."
        )
        repo_dir = self.sandbox / "repo"
        impl_commit, proto_commit = self._create_synthetic_git_repo(repo_dir)
        cmd = [
            sys.executable,
            str(script_path),
            "--repo-root",
            str(repo_dir),
            "--slice-id",
            "slice5b",
            "--run-id",
            "run1",
            "--protocol",
            str(repo_dir / "protocol.json"),
            "--implementation-commit",
            impl_commit,
            "--protocol-commit",
            proto_commit,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        diagnostic = f"{res.stdout}\n{res.stderr}".lower()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("--artifact-root", diagnostic)

    def test_70_cli_requires_slice_id(self) -> None:
        """70. prepare CLI must require --slice-id parameter."""
        script_path = Path("scripts/prepare_agentic_run.py")
        self.assertTrue(
            script_path.exists(), f"CLI script {script_path} must exist."
        )
        repo_dir = self.sandbox / "repo"
        impl_commit, proto_commit = self._create_synthetic_git_repo(repo_dir)
        cmd = [
            sys.executable,
            str(script_path),
            "--repo-root",
            str(repo_dir),
            "--artifact-root",
            str(self.sandbox / "external_artifacts"),
            "--run-id",
            "run1",
            "--protocol",
            str(repo_dir / "protocol.json"),
            "--implementation-commit",
            impl_commit,
            "--protocol-commit",
            proto_commit,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        diagnostic = f"{res.stdout}\n{res.stderr}".lower()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("--slice-id", diagnostic)

    def test_71_cli_requires_run_id(self) -> None:
        """71. prepare CLI must require --run-id parameter."""
        script_path = Path("scripts/prepare_agentic_run.py")
        self.assertTrue(
            script_path.exists(), f"CLI script {script_path} must exist."
        )
        repo_dir = self.sandbox / "repo"
        impl_commit, proto_commit = self._create_synthetic_git_repo(repo_dir)
        cmd = [
            sys.executable,
            str(script_path),
            "--repo-root",
            str(repo_dir),
            "--artifact-root",
            str(self.sandbox / "external_artifacts"),
            "--slice-id",
            "slice5b",
            "--protocol",
            str(repo_dir / "protocol.json"),
            "--implementation-commit",
            impl_commit,
            "--protocol-commit",
            proto_commit,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        diagnostic = f"{res.stdout}\n{res.stderr}".lower()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("--run-id", diagnostic)

    def test_72_cli_requires_protocol(self) -> None:
        """72. prepare CLI must require --protocol parameter."""
        script_path = Path("scripts/prepare_agentic_run.py")
        self.assertTrue(
            script_path.exists(), f"CLI script {script_path} must exist."
        )
        repo_dir = self.sandbox / "repo"
        impl_commit, proto_commit = self._create_synthetic_git_repo(repo_dir)
        cmd = [
            sys.executable,
            str(script_path),
            "--repo-root",
            str(repo_dir),
            "--artifact-root",
            str(self.sandbox / "external_artifacts"),
            "--slice-id",
            "slice5b",
            "--run-id",
            "run1",
            "--implementation-commit",
            impl_commit,
            "--protocol-commit",
            proto_commit,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        diagnostic = f"{res.stdout}\n{res.stderr}".lower()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("--protocol", diagnostic)

    def test_73_cli_requires_implementation_commit(self) -> None:
        """73. prepare CLI must require --implementation-commit parameter."""
        script_path = Path("scripts/prepare_agentic_run.py")
        self.assertTrue(
            script_path.exists(), f"CLI script {script_path} must exist."
        )
        repo_dir = self.sandbox / "repo"
        impl_commit, proto_commit = self._create_synthetic_git_repo(repo_dir)
        cmd = [
            sys.executable,
            str(script_path),
            "--repo-root",
            str(repo_dir),
            "--artifact-root",
            str(self.sandbox / "external_artifacts"),
            "--slice-id",
            "slice5b",
            "--run-id",
            "run1",
            "--protocol",
            str(repo_dir / "protocol.json"),
            "--protocol-commit",
            proto_commit,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        diagnostic = f"{res.stdout}\n{res.stderr}".lower()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("--implementation-commit", diagnostic)

    def test_74_cli_requires_protocol_commit(self) -> None:
        """74. prepare CLI must require --protocol-commit parameter."""
        script_path = Path("scripts/prepare_agentic_run.py")
        self.assertTrue(
            script_path.exists(), f"CLI script {script_path} must exist."
        )
        repo_dir = self.sandbox / "repo"
        impl_commit, proto_commit = self._create_synthetic_git_repo(repo_dir)
        cmd = [
            sys.executable,
            str(script_path),
            "--repo-root",
            str(repo_dir),
            "--artifact-root",
            str(self.sandbox / "external_artifacts"),
            "--slice-id",
            "slice5b",
            "--run-id",
            "run1",
            "--protocol",
            str(repo_dir / "protocol.json"),
            "--implementation-commit",
            impl_commit,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        diagnostic = f"{res.stdout}\n{res.stderr}".lower()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("--protocol-commit", diagnostic)

    def test_75_cli_accepts_repeatable_input(self) -> None:
        """75. prepare CLI must accept repeatable --input NAME=PATH arguments."""
        script_path = Path("scripts/prepare_agentic_run.py")
        self.assertTrue(
            script_path.exists(), f"CLI script {script_path} must exist."
        )
        repo_dir = self.sandbox / "repo"
        impl_commit, proto_commit = self._create_synthetic_git_repo(repo_dir)

        input1 = repo_dir / "qrels.json"
        input1.write_text('{"q1": "p1"}', encoding="utf-8")
        input2 = repo_dir / "passages.jsonl"
        input2.write_text('{"id": "p1"}', encoding="utf-8")

        artifact_root = self.sandbox / "external_artifacts"
        slice_id = "slice5b"
        run_id = "r1"

        cmd = [
            sys.executable,
            str(script_path),
            "--repo-root",
            str(repo_dir),
            "--artifact-root",
            str(artifact_root),
            "--slice-id",
            slice_id,
            "--run-id",
            run_id,
            "--protocol",
            str(repo_dir / "protocol.json"),
            "--implementation-commit",
            impl_commit,
            "--protocol-commit",
            proto_commit,
            "--input",
            f"qrels={input1}",
            "--input",
            f"passages={input2}",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        diagnostic = f"stdout: {res.stdout}\nstderr: {res.stderr}"
        self.assertEqual(res.returncode, 0, diagnostic)

        target_run_dir = artifact_root / slice_id / run_id
        self.assertTrue(target_run_dir.exists())
        receipt_file = target_run_dir / "run_receipt.json"
        self.assertTrue(receipt_file.exists())
        rec_data = json.loads(receipt_file.read_text(encoding="utf-8"))
        self.assertEqual(rec_data["run_id"], run_id)
        self.assertIn("qrels", rec_data["input_hashes"])
        self.assertIn("passages", rec_data["input_hashes"])

    def test_76_cli_returns_non_zero_on_invalid_preflight(self) -> None:
        """76. prepare CLI must return non-zero exit code when preflight fails."""
        script_path = Path("scripts/prepare_agentic_run.py")
        self.assertTrue(
            script_path.exists(), f"CLI script {script_path} must exist."
        )
        repo_dir = self.sandbox / "repo"
        impl_commit, proto_commit = self._create_synthetic_git_repo(repo_dir)

        # Preflight fails specifically because protocol file path does not exist
        cmd = [
            sys.executable,
            str(script_path),
            "--repo-root",
            str(repo_dir),
            "--artifact-root",
            str(self.sandbox / "external_artifacts"),
            "--slice-id",
            "slice5b",
            "--run-id",
            "r1",
            "--protocol",
            str(repo_dir / "nonexistent_protocol.json"),
            "--implementation-commit",
            impl_commit,
            "--protocol-commit",
            proto_commit,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        diagnostic = f"{res.stdout}\n{res.stderr}".lower()
        self.assertNotEqual(res.returncode, 0)
        self.assertTrue(
            "protocol" in diagnostic
            or "preflight" in diagnostic
            or "not found" in diagnostic
            or "exist" in diagnostic
        )

    def test_77_cli_creates_no_output_on_failure(self) -> None:
        """77. prepare CLI must create no output directory or receipt files after preflight failure."""
        script_path = Path("scripts/prepare_agentic_run.py")
        self.assertTrue(
            script_path.exists(), f"CLI script {script_path} must exist."
        )
        repo_dir = self.sandbox / "repo"
        impl_commit, proto_commit = self._create_synthetic_git_repo(repo_dir)
        artifact_root = self.sandbox / "external_artifacts"
        slice_id = "slice5b"
        run_id = "r1"
        target_run_dir = artifact_root / slice_id / run_id

        # Non-existent protocol triggers preflight failure
        cmd = [
            sys.executable,
            str(script_path),
            "--repo-root",
            str(repo_dir),
            "--artifact-root",
            str(artifact_root),
            "--slice-id",
            slice_id,
            "--run-id",
            run_id,
            "--protocol",
            str(repo_dir / "nonexistent_protocol.json"),
            "--implementation-commit",
            impl_commit,
            "--protocol-commit",
            proto_commit,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        diagnostic = f"{res.stdout}\n{res.stderr}".lower()
        self.assertNotEqual(res.returncode, 0)
        self.assertTrue(
            "protocol" in diagnostic
            or "preflight" in diagnostic
            or "not found" in diagnostic
            or "exist" in diagnostic
        )
        self.assertFalse(target_run_dir.exists())

    def test_78_verifier_cli_is_read_only(self) -> None:
        """78. verifier CLI must execute in strictly read-only mode without file creation or mutation."""
        script_path = Path("scripts/verify_agentic_run_lineage.py")
        self.assertTrue(
            script_path.exists(), f"CLI script {script_path} must exist."
        )
        artifact_root = self.sandbox / "external_artifacts"
        valid_run_dir = self._create_valid_run_directory(artifact_root)
        files_before, dirs_before = self._get_directory_inventory(valid_run_dir)

        cmd = [
            sys.executable,
            str(script_path),
            "--run-dir",
            str(valid_run_dir),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        diagnostic = f"stdout: {res.stdout}\nstderr: {res.stderr}"
        self.assertEqual(res.returncode, 0, diagnostic)

        files_after, dirs_after = self._get_directory_inventory(valid_run_dir)
        self.assertEqual(files_before, files_after)
        self.assertEqual(dirs_before, dirs_after)

    # =========================================================================
    # ANTI-REGRESSÃO DA IMPLEMENTAÇÃO REPROVADA (Seção 9 - Casos 79 a 90)
    # =========================================================================

    def test_79_anti_regression_reject_tmp_in_allowed_roots(self) -> None:
        """79. Anti-regression: allowed_roots containing /tmp must be explicitly rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            ExperimentalPathPolicy,
            PathPolicyError,
        )

        with self.assertRaises(PathPolicyError):
            ExperimentalPathPolicy(allowed_roots=["/tmp"])  # noqa: S108

    def test_80_anti_regression_reject_benchmarks_in_allowed_roots(
        self,
    ) -> None:
        """80. Anti-regression: allowed_roots containing benchmarks/ must be explicitly rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            ExperimentalPathPolicy,
            PathPolicyError,
        )

        with self.assertRaises(PathPolicyError):
            ExperimentalPathPolicy(allowed_roots=["benchmarks/"])

    def test_81_anti_regression_reject_checkpoints_in_allowed_roots(
        self,
    ) -> None:
        """81. Anti-regression: allowed_roots containing checkpoints/ must be explicitly rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            ExperimentalPathPolicy,
            PathPolicyError,
        )

        with self.assertRaises(PathPolicyError):
            ExperimentalPathPolicy(allowed_roots=["checkpoints/"])

    def test_82_anti_regression_reject_single_mutable_receipt(self) -> None:
        """82. Anti-regression: single mutable receipt model without receipts/ chain is rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            ReceiptStoreError,
            RunReceiptStore,
        )

        store = RunReceiptStore(self.sandbox / "legacy_single_receipt_dir")
        with self.assertRaises(ReceiptStoreError):
            store.load_history_chain()

    def test_83_anti_regression_reject_historical_receipt_overwrite(
        self,
    ) -> None:
        """83. Anti-regression: overwriting historical receipt in receipts/ must fail."""
        self._import_target_module()
        from raglab.agentic.experiments import (
            ReceiptStoreError,
            RunReceiptStore,
        )

        store = RunReceiptStore(self.sandbox / "run_hist_dir")
        with self.assertRaises(ReceiptStoreError):
            store.overwrite_receipt_file("000_PREPARED.json", {})

    def test_84_anti_regression_reject_verifier_promoting_state(
        self,
    ) -> None:
        """84. Anti-regression: verifier promoting state to AUDIT_COMPLETED is rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import LineageVerifier

        verifier = LineageVerifier(read_only=True)
        self.assertTrue(verifier.is_read_only())

    def test_85_anti_regression_reject_missing_previous_receipt_sha(
        self,
    ) -> None:
        """85. Anti-regression: receipt lacking previous_receipt_sha256 is rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import RunReceipt

        with self.assertRaises(ValueError):
            RunReceipt.from_dict({
                "schema_version": 1,
                "run_id": "r1",
                "slice_id": "s1",
                "state": "RUN_STARTED",
                "receipt_sha256": "sha1",
                # missing previous_receipt_sha256
            })

    def test_86_anti_regression_reject_missing_receipt_sha(self) -> None:
        """86. Anti-regression: receipt lacking receipt_sha256 is rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import RunReceipt

        with self.assertRaises(ValueError):
            RunReceipt.from_dict({
                "schema_version": 1,
                "run_id": "r1",
                "slice_id": "s1",
                "state": "PREPARED",
                # missing receipt_sha256
            })

    def test_87_anti_regression_reject_missing_protocol_snapshot(self) -> None:
        """87. Anti-regression: absence of protocol.snapshot.json is rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import LineageVerifier

        verifier = LineageVerifier()
        audit = verifier.verify_lineage(self.sandbox / "no_snapshot_dir")
        self.assertFalse(audit.is_valid)

    def test_88_anti_regression_reject_missing_input_hashes(self) -> None:
        """88. Anti-regression: absence of input_hashes in receipt is rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import RunReceipt

        with self.assertRaises(ValueError):
            RunReceipt.from_dict({
                "schema_version": 1,
                "run_id": "r1",
                "slice_id": "s1",
                "state": "PREPARED",
                # missing input_hashes
            })

    def test_89_anti_regression_reject_missing_exclusive_lock(self) -> None:
        """89. Anti-regression: absence of exclusive lock enforcement is rejected."""
        self._import_target_module()
        from raglab.agentic.experiments import RunController

        controller = RunController(enforce_lock=True)
        self.assertTrue(controller.requires_exclusive_lock())

    def test_90_anti_regression_reject_reduced_cli(self) -> None:
        """90. Anti-regression: CLI reduced to only --run-id/--protocol/--output-dir is rejected."""
        script_path = Path("scripts/prepare_agentic_run.py")
        self.assertTrue(
            script_path.exists(), f"CLI script {script_path} must exist."
        )
        cmd = [
            sys.executable,
            str(script_path),
            "--run-id",
            "r1",
            "--protocol",
            "p.json",
            "--output-dir",
            str(self.sandbox / "out"),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        diagnostic = f"{res.stdout}\n{res.stderr}".lower()
        self.assertNotEqual(res.returncode, 0)
        self.assertTrue(
            "unrecognized" in diagnostic
            or "unknown" in diagnostic
            or "--output-dir" in diagnostic
            or "--repo-root" in diagnostic
        )


if __name__ == "__main__":
    unittest.main()
