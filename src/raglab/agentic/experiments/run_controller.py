"""Run Lifecycle Controller Core — RAGLab V7 Experimental Readiness.

Governs experimental run directory path validation, exclusive locking,
and state transitions over append-only RunReceipt stores.

Invariants:
    NON_EMPTY_ALLOWLIST_ENFORCED
    PATH_VALIDATED_BEFORE_IO
    LOCK_REQUIRED_BEFORE_APPEND
    PERSISTED_STATE_AUTHORITATIVE
"""

from __future__ import annotations

import hashlib
import hmac
import shutil
import subprocess
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from raglab.agentic.experiments.integrity import compute_file_sha256
from raglab.agentic.experiments.path_policy import (
    ExperimentalPathPolicy,
    PathPolicyError,
)
from raglab.agentic.experiments.receipt_store import RunReceiptStore
from raglab.agentic.experiments.receipts import (
    ReceiptStoreError,
    RunReceipt,
    RunState,
    validate_state_transition,
)
from raglab.agentic.experiments.run_lock import (
    ExperimentalRunLock,
    RunLockError,
)


class RunControllerError(Exception):
    """Raised when run controller lifecycle, lock, or transition operations fail."""


class RunController:
    """Governs execution lock and state transitions for experimental run directories.

    Path validation is enforced via ``ExperimentalPathPolicy``.
    Concurrency protection is enforced via ``ExperimentalRunLock``.
    State persistence is delegated to ``RunReceiptStore``.
    """

    def __init__(
        self,
        allowed_roots: tuple[Path | str, ...] | list[Path | str] | None = None,
        enforce_lock: bool = True,
        repo_root: Path | str | None = None,
    ) -> None:
        if allowed_roots is not None:
            self._allowed_roots: list[Path | str] | None = list(allowed_roots)
        else:
            self._allowed_roots = None
        self._enforce_lock = enforce_lock
        self._active_locks: dict[Path, ExperimentalRunLock] = {}
        if repo_root is not None:
            self._repo_root: Path | None = Path(repo_root).resolve()
        else:
            self._repo_root = None

    # ------------------------------------------------------------------
    # Public Query Methods
    # ------------------------------------------------------------------

    def requires_exclusive_lock(self) -> bool:
        """Return True if this controller enforces exclusive run locking."""
        return self._enforce_lock

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_repo_root(self) -> Path:
        """Resolve the Git repository root directory.

        Uses the explicit ``repo_root`` if set, otherwise detects via
        ``git rev-parse --show-toplevel``.
        """
        if self._repo_root is not None:
            return self._repo_root
        git_bin = shutil.which("git") or "git"
        try:
            result = subprocess.run(  # noqa: S603, S607
                [git_bin, "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=True,
            )
            return Path(result.stdout.strip()).resolve()
        except (subprocess.CalledProcessError, OSError) as exc:
            raise RunControllerError(
                f"Cannot determine Git repository root: {exc}"
            ) from exc

    def _run_git(
        self, *args: str, repo_dir: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run a Git command in the repository directory.

        Uses list args, never shell=True. Returns CompletedProcess.
        Raises RunControllerError on failure.
        """
        cwd = repo_dir if repo_dir is not None else self._get_repo_root()
        git_bin = shutil.which("git") or "git"
        cmd = [git_bin, *args]
        try:
            return subprocess.run(  # noqa: S603, S607
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise RunControllerError(
                f"Git command failed: {cmd}: {exc}"
            ) from exc

    def _get_path_policy(self) -> ExperimentalPathPolicy:
        """Construct ExperimentalPathPolicy or raise error if missing."""
        if not self._allowed_roots:
            raise RunControllerError(
                "allowed_roots must be specified and non-empty for RunController"
            )
        try:
            return ExperimentalPathPolicy(allowed_roots=self._allowed_roots)
        except PathPolicyError as exc:
            raise RunControllerError(
                f"Path policy initialization failed: {exc}"
            ) from exc

    def _validate_run_dir(self, run_dir: Path | str) -> Path:
        """Validate run_dir against path policy fail-closed before any I/O."""
        policy = self._get_path_policy()
        try:
            return policy.validate_target_directory(run_dir)
        except PathPolicyError as exc:
            raise RunControllerError(
                f"Path policy validation failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Lock Management
    # ------------------------------------------------------------------

    def acquire_run_lock(self, run_dir: Path | str) -> ExperimentalRunLock:
        """Acquire exclusive lock on run_dir after path policy validation."""
        validated_dir = self._validate_run_dir(run_dir)

        if (
            validated_dir in self._active_locks
            and self._active_locks[validated_dir].is_acquired()
        ):
            return self._active_locks[validated_dir]

        run_id = validated_dir.name
        store = RunReceiptStore(validated_dir)
        if store.receipts_dir.exists():
            try:
                latest = store.load_latest(validated_dir)
                run_id = latest.run_id
            except ReceiptStoreError:
                pass

        try:
            lock = ExperimentalRunLock.acquire(validated_dir, run_id=run_id)
        except RunLockError as exc:
            raise RunControllerError(
                f"Failed to acquire run lock on '{validated_dir}': {exc}"
            ) from exc

        self._active_locks[validated_dir] = lock
        return lock

    @contextmanager
    def run_context(
        self, run_dir: Path | str
    ) -> Generator[ExperimentalRunLock, None, None]:
        """Context manager acquiring run lock on entry and releasing on clean exit.

        If an exception occurs within the block, the lock is PRESERVED (not released).
        """
        validated_dir = self._validate_run_dir(run_dir)
        lock = self.acquire_run_lock(validated_dir)
        try:
            yield lock
        except Exception:
            # Preserve lock on exception! Do not release.
            raise
        else:
            # Clean exit: release lock
            lock.release()
            self._active_locks.pop(validated_dir, None)

    # ------------------------------------------------------------------
    # State Transitions
    # ------------------------------------------------------------------

    def transition_state(
        self,
        run_dir: Path | str,
        target_state: RunState | str,
        lock: ExperimentalRunLock | None = None,
    ) -> RunReceipt:
        """Transition execution state on an existing receipt chain.

        1. Validates run_dir via ExperimentalPathPolicy.
        2. Loads history chain from RunReceiptStore.
        3. Validates transition from current state via validate_state_transition.
        4. Enforces lock acquisition if enforce_lock=True.
        5. Computes and appends new receipt.
        6. On append failure, performs 1x post-append reconciliation.
        """
        validated_dir = self._validate_run_dir(run_dir)

        target_state_enum = (
            RunState(target_state)
            if isinstance(target_state, str)
            else target_state
        )

        store = RunReceiptStore(validated_dir)
        try:
            chain = store.load_history_chain()
        except ReceiptStoreError as exc:
            raise RunControllerError(
                f"Cannot transition state: failed chain load '{validated_dir}': {exc}"
            ) from exc

        latest = chain[-1]
        prev_state = (
            RunState(latest.state)
            if isinstance(latest.state, str)
            else latest.state
        )

        try:
            validate_state_transition(prev_state, target_state_enum)
        except (ValueError, TypeError, KeyError, ReceiptStoreError) as exc:
            raise RunControllerError(
                f"Invalid state transition '{prev_state.value}' -> "
                f"'{target_state_enum.value}': {exc}"
            ) from exc

        active_lock: ExperimentalRunLock | None = None
        created_lock_internally = False

        if self._enforce_lock:
            if lock is not None:
                if not lock.is_acquired():
                    raise RunControllerError(
                        "Provided run lock is not acquired"
                    )
                # Bind supplied lock to target run directory
                lock_dir = lock.lock_path.parent.resolve()
                if lock_dir != validated_dir:
                    raise RunControllerError(
                        f"Supplied lock belongs to '{lock_dir}', "
                        f"but target run_dir is '{validated_dir}'"
                    )
                # Bind supplied lock run_id to chain's run_id
                if chain and lock.run_id != latest.run_id:
                    raise RunControllerError(
                        f"Supplied lock run_id '{lock.run_id}' does not match "
                        f"chain run_id '{latest.run_id}'"
                    )
                active_lock = lock
            elif (
                validated_dir in self._active_locks
                and self._active_locks[validated_dir].is_acquired()
            ):
                active_lock = self._active_locks[validated_dir]
            else:
                active_lock = self.acquire_run_lock(validated_dir)
                created_lock_internally = True

        now_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        started_at = (
            now_utc
            if target_state_enum == RunState.RUN_STARTED
            else latest.started_at_utc
        )
        finished_at = (
            now_utc
            if target_state_enum
            in (
                RunState.RUN_COMPLETED,
                RunState.AUDIT_COMPLETED,
                RunState.FAILED,
            )
            else None
        )
        exit_code = (
            0
            if target_state_enum
            in (RunState.RUN_COMPLETED, RunState.AUDIT_COMPLETED)
            else (1 if target_state_enum == RunState.FAILED else None)
        )

        draft = RunReceipt(
            schema_version=latest.schema_version,
            run_id=latest.run_id,
            slice_id=latest.slice_id,
            state=target_state_enum,
            artifact_root=latest.artifact_root,
            run_directory=latest.run_directory,
            implementation_commit=latest.implementation_commit,
            protocol_commit=latest.protocol_commit,
            protocol_sha256=latest.protocol_sha256,
            input_hashes=dict(latest.input_hashes),
            runner_version=latest.runner_version,
            created_at_utc=now_utc,
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            exit_code=exit_code,
            artifact_inventory=tuple(latest.artifact_inventory),
            artifact_hashes=dict(latest.artifact_hashes),
            previous_receipt_sha256=latest.receipt_sha256,
            receipt_sha256="",
        )
        computed_hash = draft.compute_hash()
        new_receipt = RunReceipt(
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
            started_at_utc=draft.started_at_utc,
            finished_at_utc=draft.finished_at_utc,
            exit_code=draft.exit_code,
            artifact_inventory=draft.artifact_inventory,
            artifact_hashes=draft.artifact_hashes,
            previous_receipt_sha256=draft.previous_receipt_sha256,
            receipt_sha256=computed_hash,
        )

        try:
            store.append(new_receipt)
        except Exception as exc:
            # Post-append reconciliation: read chain exactly 1x
            try:
                reconciled_chain = store.load_history_chain()
                if (
                    reconciled_chain
                    and reconciled_chain[-1].receipt_sha256
                    == new_receipt.receipt_sha256
                ):
                    # Commit succeeded despite post-append notification error
                    result_receipt = reconciled_chain[-1]
                elif (
                    reconciled_chain
                    and reconciled_chain[-1].receipt_sha256
                    == latest.receipt_sha256
                ):
                    # Append failed before write -> propagate failure
                    raise RunControllerError(
                        f"Receipt append failed: {exc}"
                    ) from exc
                else:
                    # Chain in unexpected state -> preserve lock and fail closed
                    raise RunControllerError(
                        f"Receipt append failure uncertain state: {exc}"
                    ) from exc
            except ReceiptStoreError as r_exc:
                # Chain corrupted or unreadable -> preserve lock and fail closed
                raise RunControllerError(
                    f"Receipt append failure unreadable chain: {r_exc}"
                ) from exc
        else:
            result_receipt = store.load_latest(validated_dir)

        if created_lock_internally and active_lock is not None:
            active_lock.release()
            self._active_locks.pop(validated_dir, None)

        return result_receipt

    def start_run(
        self,
        run_dir: Path | str,
        lock: ExperimentalRunLock | None = None,
    ) -> RunReceipt:
        """Transition state PREPARED -> RUN_STARTED for the given run directory."""
        return self.transition_state(
            run_dir, RunState.RUN_STARTED, lock=lock
        )

    def _safe_release_lock(
        self, run_dir: Path, lock: ExperimentalRunLock
    ) -> None:
        """Safely release run lock. Converts RunLockError to RunControllerError."""
        try:
            if lock.is_acquired():
                lock.release()
        except RunLockError as exc:
            raise RunControllerError(
                f"Failed to release run lock: {exc}"
            ) from exc
        finally:
            self._active_locks.pop(run_dir, None)

    def prepare_run(
        self,
        *,
        run_id: str,
        slice_id: str,
        protocol_path: Path | str,
        artifact_root: Path | str | None = None,
        inputs: dict[str, str | Path] | None = None,
        implementation_commit: str | None = None,
        protocol_commit: str | None = None,
        metadata: dict[str, Any] | None = None,
        runner_version: str = "raglab-v7",
        allow_dirty: bool = False,
        allow_staged: bool = False,
    ) -> RunReceipt:
        """Prepare a new experimental run directory layout and persist PREPARED receipt.

        Executes preflight checks, resolves path policy against allowlist,
        acquires lock, initializes layout, snapshots protocol, and appends
        000_PREPARED receipt.
        """
        # 1. PathPolicy & allowlist check FIRST (Fail-closed before Git / IO)
        policy = self._get_path_policy()

        if artifact_root is not None:
            root_path = Path(artifact_root).resolve()
        elif self._allowed_roots:
            root_path = Path(self._allowed_roots[0]).resolve()
        else:
            raise RunControllerError(
                "allowed_roots must be specified and non-empty for RunController"
            )

        try:
            run_dir = policy.derive_run_directory(root_path, slice_id, run_id)
            policy.validate_target_directory(run_dir, must_be_empty=True)
        except PathPolicyError as exc:
            raise RunControllerError(
                f"Path policy validation failed: {exc}"
            ) from exc

        # 2. Rejection of pre-existing run_dir BEFORE lock acquisition
        if run_dir.exists():
            raise RunControllerError(
                f"Run directory already exists: '{run_dir}'"
            )

        # 3. Preflight Git / Protocol / Inputs
        self.preflight_git_check(allow_dirty=allow_dirty, allow_staged=allow_staged)

        head_result = self._run_git("rev-parse", "HEAD")
        if head_result.returncode != 0:
            raise RunControllerError("Cannot determine HEAD commit")
        head_commit = head_result.stdout.strip()

        impl_commit = implementation_commit or head_commit
        proto_commit = protocol_commit or head_commit

        self.validate_commits(
            implementation_commit=impl_commit,
            protocol_commit=proto_commit,
        )
        self.validate_commit_ancestry(
            impl_commit=impl_commit,
            proto_commit=proto_commit,
        )
        self.validate_protocol_in_head_ancestry(proto_commit)

        proto_file = Path(protocol_path).resolve()
        self.validate_protocol_tracked(proto_file)
        self.validate_protocol_unmodified(str(proto_file))

        if inputs:
            self.validate_inputs(inputs)

        # 4. Metadata security check
        if metadata is not None:
            try:
                RunReceipt.validate_metadata_security(metadata)
            except ValueError as exc:
                raise RunControllerError(f"Security validation failed: {exc}") from exc

        # 5. Exclusive lock acquisition
        try:
            lock = ExperimentalRunLock.acquire(run_dir, run_id=run_id)
        except RunLockError as exc:
            raise RunControllerError(
                f"Failed to acquire run lock on '{run_dir}': {exc}"
            ) from exc

        self._active_locks[run_dir] = lock

        # 6. Layout creation, snapshot, receipt creation & append
        try:
            proto_sha256 = compute_file_sha256(proto_file)
            input_hashes: dict[str, str] = {}
            if inputs:
                for k, v in inputs.items():
                    input_hashes[k] = compute_file_sha256(v)

            (run_dir / "receipts").mkdir(parents=True, exist_ok=True)
            (run_dir / "raw").mkdir(parents=True, exist_ok=True)
            (run_dir / "derived").mkdir(parents=True, exist_ok=True)
            (run_dir / "logs").mkdir(parents=True, exist_ok=True)
            (run_dir / "hashes.sha256").touch()

            snapshot_path = run_dir / "protocol.snapshot.json"
            shutil.copyfile(proto_file, snapshot_path)

            now_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            draft = RunReceipt(
                schema_version=1,
                run_id=run_id,
                slice_id=slice_id,
                state=RunState.PREPARED,
                artifact_root=str(root_path),
                run_directory=str(run_dir),
                implementation_commit=impl_commit,
                protocol_commit=proto_commit,
                protocol_sha256=proto_sha256,
                input_hashes=input_hashes,
                runner_version=runner_version,
                created_at_utc=now_utc,
                previous_receipt_sha256=None,
                receipt_sha256="",
            )
            computed_sha = draft.compute_hash()
            new_receipt = RunReceipt(
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
                started_at_utc=draft.started_at_utc,
                finished_at_utc=draft.finished_at_utc,
                exit_code=draft.exit_code,
                artifact_inventory=draft.artifact_inventory,
                artifact_hashes=draft.artifact_hashes,
                previous_receipt_sha256=draft.previous_receipt_sha256,
                receipt_sha256=computed_sha,
            )

            store = RunReceiptStore(run_dir)

            append_exc: Exception | None = None
            try:
                store.append(new_receipt)
            except Exception as exc:
                append_exc = exc

            # 7. Post-append reconciliation & state classification
            try:
                chain = store.load_history_chain()
            except ReceiptStoreError as r_exc:
                receipts_dir = run_dir / "receipts"
                json_files = (
                    list(receipts_dir.glob("*.json"))
                    if receipts_dir.exists()
                    else []
                )
                if json_files:
                    # PREPARE_RUN_STATE_UNCERTAIN -> PRESERVE LOCK
                    raise RunControllerError(
                        "PREPARE_RUN_STATE_UNCERTAIN: Receipt store unreadable "
                        f"after write: {r_exc}"
                    ) from (append_exc or r_exc)
                else:
                    # FALHA CONFIRMADA ANTES DA PUBLICAÇÃO -> RELEASE LOCK
                    self._safe_release_lock(run_dir, lock)
                    raise RunControllerError(
                        "Receipt append failed before publication: "
                        f"{append_exc or r_exc}"
                    ) from (append_exc or r_exc)

            if chain and chain[-1].receipt_sha256 == computed_sha:
                # SUCESSO CONFIRMADO -> RELEASE LOCK & RETURN RECEIPT
                self._safe_release_lock(run_dir, lock)
                return chain[-1]
            elif not chain:
                # FALHA CONFIRMADA ANTES DA PUBLICAÇÃO -> RELEASE LOCK
                if append_exc is not None:
                    self._safe_release_lock(run_dir, lock)
                    raise RunControllerError(
                        f"Receipt append failed before publication: {append_exc}"
                    ) from append_exc
                else:
                    raise RunControllerError(
                        "PREPARE_RUN_STATE_UNCERTAIN: Receipt chain is empty "
                        "after append"
                    )
            else:
                # Unexpected/divergent receipt -> UNCERTAIN -> PRESERVE LOCK
                raise RunControllerError(
                    "PREPARE_RUN_STATE_UNCERTAIN: Receipt hash mismatch or "
                    f"divergent chain. Expected '{computed_sha}', "
                    f"got '{chain[-1].receipt_sha256}'"
                )
        except Exception as exc:
            if "PREPARE_RUN_STATE_UNCERTAIN" in str(exc):
                # PRESERVE LOCK under state uncertainty!
                if not isinstance(exc, RunControllerError):
                    raise RunControllerError(str(exc)) from exc
                raise
            # For all other setup/write errors before publication, release lock safely
            self._safe_release_lock(run_dir, lock)
            if not isinstance(exc, RunControllerError):
                raise RunControllerError(f"prepare_run failed: {exc}") from exc
            raise

    # ------------------------------------------------------------------
    # Read-Only Preflight Validations
    # ------------------------------------------------------------------

    def preflight_git_check(
        self,
        *,
        allow_dirty: bool = False,
        allow_staged: bool = False,
    ) -> None:
        """Verify Git worktree cleanliness. Read-only — no mutations.

        Raises ``RunControllerError`` if:
        - ``allow_dirty=False`` and tracked files have unstaged modifications.
        - ``allow_staged=False`` and files are staged in the index.
        """
        if not allow_dirty:
            result = self._run_git("diff", "--name-only")
            if result.returncode != 0:
                raise RunControllerError(
                    f"git diff failed (exit {result.returncode}): "
                    f"{result.stderr.strip()}"
                )
            if result.stdout.strip():
                raise RunControllerError(
                    "Git worktree has tracked dirty files: "
                    f"{result.stdout.strip()}"
                )

        if not allow_staged:
            result = self._run_git("diff", "--cached", "--name-only")
            if result.returncode != 0:
                raise RunControllerError(
                    f"git diff --cached failed (exit {result.returncode}): "
                    f"{result.stderr.strip()}"
                )
            if result.stdout.strip():
                raise RunControllerError(
                    "Git index has staged files: "
                    f"{result.stdout.strip()}"
                )

    def validate_commits(
        self,
        *,
        implementation_commit: str,
        protocol_commit: str,
    ) -> None:
        """Verify that both commit hashes exist in the repository. Read-only.

        Raises ``RunControllerError`` if either commit does not exist
        or is not a valid commit object.
        """
        for label, sha in [
            ("implementation_commit", implementation_commit),
            ("protocol_commit", protocol_commit),
        ]:
            result = self._run_git("cat-file", "-t", sha)
            if result.returncode != 0:
                raise RunControllerError(
                    f"{label} '{sha}' does not exist in the repository"
                )
            obj_type = result.stdout.strip()
            if obj_type != "commit":
                raise RunControllerError(
                    f"{label} '{sha}' is not a commit object "
                    f"(type: {obj_type})"
                )

    def validate_commit_ancestry(
        self,
        *,
        impl_commit: str,
        proto_commit: str,
    ) -> None:
        """Verify impl_commit is an ancestor of proto_commit. Read-only.

        Uses ``git merge-base --is-ancestor``.
        Raises ``RunControllerError`` if not.
        """
        result = self._run_git(
            "merge-base", "--is-ancestor", impl_commit, proto_commit
        )
        if result.returncode != 0:
            raise RunControllerError(
                f"Implementation commit '{impl_commit}' is not an ancestor "
                f"of protocol commit '{proto_commit}'"
            )

    def validate_protocol_in_head_ancestry(
        self,
        protocol_commit: str,
    ) -> None:
        """Verify protocol_commit is an ancestor of HEAD. Read-only.

        Uses ``git merge-base --is-ancestor``.
        Raises ``RunControllerError`` if not.
        """
        result = self._run_git(
            "merge-base", "--is-ancestor", protocol_commit, "HEAD"
        )
        if result.returncode != 0:
            raise RunControllerError(
                f"Protocol commit '{protocol_commit}' is not in HEAD ancestry"
            )

    def validate_protocol_tracked(
        self,
        protocol_path: Path | str,
    ) -> None:
        """Verify protocol file is tracked by Git. Read-only.

        The path must be inside the repository. Untracked files are rejected.
        Raises ``RunControllerError`` on failure.
        """
        repo_root = self._get_repo_root()
        proto = Path(protocol_path).resolve()

        # Reject files outside the repository
        try:
            rel_path = proto.relative_to(repo_root)
        except ValueError:
            raise RunControllerError(
                f"Protocol file '{proto}' is outside the repository "
                f"'{repo_root}'"
            ) from None

        result = self._run_git("ls-files", "--error-unmatch", str(rel_path))
        if result.returncode != 0:
            raise RunControllerError(
                f"Protocol file '{rel_path}' is not tracked by Git"
            )

    def validate_protocol_unmodified(
        self,
        protocol_path: str,
    ) -> None:
        """Verify protocol file has no unstaged or staged modifications. Read-only.

        Checks both worktree modifications and staging area.
        Raises ``RunControllerError`` if modified.
        """
        # Check worktree modifications
        result = self._run_git("diff", "--name-only", "--", protocol_path)
        if result.returncode != 0:
            raise RunControllerError(
                f"git diff failed for protocol '{protocol_path}': "
                f"{result.stderr.strip()}"
            )
        if result.stdout.strip():
            raise RunControllerError(
                f"Protocol file '{protocol_path}' has unstaged modifications"
            )

        # Check staged modifications
        result = self._run_git(
            "diff", "--cached", "--name-only", "--", protocol_path
        )
        if result.returncode != 0:
            raise RunControllerError(
                f"git diff --cached failed for protocol '{protocol_path}': "
                f"{result.stderr.strip()}"
            )
        if result.stdout.strip():
            raise RunControllerError(
                f"Protocol file '{protocol_path}' has staged modifications"
            )

    def validate_protocol_sha(
        self,
        *,
        protocol_path: str,
        expected_sha: str,
    ) -> None:
        """Verify protocol file SHA-256 matches expected value. Read-only.

        Reads the file bytes directly and computes SHA-256.
        Uses constant-time comparison via ``hmac.compare_digest``.
        Raises ``RunControllerError`` on mismatch or missing file.
        """
        repo_root = self._get_repo_root()
        proto_file = (repo_root / protocol_path).resolve()

        if not proto_file.exists():
            raise RunControllerError(
                f"Protocol file not found: {proto_file}"
            )
        if proto_file.is_dir():
            raise RunControllerError(
                f"Protocol path is a directory: {proto_file}"
            )

        h = hashlib.sha256()
        with proto_file.open("rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        actual_sha = h.hexdigest()

        if not hmac.compare_digest(actual_sha, expected_sha):
            raise RunControllerError(
                f"Protocol SHA-256 mismatch for '{protocol_path}': "
                f"actual '{actual_sha}' != expected '{expected_sha}'"
            )

    def validate_inputs(
        self,
        input_paths: dict[str, str | Path],
    ) -> None:
        """Verify all input files exist. Read-only.

        Raises ``RunControllerError`` if any input file is missing.
        """
        for label, path_str in input_paths.items():
            p = Path(path_str).resolve()
            if not p.exists():
                raise RunControllerError(
                    f"Input file '{label}' not found: {p}"
                )
            if not p.is_file():
                raise RunControllerError(
                    f"Input path '{label}' is not a file: {p}"
                )
