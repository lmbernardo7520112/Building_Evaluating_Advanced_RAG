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

import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

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
    ) -> None:
        if allowed_roots is not None:
            self._allowed_roots: list[Path | str] | None = list(allowed_roots)
        else:
            self._allowed_roots = None
        self._enforce_lock = enforce_lock
        self._active_locks: dict[Path, ExperimentalRunLock] = {}

    # ------------------------------------------------------------------
    # Public Query Methods
    # ------------------------------------------------------------------

    def requires_exclusive_lock(self) -> bool:
        """Return True if this controller enforces exclusive run locking."""
        return self._enforce_lock

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

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
