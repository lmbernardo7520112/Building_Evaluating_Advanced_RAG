"""Exclusive Experimental Run Lock — RAGLab V7 Experimental Readiness.

Provides an O_EXCL-based exclusive file lock for a single run directory.
The lock prevents concurrent writers from operating on the same run.

Does NOT provide:
- Automatic stale lock recovery (deferred to operational policy)
- RunController lifecycle integration (deferred to GREEN-2C.2+)
- Process-level flock/fcntl advisory locks (uses atomic file creation)

Invariant:
    LOCK_OWNERSHIP_IS_TOKEN_BASED_NOT_PID_ONLY
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import time
import uuid
from pathlib import Path


class RunLockError(Exception):
    """Raised when lock acquisition, release, or validation fails."""


_LOCK_FILENAME = ".run.lock"


class ExperimentalRunLock:
    """Exclusive file-based lock for an experimental run directory.

    Uses ``O_CREAT | O_EXCL | O_WRONLY`` with mode ``0o600`` for atomic creation.
    Ownership is enforced by a random ``owner_token`` (UUID4) and inode identity
    (``st_dev``, ``st_ino``).

    Typical usage::

        lock = ExperimentalRunLock.acquire(run_dir, run_id="my_run")
        try:
            ...  # protected work
        finally:
            lock.release()
    """

    def __init__(
        self,
        lock_path: Path,
        owner_token: str,
        run_id: str,
        st_dev: int = 0,
        st_ino: int = 0,
    ) -> None:
        self._lock_path = lock_path
        self._owner_token = owner_token
        self._run_id = run_id
        self._st_dev = st_dev
        self._st_ino = st_ino
        self._acquired = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def lock_path(self) -> Path:
        """Resolved path to the lock file."""
        return self._lock_path

    @property
    def owner_token(self) -> str:
        """Unique token proving lock ownership."""
        return self._owner_token

    @property
    def run_id(self) -> str:
        """Run ID associated with this lock."""
        return self._run_id

    @property
    def st_dev(self) -> int:
        """Device ID of the lock file at acquisition time."""
        return self._st_dev

    @property
    def st_ino(self) -> int:
        """Inode number of the lock file at acquisition time."""
        return self._st_ino

    def is_acquired(self) -> bool:
        """Return True if this instance currently holds the lock."""
        return self._acquired and self._lock_path.exists()

    def release(self) -> None:
        """Release the lock, removing the lock file.

        Only the holder with matching ``owner_token`` and inode identity
        (``st_dev``, ``st_ino``) can release.
        Raises ``RunLockError`` if the token does not match, inode identity
        differs, or the lock file is absent/malformed.
        """
        if not self._acquired:
            raise RunLockError("Lock was already released by this instance")

        if not self._lock_path.exists():
            self._acquired = False
            raise RunLockError(
                f"Lock file does not exist: {self._lock_path}"
            )

        # Read and verify ownership token (do not leak token in errors)
        current_data = _read_lock_data(self._lock_path)
        current_token = str(current_data.get("owner_token", ""))

        if current_token != self._owner_token:
            raise RunLockError(
                "Cannot release lock: owner_token mismatch."
            )

        # Verify inode and device identity to prevent deleting a successor lock
        try:
            st_current = os.stat(self._lock_path, follow_symlinks=False)
        except OSError as exc:
            self._acquired = False
            raise RunLockError(
                f"Lock file stat failed during release: {exc}"
            ) from exc

        if (
            self._st_dev != 0
            and self._st_ino != 0
            and (
                st_current.st_dev != self._st_dev
                or st_current.st_ino != self._st_ino
            )
        ):
            raise RunLockError(
                "Cannot release lock: inode or device identity mismatch. "
                "Lock path points to a successor lock."
            )

        # Safe to remove — token and inode match
        try:
            os.unlink(self._lock_path)
        except OSError as exc:
            raise RunLockError(
                f"Failed to remove lock file: {exc}"
            ) from exc

        self._acquired = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def acquire(
        cls,
        run_dir: Path | str,
        run_id: str,
    ) -> ExperimentalRunLock:
        """Acquire an exclusive lock on the given run directory.

        Creates ``<run_dir>/.run.lock`` atomically using ``O_EXCL`` with mode ``0o600``.
        If the lock file already exists, raises ``RunLockError``.

        Returns an ``ExperimentalRunLock`` instance on success.
        """
        resolved_dir = Path(run_dir).resolve()
        resolved_dir.mkdir(parents=True, exist_ok=True)
        lock_path = resolved_dir / _LOCK_FILENAME

        owner_token = uuid.uuid4().hex

        lock_data = {
            "run_id": run_id,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "owner_token": owner_token,
            "acquired_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
        }
        lock_bytes = json.dumps(
            lock_data, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

        # Atomic creation with O_EXCL and 0o600 permissions
        try:
            fd = os.open(
                str(lock_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            with contextlib.suppress(OSError):
                os.fchmod(fd, 0o600)
            st = os.fstat(fd)
            st_dev = st.st_dev
            st_ino = st.st_ino
        except FileExistsError:
            # Lock already held — read existing lock info for diagnostics
            try:
                existing = _read_lock_data(lock_path)
                holder_info = (
                    f"held by pid={existing.get('pid')}, "
                    f"host='{existing.get('hostname')}', "
                    f"since {existing.get('acquired_at_utc', '?')}"
                )
            except (RunLockError, OSError):
                holder_info = "lock file exists but is unreadable"

            raise RunLockError(
                f"Cannot acquire lock on '{resolved_dir}': "
                f"lock already exists ({holder_info})"
            ) from None

        # Write lock data with flush + fsync
        try:
            os.write(fd, lock_bytes)
            os.fsync(fd)
        except BaseException:
            # Clean up on failure
            with contextlib.suppress(OSError):
                os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(lock_path)
            raise
        finally:
            with contextlib.suppress(OSError):
                os.close(fd)

        return cls(
            lock_path=lock_path,
            owner_token=owner_token,
            run_id=run_id,
            st_dev=st_dev,
            st_ino=st_ino,
        )

    @classmethod
    def read_existing(cls, run_dir: Path | str) -> dict[str, object]:
        """Read the existing lock file data without acquiring.

        Returns the parsed lock data dictionary.
        Raises ``RunLockError`` if no lock exists or it is malformed.
        """
        resolved_dir = Path(run_dir).resolve()
        lock_path = resolved_dir / _LOCK_FILENAME
        if not lock_path.exists():
            raise RunLockError(
                f"No lock file found in '{resolved_dir}'"
            )
        return _read_lock_data(lock_path)


def _read_lock_data(lock_path: Path) -> dict[str, object]:
    """Read and parse lock file JSON. Fail-closed on malformed data."""
    try:
        raw = lock_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RunLockError(
            f"Cannot read lock file '{lock_path}': {exc}"
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RunLockError(
            f"Malformed lock file '{lock_path}': {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise RunLockError(
            f"Lock file content is not a JSON object: {lock_path}"
        )

    required = ("run_id", "pid", "hostname", "owner_token")
    for field in required:
        if field not in data:
            raise RunLockError(
                f"Lock file missing required field '{field}': {lock_path}"
            )

    return data
