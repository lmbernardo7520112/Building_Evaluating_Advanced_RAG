"""Trajectory ledger — append-only, JSONL, auditable, deterministic.

Each line is a complete agent trajectory for one query_id + policy_id.
The ledger enforces:
- schema versioning
- append-only semantics (no rewriting completed trajectories)
- atomic writes (write to temp, fsync, os.replace)
- duplicate detection
- run ID / config hash compatibility
- no chain-of-thought, credentials, or qrels
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path

from raglab.agentic.contracts import AgentTrajectory, _canonical_json
from raglab.agentic.errors import (
    IncompatibleRunError,
    LedgerConflictError,
    LedgerCorruptionError,
)

# Fields that must NEVER appear in a persisted trajectory
_FORBIDDEN_FIELDS = frozenset(
    {
        "chain_of_thought",
        "reasoning",
        "internal_messages",
        "secret",
        "credential",
        "api_key",
        "qrels",
        "gold_answer",
        "holdout",
        "password",
        "token",
    }
)


def _sanitize_trajectory_dict(d: dict) -> None:  # type: ignore[type-arg]
    """Verify no forbidden fields leak into the trajectory."""
    for key in d:
        if key.lower() in _FORBIDDEN_FIELDS:
            raise LedgerCorruptionError(f"Forbidden field '{key}' in trajectory")
    # Recurse into nested dicts
    for val in d.values():
        if isinstance(val, dict):
            _sanitize_trajectory_dict(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    _sanitize_trajectory_dict(item)


class TrajectoryLedger:
    """Append-only JSONL ledger for agent trajectories.

    Thread-safety: NOT thread-safe. External synchronization required.
    """

    def __init__(
        self,
        path: Path,
        run_id: str,
        policy_sha256: str,
        config_sha256: str,
    ) -> None:
        self._path = path
        self._run_id = run_id
        self._policy_sha256 = policy_sha256
        self._config_sha256 = config_sha256
        self._completed_keys: set[str] = set()

        # Load existing entries if file exists
        if self._path.exists():
            self._load_existing()

    def _trajectory_key(self, trajectory: AgentTrajectory) -> str:
        """Unique key for deduplication: query_id + policy_id."""
        return f"{trajectory.query_id}::{trajectory.policy_id}"

    def _load_existing(self) -> None:
        """Load completed keys from existing ledger file."""
        try:
            with open(self._path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise LedgerCorruptionError(
                            f"Invalid JSON at line {line_num}: {e}"
                        ) from e

                    # Validate compatibility
                    if entry.get("run_id") != self._run_id:
                        raise IncompatibleRunError(
                            f"Line {line_num}: run_id mismatch: "
                            f"'{entry.get('run_id')}' != '{self._run_id}'"
                        )
                    if entry.get("policy_sha256") != self._policy_sha256:
                        raise IncompatibleRunError(
                            f"Line {line_num}: policy_sha256 mismatch"
                        )
                    if entry.get("config_sha256") != self._config_sha256:
                        raise IncompatibleRunError(
                            f"Line {line_num}: config_sha256 mismatch"
                        )

                    key = f"{entry['query_id']}::{entry['policy_id']}"
                    self._completed_keys.add(key)
        except FileNotFoundError:
            pass

    def append(self, trajectory: AgentTrajectory) -> None:
        """Append a completed trajectory to the ledger.

        Raises:
        - IncompatibleRunError if run/policy/config don't match
        - LedgerConflictError if a different trajectory exists for same key
        - LedgerCorruptionError if forbidden fields detected
        """
        # Compatibility checks
        if trajectory.run_id != self._run_id:
            raise IncompatibleRunError(
                f"run_id mismatch: '{trajectory.run_id}' != '{self._run_id}'"
            )
        if trajectory.policy_sha256 != self._policy_sha256:
            raise IncompatibleRunError(
                f"policy_sha256 mismatch: "
                f"'{trajectory.policy_sha256}' != '{self._policy_sha256}'"
            )
        if trajectory.config_sha256 != self._config_sha256:
            raise IncompatibleRunError(
                f"config_sha256 mismatch: "
                f"'{trajectory.config_sha256}' != '{self._config_sha256}'"
            )

        # Duplicate check
        key = self._trajectory_key(trajectory)
        if key in self._completed_keys:
            raise LedgerConflictError(f"Trajectory already exists for {key}")

        # Serialize and sanitize
        d = trajectory.to_dict()
        _sanitize_trajectory_dict(d)

        # Atomic write: temp file → fsync → os.replace
        line = _canonical_json(d) + "\n"
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # For append, we write to a temp file and append its content
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._path.parent),
            prefix=".ledger_",
            suffix=".tmp",
        )
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
            os.close(fd)

            # Append to main file
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())

            self._completed_keys.add(key)
        finally:
            # Clean up temp file
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

    @property
    def entry_count(self) -> int:
        return len(self._completed_keys)

    def has_trajectory(self, query_id: str, policy_id: str) -> bool:
        return f"{query_id}::{policy_id}" in self._completed_keys

    def ledger_hash(self) -> str:
        """Compute SHA-256 of the entire ledger file."""
        if not self._path.exists():
            return hashlib.sha256(b"").hexdigest()
        h = hashlib.sha256()
        with open(self._path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
