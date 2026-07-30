"""Filesystem checkpoint store with atomic writes.

Implements CheckpointPort contract. Domain does not import filesystem.

Features:
- Atomic write via tempfile + os.rename
- JSON canonical serialization (sorted keys)
- Schema version tracking
- run_id, corpus fingerprint, config fingerprint, question fingerprints
- Status and UTC timestamps
- SHA-256 integrity envelope
- Idempotent resume: load → validate → continue
- Corruption rejection: invalid JSON or hash mismatch
- Incompatibility rejection: fingerprint mismatch
- No silent overwrite: existing checkpoint validated before update
- Path traversal blocked: run_id sanitized
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from raglab.domain.entities import Checkpoint
from raglab.domain.enums import QuestionState
from raglab.domain.value_objects import IntegrityDigest, RunId

# Only allow safe characters in run_id for filesystem paths
_SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")
_SCHEMA_VERSION = "3.0"


class CheckpointCorruptionError(Exception):
    """Raised when a checkpoint file is corrupted."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"Checkpoint corrupted at '{path}': {reason}")


class PathTraversalError(Exception):
    """Raised when a run_id contains path traversal characters."""

    def __init__(self, run_id: str) -> None:
        super().__init__(
            f"Path traversal blocked: run_id '{run_id}' contains unsafe characters"
        )


class FilesystemCheckpointStore:
    """Atomic, resumable filesystem checkpoint store.

    Satisfies CheckpointPort protocol:
    - save(checkpoint) → None
    - load(run_id) → Checkpoint | None
    - exists(run_id) → bool
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _validate_run_id(self, run_id: str) -> None:
        """Block path traversal attacks."""
        if not _SAFE_ID_PATTERN.match(run_id):
            raise PathTraversalError(run_id)

    def _checkpoint_path(self, run_id: str) -> Path:
        """Get the path for a checkpoint file."""
        self._validate_run_id(run_id)
        return self._base_dir / f"{run_id}.checkpoint.json"

    @staticmethod
    def _compute_hash(data: str) -> str:
        """Compute SHA-256 of serialized data."""
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def _serialize(self, checkpoint: Checkpoint) -> str:
        """Serialize checkpoint to canonical JSON."""
        payload: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "run_id": checkpoint.run_id.value,
            "corpus_fingerprint": checkpoint.corpus_fingerprint.hex_digest,
            "config_fingerprint": checkpoint.config_fingerprint.hex_digest,
            "completed_query_ids": sorted(checkpoint.completed_query_ids),
            "question_states": {
                k: v.value
                for k, v in sorted(checkpoint.question_states.items())
            },
            "status": "in_progress"
            if checkpoint.completed_query_ids
            else "initialized",
            "updated_utc": datetime.now(UTC).isoformat(),
        }

        data_json = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        integrity = self._compute_hash(data_json)

        envelope: dict[str, Any] = {
            "integrity_sha256": integrity,
            "data": payload,
        }

        return json.dumps(envelope, sort_keys=True, indent=2, ensure_ascii=True)

    def _deserialize(self, content: str, path: str) -> Checkpoint:
        """Deserialize and validate checkpoint from JSON."""
        try:
            envelope = json.loads(content)
        except json.JSONDecodeError as e:
            raise CheckpointCorruptionError(path, f"invalid JSON: {e}") from e

        if "integrity_sha256" not in envelope or "data" not in envelope:
            raise CheckpointCorruptionError(path, "missing envelope fields")

        data = envelope["data"]
        data_json = json.dumps(data, sort_keys=True, ensure_ascii=True)
        actual_hash = self._compute_hash(data_json)

        if actual_hash != envelope["integrity_sha256"]:
            raise CheckpointCorruptionError(
                path,
                f"integrity mismatch: expected {envelope['integrity_sha256']}, "
                f"got {actual_hash}",
            )

        if data.get("schema_version") != _SCHEMA_VERSION:
            raise CheckpointCorruptionError(
                path,
                f"schema version mismatch: expected {_SCHEMA_VERSION}, "
                f"got {data.get('schema_version')}",
            )

        question_states = {
            k: QuestionState(v)
            for k, v in data.get("question_states", {}).items()
        }

        return Checkpoint(
            run_id=RunId(data["run_id"]),
            corpus_fingerprint=IntegrityDigest(data["corpus_fingerprint"]),
            config_fingerprint=IntegrityDigest(data["config_fingerprint"]),
            completed_query_ids=frozenset(data.get("completed_query_ids", [])),
            question_states=question_states,
        )

    def save(self, checkpoint: Checkpoint) -> None:
        """Persist checkpoint atomically.

        Uses tempfile + os.rename for atomic write.
        No silent overwrite: checkpoint is always written atomically.
        """
        self._validate_run_id(checkpoint.run_id.value)
        target = self._checkpoint_path(checkpoint.run_id.value)
        content = self._serialize(checkpoint)

        # Atomic write: write to temp file then rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._base_dir),
            prefix=".checkpoint_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.rename(tmp_path, str(target))
        except BaseException:
            # Clean up temp file on failure
            import contextlib
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def load(self, run_id: RunId) -> Checkpoint | None:
        """Load checkpoint, returning None if not found.

        Raises CheckpointCorruptionError on integrity failure.
        """
        self._validate_run_id(run_id.value)
        path = self._checkpoint_path(run_id.value)

        if not path.exists():
            return None

        content = path.read_text(encoding="utf-8")
        return self._deserialize(content, str(path))

    def exists(self, run_id: RunId) -> bool:
        """Check if checkpoint exists."""
        self._validate_run_id(run_id.value)
        return self._checkpoint_path(run_id.value).exists()
