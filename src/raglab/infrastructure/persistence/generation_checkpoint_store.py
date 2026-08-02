"""Generation checkpoint store — idempotent per (query_id, strategy) pair.

This store tracks which questions have already been processed during
the RAG Slice 4 benchmark, enabling safe resumption after interruption.

Design:
  - One JSON file per run_id
  - Atomic writes (tempfile + rename)
  - Sanitized: no credentials, no raw LLM responses
  - Records: query_id, strategy, abstained, citation count, timestamp
  - Does NOT store: API keys, full answer text, headers, HTTP responses

SECURITY:
  - Sanitize before writing: use sanitize_answer_for_artifact()
  - Key names are query_id × strategy tuples
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "slice4_v2"


class GenerationCheckpointStore:
    """Tracks completed (query_id, strategy) pairs for idempotent resumption.

    File format (one per run):
    {
      "schema": "slice4_v2",
      "run_id": "...",
      "completed": {
        "q_dev_01::F0_baseline": {
          "query_id": "q_dev_01",
          "strategy": "F0_baseline",
          "abstained": false,
          "citation_count": 2,
          "timestamp_utc": "2026-07-31T..."
        },
        ...
      }
    }
    """

    def __init__(self, run_id: str, store_dir: Path) -> None:
        self._run_id = run_id
        self._path = store_dir / f"slice4_gen_checkpoint_{run_id}.json"
        self._store_dir = store_dir
        self._data: dict[str, dict[str, object]] = {}
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            schema = raw.get("schema") or raw.get("artifact_schema_version")
            if schema != _SCHEMA_VERSION:
                raise ValueError(
                    f"INCOMPATIBLE_CHECKPOINT_SCHEMA: found '{schema}', expected '{_SCHEMA_VERSION}'"
                )
            if raw.get("run_id") != self._run_id:
                logger.warning(
                    "Checkpoint run_id mismatch: %s != %s — ignoring",
                    raw.get("run_id"), self._run_id,
                )
                return
            self._data = raw.get("completed", {})
            logger.info(
                "Loaded %d completed entries from %s",
                len(self._data), self._path,
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load checkpoint %s: %s", self._path, exc)

    def is_completed(self, query_id: str, strategy: str) -> bool:
        """Return True if this (query_id, strategy) pair is already done."""
        key = f"{query_id}::{strategy}"
        return key in self._data

    def mark_completed(
        self,
        query_id: str,
        strategy: str,
        *,
        abstained: bool,
        citation_count: int,
    ) -> None:
        """Record that a (query_id, strategy) pair has been processed."""
        key = f"{query_id}::{strategy}"
        self._data[key] = {
            "query_id": query_id,
            "strategy": strategy,
            "abstained": abstained,
            "citation_count": citation_count,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        self._save()
        logger.debug("Checkpoint: marked %s as completed", key)

    def completed_count(self) -> int:
        """Return number of completed (query_id, strategy) pairs."""
        return len(self._data)

    def completed_pairs(self) -> list[str]:
        """Return list of completed key strings for inspection."""
        return sorted(self._data.keys())

    def _save(self) -> None:
        """Atomically write checkpoint to disk."""
        self._store_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "schema": _SCHEMA_VERSION,
                "run_id": self._run_id,
                "completed": self._data,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        # Atomic write
        fd, tmp_path = tempfile.mkstemp(
            dir=self._store_dir, prefix=".ckpt_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_path, self._path)
        except OSError:
            import contextlib
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
