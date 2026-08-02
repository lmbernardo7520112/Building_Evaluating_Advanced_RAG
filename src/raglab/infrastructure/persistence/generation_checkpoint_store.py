import hashlib
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "slice4_v3"


class GenerationCheckpointStore:
    """Tracks completed (query_id, strategy) pairs and full result rows for idempotent resumption.

    File format (one per run):
    {
      "schema": "slice4_v3",
      "run_id": "...",
      "sha256": "...",
      "completed": {
        "q_dev_01::F0_baseline": {
          "query_id": "q_dev_01",
          "strategy": "F0_baseline",
          "abstained": false,
          "citation_count": 2,
          "timestamp_utc": "2026-07-31T...",
          "result_row": { ... }
        },
        ...
      }
    }
    """

    def __init__(self, run_id: str, store_dir: Path) -> None:
        self._run_id = run_id
        self._path = store_dir / f"slice4_gen_checkpoint_{run_id}.json"
        self._store_dir = store_dir
        self._data: dict[str, dict[str, Any]] = {}
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if not self._path.exists():
            return
        try:
            raw_text = self._path.read_text(encoding="utf-8")
            raw = json.loads(raw_text)
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

            completed = raw.get("completed", {})
            expected_sha = raw.get("sha256")
            if expected_sha:
                check_bytes = json.dumps(
                    {
                        "schema": _SCHEMA_VERSION,
                        "run_id": self._run_id,
                        "completed": completed,
                    },
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
                actual_sha = hashlib.sha256(check_bytes).hexdigest()
                if actual_sha != expected_sha:
                    raise ValueError(
                        f"CHECKPOINT_CORRUPTED: SHA-256 mismatch ({actual_sha} != {expected_sha})"
                    )

            self._data = completed
            logger.info(
                "Loaded %d completed entries from %s",
                len(self._data), self._path,
            )
        except ValueError:
            raise
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load checkpoint %s: %s", self._path, exc)

    def is_completed(self, query_id: str, strategy: str) -> bool:
        """Return True if this (query_id, strategy) pair is in checkpoint."""
        key = f"{query_id}::{strategy}"
        return key in self._data

    def has_complete_result_row(self, query_id: str, strategy: str) -> bool:
        """Return True if this pair has a valid full result row with evaluation metrics."""
        key = f"{query_id}::{strategy}"
        if key not in self._data:
            return False
        entry = self._data[key]
        result_row = entry.get("result_row")
        if not isinstance(result_row, dict):
            return False
        eval_data = result_row.get("evaluation")
        return eval_data is not None and isinstance(eval_data, dict)

    def get_complete_result_row(self, query_id: str, strategy: str) -> dict[str, Any] | None:
        """Return full result row if complete, else None."""
        if self.has_complete_result_row(query_id, strategy):
            key = f"{query_id}::{strategy}"
            return self._data[key].get("result_row")
        return None

    def mark_completed(
        self,
        query_id: str,
        strategy: str,
        *,
        abstained: bool,
        citation_count: int,
    ) -> None:
        """Record a completion marker (legacy behavior)."""
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

    def mark_complete_row(
        self,
        query_id: str,
        strategy: str,
        result_entry: dict[str, Any],
    ) -> None:
        """Atomically persist a complete result row into checkpoint."""
        key = f"{query_id}::{strategy}"
        self._data[key] = {
            "query_id": query_id,
            "strategy": strategy,
            "abstained": result_entry.get("abstained", False),
            "citation_count": len(result_entry.get("citation_pages", [])),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "result_row": result_entry,
        }
        self._save()
        logger.debug("Checkpoint: persisted complete_result_row for %s", key)

    def rehydrate_complete_rows(self) -> dict[str, list[dict[str, Any]]]:
        """Rehydrate all complete result rows grouped by strategy."""
        rehydrated: dict[str, list[dict[str, Any]]] = {}
        for key, entry in sorted(self._data.items()):
            row = entry.get("result_row")
            if isinstance(row, dict) and row.get("evaluation") is not None:
                strat = str(entry.get("strategy") or row.get("strategy"))
                rehydrated.setdefault(strat, []).append(row)
        return rehydrated

    def merge_partial_artifact(self, partial_artifact_path: Path) -> int:
        """Merge complete result rows from a partial results JSON into this checkpoint store."""
        if not partial_artifact_path.exists():
            return 0
        raw = json.loads(partial_artifact_path.read_text(encoding="utf-8"))
        results = raw.get("results", {})
        merged_count = 0
        for strategy, rows in results.items():
            for row in rows:
                qid = row.get("qid")
                if not qid:
                    continue
                if row.get("evaluation") is not None and row.get("answer") is not None:
                    key = f"{qid}::{strategy}"
                    self._data[key] = {
                        "query_id": qid,
                        "strategy": strategy,
                        "abstained": row.get("abstained", False),
                        "citation_count": len(row.get("citation_pages", [])),
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                        "result_row": row,
                    }
                    merged_count += 1
        if merged_count > 0:
            self._save()
        return merged_count

    def completed_count(self) -> int:
        """Return number of completed (query_id, strategy) pairs."""
        return len(self._data)

    def complete_rows_count(self) -> int:
        """Return number of complete result rows."""
        return sum(1 for k in self._data if self.has_complete_result_row(*k.split("::", 1)))

    def completed_pairs(self) -> list[str]:
        """Return list of completed key strings for inspection."""
        return sorted(self._data.keys())

    def _save(self) -> None:
        """Atomically write checkpoint to disk using temp file + fsync + os.replace."""
        self._store_dir.mkdir(parents=True, exist_ok=True)
        check_bytes = json.dumps(
            {
                "schema": _SCHEMA_VERSION,
                "run_id": self._run_id,
                "completed": self._data,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        digest = hashlib.sha256(check_bytes).hexdigest()

        payload_obj = {
            "schema": _SCHEMA_VERSION,
            "run_id": self._run_id,
            "sha256": digest,
            "completed": self._data,
        }
        final_bytes = json.dumps(payload_obj, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")

        fd, tmp_path = tempfile.mkstemp(
            dir=self._store_dir, prefix=".ckpt_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(final_bytes)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._path)
        except OSError:
            import contextlib
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
