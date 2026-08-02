"""Tests for FilesystemCheckpointStore.

Covers: atomic write, corruption detection, resume, fingerprint
mismatch, path traversal, temporary directory isolation.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from raglab.domain.entities import Checkpoint
from raglab.domain.enums import QuestionState
from raglab.domain.value_objects import IntegrityDigest, RunId
from raglab.infrastructure.persistence.checkpoint_store import (
    CheckpointCorruptionError,
    FilesystemCheckpointStore,
    PathTraversalError,
)

_VALID_HASH = "a" * 64
_OTHER_HASH = "b" * 64


def _make_checkpoint(
    run_id: str = "test-run-1",
    corpus_fp: str = _VALID_HASH,
    config_fp: str = _OTHER_HASH,
    completed: frozenset[str] | None = None,
    states: dict[str, QuestionState] | None = None,
) -> Checkpoint:
    return Checkpoint(
        run_id=RunId(run_id),
        corpus_fingerprint=IntegrityDigest(corpus_fp),
        config_fingerprint=IntegrityDigest(config_fp),
        completed_query_ids=completed or frozenset(),
        question_states=states or {},
    )


class TestFilesystemCheckpointStore(unittest.TestCase):
    """Test checkpoint store with temporary directory."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.store = FilesystemCheckpointStore(self.tmpdir)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load(self) -> None:
        """Save and load produces equivalent checkpoint."""
        cp = _make_checkpoint()
        self.store.save(cp)

        loaded = self.store.load(RunId("test-run-1"))
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.run_id.value, "test-run-1")
        self.assertEqual(
            loaded.corpus_fingerprint.hex_digest, _VALID_HASH
        )
        self.assertEqual(
            loaded.config_fingerprint.hex_digest, _OTHER_HASH
        )

    def test_exists(self) -> None:
        """exists() returns True after save."""
        cp = _make_checkpoint()
        self.assertFalse(self.store.exists(RunId("test-run-1")))
        self.store.save(cp)
        self.assertTrue(self.store.exists(RunId("test-run-1")))

    def test_load_nonexistent_returns_none(self) -> None:
        """Loading nonexistent checkpoint returns None."""
        result = self.store.load(RunId("nonexistent"))
        self.assertIsNone(result)

    def test_completed_queries_preserved(self) -> None:
        """Completed query IDs are preserved through save/load."""
        cp = _make_checkpoint(
            completed=frozenset({"q1", "q2"}),
            states={"q1": QuestionState.COMPLETE, "q2": QuestionState.COMPLETE},
        )
        self.store.save(cp)
        loaded = self.store.load(RunId("test-run-1"))
        self.assertEqual(loaded.completed_query_ids, frozenset({"q1", "q2"}))
        self.assertEqual(
            loaded.question_states["q1"], QuestionState.COMPLETE
        )

    def test_idempotent_resume(self) -> None:
        """Saving same checkpoint twice is idempotent."""
        cp = _make_checkpoint()
        self.store.save(cp)
        self.store.save(cp)  # No error
        loaded = self.store.load(RunId("test-run-1"))
        self.assertIsNotNone(loaded)

    def test_update_checkpoint(self) -> None:
        """Updating a checkpoint with new completed queries."""
        cp1 = _make_checkpoint(completed=frozenset({"q1"}))
        self.store.save(cp1)

        cp2 = _make_checkpoint(completed=frozenset({"q1", "q2"}))
        self.store.save(cp2)

        loaded = self.store.load(RunId("test-run-1"))
        self.assertEqual(
            loaded.completed_query_ids, frozenset({"q1", "q2"})
        )

    def test_corruption_detected(self) -> None:
        """Corrupted checkpoint file raises CheckpointCorruptionError."""
        cp = _make_checkpoint()
        self.store.save(cp)

        # Corrupt the file
        path = os.path.join(self.tmpdir, "test-run-1.checkpoint.json")
        with open(path) as f:
            content = json.load(f)
        content["integrity_sha256"] = "0" * 64
        with open(path, "w") as f:
            json.dump(content, f)

        with self.assertRaises(CheckpointCorruptionError):
            self.store.load(RunId("test-run-1"))

    def test_invalid_json_detected(self) -> None:
        """Invalid JSON raises CheckpointCorruptionError."""
        path = os.path.join(self.tmpdir, "bad-json.checkpoint.json")
        with open(path, "w") as f:
            f.write("not valid json {{{")

        store = FilesystemCheckpointStore(self.tmpdir)
        with self.assertRaises(CheckpointCorruptionError):
            store.load(RunId("bad-json"))

    def test_path_traversal_blocked(self) -> None:
        """Path traversal in run_id is blocked."""
        with self.assertRaises(PathTraversalError):
            self.store.save(_make_checkpoint(run_id="../etc/passwd"))

    def test_path_traversal_dots(self) -> None:
        """Dotted paths are blocked."""
        with self.assertRaises(PathTraversalError):
            self.store.load(RunId("../../secret"))

    def test_path_traversal_slashes(self) -> None:
        """Slash paths are blocked."""
        with self.assertRaises(PathTraversalError):
            self.store.load(RunId("foo/bar"))

    def test_fingerprint_mismatch_detected_by_domain(self) -> None:
        """Fingerprint mismatch is detectable through domain entity."""
        cp = _make_checkpoint(corpus_fp=_VALID_HASH)
        self.store.save(cp)
        loaded = self.store.load(RunId("test-run-1"))

        # Domain entity check
        self.assertTrue(
            loaded.is_compatible(
                IntegrityDigest(_VALID_HASH),
                IntegrityDigest(_OTHER_HASH),
            )
        )
        self.assertFalse(
            loaded.is_compatible(
                IntegrityDigest(_OTHER_HASH),  # wrong corpus
                IntegrityDigest(_OTHER_HASH),
            )
        )

    def test_atomic_write_creates_file(self) -> None:
        """Checkpoint file is created atomically (no partial writes visible)."""
        cp = _make_checkpoint()
        self.store.save(cp)
        path = os.path.join(self.tmpdir, "test-run-1.checkpoint.json")
        self.assertTrue(os.path.exists(path))
        # Verify it's valid JSON
        with open(path) as f:
            data = json.load(f)
        self.assertIn("integrity_sha256", data)
        self.assertIn("data", data)

    def test_schema_version_present(self) -> None:
        """Checkpoint file contains schema version."""
        cp = _make_checkpoint()
        self.store.save(cp)
        path = os.path.join(self.tmpdir, "test-run-1.checkpoint.json")
        with open(path) as f:
            envelope = json.load(f)
        self.assertEqual(envelope["data"]["schema_version"], "3.0")

    def test_no_temp_files_left(self) -> None:
        """No temporary files left after save."""
        cp = _make_checkpoint()
        self.store.save(cp)
        files = os.listdir(self.tmpdir)
        temp_files = [f for f in files if f.startswith(".checkpoint_")]
        self.assertEqual(len(temp_files), 0)


if __name__ == "__main__":
    unittest.main()
