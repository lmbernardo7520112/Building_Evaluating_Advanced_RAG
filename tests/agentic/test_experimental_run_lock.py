"""Focal Unit Test Suite — ExperimentalRunLock (GREEN-2C.1).

Tests exclusive O_EXCL lock acquisition, owner_token enforcement,
release semantics, multiprocess contention, and malformed lock handling.
"""

from __future__ import annotations

import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path

from raglab.agentic.experiments import (
    ExperimentalRunLock,
    RunLockError,
)


class TestExperimentalRunLockFocal(unittest.TestCase):
    """Focal test suite for exclusive experimental run lock."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.sandbox = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # 1. First acquisition succeeds
    def test_01_first_acquisition(self) -> None:
        run_dir = self.sandbox / "run1"
        lock = ExperimentalRunLock.acquire(run_dir, run_id="r1")
        self.assertTrue(lock.is_acquired())
        self.assertTrue(lock.lock_path.exists())
        self.assertEqual(lock.run_id, "r1")
        self.assertTrue(len(lock.owner_token) > 0)

    # 2. Second acquisition rejected
    def test_02_second_acquisition_rejected(self) -> None:
        run_dir = self.sandbox / "run2"
        _lock1 = ExperimentalRunLock.acquire(run_dir, run_id="r1")
        with self.assertRaises(RunLockError) as ctx:
            ExperimentalRunLock.acquire(run_dir, run_id="r1")
        self.assertIn("already exists", str(ctx.exception))

    # 3. Two independent processes competing for the same lock
    def test_03_multiprocess_contention(self) -> None:
        run_dir = self.sandbox / "run3"
        run_dir.mkdir(parents=True, exist_ok=True)

        def _try_acquire(result_queue: multiprocessing.Queue) -> None:  # type: ignore[type-arg]
            """Child process attempts to acquire the lock."""
            try:
                lock = ExperimentalRunLock.acquire(run_dir, run_id="r1")
                result_queue.put(("acquired", lock.owner_token))
            except RunLockError as e:
                result_queue.put(("rejected", str(e)))
            except Exception as e:
                result_queue.put(("error", str(e)))

        q: multiprocessing.Queue = multiprocessing.Queue()  # type: ignore[type-arg]

        # Parent acquires first
        parent_lock = ExperimentalRunLock.acquire(run_dir, run_id="r1")
        self.assertTrue(parent_lock.is_acquired())

        # Child process tries to acquire the same lock
        child = multiprocessing.Process(target=_try_acquire, args=(q,))
        child.start()
        child.join(timeout=10)

        self.assertFalse(q.empty(), "Child process should have reported a result")
        status, detail = q.get()
        self.assertEqual(status, "rejected", f"Expected 'rejected', got '{status}': {detail}")

    # 4. Release by owner succeeds
    def test_04_release_by_owner(self) -> None:
        run_dir = self.sandbox / "run4"
        lock = ExperimentalRunLock.acquire(run_dir, run_id="r1")
        lock_path = lock.lock_path
        self.assertTrue(lock_path.exists())
        lock.release()
        self.assertFalse(lock_path.exists())
        self.assertFalse(lock.is_acquired())

    # 5. Release by non-owner rejected
    def test_05_release_by_non_owner_rejected(self) -> None:
        run_dir = self.sandbox / "run5"
        lock = ExperimentalRunLock.acquire(run_dir, run_id="r1")

        # Create a fake lock instance with wrong token
        fake_lock = ExperimentalRunLock(
            lock_path=lock.lock_path,
            owner_token="wrong_token_fake",  # noqa: S106
            run_id="r1",
        )
        with self.assertRaises(RunLockError) as ctx:
            fake_lock.release()
        self.assertIn("mismatch", str(ctx.exception))
        # Original lock file must still exist
        self.assertTrue(lock.lock_path.exists())

    # 6. Incorrect owner_token cannot release
    def test_06_incorrect_owner_token(self) -> None:
        run_dir = self.sandbox / "run6"
        lock = ExperimentalRunLock.acquire(run_dir, run_id="r1")

        impostor = ExperimentalRunLock(
            lock_path=lock.lock_path,
            owner_token="0" * 32,
            run_id="r1",
        )
        with self.assertRaises(RunLockError):
            impostor.release()
        # Lock still held
        self.assertTrue(lock.is_acquired())

    # 7. Malformed lock file
    def test_07_malformed_lock_file(self) -> None:
        run_dir = self.sandbox / "run7"
        run_dir.mkdir(parents=True)
        lock_path = run_dir / ".run.lock"
        lock_path.write_text("{not valid json!!!", encoding="utf-8")

        # Acquisition should fail because file exists
        with self.assertRaises(RunLockError):
            ExperimentalRunLock.acquire(run_dir, run_id="r1")

        # Reading existing should also fail
        with self.assertRaises(RunLockError):
            ExperimentalRunLock.read_existing(run_dir)

    # 8. Lock preserved after rejection
    def test_08_lock_preserved_after_rejection(self) -> None:
        run_dir = self.sandbox / "run8"
        lock = ExperimentalRunLock.acquire(run_dir, run_id="r1")
        lock_bytes_before = lock.lock_path.read_bytes()

        # Second attempt fails
        with self.assertRaises(RunLockError):
            ExperimentalRunLock.acquire(run_dir, run_id="r2")

        lock_bytes_after = lock.lock_path.read_bytes()
        self.assertEqual(lock_bytes_before, lock_bytes_after)

    # 9. Lock file contains required fields
    def test_09_lock_file_contents(self) -> None:
        run_dir = self.sandbox / "run9"
        lock = ExperimentalRunLock.acquire(run_dir, run_id="r1")
        data = json.loads(lock.lock_path.read_text(encoding="utf-8"))
        self.assertEqual(data["run_id"], "r1")
        self.assertIn("pid", data)
        self.assertIn("hostname", data)
        self.assertIn("owner_token", data)
        self.assertIn("acquired_at_utc", data)
        self.assertEqual(data["owner_token"], lock.owner_token)

    # 10. Re-acquire after release succeeds
    def test_10_reacquire_after_release(self) -> None:
        run_dir = self.sandbox / "run10"
        lock1 = ExperimentalRunLock.acquire(run_dir, run_id="r1")
        lock1.release()
        lock2 = ExperimentalRunLock.acquire(run_dir, run_id="r1")
        self.assertTrue(lock2.is_acquired())
        self.assertNotEqual(lock1.owner_token, lock2.owner_token)

    # 11. Release of already-released lock raises
    def test_11_double_release_raises(self) -> None:
        run_dir = self.sandbox / "run11"
        lock = ExperimentalRunLock.acquire(run_dir, run_id="r1")
        lock.release()
        with self.assertRaises(RunLockError):
            lock.release()

    # 12. read_existing returns correct data
    def test_12_read_existing(self) -> None:
        run_dir = self.sandbox / "run12"
        lock = ExperimentalRunLock.acquire(run_dir, run_id="r1")
        data = ExperimentalRunLock.read_existing(run_dir)
        self.assertEqual(data["run_id"], "r1")
        self.assertEqual(data["owner_token"], lock.owner_token)

    # 13. read_existing on missing lock raises
    def test_13_read_existing_missing(self) -> None:
        run_dir = self.sandbox / "run13"
        run_dir.mkdir(parents=True)
        with self.assertRaises(RunLockError):
            ExperimentalRunLock.read_existing(run_dir)

    # 14. Lock uses O_EXCL (verified by inspecting source)
    def test_14_o_excl_in_source(self) -> None:
        import inspect

        source = inspect.getsource(ExperimentalRunLock.acquire)
        self.assertIn("O_EXCL", source)
        self.assertIn("O_CREAT", source)


if __name__ == "__main__":
    unittest.main()
