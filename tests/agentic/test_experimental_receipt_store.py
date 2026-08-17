"""Focal Unit Test Suite — RunReceiptStore (GREEN-2B).

Tests append-only, atomic receipt persistence with chain verification,
deterministic serialization, overwrite prevention, and immutable returns.
Uses tmp_path as hermetic test infrastructure only.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from raglab.agentic.experiments import (
    ReceiptStoreError,
    RunReceipt,
    RunReceiptStore,
    RunState,
    compute_canonical_json_bytes,
)


def _make_receipt(
    *,
    state: RunState = RunState.PREPARED,
    run_id: str = "run_001",
    slice_id: str = "slice_5b",
    previous_sha: str | None = None,
    artifact_root: str = "/srv/artifacts",
    run_directory: str = "/srv/artifacts/slice_5b/run_001",
) -> RunReceipt:
    """Build a minimal valid RunReceipt with correct hash."""
    draft = RunReceipt(
        schema_version=1,
        run_id=run_id,
        slice_id=slice_id,
        state=state,
        artifact_root=artifact_root,
        run_directory=run_directory,
        implementation_commit="a" * 64,
        protocol_commit="b" * 64,
        protocol_sha256="c" * 64,
        input_hashes={"query": "d" * 64},
        runner_version="raglab-v7-test",
        created_at_utc="2026-08-10T00:00:00Z",
        previous_receipt_sha256=previous_sha,
        receipt_sha256="",
    )
    computed_hash = draft.compute_hash()
    return RunReceipt(
        schema_version=draft.schema_version,
        run_id=draft.run_id,
        slice_id=draft.slice_id,
        state=draft.state,
        artifact_root=draft.artifact_root,
        run_directory=draft.run_directory,
        implementation_commit=draft.implementation_commit,
        protocol_commit=draft.protocol_commit,
        protocol_sha256=draft.protocol_sha256,
        input_hashes=dict(draft.input_hashes),
        runner_version=draft.runner_version,
        created_at_utc=draft.created_at_utc,
        previous_receipt_sha256=draft.previous_receipt_sha256,
        receipt_sha256=computed_hash,
    )


class TestRunReceiptStoreFocal(unittest.TestCase):
    """Focal test suite for RunReceiptStore append-only persistence."""

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.sandbox = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # 1. First valid receipt
    def test_01_first_receipt_appended(self) -> None:
        run_dir = self.sandbox / "run1"
        store = RunReceiptStore(run_dir)
        r0 = _make_receipt()
        path = store.append(r0)
        self.assertTrue(path.exists())
        self.assertEqual(path.name, "000_PREPARED.json")

    # 2. Second receipt chained
    def test_02_second_receipt_chained(self) -> None:
        run_dir = self.sandbox / "run2"
        store = RunReceiptStore(run_dir)
        r0 = _make_receipt()
        store.append(r0)
        r1 = _make_receipt(
            state=RunState.RUN_STARTED,
            previous_sha=r0.receipt_sha256,
        )
        path = store.append(r1)
        self.assertEqual(path.name, "001_RUN_STARTED.json")

    # 3. Read and round-trip
    def test_03_read_round_trip(self) -> None:
        run_dir = self.sandbox / "run3"
        store = RunReceiptStore(run_dir)
        r0 = _make_receipt()
        store.append(r0)
        chain = store.load_history_chain()
        self.assertEqual(len(chain), 1)
        self.assertEqual(chain[0].receipt_sha256, r0.receipt_sha256)
        self.assertEqual(chain[0].run_id, r0.run_id)

    # 4. Determinism of bytes
    def test_04_byte_determinism(self) -> None:
        run_dir = self.sandbox / "run4"
        store = RunReceiptStore(run_dir)
        r0 = _make_receipt()
        path = store.append(r0)
        raw_bytes = path.read_bytes()
        expected_bytes = compute_canonical_json_bytes(r0.to_dict())
        self.assertEqual(raw_bytes, expected_bytes)

    # 5. Overwrite attempt rejected
    def test_05_overwrite_rejected(self) -> None:
        run_dir = self.sandbox / "run5"
        store = RunReceiptStore(run_dir)
        with self.assertRaises(ReceiptStoreError):
            store.overwrite_receipt_file("000_PREPARED.json", {})

    # 6. Duplicate receipt at same index rejected
    def test_06_duplicate_index_rejected(self) -> None:
        run_dir = self.sandbox / "run6"
        store = RunReceiptStore(run_dir)
        r0 = _make_receipt()
        store.append(r0)
        # Try appending same PREPARED receipt again
        with self.assertRaises(ReceiptStoreError):
            store.append(r0)

    # 7. Incorrect predecessor rejected
    def test_07_wrong_predecessor_rejected(self) -> None:
        run_dir = self.sandbox / "run7"
        store = RunReceiptStore(run_dir)
        r0 = _make_receipt()
        store.append(r0)
        r1_bad = _make_receipt(
            state=RunState.RUN_STARTED,
            previous_sha="f" * 64,  # wrong predecessor
        )
        with self.assertRaises(ReceiptStoreError):
            store.append(r1_bad)

    # 8. Tampered hash detected
    def test_08_tampered_hash_detected(self) -> None:
        run_dir = self.sandbox / "run8"
        store = RunReceiptStore(run_dir)
        r0 = _make_receipt()
        path = store.append(r0)
        # Tamper with the file on disk
        data = json.loads(path.read_text(encoding="utf-8"))
        data["receipt_sha256"] = "a" * 64
        path.write_text(
            json.dumps(data, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        with self.assertRaises(ReceiptStoreError):
            store.verify_chain_integrity()

    # 9. Invalid JSON rejected
    def test_09_invalid_json_rejected(self) -> None:
        run_dir = self.sandbox / "run9"
        receipts_dir = run_dir / "receipts"
        receipts_dir.mkdir(parents=True)
        (receipts_dir / "000_PREPARED.json").write_text(
            "{invalid json!!!}", encoding="utf-8"
        )
        store = RunReceiptStore(run_dir)
        with self.assertRaises(ReceiptStoreError):
            store.load_history_chain()

    # 10. Temp file not treated as receipt
    def test_10_temp_file_ignored(self) -> None:
        run_dir = self.sandbox / "run10"
        store = RunReceiptStore(run_dir)
        r0 = _make_receipt()
        store.append(r0)
        # Create a .tmp sibling — must be ignored
        (store.receipts_dir / "001_RUN_STARTED.json.tmp").write_text(
            "{}", encoding="utf-8"
        )
        chain = store.load_history_chain()
        self.assertEqual(len(chain), 1)

    # 11. Failed write leaves no partial file
    def test_11_failed_write_no_partial(self) -> None:
        target = self.sandbox / "partial_test" / "run_receipt.json"
        with self.assertRaises(RuntimeError):
            RunReceiptStore.write_atomic_failing_simulation(target)
        self.assertFalse(target.exists())

    # 12. Empty chain raises
    def test_12_empty_chain_raises(self) -> None:
        run_dir = self.sandbox / "run12"
        (run_dir / "receipts").mkdir(parents=True)
        store = RunReceiptStore(run_dir)
        with self.assertRaises(ReceiptStoreError):
            store.load_history_chain()

    # 13. Return is immutable tuple
    def test_13_return_immutable(self) -> None:
        run_dir = self.sandbox / "run13"
        store = RunReceiptStore(run_dir)
        r0 = _make_receipt()
        store.append(r0)
        chain = store.load_history_chain()
        self.assertIsInstance(chain, tuple)

    # 14. Re-open store sees persisted data
    def test_14_reopen_store(self) -> None:
        run_dir = self.sandbox / "run14"
        store1 = RunReceiptStore(run_dir)
        r0 = _make_receipt()
        store1.append(r0)
        # Open a fresh store instance
        store2 = RunReceiptStore(run_dir)
        chain = store2.load_history_chain()
        self.assertEqual(len(chain), 1)
        self.assertEqual(chain[0].receipt_sha256, r0.receipt_sha256)

    # 15. Reading does not mutate files
    def test_15_read_no_mutation(self) -> None:
        run_dir = self.sandbox / "run15"
        store = RunReceiptStore(run_dir)
        r0 = _make_receipt()
        path = store.append(r0)
        bytes_before = path.read_bytes()
        mtime_before = os.path.getmtime(path)
        _ = store.load_history_chain()
        _ = store.verify_chain_integrity()
        bytes_after = path.read_bytes()
        mtime_after = os.path.getmtime(path)
        self.assertEqual(bytes_before, bytes_after)
        self.assertEqual(mtime_before, mtime_after)

    # 16. Missing receipts dir raises
    def test_16_missing_receipts_dir_raises(self) -> None:
        run_dir = self.sandbox / "run16_no_dir"
        store = RunReceiptStore(run_dir)
        with self.assertRaises(ReceiptStoreError):
            store.load_history_chain()

    # 17. supports_fsync_atomic_write returns True
    def test_17_supports_fsync(self) -> None:
        store = RunReceiptStore(self.sandbox / "run17")
        self.assertTrue(store.supports_fsync_atomic_write())

    # 18. load_receipt_file works
    def test_18_load_receipt_file(self) -> None:
        run_dir = self.sandbox / "run18"
        store = RunReceiptStore(run_dir)
        r0 = _make_receipt()
        path = store.append(r0)
        loaded = RunReceiptStore.load_receipt_file(path)
        self.assertEqual(loaded.receipt_sha256, r0.receipt_sha256)

    # 19. load_latest works
    def test_19_load_latest(self) -> None:
        run_dir = self.sandbox / "run19"
        store = RunReceiptStore(run_dir)
        r0 = _make_receipt()
        store.append(r0)
        r1 = _make_receipt(
            state=RunState.RUN_STARTED,
            previous_sha=r0.receipt_sha256,
        )
        store.append(r1)
        latest = RunReceiptStore.load_latest(run_dir)
        self.assertEqual(latest.state, RunState.RUN_STARTED)

    # 20. Non-contiguous sequence detected
    def test_20_non_contiguous_sequence_detected(self) -> None:
        run_dir = self.sandbox / "run20"
        store = RunReceiptStore(run_dir)
        r0 = _make_receipt()
        store.append(r0)
        # Manually create a file at index 2 (skip index 1)
        r1 = _make_receipt(
            state=RunState.RUN_STARTED,
            previous_sha=r0.receipt_sha256,
        )
        data = r1.to_dict()
        gap_path = store.receipts_dir / "002_RUN_STARTED.json"
        gap_path.write_text(
            json.dumps(data, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        with self.assertRaises(ReceiptStoreError):
            store.load_history_chain()


if __name__ == "__main__":
    unittest.main()
