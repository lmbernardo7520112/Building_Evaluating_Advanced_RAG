"""Append-Only Run Receipt Store — RAGLab V7 Experimental Readiness.

Atomic, fail-closed persistence layer for RunReceipt chains.
Writes to ``<run_dir>/receipts/`` with deterministic filenames.
Does NOT provide concurrency locks (deferred to GREEN-2C).

Invariant for integration:
    RUN_CONTROLLER_MUST_SUPPLY_NON_EMPTY_ALLOWED_ROOTS
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from raglab.agentic.experiments.integrity import (
    compute_canonical_json_bytes,
)
from raglab.agentic.experiments.receipts import (
    ReceiptStoreError,
    RunReceipt,
    RunState,
    verify_receipt_chain,
)

_RECEIPT_FILENAME_RE = re.compile(
    r"^(\d{3})_([A-Z_]+)\.json$"
)

_STATE_FILENAME_ORDER: dict[RunState, str] = {
    RunState.PREPARED: "PREPARED",
    RunState.RUN_STARTED: "RUN_STARTED",
    RunState.RUN_COMPLETED: "RUN_COMPLETED",
    RunState.AUDIT_COMPLETED: "AUDIT_COMPLETED",
    RunState.FAILED: "FAILED",
}


def _receipt_filename(index: int, state: RunState) -> str:
    """Build deterministic receipt filename ``NNN_STATE.json``."""
    label = _STATE_FILENAME_ORDER.get(state)
    if label is None:
        raise ReceiptStoreError(f"Unknown RunState for filename: {state}")
    return f"{index:03d}_{label}.json"


class RunReceiptStore:
    """Append-only, atomic receipt persistence for a single run directory.

    All receipt files live under ``<run_dir>/receipts/``.
    No concurrency lock is provided; that is deferred to GREEN-2C.
    """

    def __init__(self, run_dir: Path | str) -> None:
        self._run_dir = Path(run_dir).resolve()
        self._receipts_dir = self._run_dir / "receipts"

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    @property
    def run_dir(self) -> Path:
        """Resolved run directory."""
        return self._run_dir

    @property
    def receipts_dir(self) -> Path:
        """Resolved receipts subdirectory."""
        return self._receipts_dir

    def load_history_chain(self) -> tuple[RunReceipt, ...]:
        """Load and verify the full ordered chain of receipts.

        Raises ``ReceiptStoreError`` on any integrity or ordering problem.
        Returns an immutable tuple of ``RunReceipt`` objects.
        """
        if not self._receipts_dir.exists():
            raise ReceiptStoreError(
                f"Receipts directory does not exist: {self._receipts_dir}"
            )

        files = sorted(
            f
            for f in self._receipts_dir.iterdir()
            if f.is_file() and _RECEIPT_FILENAME_RE.match(f.name)
        )

        if not files:
            raise ReceiptStoreError(
                f"No receipt files found in {self._receipts_dir}"
            )

        # Validate sequential indices and no duplicates
        seen_indices: dict[int, str] = {}
        receipts: list[RunReceipt] = []

        for f in files:
            m = _RECEIPT_FILENAME_RE.match(f.name)
            if m is None:  # pragma: no cover — guarded by filter
                continue
            idx = int(m.group(1))

            if idx in seen_indices:
                raise ReceiptStoreError(
                    f"Duplicate receipt index {idx}: "
                    f"'{seen_indices[idx]}' and '{f.name}'"
                )
            seen_indices[idx] = f.name

            try:
                raw = f.read_text(encoding="utf-8")
                data = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ReceiptStoreError(
                    f"Invalid JSON in receipt file '{f.name}': {exc}"
                ) from exc

            try:
                receipt = RunReceipt.from_dict(data)
            except (ValueError, TypeError, KeyError) as exc:
                raise ReceiptStoreError(
                    f"Cannot deserialize receipt '{f.name}': {exc}"
                ) from exc

            receipts.append(receipt)

        # Verify sequential continuity (0, 1, 2, ...)
        expected_indices = list(range(len(receipts)))
        actual_indices = sorted(seen_indices.keys())
        if actual_indices != expected_indices:
            raise ReceiptStoreError(
                f"Non-contiguous receipt indices: "
                f"expected {expected_indices}, got {actual_indices}"
            )

        # Delegate chain integrity to the core verifier
        verify_receipt_chain(receipts)

        return tuple(receipts)

    def verify_chain_integrity(self) -> bool:
        """Verify the full receipt chain. Returns True or raises."""
        self.load_history_chain()
        return True

    def get_receipt_by_index(self, index: int) -> RunReceipt:
        """Load the receipt at the given sequential index."""
        chain = self.load_history_chain()
        if index < 0 or index >= len(chain):
            raise ReceiptStoreError(
                f"Receipt index {index} out of range [0, {len(chain)})"
            )
        return chain[index]

    @classmethod
    def load_receipt_file(cls, path: Path | str) -> RunReceipt:
        """Load a single receipt from a JSON file path."""
        p = Path(path).resolve()
        if not p.exists():
            raise ReceiptStoreError(f"Receipt file not found: {p}")
        try:
            raw = p.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ReceiptStoreError(
                f"Invalid JSON in receipt file '{p}': {exc}"
            ) from exc

        try:
            return RunReceipt.from_dict(data)
        except (ValueError, TypeError, KeyError) as exc:
            raise ReceiptStoreError(
                f"Cannot deserialize receipt from '{p}': {exc}"
            ) from exc

    @classmethod
    def load_latest(cls, run_dir: Path | str) -> RunReceipt:
        """Load the most recent receipt from a run directory."""
        store = cls(run_dir)
        chain = store.load_history_chain()
        return chain[-1]

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def append(self, receipt: RunReceipt) -> Path:
        """Append a receipt to the chain atomically.

        1. Loads and verifies the existing chain.
        2. Validates the new receipt against the chain.
        3. Writes atomically via temp file + fsync + os.replace.
        4. Re-reads and re-verifies the full chain.

        Returns the path of the newly written receipt file.
        Raises ``ReceiptStoreError`` on any failure.
        """
        self._receipts_dir.mkdir(parents=True, exist_ok=True)

        # Load existing chain (empty dir is OK for the first receipt)
        existing_files = sorted(
            f
            for f in self._receipts_dir.iterdir()
            if f.is_file() and _RECEIPT_FILENAME_RE.match(f.name)
        )

        existing_chain: list[RunReceipt] = []
        seen_indices: dict[int, str] = {}

        for f in existing_files:
            m = _RECEIPT_FILENAME_RE.match(f.name)
            if m is None:  # pragma: no cover — guarded by filter
                continue
            idx = int(m.group(1))
            if idx in seen_indices:
                raise ReceiptStoreError(
                    f"Duplicate receipt index {idx} in chain"
                )
            seen_indices[idx] = f.name

            raw = f.read_text(encoding="utf-8")
            data = json.loads(raw)
            existing_chain.append(RunReceipt.from_dict(data))

        if existing_chain:
            verify_receipt_chain(existing_chain)

        # Determine new index
        new_index = len(existing_chain)

        # Validate receipt state
        state = (
            RunState(receipt.state)
            if isinstance(receipt.state, str)
            else receipt.state
        )

        # Validate receipt_sha256 is present and valid
        if not receipt.receipt_sha256:
            raise ReceiptStoreError(
                "Receipt must have a non-empty receipt_sha256 before append"
            )

        # Verify hash matches content
        expected_hash = receipt.compute_hash()
        if receipt.receipt_sha256 != expected_hash:
            raise ReceiptStoreError(
                f"Receipt receipt_sha256 mismatch: "
                f"recorded '{receipt.receipt_sha256}', "
                f"expected '{expected_hash}'"
            )

        # Validate chain linkage
        if new_index == 0:
            if state != RunState.PREPARED:
                raise ReceiptStoreError(
                    f"First receipt must be PREPARED, got {state.value}"
                )
            if receipt.previous_receipt_sha256 is not None:
                raise ReceiptStoreError(
                    "First receipt must have previous_receipt_sha256=None"
                )
        else:
            prev = existing_chain[-1]
            if receipt.previous_receipt_sha256 != prev.receipt_sha256:
                raise ReceiptStoreError(
                    f"previous_receipt_sha256 mismatch: "
                    f"'{receipt.previous_receipt_sha256}' != "
                    f"'{prev.receipt_sha256}'"
                )
            prev_state = (
                RunState(prev.state)
                if isinstance(prev.state, str)
                else prev.state
            )
            from raglab.agentic.experiments.receipts import (
                validate_state_transition,
            )

            validate_state_transition(prev_state, state)

        # Build filename
        target_name = _receipt_filename(new_index, state)
        target_path = self._receipts_dir / target_name

        if target_path.exists():
            raise ReceiptStoreError(
                f"Receipt file already exists (no overwrite): {target_path}"
            )

        # Serialize canonical JSON
        receipt_bytes = compute_canonical_json_bytes(receipt.to_dict())

        # Atomic write: temp → fsync → os.replace → dir fsync
        self._write_atomic(target_path, receipt_bytes)

        # Post-write verification: re-read and re-verify full chain
        try:
            reread_chain = self.load_history_chain()
        except ReceiptStoreError as exc:
            raise ReceiptStoreError(
                f"Post-write chain verification failed: {exc}"
            ) from exc

        if len(reread_chain) != new_index + 1:
            raise ReceiptStoreError(
                f"Post-write chain length mismatch: "
                f"expected {new_index + 1}, got {len(reread_chain)}"
            )

        return target_path

    def overwrite_receipt_file(
        self, filename: str, data: dict[str, Any]
    ) -> None:
        """Explicitly forbidden: overwriting a receipt file.

        Always raises ``ReceiptStoreError``.
        Anti-regression guard for append-only invariant.
        """
        raise ReceiptStoreError(
            f"Overwriting receipt file '{filename}' is strictly forbidden. "
            "Receipt chain is append-only."
        )

    # ------------------------------------------------------------------
    # Atomic write support
    # ------------------------------------------------------------------

    def supports_fsync_atomic_write(self) -> bool:
        """Confirm this store uses fsync + os.replace atomic writes."""
        return True

    @staticmethod
    def write_atomic_failing_simulation(target_file: Path | str) -> None:
        """Simulate a failed atomic write for testing.

        Writes to a temp file then raises before os.replace,
        ensuring the target file is never created.
        """
        target = Path(target_file).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(target.parent), suffix=".tmp"
        )
        try:
            os.write(tmp_fd, b'{"partial": true}')
            os.fsync(tmp_fd)
            os.close(tmp_fd)
            # Simulate failure before os.replace
            raise RuntimeError(
                "Simulated write failure before os.replace"
            )
        except RuntimeError:
            # Clean up temp file, re-raise
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def _write_atomic(self, target: Path, data: bytes) -> None:
        """Write data atomically: temp file → fsync → os.replace → dir fsync."""
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(target.parent), suffix=".tmp"
        )
        try:
            os.write(tmp_fd, data)
            os.fsync(tmp_fd)
            os.close(tmp_fd)
            os.replace(tmp_path, target)
        except BaseException:
            # Clean up temp file on any failure
            with contextlib.suppress(OSError):
                os.close(tmp_fd)
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

        # fsync the directory to ensure the rename is durable
        try:
            dir_fd = os.open(str(target.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass  # Best-effort dir fsync; not all FS support it
