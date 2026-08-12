"""Lineage Verifier Core Module — RAGLab V7 / Experimental Readiness V1.

Provides read-only verification of experimental run lineage, receipt chain integrity,
protocol snapshot consistency, and input hash verification.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from raglab.agentic.experiments.integrity import compute_file_sha256
from raglab.agentic.experiments.receipt_store import RunReceiptStore
from raglab.agentic.experiments.receipts import ReceiptStoreError, RunReceipt
from raglab.agentic.experiments.run_controller import RunControllerError


@dataclass(frozen=True)
class LineageAuditResult:
    """Read-only result of a lineage audit."""

    is_valid: bool
    failure_reasons: tuple[str, ...] = field(default_factory=tuple)


class LineageVerifier:
    """Read-only verifier for experimental run lineage and cryptographic integrity."""

    def __init__(self, *, read_only: bool = True) -> None:
        if not read_only:
            raise ValueError(
                "LineageVerifier must be instantiated with read_only=True"
            )
        self._read_only = True

    def is_read_only(self) -> bool:
        """Return True indicating this verifier is strictly read-only."""
        return self._read_only

    def verify_input_hashes(
        self,
        *,
        actual_inputs: Mapping[str, Path | str],
        expected_hashes: Mapping[str, str],
    ) -> None:
        """Verify that actual input files exist and match expected SHA-256 hashes.

        Raises RunControllerError or ValueError on mismatch or missing files.
        """
        if set(actual_inputs.keys()) != set(expected_hashes.keys()):
            raise RunControllerError(
                f"Input name mismatch: actual {set(actual_inputs.keys())} vs "
                f"expected {set(expected_hashes.keys())}"
            )

        for name, path_val in actual_inputs.items():
            p = Path(path_val)
            if not p.exists() or not p.is_file():
                raise RunControllerError(
                    f"Input file for '{name}' does not exist or is not a file: {p}"
                )
            expected_hash = expected_hashes[name]
            actual_hash = compute_file_sha256(p)
            if actual_hash != expected_hash:
                raise RunControllerError(
                    f"Input hash divergence for '{name}': expected '{expected_hash}', "
                    f"got '{actual_hash}'"
                )

    def verify_lineage(self, run_directory: Path | str) -> LineageAuditResult:
        """Audit run directory lineage in a strictly read-only, non-mutating manner."""
        reasons: list[str] = []
        target_dir = Path(run_directory).resolve()

        if not target_dir.exists() or not target_dir.is_dir():
            reasons.append(
                f"Run directory does not exist or is not a directory: {target_dir}"
            )
            return LineageAuditResult(
                is_valid=False, failure_reasons=tuple(reasons)
            )

        # 1. Load receipt chain using RunReceiptStore
        store = RunReceiptStore(target_dir)
        chain: Sequence[RunReceipt] = ()
        try:
            chain = store.load_history_chain()
        except ReceiptStoreError as r_exc:
            reasons.append(f"Receipt store unreadable or invalid: {r_exc}")

        if not chain:
            reasons.append("Receipt chain is empty or missing")
        else:
            try:
                store.verify_chain_integrity()
            except ReceiptStoreError as r_exc:
                reasons.append(f"Receipt chain integrity failure: {r_exc}")

        # 2. Check protocol.snapshot.json
        snapshot_path = target_dir / "protocol.snapshot.json"
        if not snapshot_path.exists() or not snapshot_path.is_file():
            reasons.append("Missing protocol.snapshot.json")
        elif chain:
            latest = chain[-1]
            try:
                actual_snapshot_sha = compute_file_sha256(snapshot_path)
                if actual_snapshot_sha != latest.protocol_sha256:
                    reasons.append(
                        "protocol.snapshot.json SHA-256 mismatch: expected "
                        f"'{latest.protocol_sha256}', got '{actual_snapshot_sha}'"
                    )
            except Exception as exc:
                reasons.append(f"Failed to read protocol.snapshot.json: {exc}")

        is_valid = len(reasons) == 0
        return LineageAuditResult(
            is_valid=is_valid, failure_reasons=tuple(reasons)
        )
