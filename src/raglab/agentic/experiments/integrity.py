"""Pure Cryptographic & Artifact Integrity Core — RAGLab V7 Experimental Readiness.

Provides deterministic canonical JSON serialization, SHA-256 computation,
and fail-closed artifact hashing/validation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


class IntegrityError(Exception):
    """Raised when an integrity check fails closed."""


def compute_bytes_sha256(data: bytes) -> str:
    """Compute 64-character lowercase hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def _assert_no_nan_or_inf(obj: Any) -> None:
    """Recursively validate that floats contain no NaN or Infinity."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(f"Invalid float value in JSON canonicalization: {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"JSON dict keys must be strings, got {type(k).__name__}"
                )
            _assert_no_nan_or_inf(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _assert_no_nan_or_inf(item)


def compute_canonical_json_bytes(data: Any) -> bytes:
    """Serialize a data structure to canonical UTF-8 JSON bytes."""
    _assert_no_nan_or_inf(data)
    try:
        json_str = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Cannot canonically serialize object: {exc}") from exc

    return json_str.encode("utf-8")


def compute_canonical_json_sha256(data: Any) -> str:
    """Compute 64-char lowercase hex SHA-256 digest of canonical JSON."""
    return compute_bytes_sha256(compute_canonical_json_bytes(data))


def compute_file_sha256(path: Path | str) -> str:
    """Compute 64-character lowercase hex SHA-256 digest of a file."""
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found for SHA-256 computation: {file_path}"
        )
    if file_path.is_dir():
        raise IsADirectoryError(
            f"Target path is a directory, not a file: {file_path}"
        )

    h = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_artifact_integrity(
    base_dir: Path | str,
    artifact_inventory: list[str],
    artifact_hashes: dict[str, str],
    allow_extra_files: bool = False,
) -> bool:
    """Verify integrity of artifacts against declared inventory and hashes."""
    root_dir = Path(base_dir).resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        raise IntegrityError(
            f"Base directory does not exist or is not a directory: {root_dir}"
        )

    for rel_path_str in artifact_inventory:
        rel_path = Path(rel_path_str)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise IntegrityError(
                f"Forbidden absolute or traversal path in inventory: {rel_path_str}"
            )

        full_path = (root_dir / rel_path).resolve()
        try:
            full_path.relative_to(root_dir)
        except ValueError:
            raise IntegrityError(
                f"Path escapes base directory: {rel_path_str}"
            ) from None

        if not full_path.exists() or not full_path.is_file():
            raise IntegrityError(f"Declared artifact missing: {rel_path_str}")

        expected_hash = artifact_hashes.get(rel_path_str)
        if not expected_hash:
            raise IntegrityError(f"No declared hash for artifact: {rel_path_str}")

        actual_hash = compute_file_sha256(full_path)
        if actual_hash.lower() != expected_hash.lower():
            raise IntegrityError(
                f"Hash mismatch for {rel_path_str}: "
                f"expected {expected_hash}, got {actual_hash}"
            )

    if not allow_extra_files:
        expected_set = {str(Path(p)) for p in artifact_inventory}
        ignored_prefixes = (
            ".",
            "receipts/",
            "hashes.sha256",
            "run_receipt.json",
            "protocol.snapshot.json",
        )
        for p in root_dir.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(root_dir))
                if rel not in expected_set and not rel.startswith(ignored_prefixes):
                    raise IntegrityError(
                        f"Unexpected unrecorded file in run directory: {rel}"
                    )

    return True
