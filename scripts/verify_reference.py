#!/usr/bin/env python3
"""Verify the v6.1 reference notebook against source_manifest.json.

This script uses ONLY the Python standard library. No external dependencies.
It validates:
  - Manifest schema (13 required fields)
  - SHA-256 hash match
  - File size match
  - Notebook JSON structure (nbformat, cell counts)
  - No sensitive data exposure in output

Exit codes:
  0 = all checks passed
  1 = one or more checks failed
  2 = script error (missing file, invalid JSON, etc.)
"""

import hashlib
import json
import os
import sys
from pathlib import Path

EXPECTED_MANIFEST_FIELDS = frozenset({
    "schema_version",
    "expected_sha256",
    "actual_sha256",
    "verified",
    "size_bytes",
    "original_filename",
    "reference_filename",
    "captured_at_utc",
    "nbformat",
    "nbformat_minor",
    "total_cells",
    "code_cells",
    "markdown_cells",
})

EXPECTED_SHA256 = (
    "c11c323e9d5362d4706c3fbbe4b11a107e7c4648407399186aef64fc1fb14db3"
)


def find_reference_dir() -> Path:
    """Locate the reference/ directory relative to this script."""
    script_dir = Path(__file__).resolve().parent
    # scripts/ is sibling to reference/
    return script_dir.parent / "reference"


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 of a file using streaming reads."""
    digest = hashlib.sha256()
    with filepath.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checks(reference_dir: Path) -> list[dict]:
    """Run all verification checks. Returns list of check results."""
    results: list[dict] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        results.append({"check": name, "passed": passed, "detail": detail})

    # --- Check 1: Manifest file exists ---
    manifest_path = reference_dir / "source_manifest.json"
    if not manifest_path.exists():
        record("manifest_exists", False, "source_manifest.json not found")
        return results
    record("manifest_exists", True)

    # --- Check 2: Manifest is valid JSON ---
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        record("manifest_valid_json", False, str(exc))
        return results
    record("manifest_valid_json", True)

    # --- Check 3: Manifest has exactly 13 required fields ---
    manifest_fields = set(manifest.keys())
    missing = EXPECTED_MANIFEST_FIELDS - manifest_fields
    extra = manifest_fields - EXPECTED_MANIFEST_FIELDS
    fields_ok = not missing and not extra
    detail_parts = []
    if missing:
        detail_parts.append(f"missing={sorted(missing)}")
    if extra:
        detail_parts.append(f"extra={sorted(extra)}")
    record(
        "manifest_schema_complete",
        fields_ok,
        "; ".join(detail_parts) if detail_parts else "13/13 fields present",
    )

    if not fields_ok:
        return results

    # --- Check 4: Reference notebook exists ---
    ref_filename = manifest.get("reference_filename", "")
    ref_path = reference_dir / ref_filename
    if not ref_path.exists():
        record("reference_file_exists", False, f"{ref_filename} not found")
        return results
    record("reference_file_exists", True)

    # --- Check 5: SHA-256 hash matches expected ---
    actual_hash = compute_sha256(ref_path)
    expected_hash = manifest.get("expected_sha256", "")
    hash_match = actual_hash == expected_hash == EXPECTED_SHA256
    record(
        "sha256_hash_match",
        hash_match,
        "hash verified"
        if hash_match
        else f"expected={expected_hash[:16]}... actual={actual_hash[:16]}...",
    )

    # --- Check 6: Manifest actual_sha256 consistent ---
    manifest_actual = manifest.get("actual_sha256", "")
    record(
        "manifest_actual_sha256_consistent",
        manifest_actual == actual_hash,
        "manifest.actual_sha256 matches recalculated hash"
        if manifest_actual == actual_hash
        else "manifest.actual_sha256 is stale",
    )

    # --- Check 7: File size matches ---
    actual_size = os.path.getsize(ref_path)
    expected_size = manifest.get("size_bytes", -1)
    record(
        "size_bytes_match",
        actual_size == expected_size,
        f"expected={expected_size} actual={actual_size}",
    )

    # --- Check 8: Notebook JSON is parseable ---
    try:
        with ref_path.open("r", encoding="utf-8") as f:
            nb = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        record("notebook_valid_json", False, str(exc))
        return results
    record("notebook_valid_json", True)

    # --- Check 9: nbformat matches ---
    nb_format = nb.get("nbformat")
    nb_minor = nb.get("nbformat_minor")
    record(
        "nbformat_match",
        nb_format == manifest["nbformat"]
        and nb_minor == manifest["nbformat_minor"],
        f"notebook={nb_format}.{nb_minor} "
        f"manifest={manifest['nbformat']}.{manifest['nbformat_minor']}",
    )

    # --- Check 10: Cell counts match ---
    cells = nb.get("cells", [])
    total = len(cells)
    code = sum(1 for c in cells if c.get("cell_type") == "code")
    markdown = sum(1 for c in cells if c.get("cell_type") == "markdown")

    record(
        "total_cells_match",
        total == manifest["total_cells"],
        f"notebook={total} manifest={manifest['total_cells']}",
    )
    record(
        "code_cells_match",
        code == manifest["code_cells"],
        f"notebook={code} manifest={manifest['code_cells']}",
    )
    record(
        "markdown_cells_match",
        markdown == manifest["markdown_cells"],
        f"notebook={markdown} manifest={manifest['markdown_cells']}",
    )

    # --- Check 11: verified flag is True ---
    record(
        "manifest_verified_flag",
        manifest.get("verified") is True,
        f"verified={manifest.get('verified')}",
    )

    # --- Check 12: schema_version present and valid ---
    record(
        "schema_version_valid",
        isinstance(manifest.get("schema_version"), str)
        and len(manifest["schema_version"]) > 0,
        f"schema_version={manifest.get('schema_version')}",
    )

    # --- Check 13: No absolute personal paths in manifest ---
    manifest_text = json.dumps(manifest)
    has_home_path = "/home/" in manifest_text or "\\Users\\" in manifest_text
    record(
        "no_personal_paths",
        not has_home_path,
        "no personal paths detected"
        if not has_home_path
        else "personal path found in manifest",
    )

    return results


def main() -> int:
    """Entry point. Returns exit code."""
    try:
        reference_dir = find_reference_dir()
    except Exception as exc:
        print(json.dumps({"error": str(exc), "exit_code": 2}))
        return 2

    results = run_checks(reference_dir)

    # Structured output
    all_passed = all(r["passed"] for r in results)
    output = {
        "script": "verify_reference.py",
        "schema_version": "1.0",
        "overall": "PASSED" if all_passed else "FAILED",
        "checks_total": len(results),
        "checks_passed": sum(1 for r in results if r["passed"]),
        "checks_failed": sum(1 for r in results if not r["passed"]),
        "results": results,
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
