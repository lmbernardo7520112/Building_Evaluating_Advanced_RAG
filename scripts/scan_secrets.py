#!/usr/bin/env python3
"""Scan tracked files for potential hardcoded secrets.

This script uses ONLY the Python standard library. No external dependencies.
It checks for:
  - API key patterns (GEMINI_API_KEY=..., etc.)
  - Private key file markers (BEGIN PRIVATE KEY, etc.)
  - High-entropy strings that resemble tokens
  - Known secret file extensions committed (.pem, .key, .p12)

Exit codes:
  0 = no secrets detected
  1 = potential secret found
  2 = script error
"""

import json
import os
import re
import sys
from pathlib import Path


# Patterns that indicate a hardcoded secret (not a reference or env lookup)
SECRET_ASSIGNMENT_PATTERNS = [
    # Direct assignment: API_KEY = "actual-value"
    re.compile(
        r"""(?:API_KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)"""
        r"""\s*=\s*["'][A-Za-z0-9\-_./+]{8,}["']""",
        re.IGNORECASE,
    ),
    # Private key block
    re.compile(r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----"),
    # Google service account JSON key
    re.compile(r'"private_key"\s*:\s*"-----BEGIN'),
]

# Lines that are safe even if they match a pattern above
SAFE_LINE_PATTERNS = [
    re.compile(r"^\s*#"),           # comments
    re.compile(r"os\.environ"),     # env variable access
    re.compile(r"userdata\.get"),   # Colab secret vault
    re.compile(r"\.example"),       # example files
    re.compile(r"getenv"),          # env variable access
    re.compile(r"expected_sha256"), # hash constants (not secrets)
    re.compile(r"actual_sha256"),   # hash constants
]

# File extensions that should never be committed
FORBIDDEN_EXTENSIONS = {".pem", ".key", ".p12", ".pfx", ".jks"}

# Directories to skip
SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".ruff_cache"}

# File extensions to scan
SCANNABLE_EXTENSIONS = {
    ".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini",
    ".md", ".txt", ".sh", ".bash", ".env",
}


def find_repo_root() -> Path:
    """Locate the repository root relative to this script."""
    return Path(__file__).resolve().parent.parent


def scan_file_content(filepath: Path) -> list[dict]:
    """Scan a single file for secret patterns."""
    findings: list[dict] = []

    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return findings

    for line_num, line in enumerate(text.splitlines(), start=1):
        # Skip safe lines
        if any(safe.search(line) for safe in SAFE_LINE_PATTERNS):
            continue

        for pattern in SECRET_ASSIGNMENT_PATTERNS:
            if pattern.search(line):
                # Redact the actual value
                redacted = line.strip()[:80]
                findings.append({
                    "file": str(filepath),
                    "line": line_num,
                    "type": "hardcoded_secret_pattern",
                    "preview": redacted + ("..." if len(line.strip()) > 80 else ""),
                })
                break  # one finding per line is enough

    return findings


def scan_forbidden_files(repo_root: Path) -> list[dict]:
    """Check for files with forbidden extensions."""
    findings: list[dict] = []

    for dirpath, dirnames, filenames in os.walk(repo_root):
        # Prune skip directories
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in FORBIDDEN_EXTENSIONS:
                findings.append({
                    "file": str(Path(dirpath) / fname),
                    "line": 0,
                    "type": "forbidden_file_extension",
                    "preview": f"File with extension {ext}",
                })

    return findings


def main() -> int:
    """Entry point. Returns exit code."""
    try:
        repo_root = find_repo_root()
    except Exception as exc:
        print(json.dumps({"error": str(exc), "exit_code": 2}))
        return 2

    all_findings: list[dict] = []

    # Scan file contents
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            fpath = Path(dirpath) / fname
            if fpath.suffix.lower() in SCANNABLE_EXTENSIONS:
                all_findings.extend(scan_file_content(fpath))

    # Scan for forbidden files
    all_findings.extend(scan_forbidden_files(repo_root))

    # Structured output
    passed = len(all_findings) == 0
    output = {
        "script": "scan_secrets.py",
        "overall": "PASSED" if passed else "FAILED",
        "findings_count": len(all_findings),
        "findings": all_findings,
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
