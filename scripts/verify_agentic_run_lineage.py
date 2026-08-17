#!/usr/bin/env python3
"""CLI runner for experimental run lineage verification.

Strictly read-only composition root over LineageVerifier.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from raglab.agentic.experiments.lineage_verifier import LineageVerifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify experimental run lineage and cryptographic integrity."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to experimental run directory to audit.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        sys.stderr.write(f"Error: Run directory does not exist: {run_dir}\n")
        return 1

    verifier = LineageVerifier(read_only=True)
    audit = verifier.verify_lineage(run_dir)

    if audit.is_valid:
        print(f"Lineage verification passed for: {run_dir}")
        return 0

    sys.stderr.write(
        f"Lineage verification failed for {run_dir}:\n"
    )
    for reason in audit.failure_reasons:
        sys.stderr.write(f"  - {reason}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
