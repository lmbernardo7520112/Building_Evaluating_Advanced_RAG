#!/usr/bin/env python3
"""CLI runner for preparing experimental runs.

Thin composition root over RunController.prepare_run().
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from raglab.agentic.experiments.run_controller import (
    RunController,
    RunControllerError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a new experimental run directory and persist "
            "000_PREPARED receipt."
        )
    )
    parser.add_argument(
        "--repo-root",
        required=True,
        help="Path to Git repository root directory.",
    )
    parser.add_argument(
        "--artifact-root",
        required=True,
        help="Path to durable external artifact root directory.",
    )
    parser.add_argument(
        "--slice-id",
        required=True,
        help="Target slice identifier (e.g. slice5b).",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Target run identifier.",
    )
    parser.add_argument(
        "--protocol",
        required=True,
        help="Path to preregistered experiment protocol JSON file.",
    )
    parser.add_argument(
        "--implementation-commit",
        required=True,
        help="Git commit SHA of implementation under test.",
    )
    parser.add_argument(
        "--protocol-commit",
        required=True,
        help="Git commit SHA containing protocol preregistration.",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        dest="inputs",
        help="Repeatable input specification in NAME=PATH format.",
    )
    return parser.parse_args()


def parse_inputs(raw_inputs: list[str]) -> dict[str, str | Path]:
    parsed: dict[str, str | Path] = {}
    for item in raw_inputs:
        if "=" not in item:
            raise ValueError(
                f"Invalid --input format: '{item}'. Expected NAME=PATH."
            )
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name:
            raise ValueError(
                f"Input name cannot be empty in '{item}'"
            )
        if not path:
            raise ValueError(
                f"Input path cannot be empty in '{item}'"
            )
        if name in parsed:
            raise ValueError(
                f"Duplicate input name '{name}' specified in --input"
            )
        parsed[name] = path
    return parsed


def main() -> int:
    try:
        args = parse_args()
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    try:
        inputs_dict = parse_inputs(args.inputs)
    except ValueError as exc:
        sys.stderr.write(f"Input validation error: {exc}\n")
        return 1

    repo_root = Path(args.repo_root).resolve()
    artifact_root = Path(args.artifact_root).resolve()

    controller = RunController(
        repo_root=repo_root,
        allowed_roots=[artifact_root],
    )

    try:
        receipt = controller.prepare_run(
            run_id=args.run_id,
            slice_id=args.slice_id,
            protocol_path=Path(args.protocol),
            artifact_root=artifact_root,
            inputs=inputs_dict if inputs_dict else None,
            implementation_commit=args.implementation_commit,
            protocol_commit=args.protocol_commit,
        )
        print(
            f"Successfully prepared run '{receipt.run_id}' at {receipt.run_directory}"
        )
        return 0
    except (RunControllerError, ValueError, OSError) as exc:
        sys.stderr.write(f"Preflight / preparation failed: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
