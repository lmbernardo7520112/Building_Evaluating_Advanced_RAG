"""Fail-closed command line interface for agentic conversational smoke canaries.

Executes known canaries against Fake or Gemini generator adapters and emits
canonical, strictly sanitized JSON reports.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from raglab.agentic.smoke_harness import run_known_canaries
from raglab.infrastructure.fakes.fake_generator_adapter import FakeGeneratorAdapter
from raglab.infrastructure.gemini.gemini_generator_adapter import (
    GeminiGeneratorAdapter,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_KNOWN_CANARIES: Final[frozenset[str]] = frozenset({
    "direct_supported_fact",
    "conversational_ellipsis",
    "unsupported_abstention",
})
_CANONICAL_KEYS: Final[frozenset[str]] = frozenset({
    "backend",
    "model_id",
    "status",
    "canaries",
    "error",
})
_PROVIDER_ERROR_TYPES: Final[frozenset[str]] = frozenset({
    "RetryExhaustedError",
    "NonRetryableError",
})
_ALLOWED_CITATION_REASONS: Final[frozenset[str]] = frozenset({
    "unknown_evidence_id",
    "missing_passage_id",
    "unknown_passage_id",
    "document_id_mismatch",
    "chunk_id_mismatch",
    "content_sha256_mismatch",
    "retrieval_rank_mismatch",
    "page_number_mismatch",
})


def _emit_json_report(
    report: dict[str, Any],
    exit_code: int,
    output_path: Path | None = None,
) -> int:
    """Serialize canonical report to stdout and optional output file atomically.

    Returns:
        Effective exit code (original exit_code on success, or 2 on file write failure).
    """
    canonical_payload: dict[str, Any] = {
        "backend": report["backend"],
        "model_id": report["model_id"],
        "status": report["status"],
        "canaries": report["canaries"],
        "error": report["error"],
    }
    serialized = json.dumps(canonical_payload, indent=2, ensure_ascii=False) + "\n"

    if output_path is not None:
        try:
            target_parent = output_path.parent
            target_parent.mkdir(parents=True, exist_ok=True)
            encoded_bytes = serialized.encode("utf-8")
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=str(target_parent),
                    prefix=".agentic_smoke_tmp_",
                    delete=False,
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    temp_file.write(encoded_bytes)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                os.replace(str(temp_path), str(output_path))
            except BaseException:
                if temp_path is not None:
                    with contextlib.suppress(OSError):
                        temp_path.unlink(missing_ok=True)
                raise
        except OSError:
            fallback_output_error: dict[str, Any] = {
                "backend": report["backend"],
                "model_id": report["model_id"],
                "status": "INCONCLUSIVE",
                "canaries": {},
                "error": {"type": "OutputWriteError"},
            }
            fallback_serialized = (
                json.dumps(fallback_output_error, indent=2, ensure_ascii=False)
                + "\n"
            )
            sys.stdout.write(fallback_serialized)
            return 2

    sys.stdout.write(serialized)
    return exit_code


def _sanitize_and_validate_report(
    raw_report: Any,
    expected_backend: str,
    expected_model_id: str | None,
) -> tuple[dict[str, Any], int]:
    """Validate and strictly sanitize the report returned by the harness.

    Returns:
        tuple of (sanitized_report_dict, exit_code)
    """
    fallback_invalid: dict[str, Any] = {
        "backend": expected_backend,
        "model_id": expected_model_id,
        "status": "FAIL",
        "canaries": {},
        "error": {"type": "InvalidSmokeReportError"},
    }

    if not isinstance(raw_report, dict):
        return fallback_invalid, 1

    if not _CANONICAL_KEYS.issubset(raw_report.keys()):
        return fallback_invalid, 1

    raw_backend = raw_report.get("backend")
    raw_model_id = raw_report.get("model_id")
    raw_status = raw_report.get("status")
    raw_canaries = raw_report.get("canaries")
    raw_error = raw_report.get("error")

    if raw_backend != expected_backend:
        return fallback_invalid, 1

    if not isinstance(raw_model_id, str) or raw_model_id != expected_model_id:
        return fallback_invalid, 1

    if raw_status not in ("PASS", "FAIL"):
        return fallback_invalid, 1

    if not isinstance(raw_canaries, dict):
        return fallback_invalid, 1

    sanitized_canaries: dict[str, dict[str, str]] = {}
    for canary_name, canary_data in raw_canaries.items():
        if canary_name not in _KNOWN_CANARIES:
            return fallback_invalid, 1
        if not isinstance(canary_data, dict):
            return fallback_invalid, 1
        c_status = canary_data.get("status")
        if c_status not in ("PASS", "FAIL"):
            return fallback_invalid, 1
        sanitized_canaries[canary_name] = {"status": c_status}

    if raw_status == "PASS":
        if raw_error is not None:
            return fallback_invalid, 1
        if set(sanitized_canaries.keys()) != _KNOWN_CANARIES:
            return fallback_invalid, 1
        for canary_entry in sanitized_canaries.values():
            if canary_entry.get("status") != "PASS":
                return fallback_invalid, 1
        return {
            "backend": expected_backend,
            "model_id": expected_model_id,
            "status": "PASS",
            "canaries": sanitized_canaries,
            "error": None,
        }, 0

    # FAIL status validation
    if not isinstance(raw_error, dict):
        return fallback_invalid, 1

    error_type = raw_error.get("type")
    if not isinstance(error_type, str) or not error_type.strip():
        return fallback_invalid, 1

    clean_error_type = error_type.strip()

    if clean_error_type in _PROVIDER_ERROR_TYPES:
        return {
            "backend": expected_backend,
            "model_id": expected_model_id,
            "status": "INCONCLUSIVE",
            "canaries": sanitized_canaries,
            "error": {"type": clean_error_type},
        }, 2

    if clean_error_type == "CitationProvenanceMismatchError":
        reason = raw_error.get("reason")
        if not isinstance(reason, str) or reason not in _ALLOWED_CITATION_REASONS:
            return fallback_invalid, 1
        return {
            "backend": expected_backend,
            "model_id": expected_model_id,
            "status": "FAIL",
            "canaries": sanitized_canaries,
            "error": {
                "type": "CitationProvenanceMismatchError",
                "reason": reason,
            },
        }, 1

    return {
        "backend": expected_backend,
        "model_id": expected_model_id,
        "status": "FAIL",
        "canaries": sanitized_canaries,
        "error": {"type": clean_error_type},
    }, 1


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for agentic smoke CLI."""
    parser = argparse.ArgumentParser(
        prog="raglab.interfaces.cli.agentic_smoke",
        description="Fail-closed agentic smoke canary execution CLI.",
    )
    parser.add_argument(
        "--backend",
        choices=["fake", "gemini"],
        required=True,
        help="Generator backend adapter to execute ('fake' or 'gemini').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write duplicate canonical JSON report.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    backend: str = args.backend
    output_path: Path | None = args.output

    if backend == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        missing_cred_report: dict[str, Any] = {
            "backend": "gemini",
            "model_id": None,
            "status": "INCONCLUSIVE",
            "canaries": {},
            "error": {"type": "MissingCredentialError"},
        }
        return _emit_json_report(
            missing_cred_report, exit_code=2, output_path=output_path
        )

    output_sink = io.StringIO()
    generator: Any = None
    expected_model_id: str | None = None

    try:
        with (
            contextlib.redirect_stdout(output_sink),
            contextlib.redirect_stderr(output_sink),
        ):
            if backend == "fake":
                generator = FakeGeneratorAdapter()
            else:
                generator = GeminiGeneratorAdapter()

            raw_model_id_prop = getattr(generator, "model_id", None)
            if (
                not isinstance(raw_model_id_prop, str)
                or not raw_model_id_prop.strip()
            ):
                raise ValueError("Invalid generator model_id property")
            expected_model_id = raw_model_id_prop.strip()

            raw_report = run_known_canaries(generator, backend=backend)
            sanitized_report, exit_code = _sanitize_and_validate_report(
                raw_report,
                expected_backend=backend,
                expected_model_id=expected_model_id,
            )

    except Exception as exc:
        exc_type = type(exc).__name__
        if exc_type in _PROVIDER_ERROR_TYPES:
            inconclusive_exc_report: dict[str, Any] = {
                "backend": backend,
                "model_id": expected_model_id,
                "status": "INCONCLUSIVE",
                "canaries": {},
                "error": {"type": exc_type},
            }
            return _emit_json_report(
                inconclusive_exc_report, exit_code=2, output_path=output_path
            )

        fail_exc_report: dict[str, Any] = {
            "backend": backend,
            "model_id": expected_model_id,
            "status": "FAIL",
            "canaries": {},
            "error": {"type": exc_type},
        }
        return _emit_json_report(
            fail_exc_report, exit_code=1, output_path=output_path
        )

    return _emit_json_report(
        sanitized_report, exit_code=exit_code, output_path=output_path
    )


if __name__ == "__main__":
    raise SystemExit(main())
