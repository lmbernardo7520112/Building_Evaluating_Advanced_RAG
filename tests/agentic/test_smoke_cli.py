"""Hermetic unit tests defining the fail-closed contract for the agentic smoke CLI.

This test suite specifies the future CLI module `raglab.interfaces.cli.agentic_smoke`
without implementing it in this step (RED phase).
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType


def _load_cli() -> ModuleType:
    """Dynamically load the future CLI module.

    Raises ModuleNotFoundError during RED phase because the production module
    has not yet been implemented.
    """
    return importlib.import_module("raglab.interfaces.cli.agentic_smoke")


def _run_cli(
    cli_module: ModuleType,
    argv: Sequence[str],
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, str, str]:
    """Execute the CLI module's main entry point and capture output/exit code."""
    try:
        exit_code = cli_module.main(argv)
    except SystemExit as exc:
        if exc.code is None:
            exit_code = 0
        elif isinstance(exc.code, int):
            exit_code = exc.code
        else:
            exit_code = 1

    captured = capsys.readouterr()
    return exit_code, captured.out, captured.err


def _assert_canonical_json_output(out: str) -> dict[str, Any]:
    """Validate that stdout contains exactly one JSON document and a single trailing newline."""
    assert out != "", "stdout must not be empty"
    assert out == out.strip() + "\n", (
        "stdout must contain exactly one JSON object followed by a single newline, "
        "with no leading/trailing whitespace, extra newlines, or banners"
    )
    payload: Any = json.loads(out)
    assert isinstance(payload, dict), "Root JSON element must be a dict"
    assert set(payload.keys()) == {
        "backend",
        "model_id",
        "status",
        "canaries",
        "error",
    }, f"JSON payload must strictly match canonical keys, got: {set(payload.keys())}"
    return payload


def test_fake_backend_passes_and_emits_single_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fake backend executes real canaries, exits 0, and emits canonical JSON."""
    if "GEMINI_API_KEY" in os.environ:
        monkeypatch.delenv("GEMINI_API_KEY")

    cli_mod = _load_cli()
    exit_code, out, err = _run_cli(cli_mod, ["--backend", "fake"], capsys)

    assert exit_code == 0
    assert err == ""

    payload = _assert_canonical_json_output(out)
    assert payload["backend"] == "fake"
    assert payload["model_id"] == "fake-generator-v1-no-network"
    assert payload["status"] == "PASS"
    assert payload["error"] is None

    canaries = payload["canaries"]
    assert isinstance(canaries, dict)
    assert set(canaries.keys()) == {
        "direct_supported_fact",
        "conversational_ellipsis",
        "unsupported_abstention",
    }
    for canary_name in canaries:
        assert canaries[canary_name] == {"status": "PASS"}


def test_invalid_backend_exits_2_without_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid backend choice must exit 2 with empty stdout without calling adapters."""
    cli_mod = _load_cli()

    def _forbidden_call(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Dispatch must not occur when CLI arguments are invalid")

    monkeypatch.setattr(cli_mod, "run_known_canaries", _forbidden_call, raising=True)
    monkeypatch.setattr(cli_mod, "FakeGeneratorAdapter", _forbidden_call, raising=True)
    monkeypatch.setattr(cli_mod, "GeminiGeneratorAdapter", _forbidden_call, raising=True)

    exit_code, out, err = _run_cli(cli_mod, ["--backend", "unsupported_backend"], capsys)

    assert exit_code == 2
    assert out == ""
    assert err != ""


def test_gemini_missing_key_is_inconclusive_without_instantiation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Gemini backend without GEMINI_API_KEY must exit 2 without instantiating adapter."""
    if "GEMINI_API_KEY" in os.environ:
        monkeypatch.delenv("GEMINI_API_KEY")

    cli_mod = _load_cli()

    def _forbidden_call(*args: Any, **kwargs: Any) -> Any:
        pytest.fail(
            "Must not instantiate adapters or execute harness when GEMINI_API_KEY is missing"
        )

    monkeypatch.setattr(cli_mod, "FakeGeneratorAdapter", _forbidden_call, raising=True)
    monkeypatch.setattr(cli_mod, "GeminiGeneratorAdapter", _forbidden_call, raising=True)
    monkeypatch.setattr(cli_mod, "run_known_canaries", _forbidden_call, raising=True)

    exit_code, out, _ = _run_cli(cli_mod, ["--backend", "gemini"], capsys)

    assert exit_code == 2
    payload = _assert_canonical_json_output(out)
    assert payload["backend"] == "gemini"
    assert payload["model_id"] is None
    assert payload["status"] == "INCONCLUSIVE"
    assert payload["canaries"] == {}
    assert payload["error"] == {"type": "MissingCredentialError"}


def test_semantic_failure_exits_1_and_sanitizes_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Semantic/provenance failure exits 1, drops stdout noise and strips secret fields."""
    import sys

    root_sentinel = "SENTINEL_ROOT_DATA_XYZ"
    error_msg_sentinel = "SENTINEL_ERROR_MSG_DATA_XYZ"
    error_extra_sentinel = "SENTINEL_ERROR_EXTRA_DATA_XYZ"
    canary_extra_sentinel = "SENTINEL_CANARY_EXTRA_DATA_XYZ"
    harness_stdout_noise = "SENTINEL_NOISY_STDOUT_LOG"

    cli_mod = _load_cli()

    def _mock_noisy_failing_canaries(
        generator: Any, *, backend: str
    ) -> dict[str, Any]:
        sys.stdout.write(harness_stdout_noise + "\n")
        print("EXTRA_UNSTRUCTURED_PRINT_LOG")
        return {
            "backend": backend,
            "model_id": getattr(generator, "model_id", "mock-model"),
            "status": "FAIL",
            "canaries": {
                "direct_supported_fact": {
                    "status": "FAIL",
                    "leaked_canary_key": canary_extra_sentinel,
                }
            },
            "error": {
                "type": "CitationProvenanceMismatchError",
                "reason": "unknown_evidence_id",
                "message": error_msg_sentinel,
                "forbidden_extra": error_extra_sentinel,
            },
            "unauthorized_root_field": root_sentinel,
        }

    monkeypatch.setattr(
        cli_mod, "run_known_canaries", _mock_noisy_failing_canaries, raising=True
    )

    exit_code, out, err = _run_cli(cli_mod, ["--backend", "fake"], capsys)

    assert exit_code == 1

    for sentinel in (
        root_sentinel,
        error_msg_sentinel,
        error_extra_sentinel,
        canary_extra_sentinel,
        harness_stdout_noise,
        "EXTRA_UNSTRUCTURED_PRINT_LOG",
    ):
        assert sentinel not in out
        assert sentinel not in err

    payload = _assert_canonical_json_output(out)
    assert payload["backend"] == "fake"
    assert payload["model_id"] == "fake-generator-v1-no-network"
    assert payload["status"] == "FAIL"
    assert payload["canaries"] == {"direct_supported_fact": {"status": "FAIL"}}
    assert payload["error"] == {
        "type": "CitationProvenanceMismatchError",
        "reason": "unknown_evidence_id",
    }


@pytest.mark.parametrize(
    "error_type",
    ["RetryExhaustedError", "NonRetryableError"],
)
def test_provider_failure_report_is_inconclusive(
    error_type: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Infrastructure/provider error in harness report is normalized to INCONCLUSIVE and exit 2."""
    synthetic_env_marker = "synthetic-fake-marker-boundary-test-only"
    monkeypatch.setenv("GEMINI_API_KEY", synthetic_env_marker)

    cli_mod = _load_cli()

    class MockGeminiAdapter:
        @property
        def model_id(self) -> str:
            return "gemini-3.1-flash-lite"

    monkeypatch.setattr(
        cli_mod, "GeminiGeneratorAdapter", MockGeminiAdapter, raising=True
    )

    def _mock_provider_error_harness(
        generator: Any, *, backend: str
    ) -> dict[str, Any]:
        return {
            "backend": backend,
            "model_id": getattr(generator, "model_id", "unknown"),
            "status": "FAIL",
            "canaries": {},
            "error": {
                "type": error_type,
                "message": f"Simulated provider failure with {synthetic_env_marker}",
            },
        }

    monkeypatch.setattr(
        cli_mod, "run_known_canaries", _mock_provider_error_harness, raising=True
    )

    exit_code, out, err = _run_cli(cli_mod, ["--backend", "gemini"], capsys)

    assert exit_code == 2
    assert synthetic_env_marker not in out
    assert synthetic_env_marker not in err

    payload = _assert_canonical_json_output(out)
    assert payload["backend"] == "gemini"
    assert payload["model_id"] == "gemini-3.1-flash-lite"
    assert payload["status"] == "INCONCLUSIVE"
    assert payload["canaries"] == {}
    assert payload["error"] == {"type": error_type}


def test_malformed_harness_report_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Malformed or incomplete report from harness fails closed with InvalidSmokeReportError and exit 1."""
    corrupt_data = "CORRUPTED_LEAK_DATA_XYZ"

    cli_mod = _load_cli()

    def _mock_malformed_harness(
        generator: Any, *, backend: str
    ) -> dict[str, Any]:
        return {"corrupt_key": corrupt_data, "extra_info": [1, 2, 3]}

    monkeypatch.setattr(
        cli_mod, "run_known_canaries", _mock_malformed_harness, raising=True
    )

    exit_code, out, err = _run_cli(cli_mod, ["--backend", "fake"], capsys)

    assert exit_code == 1
    assert corrupt_data not in out
    assert corrupt_data not in err

    payload = _assert_canonical_json_output(out)
    assert payload["backend"] == "fake"
    assert payload["model_id"] == "fake-generator-v1-no-network"
    assert payload["status"] == "FAIL"
    assert payload["canaries"] == {}
    assert payload["error"] == {"type": "InvalidSmokeReportError"}


def test_output_file_is_byte_identical_to_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Output file written via --output must be byte-for-byte identical to stdout."""
    if "GEMINI_API_KEY" in os.environ:
        monkeypatch.delenv("GEMINI_API_KEY")

    cli_mod = _load_cli()
    out_file = tmp_path / "smoke_report.json"

    exit_code, out, err = _run_cli(
        cli_mod, ["--backend", "fake", "--output", str(out_file)], capsys
    )

    assert exit_code == 0
    assert err == ""
    _assert_canonical_json_output(out)

    assert out_file.exists()
    assert out_file.read_bytes() == out.encode("utf-8")
    assert out_file.read_text(encoding="utf-8") == out


def test_divergent_model_id_in_harness_report_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Harness report with divergent model_id must fail closed with InvalidSmokeReportError and exit 1."""
    if "GEMINI_API_KEY" in os.environ:
        monkeypatch.delenv("GEMINI_API_KEY")

    cli_mod = _load_cli()
    forged_model_marker = "forged-model-id-marker-xyz"

    def _mock_forged_model_harness(
        generator: Any, *, backend: str
    ) -> dict[str, Any]:
        return {
            "backend": backend,
            "model_id": forged_model_marker,
            "status": "PASS",
            "canaries": {
                "direct_supported_fact": {"status": "PASS"},
                "conversational_ellipsis": {"status": "PASS"},
                "unsupported_abstention": {"status": "PASS"},
            },
            "error": None,
        }

    monkeypatch.setattr(
        cli_mod, "run_known_canaries", _mock_forged_model_harness, raising=True
    )

    exit_code, out, err = _run_cli(cli_mod, ["--backend", "fake"], capsys)

    assert exit_code == 1
    assert forged_model_marker not in out
    assert forged_model_marker not in err

    payload = _assert_canonical_json_output(out)
    assert payload["backend"] == "fake"
    assert payload["model_id"] == "fake-generator-v1-no-network"
    assert payload["status"] == "FAIL"
    assert payload["canaries"] == {}
    assert payload["error"] == {"type": "InvalidSmokeReportError"}


@pytest.mark.parametrize(
    "defect_mode",
    ["unknown_canary", "non_mapping_canary"],
)
def test_unknown_or_non_mapping_canary_fails_closed(
    defect_mode: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Canaries containing unknown keys or non-mapping values must fail closed with InvalidSmokeReportError."""
    if "GEMINI_API_KEY" in os.environ:
        monkeypatch.delenv("GEMINI_API_KEY")

    cli_mod = _load_cli()
    leaked_canary_marker = "unauthorized_sentinel_canary_key"

    def _mock_defective_canary_harness(
        generator: Any, *, backend: str
    ) -> dict[str, Any]:
        if defect_mode == "unknown_canary":
            return {
                "backend": backend,
                "model_id": getattr(
                    generator, "model_id", "fake-generator-v1-no-network"
                ),
                "status": "PASS",
                "canaries": {
                    "direct_supported_fact": {"status": "PASS"},
                    "conversational_ellipsis": {"status": "PASS"},
                    "unsupported_abstention": {"status": "PASS"},
                    "unauthorized_canary": {
                        "status": "PASS",
                        "leak": leaked_canary_marker,
                    },
                },
                "error": None,
            }
        return {
            "backend": backend,
            "model_id": getattr(
                generator, "model_id", "fake-generator-v1-no-network"
            ),
            "status": "FAIL",
            "canaries": {
                "direct_supported_fact": "not-a-dict",
            },
            "error": {
                "type": "CitationProvenanceMismatchError",
                "message": "simulated",
            },
        }

    monkeypatch.setattr(
        cli_mod, "run_known_canaries", _mock_defective_canary_harness, raising=True
    )

    exit_code, out, err = _run_cli(cli_mod, ["--backend", "fake"], capsys)

    assert exit_code == 1
    assert leaked_canary_marker not in out
    assert leaked_canary_marker not in err

    payload = _assert_canonical_json_output(out)
    assert payload["backend"] == "fake"
    assert payload["model_id"] == "fake-generator-v1-no-network"
    assert payload["status"] == "FAIL"
    assert payload["canaries"] == {}
    assert payload["error"] == {"type": "InvalidSmokeReportError"}


def test_model_id_property_exception_fails_closed_without_secondary_access(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exception on generator.model_id property must be accessed exactly once, emit FAIL without traceback."""
    if "GEMINI_API_KEY" in os.environ:
        monkeypatch.delenv("GEMINI_API_KEY")

    cli_mod = _load_cli()

    access_counter = 0
    property_noise_marker = "MODEL_ID_PROPERTY_INTERNAL_NOISE_XYZ"

    class DefectiveGeneratorAdapter:
        @property
        def model_id(self) -> str:
            nonlocal access_counter
            access_counter += 1
            sys.stdout.write(property_noise_marker + "\n")
            sys.stderr.write(property_noise_marker + "\n")
            raise RuntimeError("defective_model_id_access")

    monkeypatch.setattr(
        cli_mod, "FakeGeneratorAdapter", DefectiveGeneratorAdapter, raising=True
    )

    def _forbidden_harness(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("run_known_canaries must not be called when model_id raises")

    monkeypatch.setattr(
        cli_mod, "run_known_canaries", _forbidden_harness, raising=True
    )

    exit_code, out, err = _run_cli(cli_mod, ["--backend", "fake"], capsys)

    assert access_counter == 1
    assert exit_code == 1
    assert property_noise_marker not in out
    assert property_noise_marker not in err
    assert "Traceback" not in out
    assert "Traceback" not in err

    payload = _assert_canonical_json_output(out)
    assert payload["backend"] == "fake"
    assert payload["model_id"] is None
    assert payload["status"] == "FAIL"
    assert payload["canaries"] == {}
    assert payload["error"] == {"type": "RuntimeError"}


def test_output_file_write_failure_is_inconclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Failure to write to --output path must fail closed to INCONCLUSIVE exit 2 with OutputWriteError."""
    if "GEMINI_API_KEY" in os.environ:
        monkeypatch.delenv("GEMINI_API_KEY")

    cli_mod = _load_cli()

    # Pass an existing directory as --output to induce a deterministic write failure
    output_dir_target = tmp_path / "blocked_directory_as_output"
    output_dir_target.mkdir(parents=True, exist_ok=True)

    exit_code, out, err = _run_cli(
        cli_mod,
        ["--backend", "fake", "--output", str(output_dir_target)],
        capsys,
    )

    assert exit_code == 2
    assert "PASS" not in out
    assert err == ""
    assert output_dir_target.is_dir()

    payload = _assert_canonical_json_output(out)
    assert payload["backend"] == "fake"
    assert payload["model_id"] == "fake-generator-v1-no-network"
    assert payload["status"] == "INCONCLUSIVE"
    assert payload["canaries"] == {}
    assert payload["error"] == {"type": "OutputWriteError"}


@pytest.mark.parametrize(
    "bad_error_payload, unquoted_marker",
    [
        (
            {"type": "CitationProvenanceMismatchError"},
            None,
        ),
        (
            {
                "type": "CitationProvenanceMismatchError",
                "reason": "unauthorized_arbitrary_reason_marker_xyz",
            },
            "unauthorized_arbitrary_reason_marker_xyz",
        ),
        (
            {
                "type": "CitationProvenanceMismatchError",
                "reason": 12345,
            },
            "12345",
        ),
    ],
)
def test_citation_failure_with_invalid_reason_fails_closed(
    bad_error_payload: dict[str, Any],
    unquoted_marker: str | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CitationProvenanceMismatchError with missing, unallowed, or non-string reason must fail closed."""
    if "GEMINI_API_KEY" in os.environ:
        monkeypatch.delenv("GEMINI_API_KEY")

    cli_mod = _load_cli()

    def _mock_invalid_reason_harness(
        generator: Any, *, backend: str
    ) -> dict[str, Any]:
        return {
            "backend": backend,
            "model_id": getattr(
                generator, "model_id", "fake-generator-v1-no-network"
            ),
            "status": "FAIL",
            "canaries": {"direct_supported_fact": {"status": "FAIL"}},
            "error": bad_error_payload,
        }

    monkeypatch.setattr(
        cli_mod, "run_known_canaries", _mock_invalid_reason_harness, raising=True
    )

    exit_code, out, err = _run_cli(cli_mod, ["--backend", "fake"], capsys)

    assert exit_code == 1
    if unquoted_marker:
        assert unquoted_marker not in out
        assert unquoted_marker not in err

    payload = _assert_canonical_json_output(out)
    assert payload["backend"] == "fake"
    assert payload["model_id"] == "fake-generator-v1-no-network"
    assert payload["status"] == "FAIL"
    assert payload["canaries"] == {}
    assert payload["error"] == {"type": "InvalidSmokeReportError"}
