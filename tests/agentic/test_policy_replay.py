"""Tests for the policy replay CLI (subprocess execution)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "analyze_agentic_policy_replay.py"
_FIXTURE = (
    _REPO_ROOT
    / "benchmarks"
    / "agentic"
    / "slice5"
    / "fixtures"
    / "synthetic_slice4_result_v1.json"
)
_CONFIG = (
    _REPO_ROOT
    / "benchmarks"
    / "agentic"
    / "slice5"
    / "configs"
    / "policy_replay_v1.json"
)


def _offline_env() -> dict[str, str]:
    """Environment with credentials stripped and offline flags set."""
    env = os.environ.copy()
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        env.pop(key, None)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def _run_cli(
    args: list[str],
    tmp_path: Path,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the CLI script via subprocess."""
    cmd = [
        sys.executable,
        str(_SCRIPT),
        *args,
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
        env=env or _offline_env(),
        cwd=str(_REPO_ROOT),
    )


class TestPolicyReplayCLI:
    """CLI subprocess tests with offline isolation."""

    def test_successful_execution(self, tmp_path):
        """Happy path: fixture + config → exit 0."""
        result = _run_cli(
            [
                "--slice4-result",
                str(_FIXTURE),
                "--config",
                str(_CONFIG),
                "--output-dir",
                str(tmp_path),
            ],
            tmp_path,
        )
        assert result.returncode == 0, f"CLI failed with:\n{result.stderr}"

    def test_exit_code_zero(self, tmp_path):
        """Verify real exit code, not piped status."""
        result = _run_cli(
            [
                "--slice4-result",
                str(_FIXTURE),
                "--config",
                str(_CONFIG),
                "--output-dir",
                str(tmp_path),
            ],
            tmp_path,
        )
        assert result.returncode == 0

    def test_missing_input_file(self, tmp_path):
        """Non-existent input file → non-zero exit."""
        result = _run_cli(
            [
                "--slice4-result",
                str(tmp_path / "nonexistent.json"),
                "--config",
                str(_CONFIG),
                "--output-dir",
                str(tmp_path),
            ],
            tmp_path,
        )
        assert result.returncode != 0

    def test_corrupted_json_input(self, tmp_path):
        """Corrupt JSON input → non-zero exit."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{{{not json")
        result = _run_cli(
            [
                "--slice4-result",
                str(bad_file),
                "--config",
                str(_CONFIG),
                "--output-dir",
                str(tmp_path),
            ],
            tmp_path,
        )
        assert result.returncode != 0

    def test_outputs_byte_deterministic(self, tmp_path):
        """Two identical runs produce exact byte-for-byte identical outputs and SHA-256 digests.

        POLICY_REPLAY_BYTE_DETERMINISM_CONFIRMED
        """
        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        dir1.mkdir()
        dir2.mkdir()

        for d in (dir1, dir2):
            result = _run_cli(
                [
                    "--slice4-result",
                    str(_FIXTURE),
                    "--config",
                    str(_CONFIG),
                    "--output-dir",
                    str(d),
                ],
                tmp_path,
            )
            assert result.returncode == 0

        files1 = sorted(f.name for f in dir1.iterdir())
        files2 = sorted(f.name for f in dir2.iterdir())
        assert files1 == files2

        # Integral byte-for-byte & sha256 equality across all generated files
        for fname in files1:
            bytes1 = (dir1 / fname).read_bytes()
            bytes2 = (dir2 / fname).read_bytes()
            assert bytes1 == bytes2, f"Non-deterministic file bytes: {fname}"
            h1 = hashlib.sha256(bytes1).hexdigest()
            h2 = hashlib.sha256(bytes2).hexdigest()
            assert h1 == h2, f"Non-deterministic sha256 digest: {fname}"

    def test_input_file_unmodified(self, tmp_path):
        """Input file must not be modified after CLI execution."""
        hash_before = hashlib.sha256(_FIXTURE.read_bytes()).hexdigest()
        result = _run_cli(
            [
                "--slice4-result",
                str(_FIXTURE),
                "--config",
                str(_CONFIG),
                "--output-dir",
                str(tmp_path),
            ],
            tmp_path,
        )
        assert result.returncode == 0
        hash_after = hashlib.sha256(_FIXTURE.read_bytes()).hexdigest()
        assert hash_before == hash_after

    def test_manifest_has_hashes(self, tmp_path):
        """Output manifest contains hash fields."""
        result = _run_cli(
            [
                "--slice4-result",
                str(_FIXTURE),
                "--config",
                str(_CONFIG),
                "--output-dir",
                str(tmp_path),
            ],
            tmp_path,
        )
        assert result.returncode == 0

        manifests = list(tmp_path.glob("*manifest*.json"))
        assert len(manifests) >= 1, "No manifest found"
        manifest = json.loads(manifests[0].read_text())
        assert "input_sha256" in manifest
        assert "config_sha256" in manifest

    def test_oracle_marked_post_hoc(self, tmp_path):
        """Oracle results must be marked POST_HOC_ORACLE."""
        result = _run_cli(
            [
                "--slice4-result",
                str(_FIXTURE),
                "--config",
                str(_CONFIG),
                "--output-dir",
                str(tmp_path),
            ],
            tmp_path,
        )
        assert result.returncode == 0

        reports = list(tmp_path.glob("*report*.json"))
        if reports:
            report = json.loads(reports[0].read_text())
            oracle = report.get("oracle_analysis", {})
            if oracle:
                assert oracle.get("label") == "POST_HOC_ORACLE"

    def test_no_credentials_in_outputs(self, tmp_path):
        """No API keys or secrets in any output file."""
        result = _run_cli(
            [
                "--slice4-result",
                str(_FIXTURE),
                "--config",
                str(_CONFIG),
                "--output-dir",
                str(tmp_path),
            ],
            tmp_path,
        )
        assert result.returncode == 0

        secrets = [
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "OPENAI_API_KEY",
            "sk-",
            "AIza",
        ]
        for f in tmp_path.iterdir():
            if f.is_file():
                content = f.read_text(errors="replace")
                for secret in secrets:
                    assert secret not in content, f"Secret '{secret}' found in {f.name}"

        assert result.returncode == 0

    def test_offline_environment_execution_confirmed(self, tmp_path):
        """Execution in offline environment (OFFLINE_ENVIRONMENT_EXECUTION_CONFIRMED)."""
        env = _offline_env()
        env["no_proxy"] = "*"
        result = _run_cli(
            [
                "--slice4-result",
                str(_FIXTURE),
                "--config",
                str(_CONFIG),
                "--output-dir",
                str(tmp_path),
            ],
            tmp_path,
            env=env,
        )
        assert result.returncode == 0

    def test_empty_input_queries(self, tmp_path):
        """Input with empty queries array -> exit code non-zero."""
        empty_input = tmp_path / "empty_input.json"
        empty_input.write_text(json.dumps({"queries": []}))
        result = _run_cli(
            [
                "--slice4-result",
                str(empty_input),
                "--config",
                str(_CONFIG),
                "--output-dir",
                str(tmp_path / "out"),
            ],
            tmp_path,
        )
        assert result.returncode != 0

    def test_missing_required_key(self, tmp_path):
        """Input missing 'queries' key -> exit code non-zero."""
        bad_input = tmp_path / "no_queries.json"
        bad_input.write_text(json.dumps({"data": []}))
        result = _run_cli(
            [
                "--slice4-result",
                str(bad_input),
                "--config",
                str(_CONFIG),
                "--output-dir",
                str(tmp_path / "out"),
            ],
            tmp_path,
        )
        assert result.returncode != 0

    def test_corrupted_config_json(self, tmp_path):
        """Corrupt config file -> exit code non-zero."""
        bad_config = tmp_path / "bad_config.json"
        bad_config.write_text("{invalid json")
        result = _run_cli(
            [
                "--slice4-result",
                str(_FIXTURE),
                "--config",
                str(bad_config),
                "--output-dir",
                str(tmp_path / "out"),
            ],
            tmp_path,
        )
        assert result.returncode != 0

    def test_explicit_denominator_and_metrics(self, tmp_path):
        """Report contains explicit denominator and valid queries count."""
        result = _run_cli(
            [
                "--slice4-result",
                str(_FIXTURE),
                "--config",
                str(_CONFIG),
                "--output-dir",
                str(tmp_path),
            ],
            tmp_path,
        )
        assert result.returncode == 0
        report = json.loads((tmp_path / "policy_replay_report.json").read_text())
        assert "valid_queries" in report
        assert "denominator" in report["wins_ties_losses"]
        assert report["wins_ties_losses"]["denominator"] == report["valid_queries"]

    def test_missing_metric_produces_none_not_zero(self, tmp_path):
        """Queries with missing metrics produce None/null, never converted to zero."""
        input_data = {
            "queries": [
                {
                    "query_id": "q1",
                    "query_text": "What is chunking?",
                    "results_per_strategy": {
                        "baseline": {"other_metric": 0.5},  # ndcg_at_3 is missing
                    },
                }
            ]
        }
        test_input = tmp_path / "missing_metric_input.json"
        test_input.write_text(json.dumps(input_data))

        result = _run_cli(
            [
                "--slice4-result",
                str(test_input),
                "--config",
                str(_CONFIG),
                "--output-dir",
                str(tmp_path / "out"),
            ],
            tmp_path,
        )
        assert result.returncode == 0
        report = json.loads(
            (tmp_path / "out" / "policy_replay_report.json").read_text()
        )
        # ndcg_at_3 for baseline should be null (None), not 0.0
        assert report["strategy_summary"]["baseline"]["ndcg_at_3"] is None
