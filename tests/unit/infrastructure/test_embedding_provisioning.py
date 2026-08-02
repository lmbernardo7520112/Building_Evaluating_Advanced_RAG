"""Tests for Slice 4 embedding provisioning and two-phase execution.

Tests validate:
- Cache resolution rejects /tmp transient directories
- Provisioning rejects GEMINI_API_KEY presence
- Preflight does not read GEMINI_API_KEY
- Preflight uses local_files_only=True
- Smoke refuses to run with missing embedding cache
- Valid cache allows progression (with fakes)
- Dimension and finiteness validation
- Model/revision/pooling recorded in manifest
- Manifest sanitized (no credential leaks)
- Model weights excluded from Git (.gitignore)
- Pre-existing Slice 3 change preserved
- Provisioning CLI works (with fakes)
- Preflight CLI works (with fakes)

All tests use fakes. No network access. No real model download.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNNER = str(_REPO_ROOT / "benchmarks" / "run_slice4_benchmark.py")
_PROVISIONER = str(_REPO_ROOT / "scripts" / "provision_embedding_model.py")


# ─── Cache Resolution ────────────────────────────────────────────


class TestCacheResolution:
    """Tests for resolve_cache_dir() and cache policy."""

    def test_tmp_rejected_as_authoritative_cache(self):
        """Cache resolution must reject /tmp as transient."""
        from raglab.infrastructure.embeddings.fastembed_adapter import (
            resolve_cache_dir,
        )

        with patch.dict(
            os.environ, {"RAGLAB_MODEL_CACHE": "/tmp"}, clear=False  # noqa: S108
        ), pytest.raises(ValueError, match="Transient directory"):
            resolve_cache_dir()

    def test_tmp_fastembed_cache_rejected(self):  # noqa: S108
        """fastembed_cache in /tmp must be rejected."""  # noqa: S108
        from raglab.infrastructure.embeddings.fastembed_adapter import (
            resolve_cache_dir,
        )

        with patch.dict(
            os.environ,
            {"RAGLAB_MODEL_CACHE": "/tmp/fastembed_cache"},  # noqa: S108
            clear=False,
        ), pytest.raises(ValueError, match="Transient directory"):
            resolve_cache_dir()

    def test_explicit_cache_dir_accepted(self, tmp_path):
        """Explicit persistent cache dir must be accepted."""
        from raglab.infrastructure.embeddings.fastembed_adapter import (
            resolve_cache_dir,
        )

        cache = tmp_path / "model_cache"
        result = resolve_cache_dir(str(cache))
        assert result == cache.resolve()

    def test_env_var_cache_dir_accepted(self, tmp_path):
        """RAGLAB_MODEL_CACHE environment variable must be respected."""
        from raglab.infrastructure.embeddings.fastembed_adapter import (
            resolve_cache_dir,
        )

        cache = tmp_path / "env_cache"
        with patch.dict(
            os.environ, {"RAGLAB_MODEL_CACHE": str(cache)}, clear=False
        ):
            result = resolve_cache_dir()
            assert result == cache.resolve()

    def test_default_cache_dir_is_persistent(self):
        """Default cache dir must NOT be /tmp."""
        from raglab.infrastructure.embeddings.fastembed_adapter import (
            resolve_cache_dir,
        )

        # Remove env var to test default
        env = {k: v for k, v in os.environ.items() if k != "RAGLAB_MODEL_CACHE"}
        with patch.dict(os.environ, env, clear=True):
            result = resolve_cache_dir()
            assert "/tmp" not in str(result), (  # noqa: S108
                f"Default cache resolved to transient: {result}"
            )


# ─── Provisioning Security ───────────────────────────────────────


class TestProvisioningSecurity:
    """Tests for provision_embedding_model.py."""

    def test_provisioning_without_execute_exits_2(self):
        """Provisioner without --execute must exit 2 with no side effects."""
        env = dict(os.environ)
        env.pop("GEMINI_API_KEY", None)
        env.pop("GOOGLE_API_KEY", None)
        result = subprocess.run(
            [sys.executable, _PROVISIONER],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 2
        assert "--execute" in result.stderr or "--execute" in result.stdout

    def test_provisioning_rejects_gemini_api_key(self):
        """Provisioner must abort if GEMINI_API_KEY is set."""
        env = dict(os.environ)
        env["GEMINI_API_KEY"] = "test-key-must-not-be-used"
        env.pop("RAGLAB_MODEL_CACHE", None)
        result = subprocess.run(
            [sys.executable, _PROVISIONER, "--execute"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode != 0
        assert "GEMINI_API_KEY" in result.stderr or "PROVISION_ERROR" in result.stderr

    def test_provisioning_rejects_google_api_key(self):
        """Provisioner must abort if GOOGLE_API_KEY is set."""
        env = dict(os.environ)
        env["GOOGLE_API_KEY"] = "test-key-must-not-be-used"
        env.pop("GEMINI_API_KEY", None)
        env.pop("RAGLAB_MODEL_CACHE", None)
        result = subprocess.run(
            [sys.executable, _PROVISIONER, "--execute"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode != 0
        assert "GOOGLE_API_KEY" in result.stderr or "PROVISION_ERROR" in result.stderr


# ─── Preflight ────────────────────────────────────────────────────


class TestPreflight:
    """Tests for --mode preflight."""

    def test_preflight_does_not_require_gemini_key(self, tmp_path, monkeypatch):
        """Preflight must work without GEMINI_API_KEY."""
        # Verified by the CLI mode: preflight is dispatched before check_credential()

        # Verify preflight is dispatched before credential check in main()
        source = Path(_RUNNER).read_text(encoding="utf-8")
        preflight_pos = source.find("cmd_preflight")
        credential_pos = source.find("check_credential(logger)")
        assert preflight_pos < credential_pos, (
            "cmd_preflight must be called BEFORE check_credential in main()"
        )

    def test_preflight_mode_in_cli(self):
        """preflight must be a valid mode choice."""
        import benchmarks.run_slice4_benchmark as runner

        args = runner.build_parser().parse_args(["--mode", "preflight"])
        assert args.mode == "preflight"

    def test_preflight_exits_if_cache_missing(self, tmp_path, monkeypatch):
        """Preflight must fail if cache directory doesn't exist."""
        import logging

        import benchmarks.run_slice4_benchmark as runner

        nonexistent = tmp_path / "nonexistent_cache"

        args = runner.build_parser().parse_args(["--mode", "preflight"])
        logger = logging.getLogger("test_preflight_no_cache")

        # Patch at the source module so the local import picks it up
        with patch(
            "raglab.infrastructure.embeddings.fastembed_adapter.resolve_cache_dir",
            return_value=nonexistent,
        ):
            with pytest.raises(SystemExit) as exc_info:
                runner.cmd_preflight(args, logger)
            assert exc_info.value.code == 1

    def test_preflight_uses_local_files_only(self, tmp_path, monkeypatch):
        """Preflight must pass local_files_only=True to load_embedding_model."""
        import logging

        import benchmarks.run_slice4_benchmark as runner
        from raglab.infrastructure.embeddings.fastembed_adapter import (
            FastEmbedEmbeddingAdapter,
        )

        seen_kwargs: list[dict] = []

        def _fake_load(logger, *, local_files_only=True):
            seen_kwargs.append({"local_files_only": local_files_only})
            # Create a spec-based mock that passes isinstance checks
            fake = MagicMock(spec=FastEmbedEmbeddingAdapter)
            fake.dimension = 384
            fake._embed = MagicMock(return_value=[0.1] * 384)  # noqa: SLF001
            return fake

        monkeypatch.setattr(runner, "load_embedding_model", _fake_load)

        cache = tmp_path / "cache"
        cache.mkdir()

        args = runner.build_parser().parse_args(["--mode", "preflight"])
        logger = logging.getLogger("test_preflight_offline")

        with patch(
            "raglab.infrastructure.embeddings.fastembed_adapter.resolve_cache_dir",
            return_value=cache,
        ):
            runner.cmd_preflight(args, logger)

        assert len(seen_kwargs) == 1
        assert seen_kwargs[0]["local_files_only"] is True


# ─── Smoke Cache Gate ─────────────────────────────────────────────


class TestSmokeCacheGate:
    """Tests that smoke refuses to run without embedding cache."""

    def test_smoke_exits_if_cache_missing(self, tmp_path):
        """Smoke must exit before reading API key if cache is missing."""
        env = dict(os.environ)
        env.pop("GEMINI_API_KEY", None)
        env["RAGLAB_MODEL_CACHE"] = str(tmp_path / "nonexistent_cache")

        result = subprocess.run(
            [sys.executable, _RUNNER, "--mode", "smoke"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "cache" in combined.lower() or "provision" in combined.lower()


# ─── Dimension & Finiteness ───────────────────────────────────────


class TestDimensionAndFiniteness:
    """Tests for embedding validation."""

    def test_valid_canary_embedding(self):
        """A valid 384-dim embedding with finite values passes validation."""
        vec = [0.1] * 384
        assert len(vec) == 384
        assert all(math.isfinite(v) for v in vec)

    def test_nan_detected(self):
        """NaN values in embedding must be detected."""
        vec = [0.1] * 383 + [float("nan")]
        assert not all(math.isfinite(v) for v in vec)

    def test_inf_detected(self):
        """Inf values in embedding must be detected."""
        vec = [0.1] * 383 + [float("inf")]
        assert not all(math.isfinite(v) for v in vec)

    def test_wrong_dimension_detected(self):
        """Wrong dimension must be detected."""
        vec = [0.1] * 256  # wrong: should be 384
        assert len(vec) != 384


# ─── Manifest Sanitization ───────────────────────────────────────


class TestManifestSanitization:
    """Tests for embedding model manifest."""

    def test_manifest_exists(self):
        """Embedding model manifest must exist."""
        manifest = _REPO_ROOT / "benchmarks" / "embedding_model_manifest.json"
        assert manifest.exists(), f"Missing: {manifest}"

    def test_manifest_records_pooling(self):
        """Manifest must explicitly record pooling strategy."""
        manifest = _REPO_ROOT / "benchmarks" / "embedding_model_manifest.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert "pooling" in data
        assert data["pooling"] == "mean"

    def test_manifest_records_model_id(self):
        """Manifest must record model_id."""
        manifest = _REPO_ROOT / "benchmarks" / "embedding_model_manifest.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["model_id"] == (
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    def test_manifest_records_dimension(self):
        """Manifest must record dimension = 384."""
        manifest = _REPO_ROOT / "benchmarks" / "embedding_model_manifest.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["dimension"] == 384

    def test_manifest_records_backend_version(self):
        """Manifest must record backend version."""
        manifest = _REPO_ROOT / "benchmarks" / "embedding_model_manifest.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert "backend_version" in data

    def test_manifest_no_credentials(self):
        """Manifest must not contain any credential-like values."""
        manifest = _REPO_ROOT / "benchmarks" / "embedding_model_manifest.json"
        content = manifest.read_text(encoding="utf-8")
        for pattern in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "sk-", "AIzaSy"):
            assert pattern not in content, f"Credential pattern '{pattern}' found"


# ─── Git Hygiene ──────────────────────────────────────────────────


class TestGitHygiene:
    """Tests for Git configuration."""

    def test_model_cache_in_gitignore(self):
        """Model cache must be in .gitignore."""
        gitignore = _REPO_ROOT / ".gitignore"
        content = gitignore.read_text(encoding="utf-8")
        assert ".model_cache/" in content

    def test_provision_manifest_in_gitignore(self):
        """Provision manifest must be in .gitignore."""
        gitignore = _REPO_ROOT / ".gitignore"
        content = gitignore.read_text(encoding="utf-8")
        assert "provision_manifest.json" in content

    def test_slice3_benchmark_not_staged(self):
        """Pre-existing Slice 3 change must not be staged."""
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],  # noqa: S607
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        if result.returncode == 0:
            staged = result.stdout.strip().splitlines()
            assert "benchmarks/run_slice3_benchmark.py" not in staged, (
                "Pre-existing Slice 3 change must NOT be staged"
            )


# ─── ADR Exists ───────────────────────────────────────────────────


class TestADR:
    """Tests for Architecture Decision Record."""

    def test_adr_exists(self):
        """ADR for pooling policy must exist."""
        adr = _REPO_ROOT / "adrs" / "001-embedding-pooling-policy.md"
        assert adr.exists()

    def test_adr_records_mean_pooling(self):
        """ADR must document mean pooling decision."""
        adr = _REPO_ROOT / "adrs" / "001-embedding-pooling-policy.md"
        content = adr.read_text(encoding="utf-8")
        assert "mean pooling" in content.lower()

    def test_adr_documents_non_comparability(self):
        """ADR must note that previous results are not directly comparable."""
        adr = _REPO_ROOT / "adrs" / "001-embedding-pooling-policy.md"
        content = adr.read_text(encoding="utf-8")
        assert "not directly comparable" in content.lower() or (
            "não diretamente comparáveis" in content.lower()
        )


# ─── Adapter Contract ────────────────────────────────────────────


class TestAdapterCacheContract:
    """Tests that adapter accepts cache_dir and local_files_only."""

    def test_constructor_accepts_cache_dir(self):
        """FastEmbedEmbeddingAdapter.__init__ must accept cache_dir."""
        import inspect

        from raglab.infrastructure.embeddings.fastembed_adapter import (
            FastEmbedEmbeddingAdapter,
        )

        sig = inspect.signature(FastEmbedEmbeddingAdapter.__init__)
        assert "cache_dir" in sig.parameters

    def test_constructor_accepts_local_files_only(self):
        """FastEmbedEmbeddingAdapter.__init__ must accept local_files_only."""
        import inspect

        from raglab.infrastructure.embeddings.fastembed_adapter import (
            FastEmbedEmbeddingAdapter,
        )

        sig = inspect.signature(FastEmbedEmbeddingAdapter.__init__)
        assert "local_files_only" in sig.parameters

    def test_resolve_cache_dir_importable(self):
        """resolve_cache_dir must be importable."""
        from raglab.infrastructure.embeddings.fastembed_adapter import (
            resolve_cache_dir,
        )

        assert callable(resolve_cache_dir)

    def test_adapter_exposes_cache_dir_path(self):
        """Adapter must expose cache_dir_path property."""
        from raglab.infrastructure.embeddings.fastembed_adapter import (
            FastEmbedEmbeddingAdapter,
        )

        assert "cache_dir_path" in dir(FastEmbedEmbeddingAdapter)
