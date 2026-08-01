#!/usr/bin/env python3
"""Provision embedding model to persistent cache — NO CREDENTIALS ALLOWED.

This script downloads the ONNX embedding model to a persistent local cache.
It MUST be run BEFORE any scientific execution (smoke, full, preflight).
It MUST NOT be run with Gemini credentials in the environment.

Usage:
    # Show help (no model loaded, no side effects):
    .venv/bin/python scripts/provision_embedding_model.py --help

    # Dry run (shows what would happen, no side effects):
    .venv/bin/python scripts/provision_embedding_model.py

    # Execute provisioning:
    .venv/bin/python scripts/provision_embedding_model.py --execute

    # Custom cache directory:
    RAGLAB_MODEL_CACHE=/path/to/cache \\
        .venv/bin/python scripts/provision_embedding_model.py --execute

Exit codes:
    0 — Model provisioned and validated successfully (--execute), or help shown.
    1 — Error: credentials detected, cache invalid, or model load failed.
    2 — No --execute flag provided (fail-closed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# ─── Path setup ───────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ─── Constants ────────────────────────────────────────────────────
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EXPECTED_DIMENSION = 384
CANARY_TEXT = "Este é um texto canário para validação do embedding."

MANIFEST_PATH = _REPO_ROOT / "benchmarks" / "embedding_model_manifest.json"


def _abort(msg: str) -> None:
    print(f"PROVISION_ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="provision_embedding_model.py",
        description=(
            "Provision the ONNX embedding model to a persistent local cache.\n\n"
            "This script MUST be run BEFORE any benchmark execution.\n"
            "It MUST NOT be run with Gemini credentials in the environment.\n\n"
            "Without --execute, this script exits with code 2 and no side effects.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Show help:\n"
            "  .venv/bin/python scripts/provision_embedding_model.py --help\n\n"
            "  # Execute provisioning:\n"
            "  .venv/bin/python scripts/provision_embedding_model.py --execute\n\n"
            "  # Custom cache:\n"
            "  RAGLAB_MODEL_CACHE=/path/to/cache \\\\\n"
            "      .venv/bin/python scripts/provision_embedding_model.py --execute\n"
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help=(
            "Actually execute the provisioning. Without this flag, "
            "the script exits with code 2 and no side effects."
        ),
    )
    return parser


def _fingerprint_cache(cache_dir: Path) -> dict:
    """Build a deterministic fingerprint of the model cache directory."""
    entries: list[dict] = []
    if cache_dir.exists():
        for p in sorted(cache_dir.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(cache_dir))
                size = p.stat().st_size
                h = hashlib.sha256()
                with p.open("rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
                entries.append({
                    "path": rel,
                    "size_bytes": size,
                    "sha256": h.hexdigest(),
                })
    # Tree digest: hash of all individual hashes
    tree_h = hashlib.sha256()
    for e in entries:
        tree_h.update(e["sha256"].encode())
    return {
        "file_count": len(entries),
        "files": entries,
        "cache_tree_sha256": tree_h.hexdigest(),
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # --help is handled by argparse before reaching here.

    if not args.execute:
        print(
            "PROVISION: No --execute flag provided. Exiting with no side effects.\n"
            "\n"
            "To provision the embedding model, run:\n"
            "  .venv/bin/python scripts/provision_embedding_model.py --execute\n",
            file=sys.stderr,
        )
        sys.exit(2)

    # ── From here on, --execute was provided ──────────────────────
    print("=== Embedding Model Provisioning ===")
    print(f"Model: {MODEL_ID}")
    print(f"Expected dimension: {EXPECTED_DIMENSION}")

    # ── Gate 1: Reject Gemini credentials ─────────────────────────
    for key_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(key_name):
            _abort(
                f"{key_name} is set. Provisioning MUST NOT run with Gemini "
                f"credentials. Unset {key_name} and retry."
            )

    # ── Gate 2: Resolve cache directory ───────────────────────────
    from raglab.infrastructure.embeddings.fastembed_adapter import resolve_cache_dir

    try:
        cache_dir = resolve_cache_dir()
    except ValueError as exc:
        _abort(str(exc))

    print(f"Cache directory: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Gate 3: Ensure network is allowed (no HF_HUB_OFFLINE) ────
    # Temporarily allow network for download
    env_backup: dict[str, str | None] = {}
    for env_key in ("HF_HUB_OFFLINE",):
        env_backup[env_key] = os.environ.get(env_key)
        os.environ.pop(env_key, None)

    # ── Download and load model ───────────────────────────────────
    print("Downloading/loading model (network allowed)...")
    try:
        import fastembed
        import onnxruntime

        from raglab.infrastructure.embeddings.fastembed_adapter import (
            FastEmbedEmbeddingAdapter,
        )

        t0 = time.monotonic()
        adapter = FastEmbedEmbeddingAdapter(
            model_name=MODEL_ID,
            dimension=EXPECTED_DIMENSION,
            cache_dir=str(cache_dir),
            local_files_only=False,
        )
        load_time_ms = (time.monotonic() - t0) * 1000
        print(f"Model loaded in {load_time_ms:.0f} ms")
    except Exception as exc:
        _abort(f"Failed to load model: {exc}")
    finally:
        # Restore env
        for env_key, val in env_backup.items():
            if val is not None:
                os.environ[env_key] = val
            else:
                os.environ.pop(env_key, None)

    # ── Canary embedding validation ───────────────────────────────
    print(f"Generating canary embedding for: '{CANARY_TEXT[:40]}...'")
    try:
        canary_vec = adapter._embed(CANARY_TEXT)  # noqa: SLF001
    except Exception as exc:
        _abort(f"Canary embedding failed: {exc}")

    # Validate dimension
    if len(canary_vec) != EXPECTED_DIMENSION:
        _abort(
            f"Dimension mismatch: expected {EXPECTED_DIMENSION}, "
            f"got {len(canary_vec)}"
        )

    # Validate finite values
    if not all(math.isfinite(v) for v in canary_vec):
        _abort("Canary embedding contains non-finite values (NaN/Inf)")

    print(f"Canary embedding OK: dim={len(canary_vec)}, all values finite")

    # ── Cache fingerprint ─────────────────────────────────────────
    print("Computing cache fingerprint...")
    cache_fp = _fingerprint_cache(cache_dir)
    print(f"Cache files: {cache_fp['file_count']}")
    print(f"Cache tree SHA-256: {cache_fp['cache_tree_sha256']}")

    # ── Generate provisioning manifest ────────────────────────────
    manifest = {
        "provisioned_utc": datetime.now(UTC).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": "HEAD",
        "model_revision_status": "UNRESOLVED",
        "fastembed_version": fastembed.__version__,
        "onnxruntime_version": onnxruntime.__version__,
        "dimension": EXPECTED_DIMENSION,
        "pooling": "mean",
        "normalization": True,
        "cache_dir": str(cache_dir),
        "cache_tree_sha256": cache_fp["cache_tree_sha256"],
        "cache_file_count": cache_fp["file_count"],
        "cache_files": cache_fp["files"],
        "canary_text": CANARY_TEXT,
        "canary_dim_ok": len(canary_vec) == EXPECTED_DIMENSION,
        "canary_finite_ok": all(math.isfinite(v) for v in canary_vec),
        "load_time_ms": round(load_time_ms, 1),
    }

    # Sanitize: no file paths that could leak personal info
    # (cache_dir is kept since it's operator-controlled)

    manifest_output = _REPO_ROOT / "benchmarks" / "provision_manifest.json"
    manifest_output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest written: {manifest_output}")

    # ── Summary ───────────────────────────────────────────────────
    print()
    print("=== PROVISION_OK ===")
    print(f"  model_id:            {MODEL_ID}")
    print(f"  fastembed:           {fastembed.__version__}")
    print(f"  onnxruntime:         {onnxruntime.__version__}")
    print("  pooling:             mean")
    print(f"  dimension:           {EXPECTED_DIMENSION}")
    print(f"  cache_dir:           {cache_dir}")
    print(f"  cache_tree_sha256:   {cache_fp['cache_tree_sha256']}")
    print("  canary_ok:           True")
    print()
    print("Next step: run preflight to validate offline loading:")
    print("  .venv/bin/python benchmarks/run_slice4_benchmark.py --mode preflight")


if __name__ == "__main__":
    main()
