"""RAGLab CLI — minimal command-line interface.

Commands:
- raglab smoke     : Run deterministic smoke test with tiny corpus
- raglab doctor    : Check dependencies and controls
- raglab --version : Print version

No API calls, no model downloads, no credentials required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import raglab


def _find_project_root() -> Path:
    """Find the project root by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _run_smoke(args: argparse.Namespace) -> int:
    """Run the deterministic smoke test.

    1. Load tiny corpus
    2. Build baseline deterministic index
    3. Retrieve evidence for each question
    4. Compute Recall@k and MRR
    5. Save checkpoint
    6. Test resume
    7. Produce JSON summary
    8. Return 0 only if all checks pass
    """
    from raglab.domain.entities import Checkpoint
    from raglab.domain.metrics import compute_mrr, compute_recall_at_k
    from raglab.domain.value_objects import IntegrityDigest, RunId
    from raglab.infrastructure.persistence.checkpoint_store import (
        FilesystemCheckpointStore,
    )
    from raglab.infrastructure.retrieval.baseline_adapter import (
        InMemoryBaselineAdapter,
        load_tiny_corpus,
    )

    project_root = _find_project_root()
    corpus_path = project_root / "data" / "tiny_corpus" / "corpus.json"
    manifest_path = project_root / "data" / "tiny_corpus" / "manifest.json"

    results: dict[str, Any] = {
        "command": "raglab smoke",
        "version": raglab.__version__,
        "checks": [],
        "overall": "PENDING",
    }
    all_passed = True

    def _check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal all_passed
        status = "PASSED" if passed else "FAILED"
        results["checks"].append(
            {"check": name, "status": status, "detail": detail}
        )
        if not passed:
            all_passed = False
        detail_msg = f" — {detail}" if detail else ""
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name}: {status}{detail_msg}")

    print("RAGLab Smoke Test")
    print("=" * 50)

    # 1. Verify corpus exists and integrity
    _check("corpus_exists", corpus_path.exists(), str(corpus_path))
    if not corpus_path.exists():
        results["overall"] = "FAILED"
        print(json.dumps(results, indent=2))
        return 1

    corpus_hash = hashlib.sha256(
        corpus_path.read_bytes()
    ).hexdigest()

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected_hash = manifest.get("sha256", "")
        _check(
            "corpus_integrity",
            corpus_hash == expected_hash,
            f"expected={expected_hash[:16]}... got={corpus_hash[:16]}...",
        )
    else:
        _check("corpus_integrity", False, "manifest not found")

    # 2. Load corpus
    chunks, corpus_data = load_tiny_corpus(str(corpus_path))
    _check("corpus_loaded", len(chunks) > 0, f"{len(chunks)} chunks")

    # 3. Build baseline index according to selected backend
    backend = getattr(args, "backend", "deterministic")
    if backend == "llamaindex":
        from raglab.infrastructure.retrieval.llamaindex_adapter import (
            LlamaIndexBaselineAdapter,
        )
        adapter: Any = LlamaIndexBaselineAdapter()
    else:
        adapter = InMemoryBaselineAdapter()

    adapter.index_chunks(chunks)
    _check(
        "index_built",
        True,
        f"{len(chunks)} chunks indexed (backend={backend})",
    )

    # 4. Retrieve and evaluate
    questions = corpus_data.get("questions", [])
    _check("questions_loaded", len(questions) > 0, f"{len(questions)} questions")

    mrr_queries: list[tuple[list[str], set[str]]] = []
    recall_results: list[dict[str, Any]] = []
    top_k = 3

    for q in questions:
        retrieved = adapter.retrieve(q["text"], top_k=top_k)
        retrieved_ids = [r.chunk_id.value for r in retrieved]
        relevant_ids = set(q.get("relevant_chunks", []))

        recall = compute_recall_at_k(retrieved_ids, relevant_ids, k=top_k)
        mrr_queries.append((retrieved_ids, relevant_ids))

        recall_results.append({
            "question_id": q["question_id"],
            "recall_at_k": recall.value,
            "retrieved_relevant": recall.retrieved_relevant,
            "total_relevant": recall.total_relevant,
            "skipped": recall.skipped,
            "is_abstention": q.get("abstention_expected", False),
        })

    mrr_result = compute_mrr(mrr_queries)

    _check(
        "recall_computed",
        all(r["recall_at_k"] is not None or r["skipped"] for r in recall_results),
        f"{len(recall_results)} queries evaluated",
    )
    _check(
        "mrr_computed",
        mrr_result.value is not None,
        f"MRR={mrr_result.value:.4f}" if mrr_result.value is not None else "N/A",
    )

    results["metrics"] = {
        "recall_at_k": [
            {
                "qid": r["question_id"],
                "value": r["recall_at_k"],
                "skipped": r["skipped"],
            }
            for r in recall_results
        ],
        "mrr": mrr_result.value,
        "top_k": top_k,
    }

    # 5. Checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        store = FilesystemCheckpointStore(tmpdir)
        config_fp = IntegrityDigest("c" * 64)
        corpus_fp = IntegrityDigest(corpus_hash)
        run_id = RunId("smoke-test-run")

        cp = Checkpoint(
            run_id=run_id,
            corpus_fingerprint=corpus_fp,
            config_fingerprint=config_fp,
            completed_query_ids=frozenset(q["question_id"] for q in questions),
        )
        store.save(cp)
        _check("checkpoint_saved", store.exists(run_id))

        # 6. Test resume
        loaded = store.load(run_id)
        _check(
            "checkpoint_resumed",
            loaded is not None
            and loaded.completed_query_ids == cp.completed_query_ids,
        )

    # 7. Abstention check
    abstention_qs = [q for q in questions if q.get("abstention_expected")]
    if abstention_qs:
        _check(
            "abstention_question_present",
            True,
            f"{len(abstention_qs)} abstention question(s)",
        )
    else:
        _check(
            "abstention_question_present",
            False,
            "no abstention questions in corpus",
        )

    # 8. Determinism check
    retrieved1 = adapter.retrieve(questions[0]["text"], top_k=top_k)
    retrieved2 = adapter.retrieve(questions[0]["text"], top_k=top_k)
    ids1 = [r.chunk_id.value for r in retrieved1]
    ids2 = [r.chunk_id.value for r in retrieved2]
    _check("deterministic", ids1 == ids2)

    # Summary
    results["overall"] = "PASSED" if all_passed else "FAILED"

    print()
    print("Summary")
    print("-" * 50)
    print(json.dumps(results, indent=2))

    return 0 if all_passed else 1


def _run_doctor(args: argparse.Namespace) -> int:
    """Check dependencies and controls."""
    print("RAGLab Doctor")
    print("=" * 50)

    checks: list[dict[str, str]] = []

    def _check(name: str, status: str, detail: str = "") -> None:
        checks.append({"check": name, "status": status, "detail": detail})
        status_icons = {
            "PASSED": "✅",
            "FAILED": "❌",
            "NOT_CONFIGURED": "⚠️",
            "NOT_EXECUTED": "⏸️",
        }
        icon = status_icons.get(status, "?")
        detail_msg = f" — {detail}" if detail else ""
        print(f"  {icon} {name}: {status}{detail_msg}")

    # Python version
    v_info = sys.version_info
    py_ver = f"{v_info.major}.{v_info.minor}.{v_info.micro}"
    _check("python_version", "PASSED" if v_info >= (3, 11) else "FAILED", py_ver)

    # pytest
    try:
        import pytest
        _check("pytest", "PASSED", f"v{pytest.__version__}")
    except ImportError:
        _check("pytest", "NOT_CONFIGURED")

    # ruff
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "--version"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode == 0:
            _check("ruff", "PASSED", result.stdout.strip())
        else:
            _check("ruff", "FAILED", result.stderr.strip()[:80])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _check("ruff", "NOT_CONFIGURED")

    # mypy
    try:
        from mypy.version import __version__ as mypy_ver
        _check("mypy", "PASSED", f"v{mypy_ver}")
    except ImportError:
        _check("mypy", "NOT_CONFIGURED")

    # pip-audit
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip_audit", "--version"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode == 0:
            _check("pip_audit", "PASSED", result.stdout.strip()[:60])
        else:
            _check("pip_audit", "FAILED")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _check("pip_audit", "NOT_CONFIGURED")

    # Reference integrity
    project_root = _find_project_root()
    verify_script = project_root / "scripts" / "verify_reference.py"
    if verify_script.exists():
        try:
            result = subprocess.run(  # noqa: S603
                [sys.executable, str(verify_script)],
                capture_output=True, text=True, timeout=30, check=False,
                cwd=str(project_root),
            )
            ok = result.returncode == 0
            _check("reference_integrity", "PASSED" if ok else "FAILED")
        except subprocess.TimeoutExpired:
            _check("reference_integrity", "FAILED", "timeout")
    else:
        _check("reference_integrity", "NOT_CONFIGURED")

    # Secret scan
    scan_script = project_root / "scripts" / "scan_secrets.py"
    if scan_script.exists():
        try:
            result = subprocess.run(  # noqa: S603
                [sys.executable, str(scan_script)],
                capture_output=True, text=True, timeout=30, check=False,
                cwd=str(project_root),
            )
            _check("secret_scan", "PASSED" if result.returncode == 0 else "FAILED")
        except subprocess.TimeoutExpired:
            _check("secret_scan", "FAILED", "timeout")
    else:
        _check("secret_scan", "NOT_CONFIGURED")

    # Tiny corpus
    corpus_path = project_root / "data" / "tiny_corpus" / "corpus.json"
    _check("tiny_corpus", "PASSED" if corpus_path.exists() else "NOT_CONFIGURED")

    # Lockfile
    lockfile = project_root / "requirements.lock"
    _check("lockfile", "PASSED" if lockfile.exists() else "NOT_CONFIGURED")

    # Git remotes
    git_bin = shutil.which("git")
    if git_bin:
        try:
            result = subprocess.run(  # noqa: S603
                [git_bin, "remote", "-v"],
                capture_output=True, text=True, timeout=5, check=False,
                cwd=str(project_root),
            )
            lines = result.stdout.strip().splitlines()
            remote_count = len(lines) if result.stdout.strip() else 0
            # Do not reveal credential values
            _check(
                "zero_remotes",
                "PASSED" if remote_count == 0 else "FAILED",
                f"{remote_count} remote(s)",
            )
        except subprocess.TimeoutExpired:
            _check("zero_remotes", "NOT_EXECUTED")
    else:
        _check("zero_remotes", "NOT_EXECUTED")

    print()
    all_ok = all(c["status"] == "PASSED" for c in checks)
    overall_status = "PASSED" if all_ok else "ISSUES_FOUND"
    print(f"Overall: {overall_status}")

    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="raglab",
        description="RAGLab — Advanced RAG Pipeline with Scientific Evaluation",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"raglab {raglab.__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Run deterministic smoke test with tiny corpus",
    )
    smoke_parser.add_argument(
        "--backend",
        choices=["deterministic", "llamaindex"],
        default="deterministic",
        help="Retrieval backend adapter to use (default: deterministic)",
    )
    smoke_parser.set_defaults(func=_run_smoke)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check dependencies and controls",
    )
    doctor_parser.set_defaults(func=_run_doctor)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    res: int = args.func(args)
    return res


if __name__ == "__main__":
    sys.exit(main())
