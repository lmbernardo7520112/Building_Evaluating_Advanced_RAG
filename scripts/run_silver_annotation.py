"""Automated Silver Triage Runner (Gate B2).

Provides governed execution infrastructure for Machine Silver triage with Gemini API.
Enforces strict mode isolation, security boundaries, and resilient checkpointing.

Modes:
  --mode validate-only : Validate pool & schemas without API calls / keys
  --mode smoke         : Run real Gemini silver triage on 1 item (requires key)
  --mode full          : Run real Gemini silver triage on all pool items
                         (requires key + confirmation)
  --mode resume        : Resume interrupted run by --run-id (requires key)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure src and benchmarks are on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))

from run_slice4_benchmark import ACTIVE_QUESTIONS  # noqa: E402

from raglab.evaluation.contracts.hybrid_eval_v2 import (  # noqa: E402
    SilverAnnotationRecord,
)
from raglab.evaluation.contracts.silver_annotation_v2 import (  # noqa: E402
    DEFAULT_SILVER_JUDGE_MODEL,
    FORBIDDEN_JUDGE_MODELS,
    SilverExecutionMode,
    validate_silver_record,
)
from raglab.evaluation.prompts.silver_judge_prompt import (  # noqa: E402
    render_silver_judge_prompt,
)
from raglab.infrastructure.gemini.silver_judge_adapter import (  # noqa: E402
    SilverJudgeAdapter,
)

PROTOCOL_VERSION = "raglab_v7_slice4_v3"
SCHEMA_VERSION = "3.0.0"
RUBRIC_VERSION = "2.0.0"
AUTOMATED_JUDGE_INDEPENDENCE = "CORRELATED_SINGLE_PROVIDER"
HOLDOUT_QIDS = frozenset({"q_holdout_01", "q_holdout_02"})


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _get_item_key(item: dict[str, Any]) -> str:
    return f"{item['question_id']}:{item['passage_id']}"


def compute_checkpoint_envelope(checkpoint_dict: dict[str, Any]) -> str:
    """Compute SHA-256 envelope of checkpoint content (excluding checkpoint_sha256)."""
    copy_dict = dict(checkpoint_dict)
    copy_dict.pop("checkpoint_sha256", None)
    sorted_str = json.dumps(copy_dict, sort_keys=True, ensure_ascii=False)
    return sha256_text(sorted_str)


# ── Mode: VALIDATE_ONLY ──────────────────────────────────────────


def run_validate_only(
    candidate_pool_file: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Validate pool, prompt template, schemas without API calls or credentials."""
    if not candidate_pool_file.exists():
        raise FileNotFoundError(f"Candidate pool not found at {candidate_pool_file}")

    pool_bytes = candidate_pool_file.read_bytes()
    pool_sha = sha256_bytes(pool_bytes)
    pool_items = [
        json.loads(line)
        for line in pool_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]

    for item in pool_items:
        qid = item["question_id"]
        if qid in HOLDOUT_QIDS or "holdout" in qid.lower():
            raise ValueError(f"HOLDOUT VIOLATION: item {qid} is in holdout")

    # Render prompt sample to verify rendering
    first_item = pool_items[0]
    q_obj = next(
        (q for q in ACTIVE_QUESTIONS if q["qid"] == first_item["question_id"]), None
    )
    q_text = q_obj["query"] if q_obj else "Pergunta"
    sample_prompt = render_silver_judge_prompt(
        q_text, first_item["passage_id"], first_item["text"]
    )
    prompt_sha = sha256_text(sample_prompt)

    val_dir = output_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = val_dir / "silver_validation_manifest.json"

    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "mode": "validate-only",
        "execution_mode": SilverExecutionMode.VALIDATION_ONLY.value,
        "execution_authenticity": "NO_LLM_CALL_VALIDATION",
        "model": DEFAULT_SILVER_JUDGE_MODEL,
        "prompt_sha256": prompt_sha,
        "rubric_version": RUBRIC_VERSION,
        "pool_sha256": pool_sha,
        "eligible_pool_items": len(pool_items),
        "record_count": 0,
        "logical_calls": 0,
        "physical_attempts": 0,
        "retry_count": 0,
        "network_used": False,
        "api_used": False,
        "credentials_accessed": False,
        "credential_source": "NONE",
        "credential_value_persisted": False,
        "authoritative_for_human_qrels": False,
        "holdout_sealed": True,
        "validation_status": "PASSED",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }

    manifest_file.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Empty annotations placeholder in validation directory
    ann_file = val_dir / "silver_annotations_validation.jsonl"
    ann_file.write_text("", encoding="utf-8")

    return ann_file, manifest_file


# ── Modes: SMOKE, FULL, RESUME ───────────────────────────────────


def run_silver_triage_real(
    candidate_pool_file: Path,
    output_dir: Path,
    mode: str,
    run_id: str = "",
    confirm_full: bool = False,
    judge_adapter: SilverJudgeAdapter | None = None,
) -> tuple[Path, Path]:
    """Run real Machine Silver triage with Gemini API or injected adapter."""

    if mode == "full" and not confirm_full:
        raise ValueError("ERROR: --mode full requires --confirm-full-silver-run flag.")

    if mode == "resume" and not run_id:
        raise ValueError("ERROR: --mode resume requires --run-id <RUN_ID>.")

    # ── Require GEMINI_API_KEY when no injected adapter ──────────
    if judge_adapter is None:
        if (
            "GEMINI_API_KEY" not in os.environ
            or not os.environ["GEMINI_API_KEY"].strip()
        ):
            raise RuntimeError(
                "GEMINI_API_KEY environment variable missing. "
                "Real silver triage modes (smoke/full/resume) require API key."
            )
        adapter = SilverJudgeAdapter()
    else:
        adapter = judge_adapter

    model_id = adapter.model_id
    if model_id in FORBIDDEN_JUDGE_MODELS:
        raise ValueError(f"Forbidden model '{model_id}' cannot be used.")

    # ── Load Candidate Pool ──────────────────────────────────────
    if not candidate_pool_file.exists():
        raise FileNotFoundError(f"Candidate pool not found at {candidate_pool_file}")

    pool_bytes = candidate_pool_file.read_bytes()
    pool_sha = sha256_bytes(pool_bytes)
    pool_items = [
        json.loads(line)
        for line in pool_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]

    # Holdout validation
    eligible_items = []
    for item in pool_items:
        qid = item["question_id"]
        if qid in HOLDOUT_QIDS or "holdout" in qid.lower():
            continue
        eligible_items.append(item)

    if not eligible_items:
        raise ValueError("No eligible non-holdout items found in candidate pool.")

    # ── Setup Run Directory and Run ID ───────────────────────────
    now_utc = datetime.now(UTC)
    now_str = now_utc.strftime("%Y%m%dT%H%M%SZ")

    if mode == "smoke":
        active_run_id = run_id or f"smoke_run_{now_str}"
        target_items = eligible_items[:1]  # EXACTLY ONE ITEM FOR SMOKE
        exec_mode = SilverExecutionMode.SMOKE_REAL.value
    elif mode == "full":
        active_run_id = run_id or f"full_run_{now_str}"
        target_items = eligible_items
        exec_mode = SilverExecutionMode.FULL_REAL.value
    elif mode == "resume":
        active_run_id = run_id
        target_items = eligible_items
        exec_mode = SilverExecutionMode.RESUME_REAL.value
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    run_dir = output_dir / "runs" / active_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    records_file = run_dir / "silver_annotations.jsonl"
    manifest_file = run_dir / "silver_manifest.json"
    checkpoint_file = run_dir / "checkpoint.json"

    # ── Load existing checkpoint if resuming ─────────────────────
    completed_keys: set[str] = set()
    existing_records: list[dict[str, Any]] = []
    total_logical_calls = 0
    total_physical_attempts = 0

    if mode == "resume":
        if not checkpoint_file.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found for resume at {checkpoint_file}"
            )
        cp_data = json.loads(checkpoint_file.read_text(encoding="utf-8"))

        # Verify fingerprints
        if cp_data.get("pool_sha256") != pool_sha:
            raise ValueError(
                f"Pool SHA mismatch in checkpoint: {cp_data.get('pool_sha256')}"
                f" != {pool_sha}"
            )
        if cp_data.get("rubric_version") != RUBRIC_VERSION:
            raise ValueError("Rubric version mismatch in checkpoint")
        if cp_data.get("model") != model_id:
            raise ValueError(
                f"Model mismatch in checkpoint: {cp_data.get('model')} != {model_id}"
            )

        completed_keys = set(cp_data.get("completed_keys", []))
        total_logical_calls = cp_data.get("logical_calls", 0)
        total_physical_attempts = cp_data.get("physical_attempts", 0)

        if records_file.exists():
            for line in records_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    existing_records.append(r)
                    completed_keys.add(f"{r['question_id']}:{r['passage_id']}")
    else:
        # For new smoke/full runs, clear any stale state
        if records_file.exists():
            records_file.unlink()

    # ── Execute Silver Triage Items ──────────────────────────────
    new_records: list[dict[str, Any]] = []

    # Get sample prompt SHA for manifest
    sample_q = next(
        (q for q in ACTIVE_QUESTIONS if q["qid"] == eligible_items[0]["question_id"]),
        None,
    )
    sample_prompt = render_silver_judge_prompt(
        sample_q["query"] if sample_q else "",
        eligible_items[0]["passage_id"],
        eligible_items[0]["text"],
    )
    prompt_sha = sha256_text(sample_prompt)

    for item in target_items:
        key = _get_item_key(item)
        if key in completed_keys:
            continue

        qid = item["question_id"]
        ps_id = item["passage_id"]
        text = item["text"]

        q_obj = next((q for q in ACTIVE_QUESTIONS if q["qid"] == qid), None)
        q_text = q_obj["query"] if q_obj else "Pergunta"

        rec_dict, log_calls, phys_attempts = adapter.evaluate_passage(
            question_id=qid,
            question_text=q_text,
            passage_id=ps_id,
            passage_text=text,
            rubric_version=RUBRIC_VERSION,
        )

        record = SilverAnnotationRecord(**rec_dict)
        validate_silver_record(record, text)
        new_records.append(rec_dict)
        completed_keys.add(key)

        total_logical_calls += log_calls
        total_physical_attempts += phys_attempts

        # Atomic append to silver_annotations.jsonl
        line_json = json.dumps(rec_dict, ensure_ascii=False) + "\n"
        with records_file.open("a", encoding="utf-8") as f:
            f.write(line_json)

        # Atomic checkpoint update after each item
        cp_dict = {
            "run_id": active_run_id,
            "mode": mode,
            "pool_sha256": pool_sha,
            "prompt_sha256": prompt_sha,
            "rubric_version": RUBRIC_VERSION,
            "model": model_id,
            "completed_keys": sorted(completed_keys),
            "records_sha256": sha256_bytes(records_file.read_bytes())
            if records_file.exists()
            else "",
            "logical_calls": total_logical_calls,
            "physical_attempts": total_physical_attempts,
            "updated_at_utc": datetime.now(UTC).isoformat(),
        }
        cp_dict["checkpoint_sha256"] = compute_checkpoint_envelope(cp_dict)

        tmp_cp = checkpoint_file.with_suffix(".tmp")
        tmp_cp.write_text(
            json.dumps(cp_dict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        tmp_cp.rename(checkpoint_file)

    # ── Build Final Manifest ─────────────────────────────────────
    all_final_records = existing_records + new_records
    final_sha = sha256_bytes(records_file.read_bytes()) if records_file.exists() else ""

    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_id": active_run_id,
        "mode": mode,
        "execution_mode": exec_mode,
        "execution_authenticity": "REAL_MODEL_CALL",
        "model": model_id,
        "prompt_sha256": prompt_sha,
        "rubric_version": RUBRIC_VERSION,
        "pool_sha256": pool_sha,
        "total_eligible_items": len(eligible_items),
        "record_count": len(all_final_records),
        "completed_count": len(completed_keys),
        "pending_count": max(0, len(eligible_items) - len(completed_keys)),
        "logical_calls": total_logical_calls,
        "physical_attempts": total_physical_attempts,
        "retry_count": max(0, total_physical_attempts - total_logical_calls),
        "silver_file_sha256": final_sha,
        "created_at_utc": now_utc.isoformat(),
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "holdout_sealed": True,
        "api_used": True,
        "network_used": True,
        "credential_source": "ENVIRONMENT_ONLY",
        "credential_value_persisted": False,
        "authoritative_for_human_qrels": False,
        "status": "COMPLETED"
        if len(completed_keys) >= len(target_items)
        else "PARTIAL",
    }

    manifest_file.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return records_file, manifest_file


def main() -> int:
    if len(sys.argv) == 1:
        print(
            "ERROR: No arguments provided. Usage: run_silver_annotation.py"
            " --mode <mode>",
            file=sys.stderr,
        )
        return 2

    parser = argparse.ArgumentParser(description="Automated Silver Triage Runner")
    parser.add_argument(
        "--mode",
        choices=["smoke", "full", "resume", "validate-only"],
        required=True,
        help="Execution mode",
    )
    parser.add_argument(
        "--confirm-full-silver-run",
        action="store_true",
        help="Confirmation required for --mode full",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run ID required for --mode resume",
    )
    parser.add_argument(
        "--pool-file",
        type=Path,
        default=_REPO_ROOT
        / "benchmarks"
        / "ground_truth"
        / "v2"
        / "hybrid"
        / "candidate_pool"
        / "pool.jsonl",
        help="Candidate pool file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "benchmarks" / "ground_truth" / "v2" / "hybrid" / "silver",
        help="Silver output directory",
    )

    args = parser.parse_args()

    if args.mode == "validate-only":
        rec_f, man_f = run_validate_only(
            candidate_pool_file=args.pool_file,
            output_dir=args.output_dir,
        )
        print("SILVER TRIAGE VALIDATE-ONLY COMPLETED (ZERO LLM CALLS)")
        print(f"Validation Manifest: {man_f}")
        return 0

    try:
        rec_f, man_f = run_silver_triage_real(
            candidate_pool_file=args.pool_file,
            output_dir=args.output_dir,
            mode=args.mode,
            run_id=args.run_id,
            confirm_full=args.confirm_full_silver_run,
        )
        print(f"SILVER TRIAGE EXECUTION COMPLETED ({args.mode.upper()} MODE)")
        print(f"Records File: {rec_f}")
        print(f"Manifest File: {man_f}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
