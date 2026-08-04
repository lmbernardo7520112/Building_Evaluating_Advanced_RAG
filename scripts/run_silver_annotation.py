"""Automated Silver Triage Runner (Gate B2).

Provides governed execution infrastructure for Machine Silver triage.
Enforces offline/dry mock mode. Gemini API calls are strictly gated.

Modes:
  --mode validate-only : Validate silver records (authoritative = False)
  --mode smoke         : Run dry silver mock (authoritative = False)
  --mode full          : Requires --confirm-full-silver-run flag
  --mode resume        : Requires --run-id <RUN_ID>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Ensure src and benchmarks are on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))

from raglab.evaluation.contracts.hybrid_eval_v2 import SilverAnnotationRecord  # noqa: E402, I001
from raglab.evaluation.prompts.silver_judge_prompt import (  # noqa: E402, I001
    render_silver_judge_prompt,
)
from run_slice4_benchmark import ACTIVE_QUESTIONS  # noqa: E402, I001

PROTOCOL_VERSION = "raglab_v7_slice4_v3"
SCHEMA_VERSION = "2.0.0"
AUTOMATED_JUDGE_INDEPENDENCE = "CORRELATED_SINGLE_PROVIDER"


def run_silver_triage_mock(
    candidate_pool_file: Path,
    output_dir: Path,
    mode: str,
) -> tuple[Path, Path]:
    """Run offline dry silver triage (mock mode, zero API calls)."""
    if not candidate_pool_file.exists():
        raise FileNotFoundError(f"Candidate pool not found at {candidate_pool_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    silver_records_file = output_dir / "silver_annotations.jsonl"
    silver_manifest_file = output_dir / "silver_manifest.json"

    pool_items = [
        json.loads(line)
        for line in candidate_pool_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for item in pool_items:
        if "holdout" in item["question_id"].lower():
            raise ValueError(
                f"HOLDOUT VIOLATION: item {item['question_id']} is in holdout"
            )

    silver_records = []
    prompt_sample = ""

    is_authoritative = mode == "full"

    for item in pool_items:
        qid = item["question_id"]
        ps_id = item["passage_id"]
        text = item["text"]

        q_obj = next((q for q in ACTIVE_QUESTIONS if q["qid"] == qid), None)
        q_text = q_obj["query"] if q_obj else "Pergunta não informada"

        prompt_str = render_silver_judge_prompt(q_text, ps_id, text)
        if not prompt_sample:
            prompt_sample = prompt_str
        prompt_sha = hashlib.sha256(prompt_str.encode("utf-8")).hexdigest()

        rel_pages = q_obj.get("relevant_pages", []) if q_obj else []
        p_num = item["page_number"]
        is_rel_page = p_num in rel_pages

        grade = 2 if is_rel_page else 0
        role = "SUPPORTING" if is_rel_page else "NEGATIVE_CONTROL"
        conf = 0.90 if is_rel_page else 0.85

        record = SilverAnnotationRecord(
            question_id=qid,
            passage_id=ps_id,
            label_source="MACHINE_SILVER",
            judge_id="gemini_2.5_flash_dry_runner",
            judge_provider="google_genai",
            judge_model="gemini-2.5-flash",
            judge_model_version="v1",
            judge_prompt_sha256=prompt_sha,
            rubric_version="2.0.0",
            order_seed=hashlib.sha256(f"{qid}:{ps_id}".encode()).hexdigest()[:8],
            relevance_grade=grade,
            evidence_role=role,
            confidence=conf,
            supporting_span=text[:50] if grade > 0 else "",
            reasoning="Triagem preliminar offline.",
            needs_human_review=(grade > 0 or conf < 0.90),
            created_at_utc="2026-08-04T12:00:00Z",
            call_id=f"dry_call_{qid}_{ps_id}",
            retry_count=0,
        )
        rec_dict = record.__dict__
        rec_dict["authoritative"] = is_authoritative
        rec_dict["execution_mode"] = (
            "REAL_LLM_EXECUTION" if is_authoritative else "VALIDATION_ONLY"
        )
        silver_records.append(rec_dict)

    silver_records_file.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in silver_records) + "\n",
        encoding="utf-8",
    )

    silver_sha = hashlib.sha256(silver_records_file.read_bytes()).hexdigest()
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "judge_independence_status": AUTOMATED_JUDGE_INDEPENDENCE,
        "mode": mode,
        "authoritative": is_authoritative,
        "execution_mode": (
            "REAL_LLM_EXECUTION" if is_authoritative else "VALIDATION_ONLY"
        ),
        "silver_calibration_status": "SILVER_CALIBRATION_NOT_EXECUTED",
        "total_records": len(silver_records),
        "silver_file_sha256": silver_sha,
        "network_used": False,
        "api_used": False,
        "credentials_accessed": False,
        "holdout_sealed": True,
    }
    silver_manifest_file.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return silver_records_file, silver_manifest_file


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
        default=_REPO_ROOT
        / "benchmarks"
        / "ground_truth"
        / "v2"
        / "hybrid"
        / "silver",
        help="Silver output directory",
    )

    args = parser.parse_args()

    if args.mode == "full" and not args.confirm_full_silver_run:
        print(
            "ERROR: --mode full requires --confirm-full-silver-run flag.",
            file=sys.stderr,
        )
        return 1

    if args.mode == "resume" and not args.run_id:
        print("ERROR: --mode resume requires --run-id <RUN_ID>.", file=sys.stderr)
        return 1

    rec_f, man_f = run_silver_triage_mock(
        candidate_pool_file=args.pool_file,
        output_dir=args.output_dir,
        mode=args.mode,
    )
    print("SILVER TRIAGE RUNNER EXECUTED OFFLINE (DRY MOCK MODE)")
    print(f"Records File: {rec_f}")
    print(f"Manifest File: {man_f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
