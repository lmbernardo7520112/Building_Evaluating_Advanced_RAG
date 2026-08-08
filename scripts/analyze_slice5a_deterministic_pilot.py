"""CLI: Analyze deterministic pilot results.

Post-execution analysis — may access qrels for evaluation.
Takes pilot output and produces analysis artifacts.

Usage:
  python scripts/analyze_slice5a_deterministic_pilot.py \
    --pilot-result benchmarks/agentic/slice5/pilot_output/pilot_result.json \
    --output-dir benchmarks/agentic/slice5/pilot_analysis
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def analyze_pilot(
    pilot_result_path: Path,
    output_dir: Path,
) -> int:
    """Analyze deterministic pilot results.

    Returns 0 on success, non-zero on failure.
    """
    with open(pilot_result_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    routing = data.get("routing_decisions", [])
    errors = data.get("errors", [])

    if not results:
        print("ERROR: No results in pilot output", file=sys.stderr)
        return 1

    # Routing distribution
    strategy_counts: dict[str, int] = defaultdict(int)
    fallback_count = 0
    for rd in routing:
        strategy_counts[rd["selected_strategy"]] += 1
        if rd.get("fallback_used"):
            fallback_count += 1

    # Evidence stats
    evidence_counts = [r["evidence_count"] for r in results]
    avg_evidence = sum(evidence_counts) / len(evidence_counts) if evidence_counts else 0

    # Split analysis
    dev_results = [r for r in results if r.get("split") == "development"]
    test_results = [r for r in results if r.get("split") == "test"]

    analysis = {
        "pilot_result_sha256": _file_sha256(pilot_result_path),
        "total_questions": len(results),
        "dev_questions": len(dev_results),
        "test_questions": len(test_results),
        "error_count": len(errors),
        "strategy_distribution": dict(strategy_counts),
        "fallback_count": fallback_count,
        "evidence_stats": {
            "total": sum(evidence_counts),
            "average": avg_evidence,
            "min": min(evidence_counts) if evidence_counts else 0,
            "max": max(evidence_counts) if evidence_counts else 0,
        },
        "generation_status": "NOT_EXECUTED",
        "per_question": [
            {
                "qid": r["qid"],
                "split": r["split"],
                "strategy": r["strategy_selected"],
                "evidence_count": r["evidence_count"],
                "stop_reason": r["stop_reason"],
                "error": r.get("error"),
            }
            for r in results
        ],
    }

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "pilot_analysis.json"
    out_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print("DETERMINISTIC PILOT ANALYSIS")
    print(f"{'=' * 60}")
    print(
        f"Questions: {len(results)} ({len(dev_results)} dev, {len(test_results)} test)"
    )
    print(f"Errors: {len(errors)}")
    print(f"Fallbacks: {fallback_count}")
    print(f"Evidence: avg={avg_evidence:.1f}")
    print("\nStrategy distribution:")
    for s, c in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        print(f"  {s:30s} {c}")
    print("\nGeneration: NOT_EXECUTED")
    print(f"Output: {out_path}")
    print(f"{'=' * 60}\n")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze deterministic pilot results",
    )
    parser.add_argument(
        "--pilot-result",
        type=Path,
        required=True,
        help="Path to pilot_result.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for analysis",
    )

    args = parser.parse_args()

    if not args.pilot_result.exists():
        print(f"ERROR: Pilot result not found: {args.pilot_result}", file=sys.stderr)
        sys.exit(1)

    exit_code = analyze_pilot(args.pilot_result, args.output_dir)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
