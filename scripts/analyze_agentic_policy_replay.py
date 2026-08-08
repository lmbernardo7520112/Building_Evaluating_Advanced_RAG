"""Offline policy replay for agentic routing analysis.

Two modes:
  A. CI/synthetic: uses synthetic fixtures (small, deterministic)
  B. local/authoritative: uses real Slice 4 results (not required by CI)

Output: strategy summary, oracle ceiling, routing analysis, manifesto.
Does NOT alter input files. Does NOT invent absent metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from raglab.agentic.contracts import SCHEMA_VERSION, _sha256
from raglab.agentic.enums import OracleLabel
from raglab.agentic.router import classify_query, get_deterministic_policy_metadata


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_input(path: Path) -> dict[str, Any]:
    """Load and validate the input file."""
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    if "queries" not in data:
        print(f"ERROR: Input file missing 'queries' key: {path}", file=sys.stderr)
        sys.exit(1)

    return data


def _load_config(path: Path) -> dict[str, Any]:
    """Load replay configuration."""
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
        return data


def analyze_policy_replay(
    input_path: Path,
    config_path: Path,
    output_dir: Path,
) -> int:
    """Run offline policy replay analysis.

    Returns 0 on success, non-zero on failure.
    """
    # Load inputs
    data = _load_input(input_path)
    config = _load_config(config_path)

    input_hash = _file_sha256(input_path)
    config_hash = _file_sha256(config_path)

    queries = data["queries"]
    if not queries:
        print("ERROR: No queries found in input", file=sys.stderr)
        return 1

    # Extract metrics and strategies from config
    metrics = config.get("metrics", ["ndcg_at_3"])
    primary_metric = config.get("primary_metric", metrics[0])

    # Policy metadata
    policy_meta = get_deterministic_policy_metadata()

    # --- Strategy Summary ---
    strategy_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    router_scores: dict[str, list[float]] = defaultdict(list)
    oracle_scores: dict[str, list[float]] = defaultdict(list)

    valid_queries = 0
    na_queries: list[dict[str, str]] = []
    wins = 0
    ties = 0
    losses = 0

    for q in queries:
        qid = q["query_id"]
        query_text = q.get("query_text", "")
        results = q.get("results_per_strategy", {})

        if not results:
            na_queries.append({"query_id": qid, "reason": "no_results"})
            continue

        # Collect per-strategy scores
        for strat, scores in results.items():
            for metric_name, val in scores.items():
                strategy_scores[strat][metric_name].append(val)

        # Router decision
        query_class = classify_query(query_text)
        from raglab.agentic.router import _STRATEGY_MAP

        routed_strategy = _STRATEGY_MAP.get(query_class.value)

        if routed_strategy and routed_strategy in results:
            routed_scores = results[routed_strategy]
            for m in metrics:
                if m in routed_scores:
                    router_scores[m].append(routed_scores[m])
        else:
            na_queries.append(
                {
                    "query_id": qid,
                    "reason": f"routed_strategy '{routed_strategy}' not in results",
                }
            )
            continue

        # Oracle (post-hoc best per query)
        for m in metrics:
            best_val = max(
                (results[s].get(m, 0.0) for s in results if m in results.get(s, {})),
                default=0.0,
            )
            oracle_scores[m].append(best_val)

        # Win/tie/loss vs fixed-best baseline
        fixed_best_val = -1.0
        for _strat, scores in results.items():
            val = scores.get(primary_metric, 0.0)
            if val > fixed_best_val:
                fixed_best_val = val

        router_val = results.get(routed_strategy, {}).get(primary_metric, 0.0)
        if router_val > fixed_best_val:
            wins += 1
        elif abs(router_val - fixed_best_val) < 1e-9:
            ties += 1
        else:
            losses += 1

        valid_queries += 1

    # --- Compute averages ---
    def _avg(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    strategy_summary: dict[str, dict[str, float | None]] = {}
    for strat, metrics_dict in sorted(strategy_scores.items()):
        strategy_summary[strat] = {m: _avg(metrics_dict[m]) for m in metrics}

    router_summary: dict[str, float | None] = {
        m: _avg(router_scores[m]) for m in metrics
    }

    oracle_summary: dict[str, float | None] = {
        m: _avg(oracle_scores[m]) for m in metrics
    }

    # Fixed-best baseline: strategy with highest average on primary metric
    fixed_best_name = None
    fixed_best_avg = -1.0
    for strat, avgs in strategy_summary.items():
        val = avgs.get(primary_metric)
        if val is not None and val > fixed_best_avg:
            fixed_best_avg = val
            fixed_best_name = strat

    # --- Regret ---
    router_primary = router_summary.get(primary_metric)
    oracle_primary = oracle_summary.get(primary_metric)
    regret = None
    if router_primary is not None and oracle_primary is not None and oracle_primary > 0:
        regret = oracle_primary - router_primary

    # --- Build report ---
    report = {
        "schema_version": SCHEMA_VERSION,
        "input_file": str(input_path),
        "input_sha256": input_hash,
        "config_file": str(config_path),
        "config_sha256": config_hash,
        "policy_id": policy_meta.policy_id,
        "policy_version": policy_meta.policy_version,
        "policy_sha256": policy_meta.policy_sha256,
        "total_queries": len(queries),
        "valid_queries": valid_queries,
        "na_queries": na_queries,
        "primary_metric": primary_metric,
        "strategy_summary": strategy_summary,
        "fixed_best_baseline": {
            "strategy": fixed_best_name,
            "average": fixed_best_avg,
            "metric": primary_metric,
        },
        "router_summary": router_summary,
        "oracle_summary": {
            "label": OracleLabel.POST_HOC_ORACLE.value,
            "disclaimer": (
                "Post-hoc oracle selects the best strategy PER QUERY using "
                "evaluation outcomes. This is a CEILING, not a deployable policy."
            ),
            "averages": oracle_summary,
        },
        "routing_ceiling": {
            "oracle_avg": oracle_primary,
            "router_avg": router_primary,
            "regret": regret,
        },
        "wins_ties_losses": {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "denominator": valid_queries,
        },
    }

    # --- Write output ---
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "policy_replay_report.json"
    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    report_path.write_text(report_json, encoding="utf-8")

    # Manifest
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "input_sha256": input_hash,
        "config_sha256": config_hash,
        "output_sha256": _sha256(report_json),
        "policy_sha256": policy_meta.policy_sha256,
    }
    manifest_path = output_dir / "replay_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"POLICY REPLAY REPORT — {SCHEMA_VERSION}")
    print(f"{'=' * 60}")
    print(f"Input:  {input_path}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"\nQueries: {valid_queries}/{len(queries)} valid")
    print(f"\n--- Strategy Summary ({primary_metric}) ---")
    for strat, avgs in sorted(strategy_summary.items()):
        val = avgs.get(primary_metric)
        print(f"  {strat:30s} {val:.4f}" if val is not None else f"  {strat:30s} NA")
    print("\n--- Fixed-best Baseline ---")
    print(f"  {fixed_best_name}: {fixed_best_avg:.4f}")
    print("\n--- Router (deterministic_v1) ---")
    print(f"  {primary_metric}: {router_primary:.4f}" if router_primary else "  NA")
    print("\n--- Oracle (POST_HOC — ceiling only) ---")
    print(f"  {primary_metric}: {oracle_primary:.4f}" if oracle_primary else "  NA")
    print("\n--- Regret ---")
    print(f"  {regret:.4f}" if regret is not None else "  NA")
    print("\n--- Win/Tie/Loss ---")
    print(f"  Wins: {wins}, Ties: {ties}, Losses: {losses} (n={valid_queries})")
    print(f"\nReport: {report_path}")
    print(f"Manifest: {manifest_path}")
    print(f"{'=' * 60}\n")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline policy replay for agentic routing analysis",
        epilog=(
            "Two modes:\n"
            "  A. CI/synthetic: use the synthetic fixture\n"
            "  B. local/authoritative: use the local result\n\n"
            "Does NOT alter input files or invent metrics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--slice4-result",
        type=Path,
        required=True,
        help="Path to Slice 4 result JSON (synthetic or authoritative)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to replay configuration JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output files",
    )

    args = parser.parse_args()

    if not args.slice4_result.exists():
        print(f"ERROR: Input file not found: {args.slice4_result}", file=sys.stderr)
        sys.exit(1)
    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    exit_code = analyze_policy_replay(args.slice4_result, args.config, args.output_dir)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
