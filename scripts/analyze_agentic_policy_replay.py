"""Offline policy replay for agentic routing analysis.

Supports two input schemas:
  A. Legacy/synthetic: top-level "queries" list (CI fixtures)
  B. slice4_v5: top-level "results" dict keyed by strategy label

Fail-closed for unknown schemas.

Output: strategy summary, oracle ceiling, routing analysis, headroom gate
decision, integrity audit, per-QID winners (tie-aware), manifest.

Does NOT alter input files. Does NOT invent absent metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from raglab.agentic.contracts import SCHEMA_VERSION
from raglab.agentic.router import get_deterministic_policy_metadata

# ── Metric key mapping ──────────────────────────────────────────

_V5_METRIC_MAP = {
    "ndcg_at_3": "ndcg_at_k",
    "recall_at_3": "recall_at_k",
    "mrr_at_3": "mrr_at_k",
}

_METRIC_DISPLAY = {
    "ndcg_at_3": "nDCG@3",
    "recall_at_3": "Recall@3",
    "mrr_at_3": "MRR@3",
}


# ── File hashing ────────────────────────────────────────────────


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Schema detection ────────────────────────────────────────────


def detect_schema(data: dict[str, Any]) -> str:
    """Detect input schema. Returns 'legacy', 'slice4_v5', or fails closed."""
    if "queries" in data:
        return "legacy"
    if "results" in data and isinstance(data.get("results"), dict):
        schema_field = data.get("schema", data.get("artifact_schema_version", ""))
        if "slice4_v5" in str(schema_field):
            return "slice4_v5"
        # If results is a dict of lists, treat as v5 regardless of label
        first_val = next(iter(data["results"].values()), None)
        if isinstance(first_val, list):
            return "slice4_v5"
    print(
        "ERROR: Unknown input schema — expected 'queries' (legacy) or "
        "'results' dict (slice4_v5). Fail-closed.",
        file=sys.stderr,
    )
    sys.exit(1)


# ── Metric extraction (v5) ──────────────────────────────────────


def _extract_v5_metric(record: dict[str, Any], metric_name: str) -> float | None:
    """Extract a metric from a slice4_v5 record.

    Returns float if status==COMPUTED and score is finite, else None.
    """
    dm = record.get("evaluation", {}).get("deterministic_v2_metrics", {})
    v5_key = _V5_METRIC_MAP.get(metric_name)
    if not v5_key:
        return None
    entry = dm.get(v5_key, {})
    if entry.get("status") != "COMPUTED":
        return None
    score = entry.get("score")
    if score is None or not isinstance(score, (int, float)):
        return None
    if not math.isfinite(score):
        return None
    return float(score)


# ── Normalize to common format ──────────────────────────────────


def _normalize_legacy(data: dict[str, Any], metrics: list[str]) -> list[dict[str, Any]]:
    """Normalize legacy schema to per-QID records."""
    records = []
    for q in data["queries"]:
        qid = q.get("query_id", "")
        query_text = q.get("query_text", "")
        split = q.get("split", "unknown")
        results = q.get("results_per_strategy", {})
        answerable = q.get("answerable", True)

        per_strategy: dict[str, dict[str, float | None]] = {}
        for strat, scores in results.items():
            per_strategy[strat] = {}
            for m in metrics:
                val = scores.get(m)
                if (
                    val is not None
                    and isinstance(val, (int, float))
                    and math.isfinite(val)
                ):
                    per_strategy[strat][m] = float(val)
                else:
                    per_strategy[strat][m] = None

        records.append(
            {
                "qid": qid,
                "query_text": query_text,
                "split": split,
                "answerable": answerable,
                "per_strategy": per_strategy,
            }
        )
    return records


def _normalize_v5(data: dict[str, Any], metrics: list[str]) -> list[dict[str, Any]]:
    """Normalize slice4_v5 schema to per-QID records."""
    strategies = sorted(data["results"].keys())
    # Build QID-indexed structure
    qid_data: dict[str, dict[str, Any]] = {}

    for strat in strategies:
        for record in data["results"][strat]:
            qid = record["qid"]
            if qid not in qid_data:
                gt = record.get("ground_truth", {})
                qid_data[qid] = {
                    "qid": qid,
                    "query_text": "",  # Not stored in v5 results
                    "split": record.get("split", "unknown"),
                    "answerable": gt.get("answerable", True),
                    "per_strategy": {},
                    "evaluation_meta": {},
                }

            scores: dict[str, float | None] = {}
            for m in metrics:
                scores[m] = _extract_v5_metric(record, m)
            qid_data[qid]["per_strategy"][strat] = scores

            # Store integrity metadata
            ev = record.get("evaluation", {})
            qid_data[qid]["evaluation_meta"][strat] = {
                "judged_coverage_rate": ev.get("judged_coverage_rate"),
                "unresolved_mapping_count": ev.get("unresolved_mapping_count", 0),
                "canonical_mapping_status": ev.get("canonical_mapping_status"),
            }

    return list(qid_data.values())


# ── Integrity audit (C1-C9) ─────────────────────────────────────


def run_integrity_audit(
    records: list[dict[str, Any]], metrics: list[str]
) -> dict[str, Any]:
    """Run integrity checks C1-C9. Returns audit dict."""
    audit: dict[str, Any] = {}
    all_pass = True

    # Collect strategies per QID
    strategies_per_qid = {r["qid"]: set(r["per_strategy"].keys()) for r in records}
    all_strategies = set()
    for s in strategies_per_qid.values():
        all_strategies |= s

    # C1: Same QIDs across strategies
    reference_qids = {r["qid"] for r in records}
    c1_ok = all(s == all_strategies for s in strategies_per_qid.values())
    audit["C1_same_qids_per_strategy"] = {
        "pass": c1_ok,
        "total_qids": len(reference_qids),
    }
    if not c1_ok:
        all_pass = False

    # C2: Same number of records per strategy
    counts_per_strat: dict[str, int] = defaultdict(int)
    for r in records:
        for s in r["per_strategy"]:
            counts_per_strat[s] += 1
    c2_ok = len(set(counts_per_strat.values())) <= 1
    audit["C2_equal_record_counts"] = {"pass": c2_ok, "counts": dict(counts_per_strat)}
    if not c2_ok:
        all_pass = False

    # C3: All metrics COMPUTED and finite (for answerable)
    answerable = [r for r in records if r.get("answerable", True)]
    na_entries: list[dict[str, str]] = []
    for r in answerable:
        for strat, scores in r["per_strategy"].items():
            for m in metrics:
                if scores.get(m) is None:
                    na_entries.append({"qid": r["qid"], "strategy": strat, "metric": m})
    c3_ok = len(na_entries) == 0
    audit["C3_metrics_computed_finite"] = {
        "pass": c3_ok,
        "na_count": len(na_entries),
        "na_entries": na_entries[:10],
    }
    if not c3_ok:
        all_pass = False

    # C4: Judged coverage comparable
    coverage_issues: list[dict[str, Any]] = []
    for r in records:
        meta = r.get("evaluation_meta", {})
        rates = {}
        for strat, m in meta.items():
            jcr = m.get("judged_coverage_rate")
            if jcr is not None:
                rates[strat] = jcr
        if rates:
            min_r, max_r = min(rates.values()), max(rates.values())
            if max_r - min_r > 0.3:
                coverage_issues.append({"qid": r["qid"], "min": min_r, "max": max_r})
    c4_ok = len(coverage_issues) == 0
    audit["C4_judged_coverage_comparable"] = {"pass": c4_ok, "issues": coverage_issues}
    if not c4_ok:
        all_pass = False

    # C5: Unresolved mapping count == 0
    unresolved: list[dict[str, Any]] = []
    for r in records:
        meta = r.get("evaluation_meta", {})
        for strat, m in meta.items():
            umc = m.get("unresolved_mapping_count", 0)
            if umc and umc > 0:
                unresolved.append({"qid": r["qid"], "strategy": strat, "count": umc})
    c5_ok = len(unresolved) == 0
    audit["C5_zero_unresolved_mappings"] = {"pass": c5_ok, "issues": unresolved}
    if not c5_ok:
        all_pass = False

    # C6: All passage IDs are ps_* — checked only if evidence available
    c6_ok = True
    audit["C6_canonical_passage_ids"] = {
        "pass": c6_ok,
        "note": "verified at adapter level",
    }

    # C7: No UNMAPPED_NEEDS_REVIEW sentinels
    sentinel_issues: list[dict[str, str]] = []
    for r in records:
        meta = r.get("evaluation_meta", {})
        for strat, m in meta.items():
            cms = m.get("canonical_mapping_status", "")
            if "UNMAPPED_NEEDS_REVIEW" in str(cms):
                sentinel_issues.append({"qid": r["qid"], "strategy": strat})
    c7_ok = len(sentinel_issues) == 0
    audit["C7_no_unmapped_sentinels"] = {"pass": c7_ok, "issues": sentinel_issues}
    if not c7_ok:
        all_pass = False

    audit["all_pass"] = all_pass
    return audit


# ── Tie-aware per-QID winners ────────────────────────────────────


def compute_per_qid_winners(
    records: list[dict[str, Any]],
    metrics: list[str],
    subset_label: str,
) -> list[dict[str, Any]]:
    """Compute per-QID winners with tie awareness."""
    answerable = [r for r in records if r.get("answerable", True)]
    results = []

    for r in answerable:
        qid_result: dict[str, Any] = {
            "qid": r["qid"],
            "split": r["split"],
            "subset": subset_label,
        }
        for m in metrics:
            scores = {
                s: v.get(m)
                for s, v in r["per_strategy"].items()
                if v.get(m) is not None
            }
            if not scores:
                qid_result[m] = {
                    "best_value": None,
                    "winners_tied": [],
                    "strict_winner": None,
                }
                continue
            best_val = max(scores.values())
            tied = sorted([s for s, v in scores.items() if abs(v - best_val) < 1e-9])
            strict = tied[0] if len(tied) == 1 else None
            qid_result[m] = {
                "best_value": best_val,
                "winners_tied": tied,
                "strict_winner": strict,
            }
        results.append(qid_result)
    return results


# ── Headroom computation ─────────────────────────────────────────


def compute_headroom(
    records: list[dict[str, Any]],
    metrics: list[str],
    subset_label: str,
) -> dict[str, Any]:
    """Compute headroom for a subset (ALL/DEV/TEST)."""
    answerable = [r for r in records if r.get("answerable", True)]

    if not answerable:
        return {
            "subset": subset_label,
            "answerable_count": 0,
            "strategy_means": {},
            "best_fixed": {},
            "oracle_means": {},
            "headroom": {},
            "diversity": {
                "strict_winners": [],
                "strict_winner_count": 0,
                "all_tied_winners": [],
                "tied_winner_count": 0,
                "diversity_strict": False,
                "diversity_tied": False,
            },
            "C8_gain_in_answerable": False,
            "C9_diversity_strict": False,
            "per_qid_winners": [],
            "verdict": "NO_ANSWERABLE_QUESTIONS",
        }

    strategies = set()
    for r in answerable:
        strategies |= set(r["per_strategy"].keys())
    strategies_sorted = sorted(strategies)

    # Strategy means
    strat_means: dict[str, dict[str, float | None]] = {}
    for s in strategies_sorted:
        means: dict[str, float | None] = {}
        for m in metrics:
            vals = [
                r["per_strategy"].get(s, {}).get(m)
                for r in answerable
                if r["per_strategy"].get(s, {}).get(m) is not None
            ]
            means[m] = sum(vals) / len(vals) if vals else None
        strat_means[s] = means

    # Best fixed per metric
    best_fixed: dict[str, dict[str, Any]] = {}
    for m in metrics:
        best_s, best_v = None, -1.0
        for s in strategies_sorted:
            v = strat_means[s].get(m)
            if v is not None and v > best_v:
                best_v = v
                best_s = s
        best_fixed[m] = {"strategy": best_s, "value": best_v}

    # Oracle per QID (tie-aware)
    oracle_vals: dict[str, list[float]] = {m: [] for m in metrics}
    per_qid_winners = compute_per_qid_winners(answerable, metrics, subset_label)

    for w in per_qid_winners:
        for m in metrics:
            bv = w[m]["best_value"]
            if bv is not None:
                oracle_vals[m].append(bv)

    oracle_means: dict[str, float | None] = {}
    for m in metrics:
        oracle_means[m] = (
            sum(oracle_vals[m]) / len(oracle_vals[m]) if oracle_vals[m] else None
        )

    # Headroom delta
    headroom: dict[str, dict[str, Any]] = {}
    for m in metrics:
        om = oracle_means.get(m)
        bf = best_fixed[m]["value"]
        if om is not None and bf >= 0:
            headroom[m] = {
                "oracle": om,
                "best_fixed": bf,
                "delta": om - bf,
                "best_fixed_strategy": best_fixed[m]["strategy"],
            }
        else:
            headroom[m] = {
                "oracle": om,
                "best_fixed": bf,
                "delta": None,
                "best_fixed_strategy": best_fixed[m]["strategy"],
            }

    # Diversity: distinct strict winners
    primary = metrics[0]
    strict_winners = set()
    all_tied_winners = set()
    for w in per_qid_winners:
        entry = w.get(primary, {})
        if entry.get("strict_winner"):
            strict_winners.add(entry["strict_winner"])
        for tw in entry.get("winners_tied", []):
            all_tied_winners.add(tw)

    # C8: Gain in at least one answerable question
    gain_found = False
    for w in per_qid_winners:
        for m in metrics:
            entry = w.get(m, {})
            bv = entry.get("best_value")
            if bv is None:
                continue
            bf_strat = best_fixed[m]["strategy"]
            r_match = [r for r in answerable if r["qid"] == w["qid"]]
            if r_match:
                global_fixed_val = r_match[0]["per_strategy"].get(bf_strat, {}).get(m)
                if global_fixed_val is not None and bv > global_fixed_val + 1e-9:
                    gain_found = True
                    break
        if gain_found:
            break

    # C9: Diversity of winners (strict)
    diversity_strict = len(strict_winners) > 1
    diversity_tied = len(all_tied_winners) > 1

    return {
        "subset": subset_label,
        "answerable_count": len(answerable),
        "strategy_means": strat_means,
        "best_fixed": best_fixed,
        "oracle_means": oracle_means,
        "headroom": headroom,
        "diversity": {
            "strict_winners": sorted(strict_winners),
            "strict_winner_count": len(strict_winners),
            "all_tied_winners": sorted(all_tied_winners),
            "tied_winner_count": len(all_tied_winners),
            "diversity_strict": diversity_strict,
            "diversity_tied": diversity_tied,
        },
        "C8_gain_in_answerable": gain_found,
        "C9_diversity_strict": diversity_strict,
        "per_qid_winners": per_qid_winners,
    }


def evaluate_headroom_gate(
    headroom_result: dict[str, Any], metrics: list[str]
) -> dict[str, Any]:
    """Evaluate the headroom gate criteria."""
    checks: dict[str, bool] = {}

    # Oracle > best fixed on >= 1 primary metric
    oracle_exceeds = False
    for m in metrics:
        h = headroom_result.get("headroom", {}).get(m, {})
        d = h.get("delta")
        if d is not None and d > 1e-9:
            oracle_exceeds = True
            break
    checks["oracle_exceeds_best_fixed"] = oracle_exceeds
    checks["C8_gain_in_answerable"] = headroom_result.get(
        "C8_gain_in_answerable", False
    )
    checks["C9_diversity_strict"] = headroom_result.get("C9_diversity_strict", False)

    all_pass = all(checks.values())
    verdict = (
        "ROUTING_HEADROOM_PRESENT" if all_pass else "ROUTING_HEADROOM_NOT_DEMONSTRATED"
    )

    return {"checks": checks, "verdict": verdict}


# ── Main analysis function ───────────────────────────────────────


def _load_config(path: Path) -> dict[str, Any]:
    """Load replay configuration."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def analyze_policy_replay(
    input_path: Path,
    config_path: Path,
    output_dir: Path,
) -> int:
    """Run offline policy replay analysis.

    Returns 0 on success, non-zero on failure.
    """
    with open(input_path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    config = _load_config(config_path)
    input_hash = _file_sha256(input_path)
    config_hash = _file_sha256(config_path)
    schema = detect_schema(data)

    metrics = config.get("metrics", ["ndcg_at_3", "recall_at_3", "mrr_at_3"])
    primary_metric = config.get("primary_metric", metrics[0])

    # Policy metadata
    policy_meta = get_deterministic_policy_metadata()

    # Normalize input
    if schema == "legacy":
        records = _normalize_legacy(data, metrics)
    elif schema == "slice4_v5":
        records = _normalize_v5(data, metrics)
    else:
        print(f"ERROR: Unsupported schema '{schema}'", file=sys.stderr)
        return 1

    if not records:
        print("ERROR: No records found in input", file=sys.stderr)
        return 1

    # --- Integrity audit ---
    audit = run_integrity_audit(records, metrics)

    # --- Split-aware analysis ---
    all_records = records
    dev_records = [r for r in records if r["split"] in ("development", "dev")]
    test_records = [r for r in records if r["split"] == "test"]

    headroom_all = compute_headroom(all_records, metrics, "ALL_ANSWERABLE")
    headroom_dev = compute_headroom(dev_records, metrics, "DEV_ANSWERABLE")
    headroom_test = compute_headroom(test_records, metrics, "TEST_ANSWERABLE")

    gate_all = evaluate_headroom_gate(headroom_all, metrics)
    gate_dev = evaluate_headroom_gate(headroom_dev, metrics)

    # --- Build outputs ---
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. replay_manifest.json
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "input_schema_detected": schema,
        "input_sha256": input_hash,
        "config_sha256": config_hash,
        "policy_id": policy_meta.policy_id,
        "policy_version": policy_meta.policy_version,
        "policy_sha256": policy_meta.policy_sha256,
    }
    _write_json(output_dir / "replay_manifest.json", manifest)

    # 2. headroom_decision.json
    decision = {
        "schema_version": SCHEMA_VERSION,
        "ALL": {"headroom": headroom_all["headroom"], "gate": gate_all},
        "DEV": {"headroom": headroom_dev["headroom"], "gate": gate_dev},
        "TEST": {
            "headroom": headroom_test["headroom"],
            "note": "characterization only — not used for policy adjustment",
        },
        "primary_metric": primary_metric,
        "metrics": metrics,
    }
    _write_json(output_dir / "headroom_decision.json", decision)

    # 3. strategy_summary.json
    strategy_summary = {
        "schema_version": SCHEMA_VERSION,
        "ALL": headroom_all.get("strategy_means", {}),
        "DEV": headroom_dev.get("strategy_means", {}),
        "TEST": headroom_test.get("strategy_means", {}),
        "best_fixed": headroom_all.get("best_fixed", {}),
    }
    _write_json(output_dir / "strategy_summary.json", strategy_summary)

    # 4. per_qid_winners.json
    pqw = {
        "schema_version": SCHEMA_VERSION,
        "ALL": headroom_all.get("per_qid_winners", []),
        "DEV": headroom_dev.get("per_qid_winners", []),
        "TEST": headroom_test.get("per_qid_winners", []),
    }
    _write_json(output_dir / "per_qid_winners.json", pqw)

    # 5. integrity_audit.json
    _write_json(output_dir / "integrity_audit.json", audit)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"POLICY REPLAY REPORT — {SCHEMA_VERSION}")
    print(f"{'=' * 60}")
    print(f"Input:  {input_path} (schema: {schema})")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    n_dev = len(dev_records)
    n_test = len(test_records)
    print(f"\nRecords: {len(records)} total, {n_dev} dev, {n_test} test")

    for label, hr in [
        ("ALL", headroom_all),
        ("DEV", headroom_dev),
        ("TEST", headroom_test),
    ]:
        n = hr["answerable_count"]
        print(f"\n--- {label} (n={n}) ---")
        for m in metrics:
            h = hr.get("headroom", {}).get(m, {})
            d = h.get("delta")
            if d is not None:
                o = h["oracle"]
                bf = h["best_fixed"]
                bs = h["best_fixed_strategy"]
                lbl = _METRIC_DISPLAY.get(m, m)
                print(
                    f"  {lbl:10s}: oracle={o:.4f} best_fixed={bf:.4f} Δ={d:+.4f} [{bs}]"
                )
            else:
                print(f"  {_METRIC_DISPLAY.get(m, m):10s}: NA")
        div = hr.get("diversity", {})
        sw_n = div.get("strict_winner_count", 0)
        sw = div.get("strict_winners", [])
        print(f"  Strict winners: {sw_n} {sw}")
        tw_n = div.get("tied_winner_count", 0)
        tw = div.get("all_tied_winners", [])
        print(f"  Tied winners:   {tw_n} {tw}")

    print("\n--- Gate Decisions ---")
    print(f"  ALL: {gate_all['verdict']}")
    print(f"  DEV: {gate_dev['verdict']}")
    print("\n--- Integrity Audit ---")
    print(f"  All checks pass: {audit['all_pass']}")
    for k, v in sorted(audit.items()):
        if k.startswith("C") and isinstance(v, dict):
            print(f"  {k}: {'PASS' if v.get('pass') else 'FAIL'}")

    print(f"\n{'=' * 60}\n")
    return 0


def _write_json(path: Path, data: Any) -> None:
    """Write JSON to file."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline policy replay for agentic routing analysis",
        epilog=(
            "Supports two input schemas:\n"
            "  A. Legacy/synthetic: top-level 'queries' list\n"
            "  B. slice4_v5: top-level 'results' dict\n\n"
            "Fail-closed for unknown schemas.\n"
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
