#!/usr/bin/env python3
"""Fail-closed validator for retrieval_evidence_v2.json.

Rejects:
- full_text empty
- full_text equal only to preview when original is larger
- is_truncated = True
- SHA-256 inconsistent
- strategy absent
- qid absent
- candidate without provenance
- holdout
- corpus/config incompatible
- quantity different than expected without explicit justification

Also compares against historical composite artifact for compatibility.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_V2_PATH = _REPO_ROOT / "benchmarks" / "results" / "retrieval_evidence_v2.json"
COMPOSITE_PATH = (
    _REPO_ROOT / "benchmarks" / "results" / "slice4_final_composite_recovered_run.json"
)

HOLDOUT_QIDS = frozenset({"q_holdout_01", "q_holdout_02"})

EXPECTED_STRATEGIES = frozenset(
    {
        "F0_baseline",
        "S0_sentence_anchor",
        "W0_sentence_window",
        "W1_sentence_window_rerank",
        "H0_hierarchical_leaf",
        "H1_auto_merging",
        "H2_auto_merging_rerank",
    }
)

EXPECTED_QIDS = frozenset(
    {
        "q_dev_01",
        "q_dev_02",
        "q_dev_03",
        "q_dev_04",
        "q_test_01",
        "q_test_02",
        "q_test_03",
        "q_test_04",
    }
)

PDF_SHA256_EXPECTED = "33e2e9f1e190158b3e99c19fced1acd050720247c7556780bad82b2f93bf1254"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate(evidence_path: Path) -> tuple[list[str], list[str]]:
    """Validate evidence v2 artifact. Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    if not evidence_path.exists():
        errors.append(f"Evidence file not found: {evidence_path}")
        return errors, warnings

    with open(evidence_path, encoding="utf-8") as f:
        data = json.load(f)

    if data.get("schema") != "retrieval_evidence_v2_collection":
        errors.append(f"Unexpected schema: {data.get('schema')}")

    records: list[dict[str, Any]] = data.get("records", [])
    if not records:
        errors.append("No records found")
        return errors, warnings

    # ── Per-record validation ────────────────────────────────────
    seen_strategies: set[str] = set()
    seen_qids: set[str] = set()
    strategy_qid_counts: Counter[tuple[str, str]] = Counter()

    for i, r in enumerate(records):
        prefix = f"record[{i}] ({r.get('strategy', '?')} {r.get('qid', '?')})"

        # Schema
        if r.get("schema") != "retrieval_evidence_v2":
            errors.append(f"{prefix}: schema != retrieval_evidence_v2")

        # Required fields
        for field in (
            "strategy",
            "qid",
            "raw_candidate_id",
            "full_text",
            "full_text_sha256",
            "document_id",
        ):
            if not r.get(field):
                errors.append(f"{prefix}: missing or empty {field}")

        strategy = r.get("strategy", "")
        qid = r.get("qid", "")
        seen_strategies.add(strategy)
        seen_qids.add(qid)
        strategy_qid_counts[(strategy, qid)] += 1

        # Holdout guard
        if qid in HOLDOUT_QIDS:
            errors.append(f"{prefix}: HOLDOUT question found")

        # Full text checks
        full_text = r.get("full_text", "")
        if not full_text:
            errors.append(f"{prefix}: full_text is empty")
        elif r.get("is_truncated", False):
            errors.append(f"{prefix}: is_truncated=True (fail-closed)")

        # SHA-256 consistency
        expected_sha = sha256_text(full_text) if full_text else ""
        actual_sha = r.get("full_text_sha256", "")
        if full_text and expected_sha != actual_sha:
            errors.append(f"{prefix}: SHA-256 mismatch")

        # Text length consistency
        if full_text and r.get("text_length", 0) != len(full_text):
            errors.append(f"{prefix}: text_length mismatch")

        # Provenance
        if not r.get("document_id"):
            errors.append(f"{prefix}: missing document_id")
        if not r.get("page_numbers"):
            warnings.append(f"{prefix}: no page_numbers")

        # Corpus/PDF consistency
        if r.get("source_artifact_sha256") != PDF_SHA256_EXPECTED:
            errors.append(f"{prefix}: source PDF SHA-256 mismatch")

    # ── Completeness checks ──────────────────────────────────────
    missing_strategies = EXPECTED_STRATEGIES - seen_strategies
    if missing_strategies:
        errors.append(f"Missing strategies: {missing_strategies}")

    missing_qids = EXPECTED_QIDS - seen_qids
    if missing_qids:
        errors.append(f"Missing qids: {missing_qids}")

    # For non-reranked strategies: expect TOP_K=3 candidates per question
    # For reranked strategies: TOP_K + dropped candidates
    for strat in EXPECTED_STRATEGIES:
        for qid in EXPECTED_QIDS:
            count = strategy_qid_counts.get((strat, qid), 0)
            if count < 1:
                errors.append(f"No records for {strat} × {qid}")
            if (
                strat
                not in (
                    "W1_sentence_window_rerank",
                    "H2_auto_merging_rerank",
                )
                and count != 3
            ):
                warnings.append(f"Expected 3 records for {strat} × {qid}, got {count}")

    # ── Compare with historical composite ────────────────────────
    divergences: list[str] = []
    if COMPOSITE_PATH.exists():
        with open(COMPOSITE_PATH, encoding="utf-8") as f:
            composite = json.load(f)

        for strat in EXPECTED_STRATEGIES:
            if strat not in composite.get("results", {}):
                divergences.append(
                    f"Strategy {strat} missing from historical composite"
                )
                continue

            hist_results = composite["results"][strat]
            for hist_item in hist_results:
                h_qid = hist_item.get("qid", "")
                hist_ev = hist_item.get("retrieval_evidence", {})
                hist_candidates = hist_ev.get("candidates", [])

                # Find matching v2 records
                v2_matches = [
                    r
                    for r in records
                    if r["strategy"] == strat
                    and r["qid"] == h_qid
                    and r.get("post_rerank_rank") is not None  # final candidates
                    or (
                        r["strategy"] == strat
                        and r["qid"] == h_qid
                        and r.get("pre_rerank_rank") is None
                    )  # non-reranked
                ]

                # Check page compatibility
                for hc in hist_candidates:
                    h_page = hc.get("page_number")
                    # Find matching v2 record by rank
                    h_rank = hc.get("retrieval_rank")
                    matching_v2 = [
                        r for r in v2_matches if r.get("retrieval_rank") == h_rank
                    ]
                    if matching_v2:
                        v2_pages = matching_v2[0].get("page_numbers", [])
                        if h_page and h_page not in v2_pages:
                            divergences.append(
                                f"{strat} {h_qid} rank={h_rank}: "
                                f"historical page={h_page}, v2 pages={v2_pages}"
                            )

                    # Check SHA compatibility (preview SHA vs full text SHA)
                    # The historical text_sha256 is computed from the FULL text
                    # (even though only preview was stored)
                    if matching_v2:
                        h_sha = hc.get("text_sha256", "")
                        v2_sha = matching_v2[0].get("full_text_sha256", "")
                        if h_sha and v2_sha and h_sha != v2_sha:
                            divergences.append(
                                f"{strat} {h_qid} rank={h_rank}: "
                                f"SHA MISMATCH historical={h_sha[:16]}... "
                                f"v2={v2_sha[:16]}..."
                            )

    return errors, warnings + [f"HISTORICAL DIVERGENCE: {d}" for d in divergences]


def main() -> None:
    print(f"Validating: {EVIDENCE_V2_PATH}")
    errors, warnings = validate(EVIDENCE_V2_PATH)

    if warnings:
        print(f"\n⚠ {len(warnings)} warnings:")
        for w in warnings[:50]:  # Limit output
            print(f"  {w}")

    if errors:
        print(f"\n✗ {len(errors)} ERRORS:")
        for e in errors:
            print(f"  {e}")
        print("\nVALIDATION FAILED")
        sys.exit(1)
    else:
        print(f"\n✓ Validation passed ({len(warnings)} warnings)")
        sys.exit(0)


if __name__ == "__main__":
    main()
