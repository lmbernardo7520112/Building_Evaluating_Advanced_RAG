"""Unit tests for the extended policy replay analyzer.

Tests both schema formats, integrity audit, tie-aware winners,
and fail-closed behavior.
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.analyze_agentic_policy_replay import (
    _extract_v5_metric,
    _normalize_legacy,
    _normalize_v5,
    compute_headroom,
    compute_per_qid_winners,
    detect_schema,
    evaluate_headroom_gate,
    run_integrity_audit,
)

# ── Fixtures ────────────────────────────────────────────────────


def _legacy_fixture(
    metrics: dict[str, float] | None = None,
    *,
    answerable: bool = True,
    split: str = "development",
    strategies: list[str] | None = None,
) -> dict[str, Any]:
    """Build a minimal legacy fixture."""
    strats = strategies or ["baseline", "auto_merging"]
    if metrics is None:
        metrics = {"ndcg_at_3": 0.5, "recall_at_3": 0.4, "mrr_at_3": 1.0}
    results = {s: dict(metrics) for s in strats}
    return {
        "queries": [
            {
                "query_id": "q1",
                "query_text": "What is chunking?",
                "split": split,
                "answerable": answerable,
                "results_per_strategy": results,
            }
        ]
    }


def _v5_fixture(
    *,
    num_qids: int = 2,
    dev_count: int = 1,
    strategies: list[str] | None = None,
    score_fn: Any = None,
) -> dict[str, Any]:
    """Build a minimal slice4_v5 fixture."""
    strats = strategies or [
        "F0_baseline",
        "H0_hierarchical_leaf",
        "W1_sentence_window_rerank",
    ]

    results: dict[str, list[dict[str, Any]]] = {s: [] for s in strats}

    for i in range(num_qids):
        qid = f"q_{'dev' if i < dev_count else 'test'}_{i + 1:02d}"
        split = "development" if i < dev_count else "test"
        for j, s in enumerate(strats):
            sc = score_fn(qid, s, j) if score_fn else 0.3 + j * 0.1
            results[s].append(
                {
                    "qid": qid,
                    "split": split,
                    "ground_truth": {"answerable": True},
                    "evaluation": {
                        "deterministic_v2_metrics": {
                            "ndcg_at_k": {"status": "COMPUTED", "score": sc},
                            "recall_at_k": {"status": "COMPUTED", "score": sc * 0.8},
                            "mrr_at_k": {"status": "COMPUTED", "score": 1.0},
                        },
                        "judged_coverage_rate": 1.0,
                        "unresolved_mapping_count": 0,
                    },
                }
            )

    return {"results": results, "schema": "slice4_v5"}


METRICS = ["ndcg_at_3", "recall_at_3", "mrr_at_3"]


# ── Schema detection ────────────────────────────────────────────


class TestDetectSchema:
    def test_legacy_schema(self) -> None:
        assert detect_schema({"queries": []}) == "legacy"

    def test_v5_explicit(self) -> None:
        assert (
            detect_schema({"results": {"strat": []}, "schema": "slice4_v5"})
            == "slice4_v5"
        )

    def test_v5_implicit_list(self) -> None:
        assert detect_schema({"results": {"strat": []}}) == "slice4_v5"

    def test_unknown_exits(self) -> None:
        with pytest.raises(SystemExit):
            detect_schema({"data": []})

    def test_unknown_results_not_dict(self) -> None:
        with pytest.raises(SystemExit):
            detect_schema({"results": "not a dict"})


# ── Metric extraction ───────────────────────────────────────────


class TestExtractV5Metric:
    def test_computed_finite(self) -> None:
        rec = {
            "evaluation": {
                "deterministic_v2_metrics": {
                    "ndcg_at_k": {"status": "COMPUTED", "score": 0.5}
                }
            }
        }
        assert _extract_v5_metric(rec, "ndcg_at_3") == 0.5

    def test_not_computed_returns_none(self) -> None:
        rec = {
            "evaluation": {
                "deterministic_v2_metrics": {
                    "ndcg_at_k": {"status": "NA", "score": None}
                }
            }
        }
        assert _extract_v5_metric(rec, "ndcg_at_3") is None

    def test_infinite_returns_none(self) -> None:
        rec = {
            "evaluation": {
                "deterministic_v2_metrics": {
                    "ndcg_at_k": {"status": "COMPUTED", "score": float("inf")}
                }
            }
        }
        assert _extract_v5_metric(rec, "ndcg_at_3") is None

    def test_nan_returns_none(self) -> None:
        rec = {
            "evaluation": {
                "deterministic_v2_metrics": {
                    "ndcg_at_k": {"status": "COMPUTED", "score": float("nan")}
                }
            }
        }
        assert _extract_v5_metric(rec, "ndcg_at_3") is None

    def test_missing_metric_key(self) -> None:
        rec = {"evaluation": {"deterministic_v2_metrics": {}}}
        assert _extract_v5_metric(rec, "ndcg_at_3") is None

    def test_unknown_metric_name(self) -> None:
        rec = {
            "evaluation": {
                "deterministic_v2_metrics": {
                    "ndcg_at_k": {"status": "COMPUTED", "score": 0.5}
                }
            }
        }
        assert _extract_v5_metric(rec, "unknown_metric") is None


# ── Normalization ────────────────────────────────────────────────


class TestNormalize:
    def test_legacy_normalization(self) -> None:
        data = _legacy_fixture()
        records = _normalize_legacy(data, METRICS)
        assert len(records) == 1
        assert records[0]["qid"] == "q1"
        assert records[0]["split"] == "development"
        assert records[0]["answerable"] is True
        assert "baseline" in records[0]["per_strategy"]
        assert records[0]["per_strategy"]["baseline"]["ndcg_at_3"] == 0.5

    def test_v5_normalization(self) -> None:
        data = _v5_fixture(num_qids=3, dev_count=2)
        records = _normalize_v5(data, METRICS)
        assert len(records) == 3
        dev = [r for r in records if r["split"] == "development"]
        test = [r for r in records if r["split"] == "test"]
        assert len(dev) == 2
        assert len(test) == 1

    def test_legacy_infinite_score_becomes_none(self) -> None:
        data = _legacy_fixture(
            metrics={"ndcg_at_3": float("inf"), "recall_at_3": 0.4, "mrr_at_3": 1.0}
        )
        records = _normalize_legacy(data, METRICS)
        assert records[0]["per_strategy"]["baseline"]["ndcg_at_3"] is None


# ── Integrity audit ──────────────────────────────────────────────


class TestIntegrityAudit:
    def test_clean_data_passes(self) -> None:
        data = _v5_fixture(num_qids=2)
        records = _normalize_v5(data, METRICS)
        audit = run_integrity_audit(records, METRICS)
        assert audit["all_pass"] is True
        assert audit["C1_same_qids_per_strategy"]["pass"] is True
        assert audit["C2_equal_record_counts"]["pass"] is True
        assert audit["C3_metrics_computed_finite"]["pass"] is True

    def test_missing_strategy_fails_c1(self) -> None:
        data = _v5_fixture(num_qids=2, strategies=["A", "B"])
        records = _normalize_v5(data, METRICS)
        # Remove strategy B from one QID
        records[0]["per_strategy"].pop("B", None)
        audit = run_integrity_audit(records, METRICS)
        assert audit["C1_same_qids_per_strategy"]["pass"] is False

    def test_unequal_record_count_fails_c2(self) -> None:
        data = _v5_fixture(num_qids=2, strategies=["A", "B"])
        records = _normalize_v5(data, METRICS)
        # Add extra strategy to one QID only
        records[0]["per_strategy"]["C"] = {
            "ndcg_at_3": 0.5,
            "recall_at_3": 0.4,
            "mrr_at_3": 1.0,
        }
        audit = run_integrity_audit(records, METRICS)
        assert audit["C2_equal_record_counts"]["pass"] is False

    def test_na_metric_fails_c3(self) -> None:
        data = _v5_fixture(num_qids=1)
        records = _normalize_v5(data, METRICS)
        # Set one metric to None
        first_strat = next(iter(records[0]["per_strategy"]))
        records[0]["per_strategy"][first_strat]["ndcg_at_3"] = None
        audit = run_integrity_audit(records, METRICS)
        assert audit["C3_metrics_computed_finite"]["pass"] is False


# ── Tie-aware winners ────────────────────────────────────────────


class TestPerQidWinners:
    def test_strict_winner(self) -> None:
        """When one strategy is strictly best, it should be the strict winner."""

        def _diff_scores(qid: str, strat: str, j: int) -> float:
            return 0.1 * (j + 1)  # monotonically increasing by strategy index

        data = _v5_fixture(num_qids=1, dev_count=1, score_fn=_diff_scores)
        records = _normalize_v5(data, METRICS)
        winners = compute_per_qid_winners(records, METRICS, "ALL")
        assert len(winners) == 1
        # Last strategy has highest score
        assert winners[0]["ndcg_at_3"]["strict_winner"] is not None

    def test_tied_winners(self) -> None:
        """When all strategies tie, no strict winner but all are tied."""

        def _equal_scores(qid: str, strat: str, j: int) -> float:
            return 0.5  # all same

        data = _v5_fixture(num_qids=1, dev_count=1, score_fn=_equal_scores)
        records = _normalize_v5(data, METRICS)
        winners = compute_per_qid_winners(records, METRICS, "ALL")
        assert len(winners) == 1
        ndcg = winners[0]["ndcg_at_3"]
        assert ndcg["strict_winner"] is None  # tie: no strict winner
        assert len(ndcg["winners_tied"]) == 3  # all 3 strategies tied


# ── Headroom computation ─────────────────────────────────────────


class TestHeadroom:
    def test_positive_headroom(self) -> None:
        """Different strategies winning on different QIDs → positive headroom."""

        def _varied_scores(qid: str, strat: str, j: int) -> float:
            # QID1: strat 0 strictly best; QID2: strat 2 strictly best
            if "01" in qid:
                return [0.9, 0.3, 0.2][j]
            return [0.2, 0.3, 0.9][j]

        data = _v5_fixture(num_qids=2, dev_count=2, score_fn=_varied_scores)
        records = _normalize_v5(data, METRICS)
        hr = compute_headroom(records, METRICS, "DEV")
        assert hr["answerable_count"] == 2
        assert hr["headroom"]["ndcg_at_3"]["delta"] > 0
        assert hr["C8_gain_in_answerable"] is True
        assert hr["C9_diversity_strict"] is True

    def test_no_headroom_when_one_strategy_always_best(self) -> None:
        """Single winner on all QIDs → zero headroom, no diversity."""

        def _single_winner(qid: str, strat: str, j: int) -> float:
            return 0.9 if j == 2 else 0.3

        data = _v5_fixture(num_qids=2, dev_count=2, score_fn=_single_winner)
        records = _normalize_v5(data, METRICS)
        hr = compute_headroom(records, METRICS, "DEV")
        assert hr["headroom"]["ndcg_at_3"]["delta"] < 1e-9
        assert hr["C9_diversity_strict"] is False

    def test_empty_subset(self) -> None:
        """Empty subset returns zero headroom."""
        hr = compute_headroom([], METRICS, "EMPTY")
        assert hr["answerable_count"] == 0
        assert hr["headroom"] == {}

    def test_gate_evaluation_pass(self) -> None:
        """Gate passes when all criteria met."""

        def _varied_scores(qid: str, strat: str, j: int) -> float:
            if "01" in qid:
                return [0.9, 0.3, 0.2][j]
            return [0.2, 0.3, 0.9][j]

        data = _v5_fixture(num_qids=2, dev_count=2, score_fn=_varied_scores)
        records = _normalize_v5(data, METRICS)
        hr = compute_headroom(records, METRICS, "DEV")
        gate = evaluate_headroom_gate(hr, METRICS)
        assert gate["verdict"] == "ROUTING_HEADROOM_PRESENT"

    def test_gate_evaluation_fail(self) -> None:
        """Gate fails when one strategy always wins (no diversity)."""

        def _single_winner(qid: str, strat: str, j: int) -> float:
            return 0.9 if j == 2 else 0.3

        data = _v5_fixture(num_qids=2, dev_count=2, score_fn=_single_winner)
        records = _normalize_v5(data, METRICS)
        hr = compute_headroom(records, METRICS, "DEV")
        gate = evaluate_headroom_gate(hr, METRICS)
        assert gate["verdict"] == "ROUTING_HEADROOM_NOT_DEMONSTRATED"
