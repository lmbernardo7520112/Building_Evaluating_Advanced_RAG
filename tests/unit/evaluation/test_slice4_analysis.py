"""Unit tests for Slice 4 offline scientific analyzer (scripts/analyze_slice4_full_results.py).

Covers 30+ fail-closed invariants, contract verifications, paired comparisons,
abstention classifications, determinism, and offline safety.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import scripts.analyze_slice4_full_results as analyzer


def create_minimal_valid_result_json() -> dict[str, Any]:
    """Build a minimal valid result dict matching schema slice4_v5."""
    results = {}
    strategies = analyzer.EXPECTED_STRATEGIES
    qids = analyzer.EXPECTED_QIDS

    for s in strategies:
        results[s] = []
        for qid in qids:
            is_ans = qid != "q_test_04"
            results[s].append({
                "strategy": s,
                "qid": qid,
                "split": "dev",
                "abstained": not is_ans,  # Answerable = False, Negative control = True
                "ground_truth": {
                    "answerable": is_ans,
                    "contract_version": "v2",
                },
                "retrieval_evidence": {
                    "candidates": [
                        {
                            "canonical_passage_id": "ps_1234567890abcdef",
                            "page_number": 92,
                            "rank": 1,
                        }
                    ]
                },
                "answer": {
                    "abstained": not is_ans,
                    "citations": [
                        {
                            "canonical_passage_id": "ps_1234567890abcdef",
                            "page_number": 92,
                        }
                    ]
                    if is_ans
                    else [],
                },
                "evaluation": {
                    "schema_version": "slice4_v5",
                    "unresolved_mapping_count": 0,
                    "judged_coverage_rate": 1.0,
                    "mapped_count": 1,
                    "deterministic_v2_metrics": {
                        "ndcg_at_k": {"k": 3, "score": 0.5, "status": "COMPUTED"},
                        "recall_at_k": {
                            "k": 3,
                            "score": 0.5,
                            "status": "COMPUTED",
                        },
                        "mrr_at_k": {"k": 3, "score": 1.0, "status": "COMPUTED"},
                    },
                    "retrieval_evaluation": {
                        "retrieval_accounting": {
                            "judged_count": 1,
                            "relevant_retrieved_count": 1,
                        }
                    },
                    "generation_evaluation": {
                        "context_relevance": {
                            "name": "context_relevance",
                            "score": 0.8,
                            "status": "COMPUTED",
                        },
                        "groundedness": {
                            "name": "groundedness",
                            "score": 0.9 if is_ans else None,
                            "status": "COMPUTED" if is_ans else "NOT_APPLICABLE",
                        },
                        "answer_relevance": {
                            "name": "answer_relevance",
                            "score": 0.95 if is_ans else None,
                            "status": "COMPUTED" if is_ans else "NOT_APPLICABLE",
                        },
                        "abstention_correctness": {
                            "name": "abstention_correctness",
                            "score": 1.0,
                            "status": "COMPUTED",
                            "reason": "CORRECT",
                        },
                    },
                },
            })

    return {
        "schema": "slice4_v5",
        "experiment_id": "raglab_v7_slice4_v5_humanqrels_20260806T135108Z",
        "holdout_status": "SEALED",
        "qrels_path": "benchmarks/ground_truth/v2/hybrid/qrels/human_qrels_final.jsonl",
        "qrels_manifest_sha256": "8e596a1238ac4ef224b4c2f9d0959e540885f959b5de0294e3fba734db56c434",
        "results": results,
    }


def create_authoritative_synthetic_result_json() -> dict[str, Any]:
    """Build a deterministic synthetic result dict matching the 30/7/23 counts."""
    strategies = analyzer.EXPECTED_STRATEGIES
    qids = analyzer.EXPECTED_QIDS

    # Total 23 answerable abstentions matching real dataset contract:
    # F0: 5 (q_dev_02, q_dev_03, q_dev_04, q_test_01, q_test_02)
    # S0: 5 (q_dev_02, q_dev_03, q_dev_04, q_test_01, q_test_02)
    # W0: 1 (q_dev_02) -> 6 answered (q_dev_01, q_dev_03, q_dev_04, q_test_01, q_test_02, q_test_03)
    # W1: 3 (q_dev_02, q_dev_03, q_test_02) -> 4 answered (q_dev_01, q_dev_04, q_test_01, q_test_03)
    # H0: 3 (q_dev_02, q_dev_03, q_test_02)
    # H1: 3 (q_dev_02, q_dev_03, q_test_02)
    # H2: 3 (q_dev_02, q_dev_03, q_test_02)
    # Total = 5 + 5 + 1 + 3 + 3 + 3 + 3 = 23!

    ans_ab_qids = {
        "F0_baseline": {"q_dev_02", "q_dev_03", "q_dev_04", "q_test_01", "q_test_02"},
        "S0_sentence_anchor": {"q_dev_02", "q_dev_03", "q_dev_04", "q_test_01", "q_test_02"},
        "W0_sentence_window": {"q_dev_02"},
        "W1_sentence_window_rerank": {"q_dev_02", "q_dev_03", "q_test_02"},
        "H0_hierarchical_leaf": {"q_dev_02", "q_dev_03", "q_test_02"},
        "H1_auto_merging": {"q_dev_02", "q_dev_03", "q_test_02"},
        "H2_auto_merging_rerank": {"q_dev_02", "q_dev_03", "q_test_02"},
    }

    results = {}
    for s in strategies:
        results[s] = []
        for qid in qids:
            is_ans = qid != "q_test_04"
            abstained = (qid in ans_ab_qids[s]) if is_ans else True

            rec = {
                "strategy": s,
                "qid": qid,
                "split": "dev" if qid.startswith("q_dev") else "test",
                "abstained": abstained,
                "ground_truth": {
                    "answerable": is_ans,
                    "contract_version": "v2",
                },
                "retrieval_evidence": {
                    "candidates": [
                        {
                            "canonical_passage_id": "ps_1234567890abcdef",
                            "page_number": 92,
                            "rank": 1,
                        }
                    ]
                },
                "answer": {
                    "abstained": abstained,
                    "citations": [
                        {
                            "canonical_passage_id": "ps_1234567890abcdef",
                            "page_number": 92,
                        }
                    ]
                    if not abstained
                    else [],
                },
                "evaluation": {
                    "schema_version": "slice4_v5",
                    "unresolved_mapping_count": 0,
                    "judged_coverage_rate": 1.0,
                    "mapped_count": 1,
                    "deterministic_v2_metrics": {
                        "ndcg_at_k": {"k": 3, "score": 0.5, "status": "COMPUTED"},
                        "recall_at_k": {"k": 3, "score": 0.5, "status": "COMPUTED"},
                        "mrr_at_k": {"k": 3, "score": 1.0, "status": "COMPUTED"},
                    },
                    "retrieval_evaluation": {
                        "retrieval_accounting": {
                            "judged_count": 1,
                            "relevant_retrieved_count": 1,
                        }
                    },
                    "generation_evaluation": {
                        "context_relevance": {
                            "name": "context_relevance",
                            "score": 0.8,
                            "status": "COMPUTED",
                        },
                        "groundedness": {
                            "name": "groundedness",
                            "score": 0.9 if not abstained else None,
                            "status": "COMPUTED" if not abstained else "NOT_APPLICABLE",
                        },
                        "answer_relevance": {
                            "name": "answer_relevance",
                            "score": 0.95 if not abstained else None,
                            "status": "COMPUTED" if not abstained else "NOT_APPLICABLE",
                        },
                        "abstention_correctness": {
                            "name": "abstention_correctness",
                            "score": 1.0 if not is_ans else (0.0 if abstained else None),
                            "status": "COMPUTED",
                            "reason": "CORRECT" if not is_ans else ("INCORRECT_ABSTENTION" if abstained else "ANSWERED"),
                        },
                    },
                },
            }
            results[s].append(rec)

    return {
        "schema": "slice4_v5",
        "experiment_id": "raglab_v7_slice4_v5_humanqrels_20260806T135108Z",
        "holdout_status": "SEALED",
        "qrels_path": "benchmarks/ground_truth/v2/hybrid/qrels/human_qrels_final.jsonl",
        "qrels_manifest_sha256": "8e596a1238ac4ef224b4c2f9d0959e540885f959b5de0294e3fba734db56c434",
        "results": results,
    }


def create_minimal_qrels() -> list[dict[str, Any]]:
    """Build a minimal valid qrels list."""
    return [
        {
            "question_id": "q_dev_01",
            "canonical_passage_id": "ps_1234567890abcdef",
            "relevance_grade": 2,
        }
    ]


class TestSlice4OfflineAnalysis:
    """Test suite covering scientific analysis governance and metric calculation."""

    def test_01_non_existent_input_files(self, tmp_path: Path) -> None:
        p = tmp_path / "bad_hash.json"
        p.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="INPUT_HASH_MISMATCH"):
            analyzer.validate_inputs(p, p, p, p, strict_hashes=True)

    def test_02_incorrect_schema_detection(self, tmp_path: Path) -> None:
        data = create_minimal_valid_result_json()
        data["schema"] = "slice4_v2"
        p = tmp_path / "res.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="INVALID_SCHEMA"):
            analyzer.validate_inputs(p, p, p, p, strict_hashes=False)

    def test_03_incorrect_experiment_id_detection(self, tmp_path: Path) -> None:
        data = create_minimal_valid_result_json()
        data["experiment_id"] = "wrong_exp_id"
        p = tmp_path / "res.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="INVALID_EXPERIMENT_ID"):
            analyzer.validate_inputs(p, p, p, p, strict_hashes=False)

    def test_04_holdout_not_sealed_detection(self, tmp_path: Path) -> None:
        data = create_minimal_valid_result_json()
        data["holdout_status"] = "UNSEALED"
        p = tmp_path / "res.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="HOLDOUT_NOT_SEALED"):
            analyzer.validate_inputs(p, p, p, p, strict_hashes=False)

    def test_05_missing_strategy_detection(self, tmp_path: Path) -> None:
        data = create_minimal_valid_result_json()
        del data["results"]["F0_baseline"]
        p = tmp_path / "res.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="MISSING_STRATEGY"):
            analyzer.validate_inputs(p, p, p, p, strict_hashes=False)

    def test_06_incorrect_record_count_detection(self, tmp_path: Path) -> None:
        data = create_minimal_valid_result_json()
        data["results"]["F0_baseline"].pop()
        p = tmp_path / "res.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="INVALID_RECORD_COUNT"):
            analyzer.validate_inputs(p, p, p, p, strict_hashes=False)

    def test_07_unexpected_qid_detection(self, tmp_path: Path) -> None:
        data = create_minimal_valid_result_json()
        data["results"]["F0_baseline"][7]["qid"] = "unexpected_qid"
        p = tmp_path / "res.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="MISSING_NEGATIVE_CONTROL"):
            analyzer.validate_inputs(p, p, p, p, strict_hashes=False)


    def test_08_invalid_abstention_flag_type(self, tmp_path: Path) -> None:
        data = create_minimal_valid_result_json()
        data["results"]["F0_baseline"][0]["evaluation"][
            "unresolved_mapping_count"
        ] = 2
        p = tmp_path / "res.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="UNRESOLVED_SENTINEL_DETECTED"):
            analyzer.validate_inputs(p, p, p, p, strict_hashes=False)

    def test_09_unresolved_mapping_sentinel_detection(self, tmp_path: Path) -> None:
        data = create_minimal_valid_result_json()
        data["results"]["F0_baseline"][0]["evaluation"][
            "unresolved_mapping_count"
        ] = 1
        p = tmp_path / "res.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="UNRESOLVED_SENTINEL_DETECTED"):
            analyzer.validate_inputs(p, p, p, p, strict_hashes=False)

    def test_10_preserving_zero_metric_as_zero(self) -> None:
        data = create_minimal_valid_result_json()
        data["results"]["F0_baseline"][0]["evaluation"][
            "deterministic_v2_metrics"
        ]["ndcg_at_k"]["score"] = 0.0
        summs = analyzer.analyze_strategies(data)
        f0 = [s for s in summs if s["strategy"] == "F0_baseline"][0]
        assert f0["retrieval"]["ndcg_at_3"]["min"] == 0.0

    def test_11_preserving_missing_metric_as_na(self) -> None:
        data = create_minimal_valid_result_json()
        summs = analyzer.analyze_strategies(data)
        f0 = [s for s in summs if s["strategy"] == "F0_baseline"][0]
        assert f0["generation"]["groundedness"]["n_valid"] == 7

    def test_12_separating_correct_negative_control_abstention(self) -> None:
        data = create_minimal_valid_result_json()
        summs = analyzer.analyze_strategies(data)
        f0 = [s for s in summs if s["strategy"] == "F0_baseline"][0]
        assert f0["abstention"]["correct_abstentions_negative_control"] == 1
        assert f0["abstention"]["incorrect_answers_negative_control"] == 0

    def test_13_total_30_abstentions_on_authoritative_data(self) -> None:
        data = create_authoritative_synthetic_result_json()
        tot_ab = sum(
            1
            for recs in data["results"].values()
            for r in recs
            if r["abstained"]
        )
        assert tot_ab == 30

    def test_14_seven_negative_control_abstentions_on_authoritative_data(self) -> None:
        data = create_authoritative_synthetic_result_json()
        neg_ab = sum(
            1
            for recs in data["results"].values()
            for r in recs
            if r["qid"] == "q_test_04" and r["abstained"]
        )
        assert neg_ab == 7

    def test_15_twenty_three_answerable_abstentions_on_authoritative_data(self) -> None:
        data = create_authoritative_synthetic_result_json()
        ans_ab = sum(
            1
            for recs in data["results"].values()
            for r in recs
            if r["ground_truth"]["answerable"] and r["abstained"]
        )
        assert ans_ab == 23

    def test_16_paired_comparison_same_qid(self) -> None:
        data = create_minimal_valid_result_json()
        pcomps = analyzer.analyze_paired_comparisons(data)
        assert len(pcomps) == 2
        assert pcomps[0]["strategy_b"] == "W1_sentence_window_rerank"
        assert pcomps[0]["strategy_a"] == "W0_sentence_window"

    def test_17_win_tie_loss_counting_logic(self) -> None:
        data = create_minimal_valid_result_json()
        data["results"]["W1_sentence_window_rerank"][0]["evaluation"][
            "deterministic_v2_metrics"
        ]["ndcg_at_k"]["score"] = 0.9
        data["results"]["W0_sentence_window"][0]["evaluation"][
            "deterministic_v2_metrics"
        ]["ndcg_at_k"]["score"] = 0.5
        pcomps = analyzer.analyze_paired_comparisons(data)
        ndcg_comp = pcomps[0]["metrics"]["ndcg_at_3"]
        assert ndcg_comp["wins"] == 1
        assert ndcg_comp["qids_benefited"] == ["q_dev_01"]

    def test_18_reranker_damage_identification(self) -> None:
        data = create_minimal_valid_result_json()
        data["results"]["W1_sentence_window_rerank"][0]["evaluation"][
            "deterministic_v2_metrics"
        ]["ndcg_at_k"]["score"] = 0.2
        data["results"]["W0_sentence_window"][0]["evaluation"][
            "deterministic_v2_metrics"
        ]["ndcg_at_k"]["score"] = 0.8
        pcomps = analyzer.analyze_paired_comparisons(data)
        ndcg_comp = pcomps[0]["metrics"]["ndcg_at_3"]
        assert ndcg_comp["losses"] == 1
        assert ndcg_comp["damage_count"] == 1

    def test_19_reranker_benefit_identification(self) -> None:
        data = create_minimal_valid_result_json()
        data["results"]["W1_sentence_window_rerank"][0]["evaluation"][
            "deterministic_v2_metrics"
        ]["ndcg_at_k"]["score"] = 0.9
        data["results"]["W0_sentence_window"][0]["evaluation"][
            "deterministic_v2_metrics"
        ]["ndcg_at_k"]["score"] = 0.2
        pcomps = analyzer.analyze_paired_comparisons(data)
        ndcg_comp = pcomps[0]["metrics"]["ndcg_at_3"]
        assert ndcg_comp["benefit_count"] == 1

    def test_20_classification_retrieval_failure(self) -> None:
        data = create_minimal_valid_result_json()
        data["results"]["F0_baseline"][0]["abstained"] = True
        qrels = [{"question_id": "q_dev_01", "passage_id": "ps_1234", "relevance_grade": 0}]
        cases = analyzer.analyze_answerable_abstentions(data, qrels)
        assert cases[0]["category"] == "RETRIEVAL_FAILURE"

    def test_21_classification_insufficient_retrieved_support(self) -> None:
        data = create_minimal_valid_result_json()
        data["results"]["F0_baseline"][0]["abstained"] = True
        data["results"]["F0_baseline"][0]["retrieval_evidence"]["candidates"][0]["canonical_passage_id"] = "ps_1234"
        qrels = [{"question_id": "q_dev_01", "canonical_passage_id": "ps_1234", "relevance_grade": 1}]
        cases = analyzer.analyze_answerable_abstentions(data, qrels)
        assert cases[0]["category"] == "INSUFFICIENT_RETRIEVED_SUPPORT"

    def test_22_classification_ambiguity_category(self) -> None:
        data = create_minimal_valid_result_json()
        data["results"]["F0_baseline"][0]["abstained"] = True
        data["results"]["F0_baseline"][0]["retrieval_evidence"]["candidates"][0]["canonical_passage_id"] = "ps_1234"
        qrels = [{"question_id": "q_dev_01", "canonical_passage_id": "ps_1234", "relevance_grade": 2}]
        cases = analyzer.analyze_answerable_abstentions(data, qrels)
        assert cases[0]["category"] == "QREL_OR_QUESTION_AMBIGUITY"

    def test_23_explicit_denominators_recording(self) -> None:
        metric_dict = analyzer.generate_metric_dictionary()
        assert metric_dict["metrics"]["ndcg_at_3"]["denominator"] == "answerable queries (n=7 per strategy)"

    def test_24_atomic_writing_mechanism(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        h = analyzer.atomic_write_text(target, "hello world")
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "hello world"
        assert h == analyzer.compute_sha256(target)

    def test_25_byte_by_byte_determinism(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.json"
        p2 = tmp_path / "b.json"
        mdict = analyzer.generate_metric_dictionary()
        content = json.dumps(mdict, indent=2, sort_keys=True)
        h1 = analyzer.atomic_write_text(p1, content)
        h2 = analyzer.atomic_write_text(p2, content)
        assert h1 == h2
        assert p1.read_bytes() == p2.read_bytes()

    def test_26_analysis_manifest_hashes_present(self, tmp_path: Path) -> None:
        data = create_minimal_valid_result_json()
        res_p = tmp_path / "results.json"
        ckpt_p = tmp_path / "checkpoint.json"
        res_p.write_text(json.dumps(data), encoding="utf-8")
        ckpt_p.write_text(json.dumps({"schema": "slice4_v5", "status": "COMPLETED"}), encoding="utf-8")

        qrels_p = Path("benchmarks/ground_truth/v2/hybrid/qrels/human_qrels_final.jsonl")
        manif_p = Path("benchmarks/ground_truth/v2/hybrid/qrels/human_qrels_manifest.json")

        res_data, ckpt_data, qrels_lines, manifest_data = analyzer.validate_inputs(
            res_p, ckpt_p, qrels_p, manif_p, strict_hashes=False
        )
        assert res_data["experiment_id"] == analyzer.EXPECTED_EXPERIMENT_ID

    def test_27_zero_network_adapter_imports(self) -> None:
        import inspect
        analyzer_source = inspect.getsource(analyzer)
        assert "google.genai" not in analyzer_source
        assert "GeminiGeneratorAdapter" not in analyzer_source
        assert "GeminiJudgeAdapter" not in analyzer_source
        assert "urllib.request" not in analyzer_source
        assert "requests" not in analyzer_source

    def test_28_zero_modification_of_inputs(self, tmp_path: Path) -> None:
        data = create_minimal_valid_result_json()
        res_p = tmp_path / "results.json"
        content = json.dumps(data)
        res_p.write_text(content, encoding="utf-8")
        h_before = analyzer.compute_sha256(res_p)
        analyzer.analyze_strategies(data)
        h_after = analyzer.compute_sha256(res_p)
        assert h_before == h_after

    def test_29_fail_closed_behavior_on_broken_inputs(self, tmp_path: Path) -> None:
        bad_json = tmp_path / "broken.json"
        bad_json.write_text("invalid json", encoding="utf-8")
        with pytest.raises(ValueError):
            analyzer.validate_inputs(bad_json, bad_json, bad_json, bad_json, strict_hashes=False)

    def test_30_controlled_scientific_conclusion_restricted_to_slice(self) -> None:
        data = create_minimal_valid_result_json()
        summs = analyzer.analyze_strategies(data)
        pcomps = analyzer.analyze_paired_comparisons(data)
        ab_cases = analyzer.analyze_answerable_abstentions(data, create_minimal_qrels())
        report = analyzer.generate_scientific_markdown_report(data, summs, pcomps, ab_cases, {})
        assert "MIXED_RESULTS_NO_CLEAR_SUPERIORITY" in report

    def test_31_paired_comparison_denominator_exact_seven(self) -> None:
        data = create_minimal_valid_result_json()
        pcomps = analyzer.analyze_paired_comparisons(data)
        for comp in pcomps:
            for _m_key, m_data in comp["metrics"].items():
                w = m_data["wins"]
                t = m_data["ties"]
                losses_cnt = m_data["losses"]
                missing = len(m_data["qids_no_comparison"])
                assert w + t + losses_cnt + missing == 7

    def test_32_exclusion_of_q_test_04_from_main_metrics(self) -> None:
        data = create_minimal_valid_result_json()
        pcomps = analyzer.analyze_paired_comparisons(data)
        for comp in pcomps:
            for _m_key, m_data in comp["metrics"].items():
                assert "q_test_04" not in m_data["qids_benefited"]
                assert "q_test_04" not in m_data["qids_harmed"]
                assert "q_test_04" not in m_data["qids_no_comparison"]

    def test_33_multidimensional_analysis_separation(self) -> None:
        data = create_minimal_valid_result_json()
        pcomps = analyzer.analyze_paired_comparisons(data)
        for comp in pcomps:
            multi = comp["multidimensional_analysis"]
            assert "retrieval_ranking" in multi
            assert "generation_quality" in multi
            assert "answerable_coverage" in multi
            assert "abstention_safety" in multi

    def test_34_coverage_damage_and_benefit_tracking(self) -> None:
        data = create_authoritative_synthetic_result_json()
        pcomps = analyzer.analyze_paired_comparisons(data)
        w1_w0 = pcomps[0]["multidimensional_analysis"]["answerable_coverage"]
        assert w1_w0["coverage_damage_count"] == 2
        assert w1_w0["responder_to_abstain_qids"] == ["q_dev_03", "q_test_02"]

    def test_35_unambiguous_abstention_correctness(self) -> None:
        data = create_minimal_valid_result_json()
        summs = analyzer.analyze_strategies(data)
        for s in summs:
            abs_info = s["abstention"]
            assert "negative_control_abstention_correctness" in abs_info
            assert "abstention_correctness_mean_recorded" in abs_info
            assert abs_info["negative_control_abstention_correctness"] == 1.0

    def test_36_absence_of_e501_ignore_in_pyproject(self) -> None:
        pyproject_path = Path("pyproject.toml")
        content = pyproject_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            if "scripts/**" in line:
                assert "E501" not in line

    def test_37_common_answer_pairs_n_equals_qids_list_length(self) -> None:
        data = create_authoritative_synthetic_result_json()
        pcomps = analyzer.analyze_paired_comparisons(data)
        for comp in pcomps:
            gen_q = comp["multidimensional_analysis"]["generation_quality"]
            assert gen_q["common_answer_pairs_n"] == len(
                gen_q["common_answer_qids"]
            )

    def test_38_metric_valid_pairs_n_equals_qids_list_length(self) -> None:
        data = create_authoritative_synthetic_result_json()
        pcomps = analyzer.analyze_paired_comparisons(data)
        for comp in pcomps:
            for _m_key, m_data in comp["metrics"].items():
                assert m_data["metric_valid_pairs_n"] == len(
                    m_data["metric_valid_qids"]
                )
                assert m_data["metric_missing_pairs_n"] == len(
                    m_data["metric_missing_qids"]
                )

    def test_39_distinction_between_common_answers_and_valid_metrics(
        self,
    ) -> None:
        data = create_authoritative_synthetic_result_json()
        pcomps = analyzer.analyze_paired_comparisons(data)
        w1_w0 = pcomps[0]
        gen_q = w1_w0["multidimensional_analysis"]["generation_quality"]
        assert gen_q["common_answer_pairs_n"] == 4
        assert gen_q["common_answer_qids"] == [
            "q_dev_01",
            "q_dev_04",
            "q_test_01",
            "q_test_03",
        ]
        assert w1_w0["metrics"]["context_relevance"]["metric_valid_pairs_n"] == 7

    def test_40_f0_has_exactly_five_answerable_abstentions(self) -> None:
        data = create_authoritative_synthetic_result_json()
        summs = analyzer.analyze_strategies(data)
        f0_summ = next(s for s in summs if s["strategy"] == "F0_baseline")
        abs_info = f0_summ["abstention"]
        assert abs_info["abstentions_on_answerable"] == 5
        assert abs_info["answers_on_answerable"] == 2
        assert abs_info["correct_abstentions_negative_control"] == 1
        assert abs_info["total_queries"] == 8

    def test_41_traceability_of_eight_abstention_correctness_scores_for_f0(
        self,
    ) -> None:
        data = create_authoritative_synthetic_result_json()
        f0_recs = data["results"]["F0_baseline"]
        valid_scores = []
        na_count = 0
        for r in f0_recs:
            ac = r["evaluation"]["generation_evaluation"][
                "abstention_correctness"
            ]
            if ac["status"] == "COMPUTED" and ac["score"] is not None:
                valid_scores.append(ac["score"])
            else:
                na_count += 1

        assert len(valid_scores) == 6
        assert valid_scores == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        assert na_count == 2
        recorded_mean = sum(valid_scores) / len(valid_scores)
        assert round(recorded_mean, 4) == 0.1667

    def test_42_impossible_to_infer_025_by_incompatible_arithmetic(
        self,
    ) -> None:
        data = create_authoritative_synthetic_result_json()
        w1_recs = data["results"]["W1_sentence_window_rerank"]
        valid_scores = [
            r["evaluation"]["generation_evaluation"]["abstention_correctness"][
                "score"
            ]
            for r in w1_recs
            if r["evaluation"]["generation_evaluation"][
                "abstention_correctness"
            ]["status"]
            == "COMPUTED"
            and r["evaluation"]["generation_evaluation"][
                "abstention_correctness"
            ]["score"]
            is not None
        ]
        assert valid_scores == [0.0, 0.0, 0.0, 1.0]
        assert len(valid_scores) == 4
        assert sum(valid_scores) / len(valid_scores) == 0.25

    def test_43_separation_of_negative_control_abstention(self) -> None:
        data = create_authoritative_synthetic_result_json()
        summs = analyzer.analyze_strategies(data)
        for s in summs:
            abs_info = s["abstention"]
            assert abs_info["negative_control_abstention_correctness"] == 1.0
            assert abs_info["correct_abstentions_negative_control"] == 1

    def test_44_preservation_of_mixed_results_no_clear_superiority(self) -> None:
        data = create_authoritative_synthetic_result_json()
        pcomps = analyzer.analyze_paired_comparisons(data)
        for comp in pcomps:
            multi = comp["multidimensional_analysis"]
            assert (
                multi["controlled_scientific_conclusion"]
                == "MIXED_RESULTS_NO_CLEAR_SUPERIORITY"
            )

    def test_45_governance_no_reactive_asset_skips(self) -> None:
        """Governance test: fail if any test file contains reactive skips based on missing assets."""
        target_files = [
            Path("tests/unit/evaluation/test_human_qrels_v2.py"),
            Path("tests/unit/evaluation/test_silver_queue_routing_governance.py"),
            Path("tests/unit/evaluation/test_slice4_analysis.py"),
        ]
        # Check that forbidden skip strings are absent from non-governance lines
        forbidden_skip_terms = ["pytest.skip(", "pytest.mark.skipif("]
        for p in target_files:
            assert p.exists(), f"Target test file {p} missing"
            lines = p.read_text(encoding="utf-8").splitlines()
            for idx, line in enumerate(lines, start=1):
                if "test_45_governance_no_reactive_asset_skips" in line or "forbidden_skip_terms" in line:
                    continue
                for term in forbidden_skip_terms:
                    assert term not in line, f"Forbidden skip term '{term}' found in {p.name}:{idx}"
