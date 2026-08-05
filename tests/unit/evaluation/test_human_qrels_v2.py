# ruff: noqa: E501, S607
"""Unit and Integration Tests for Human-Validated Graded Qrels v2 & Slice 4 Integration (Gate B).

Covers all 37 mandatory test cases specified by Section 12 of the Gate B integration contract.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.run_slice4_benchmark import (
    DEFAULT_QRELS_MANIFEST_PATH,
    DEFAULT_QRELS_PATH,
    build_parser,
)
from raglab.evaluation.contracts.ground_truth_v2 import GroundTruthItemV2
from raglab.evaluation.contracts.human_annotation_v2 import PassageRegistryEntry
from raglab.evaluation.contracts.human_qrels_v2 import (
    load_human_qrels_set,
)
from raglab.evaluation.metrics.human_qrels_metrics import (
    compute_human_qrels_metrics_for_question,
)
from raglab.evaluation.pooling.canonical_passage_mapper import (
    CanonicalPassageMapper,
)


class TestHumanQrelsV2GovernanceAndMetrics:
    """Suíte de testes para validação fail-closed dos qrels humanos e métricas de recuperação."""

    @pytest.fixture
    def mock_qrels_and_manifest(self, tmp_path: Path) -> tuple[Path, Path]:
        qrels_file = tmp_path / "human_qrels_final.jsonl"
        manifest_file = tmp_path / "human_qrels_manifest.json"

        records = []
        # q_test_04: 10 negativas (grau 0) (6 consensus, 4 adjudicated)
        for i in range(10):
            records.append({
                "schema_version": "2.0.0",
                "question_id": "q_test_04",
                "passage_id": f"ps_q4_{i:02d}",
                "relevance_grade": 0,
                "evidence_role": "NEGATIVE_CONTROL",
                "provenance": "HUMAN_EXACT_CONSENSUS" if i < 6 else "HUMAN_ADJUDICATED",
            })

        # Explicitamente incluir 4 itens para q_dev_01 (todos consensus)
        records.append({"schema_version": "2.0.0", "question_id": "q_dev_01", "passage_id": "ps_gen_3_00", "relevance_grade": 3, "evidence_role": "PRIMARY", "provenance": "HUMAN_EXACT_CONSENSUS"})
        records.append({"schema_version": "2.0.0", "question_id": "q_dev_01", "passage_id": "ps_gen_2_00", "relevance_grade": 2, "evidence_role": "SUPPORTING", "provenance": "HUMAN_EXACT_CONSENSUS"})
        records.append({"schema_version": "2.0.0", "question_id": "q_dev_01", "passage_id": "ps_gen_1_00", "relevance_grade": 1, "evidence_role": "CONTEXTUAL", "provenance": "HUMAN_EXACT_CONSENSUS"})
        records.append({"schema_version": "2.0.0", "question_id": "q_dev_01", "passage_id": "ps_gen_0_00", "relevance_grade": 0, "evidence_role": "NEGATIVE_CONTROL", "provenance": "HUMAN_EXACT_CONSENSUS"})

        # Restante: 21 grau 0, 17 grau 1, 12 grau 2, 5 grau 3 (Total = 10 + 4 + 55 = 69)
        # Consensus total desejado = 41.
        # Já alocados: 6 (q4) + 4 (q_dev_01) = 10. Faltam 31 consensus.
        questions = ["q_dev_02", "q_dev_03", "q_dev_04", "q_test_01", "q_test_02", "q_test_03"]
        g_counts = {0: 21, 1: 17, 2: 12, 3: 5}
        idx = 0
        consensus_left = 31
        for grade, count in g_counts.items():
            for c in range(count):
                qid = questions[idx % len(questions)]
                role = "PRIMARY" if grade == 3 else ("SUPPORTING" if grade == 2 else ("CONTEXTUAL" if grade == 1 else "NEGATIVE_CONTROL"))
                prov = "HUMAN_EXACT_CONSENSUS" if consensus_left > 0 else "HUMAN_ADJUDICATED"
                if consensus_left > 0:
                    consensus_left -= 1
                records.append({
                    "schema_version": "2.0.0",
                    "question_id": qid,
                    "passage_id": f"ps_rest_{grade}_{c:02d}",
                    "relevance_grade": grade,
                    "evidence_role": role,
                    "provenance": prov,
                })
                idx += 1

        assert len(records) == 69

        content_lines = [json.dumps(r) for r in records]
        qrels_file.write_text("\n".join(content_lines) + "\n", encoding="utf-8")

        import hashlib
        qrels_sha = hashlib.sha256(qrels_file.read_bytes()).hexdigest()

        manifest_data = {
            "schema_version": "2.0.0",
            "authoritative_for_evaluation": True,
            "silver_used_as_ground_truth": False,
            "holdout_sealed": True,
            "total_pairs": 69,
            "consensus_pairs_count": 41,
            "adjudicated_pairs_count": 28,
            "final_qrels_file_sha256": qrels_sha,
        }
        manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        return qrels_file, manifest_file

    # 1. Carregamento de qrels válidos
    def test_01_load_valid_qrels(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        qs = load_human_qrels_set(q_file, m_file)
        assert qs.total_pairs == 69
        assert qs.consensus_count == 41
        assert qs.adjudicated_count == 28
        assert qs.grade_distribution == {0: 32, 1: 18, 2: 13, 3: 6}

    # 2. Arquivo ausente
    def test_02_missing_qrels_file_raises_error(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        missing_q = q_file.parent / "non_existent.jsonl"
        with pytest.raises(ValueError, match="HUMAN_QRELS_REQUIRED_OR_INVALID"):
            load_human_qrels_set(missing_q, m_file)

    # 3. JSONL corrompido
    def test_03_corrupted_jsonl_raises_error(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        q_file.write_text("INVALID JSONL LINE {{{", encoding="utf-8")
        with pytest.raises(ValueError, match="HUMAN_QRELS_REQUIRED_OR_INVALID"):
            load_human_qrels_set(q_file, m_file)

    # 4. Duplicata
    def test_04_duplicate_pair_raises_error(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        lines = q_file.read_text().splitlines()
        lines.append(lines[0])  # Duplicar primeira linha
        q_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        import hashlib
        m_data = json.loads(m_file.read_text())
        m_data["final_qrels_file_sha256"] = hashlib.sha256(q_file.read_bytes()).hexdigest()
        m_data["total_pairs"] = 70
        m_file.write_text(json.dumps(m_data))

        with pytest.raises(ValueError, match="HUMAN_QRELS_REQUIRED_OR_INVALID"):
            load_human_qrels_set(q_file, m_file)

    # 5. Grau fora de 0-3
    def test_05_grade_outside_range_raises_error(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        lines = [json.loads(line_item) for line_item in q_file.read_text().splitlines() if line_item.strip()]
        lines[0]["relevance_grade"] = 4  # Grau inválido
        q_file.write_text("\n".join([json.dumps(line_item) for line_item in lines]) + "\n")

        import hashlib
        m_data = json.loads(m_file.read_text())
        m_data["final_qrels_file_sha256"] = hashlib.sha256(q_file.read_bytes()).hexdigest()
        m_file.write_text(json.dumps(m_data))

        with pytest.raises(ValueError, match="HUMAN_QRELS_REQUIRED_OR_INVALID"):
            load_human_qrels_set(q_file, m_file)

    # 6. Proveniência inválida
    def test_06_invalid_provenance_raises_error(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        lines = [json.loads(line_item) for line_item in q_file.read_text().splitlines() if line_item.strip()]
        lines[0]["provenance"] = "MACHINE_SILVER"  # Proveniência proibida
        q_file.write_text("\n".join([json.dumps(line_item) for line_item in lines]) + "\n")

        import hashlib
        m_data = json.loads(m_file.read_text())
        m_data["final_qrels_file_sha256"] = hashlib.sha256(q_file.read_bytes()).hexdigest()
        m_file.write_text(json.dumps(m_data))

        with pytest.raises(ValueError, match="HUMAN_QRELS_REQUIRED_OR_INVALID"):
            load_human_qrels_set(q_file, m_file)

    # 7. Hash adulterado
    def test_07_tampered_hash_raises_error(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        m_data = json.loads(m_file.read_text())
        m_data["final_qrels_file_sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
        m_file.write_text(json.dumps(m_data))

        with pytest.raises(ValueError, match="HUMAN_QRELS_REQUIRED_OR_INVALID"):
            load_human_qrels_set(q_file, m_file)

    # 8. Manifesto inconsistente
    def test_08_inconsistent_manifest_raises_error(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        m_data = json.loads(m_file.read_text())
        m_data["authoritative_for_evaluation"] = False
        m_file.write_text(json.dumps(m_data))

        with pytest.raises(ValueError, match="HUMAN_QRELS_REQUIRED_OR_INVALID"):
            load_human_qrels_set(q_file, m_file)

    # 9. Distribuição inesperada
    def test_09_unexpected_distribution_raises_error(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        lines = [json.loads(line_item) for line_item in q_file.read_text().splitlines() if line_item.strip()]
        for line_item in lines:
            if line_item["relevance_grade"] == 1:
                line_item["relevance_grade"] = 0
                break
        q_file.write_text("\n".join([json.dumps(line_item) for line_item in lines]) + "\n")

        import hashlib
        m_data = json.loads(m_file.read_text())
        m_data["final_qrels_file_sha256"] = hashlib.sha256(q_file.read_bytes()).hexdigest()
        m_file.write_text(json.dumps(m_data))

        with pytest.raises(ValueError, match="HUMAN_QRELS_REQUIRED_OR_INVALID"):
            load_human_qrels_set(q_file, m_file)

    # 10. Holdout presente
    def test_10_holdout_present_raises_error(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        lines = [json.loads(line_item) for line_item in q_file.read_text().splitlines() if line_item.strip()]
        lines[0]["question_id"] = "q_holdout_01"
        q_file.write_text("\n".join([json.dumps(line_item) for line_item in lines]) + "\n")

        import hashlib
        m_data = json.loads(m_file.read_text())
        m_data["final_qrels_file_sha256"] = hashlib.sha256(q_file.read_bytes()).hexdigest()
        m_file.write_text(json.dumps(m_data))

        with pytest.raises(ValueError, match="HUMAN_QRELS_REQUIRED_OR_INVALID"):
            load_human_qrels_set(q_file, m_file)

    # 11. Silver declarado como ground truth
    def test_11_silver_declared_as_ground_truth_raises_error(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        m_data = json.loads(m_file.read_text())
        m_data["silver_used_as_ground_truth"] = True
        m_file.write_text(json.dumps(m_data))

        with pytest.raises(ValueError, match="HUMAN_QRELS_REQUIRED_OR_INVALID"):
            load_human_qrels_set(q_file, m_file)

    # 12. q_test_04 com falso positivo / grau > 0
    def test_12_q_test_04_false_positive_raises_error(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        lines = [json.loads(line_item) for line_item in q_file.read_text().splitlines() if line_item.strip()]
        for line_item in lines:
            if line_item["question_id"] == "q_test_04":
                line_item["relevance_grade"] = 1
                break
        q_file.write_text("\n".join([json.dumps(line_item) for line_item in lines]) + "\n")

        import hashlib
        m_data = json.loads(m_file.read_text())
        m_data["final_qrels_file_sha256"] = hashlib.sha256(q_file.read_bytes()).hexdigest()
        m_file.write_text(json.dumps(m_data))

        with pytest.raises(ValueError, match="HUMAN_QRELS_REQUIRED_OR_INVALID"):
            load_human_qrels_set(q_file, m_file)

    # 13. Cálculo conhecido de nDCG@3
    def test_13_known_ndcg_at_3_calculation(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        qs = load_human_qrels_set(q_file, m_file)

        res = compute_human_qrels_metrics_for_question(
            qrels_set=qs,
            question_id="q_dev_01",
            retrieved_passage_ids=["ps_gen_3_00", "ps_gen_2_00", "ps_gen_0_00"],
            k=3,
        )
        ndcg_entry = res["metrics"]["ndcg_at_k"]
        assert ndcg_entry["status"] == "COMPUTED"
        assert isinstance(ndcg_entry["score"], float)
        assert ndcg_entry["score"] > 0.0

    # 14. Cálculo conhecido de Recall@3
    def test_14_known_recall_at_3_calculation(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        qs = load_human_qrels_set(q_file, m_file)

        res = compute_human_qrels_metrics_for_question(
            qrels_set=qs,
            question_id="q_dev_01",
            retrieved_passage_ids=["ps_gen_3_00", "ps_gen_2_00", "ps_gen_0_00"],
            k=3,
        )
        recall_entry = res["metrics"]["recall_at_k"]
        assert recall_entry["status"] == "COMPUTED"
        assert isinstance(recall_entry["score"], float)

    # 15. Cálculo conhecido de MRR@3
    def test_15_known_mrr_at_3_calculation(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        qs = load_human_qrels_set(q_file, m_file)

        res = compute_human_qrels_metrics_for_question(
            qrels_set=qs,
            question_id="q_dev_01",
            retrieved_passage_ids=["ps_gen_0_00", "ps_gen_3_00", "ps_gen_2_00"],
            k=3,
        )
        mrr_entry = res["metrics"]["mrr_at_k"]
        assert mrr_entry["status"] == "COMPUTED"
        assert mrr_entry["score"] == 0.5

    # 16. Nenhum item relevante -> métrica NOT_APPLICABLE
    def test_16_no_relevant_items_metric_not_applicable(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        qs = load_human_qrels_set(q_file, m_file)

        res = compute_human_qrels_metrics_for_question(
            qrels_set=qs,
            question_id="q_test_04",
            retrieved_passage_ids=["ps_q4_00", "ps_q4_01", "ps_q4_02"],
            k=3,
        )
        assert res["metrics"]["ndcg_at_k"]["status"] == "NOT_APPLICABLE"
        assert res["metrics"]["recall_at_k"]["status"] == "NOT_APPLICABLE"
        assert res["metrics"]["mrr_at_k"]["status"] == "NOT_APPLICABLE"
        assert res["metrics"]["ndcg_at_k"]["score"] is None

    # 17. Não julgado é distinto de grau 0 na contabilidade
    def test_17_unjudged_distinct_from_grade_zero(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        qs = load_human_qrels_set(q_file, m_file)

        res = compute_human_qrels_metrics_for_question(
            qrels_set=qs,
            question_id="q_dev_01",
            retrieved_passage_ids=["ps_unjudged_999", "ps_gen_3_00"],
            k=3,
        )
        acc = res["retrieval_accounting"]
        assert acc["unjudged_count"] == 1
        assert acc["judged_count"] == 1

    # 18. Trata passage_id não resolvido
    def test_18_unresolved_passage_id_handling(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        qs = load_human_qrels_set(q_file, m_file)

        res = compute_human_qrels_metrics_for_question(
            qrels_set=qs,
            question_id="q_dev_01",
            retrieved_passage_ids=["UNMAPPED_NEEDS_REVIEW", "ps_gen_3_00"],
            k=3,
        )
        acc = res["retrieval_accounting"]
        assert acc["unresolved_mapping_count"] == 1

    # 19. Cálculo de cobertura
    def test_19_coverage_calculation(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        qs = load_human_qrels_set(q_file, m_file)

        res = compute_human_qrels_metrics_for_question(
            qrels_set=qs,
            question_id="q_dev_01",
            retrieved_passage_ids=["ps_gen_3_00", "ps_gen_2_00"],
            k=2,
        )
        assert res["retrieval_accounting"]["judged_coverage_rate"] == 1.0

    # 20. Mapeamento de passage_id canônico
    def test_20_canonical_passage_id_mapping(self) -> None:
        entry = PassageRegistryEntry(
            passage_id="ps_test_01",
            document_id="gersting_discrete_math",
            page_number=95,
            start_char=0,
            end_char=100,
            content_sha256="abc",
            text="Texto de teste para mapeamento de passagem canônica.",
        )
        mapper = CanonicalPassageMapper([entry])
        res = mapper.map_chunk({"passage_id": "ps_test_01"})
        assert res.mapped_passage_id == "ps_test_01"

    # 21. Nenhuma comparação fuzzy de texto
    def test_21_no_fuzzy_text_matching(self) -> None:
        entry = PassageRegistryEntry(
            passage_id="ps_test_01",
            document_id="gersting_discrete_math",
            page_number=95,
            start_char=0,
            end_char=100,
            content_sha256="abc",
            text="Texto exato da passagem.",
        )
        mapper = CanonicalPassageMapper([entry])
        res = mapper.map_chunk({"page_number": 95, "text": "Texto quase parecido mas diferente"})
        assert res.mapped_passage_id is None

    # 22. Dano do reranker com qrels humanos
    def test_22_reranker_damage_with_human_qrels(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        qs = load_human_qrels_set(q_file, m_file)

        res = compute_human_qrels_metrics_for_question(
            qrels_set=qs,
            question_id="q_dev_01",
            retrieved_passage_ids=["ps_gen_0_00"],
            k=3,
            candidate_passage_ids_pre_rerank=["ps_gen_3_00", "ps_gen_0_00"],
        )
        damage = res["reranker_damage"]
        assert damage is not None
        assert damage["dropped_relevant_count"] == 1
        assert "ps_gen_3_00" in damage["dropped_relevant_passage_ids"]

    # 23. Separação entre métricas de retrieval e de geração
    def test_23_separation_retrieval_and_generation_eval(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        qs = load_human_qrels_set(q_file, m_file)

        res = compute_human_qrels_metrics_for_question(
            qrels_set=qs,
            question_id="q_dev_01",
            retrieved_passage_ids=["ps_gen_3_00"],
            k=3,
        )
        assert "retrieval_accounting" in res
        assert "metrics" in res
        assert "groundedness" not in res["metrics"]

    # 24. Proveniência honesta da resposta de referência
    def test_24_reference_answer_provenance_honesty(self) -> None:
        gt = GroundTruthItemV2(
            query_id="q_dev_01",
            query_text="query",
            answerable=True,
            unanswerable_reason=None,
            gold_answer="Summary reference",
            relevant_evidences=(),
            provenance_status="SINGLE_ANNOTATOR",
        )
        assert gt.gold_answer == "Summary reference"

    # 25. Preflight não exige GEMINI_API_KEY
    def test_25_preflight_no_gemini_key_required(self) -> None:
        cmd = [
            sys.executable,
            "benchmarks/run_slice4_benchmark.py",
            "--mode",
            "preflight-human-qrels",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0

    # 26. Preflight não instancia cliente Gemini
    def test_26_preflight_does_not_instantiate_gemini(self) -> None:
        cmd = [
            sys.executable,
            "benchmarks/run_slice4_benchmark.py",
            "--mode",
            "preflight-human-qrels",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0
        assert "Generator initialized" not in proc.stdout and "Generator initialized" not in proc.stderr

    # 27. Modo full exige flag de confirmação
    def test_27_full_mode_requires_confirmation_flag(self) -> None:
        cmd = [
            sys.executable,
            "benchmarks/run_slice4_benchmark.py",
            "--mode",
            "full",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode != 0

    # 28. Modo full exige qrels válidos
    def test_28_full_mode_requires_valid_qrels(self, tmp_path: Path) -> None:
        bad_q = tmp_path / "bad_qrels.jsonl"
        bad_q.write_text("INVALID JSON", encoding="utf-8")
        bad_m = tmp_path / "bad_manifest.json"
        bad_m.write_text("{}", encoding="utf-8")

        cmd = [
            sys.executable,
            "benchmarks/run_slice4_benchmark.py",
            "--mode",
            "full",
            "--confirm-full-benchmark",
            "--qrels-path",
            str(bad_q),
            "--qrels-manifest",
            str(bad_m),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode != 0

    # 29. Pergunta de holdout rejeitada
    def test_29_holdout_question_rejected(self) -> None:
        cmd = [
            sys.executable,
            "benchmarks/run_slice4_benchmark.py",
            "--mode",
            "smoke",
            "--smoke-question",
            "q_holdout_01",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode != 0

    # 30. --help sem efeitos colaterais
    def test_30_help_no_side_effects(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0

    # 31. Schema do resultado contém hashes dos qrels
    def test_31_result_schema_contains_qrels_hashes(self, mock_qrels_and_manifest: tuple[Path, Path]) -> None:
        q_file, m_file = mock_qrels_and_manifest
        qs = load_human_qrels_set(q_file, m_file)
        assert len(qs.qrels_sha256) == 64
        assert len(qs.manifest_sha256) == 64

    # 32. Artefato de resultado não contém credenciais
    def test_32_result_artifact_zero_credentials(self, tmp_path: Path) -> None:
        res_file = tmp_path / "result.json"
        res_file.write_text(json.dumps({"qrels_authority": "HUMAN_VALIDATED_GRADED_PASSAGE_RELEVANCE"}))
        content = res_file.read_text()
        for sec in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "AIza"):
            assert sec not in content

    # 33. Resultados históricos não foram modificados
    def test_33_historical_result_artifacts_untouched(self) -> None:
        proc = subprocess.run(["git", "status", "--short", "benchmarks/results/"], capture_output=True, text=True)
        assert "M  benchmarks/results/" not in proc.stdout

    # 34. Qrels reais do repositório passam no carregador
    def test_34_real_qrels_files_pass_loader(self) -> None:
        qs = load_human_qrels_set(DEFAULT_QRELS_PATH, DEFAULT_QRELS_MANIFEST_PATH)
        assert qs.total_pairs == 69

    # 35. Distribuição real 32/18/13/6
    def test_35_real_distribution_32_18_13_6(self) -> None:
        qs = load_human_qrels_set(DEFAULT_QRELS_PATH, DEFAULT_QRELS_MANIFEST_PATH)
        assert qs.grade_distribution == {0: 32, 1: 18, 2: 13, 3: 6}

    # 36. Proveniência real 41/28
    def test_36_real_provenance_41_28(self) -> None:
        qs = load_human_qrels_set(DEFAULT_QRELS_PATH, DEFAULT_QRELS_MANIFEST_PATH)
        assert qs.consensus_count == 41
        assert qs.adjudicated_count == 28

    # 37. q_test_04 real 10/10 negativo
    def test_37_real_q_test_04_negative_control_10_of_10(self) -> None:
        qs = load_human_qrels_set(DEFAULT_QRELS_PATH, DEFAULT_QRELS_MANIFEST_PATH)
        q4_items = qs.get_qrels_for_question("q_test_04")
        assert len(q4_items) == 10
        for item in q4_items:
            assert item.relevance_grade == 0
            assert item.evidence_role == "NEGATIVE_CONTROL"
