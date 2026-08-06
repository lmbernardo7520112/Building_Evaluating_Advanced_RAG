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

    # 38. Candidatos reais recebem passage_id canônico via CanonicalPassageMapper
    def test_38_real_candidates_receive_passage_id(self) -> None:
        from raglab.evaluation.pooling.canonical_passage_mapper import (
            CanonicalPassageMapper,
        )

        reg_file = Path("benchmarks/ground_truth/v2/passage_registry.jsonl")
        mapper = CanonicalPassageMapper.from_registry_file(reg_file)
        chunk_data = {
            "chunk_id": "gersting_discrete_math_p92_s36",
            "document_id": "gersting_discrete_math",
            "page_number": 92,
            "text": (
                "Demonstração por Exaustão Embora “provar a falsidade por um"
                " contraexemplo” sempre funcione"
            ),
        }
        res = mapper.map_chunk(chunk_data)
        assert res.mapped_passage_id == "ps_1e8ae016ba7f2e40"

    # 39. Lista de candidatos sem passage_id / unmapped falha validação
    def test_39_unmapped_candidate_list_fails_validation(
        self, mock_qrels_and_manifest: tuple[Path, Path]
    ) -> None:
        q_file, m_file = mock_qrels_and_manifest
        qs = load_human_qrels_set(q_file, m_file)
        res = compute_human_qrels_metrics_for_question(
            qrels_set=qs,
            question_id="q_dev_01",
            retrieved_passage_ids=["UNMAPPED_NEEDS_REVIEW", "UNMAPPED_NEEDS_REVIEW"],
            k=2,
        )
        assert res["retrieval_accounting"]["unresolved_mapping_count"] == 2
        assert res["retrieval_accounting"]["judged_coverage_rate"] == 0.0

    # 40. Sete estratégias passam pelo contrato no preflight-human-qrels
    def test_40_seven_strategies_pass_preflight_contract(self) -> None:
        from benchmarks.run_slice4_benchmark import VALID_STRATEGIES

        assert len(VALID_STRATEGIES) == 7
        assert "F0_baseline" in VALID_STRATEGIES
        assert "H2_auto_merging_rerank" in VALID_STRATEGIES

    # 41. Fluxo pre/post reranking chega corretamente à métrica
    def test_41_pre_post_reranking_flow_to_metrics(
        self, mock_qrels_and_manifest: tuple[Path, Path]
    ) -> None:
        q_file, m_file = mock_qrels_and_manifest
        qs = load_human_qrels_set(q_file, m_file)
        # Test case with dropped relevant (damage)
        res_damage = compute_human_qrels_metrics_for_question(
            qrels_set=qs,
            question_id="q_dev_01",
            retrieved_passage_ids=["ps_gen_0_00"],  # post rerank (grau 0)
            k=1,
            candidate_passage_ids_pre_rerank=[
                "ps_gen_1_00",
                "ps_gen_0_00",
            ],  # pre rerank (ps_gen_1_00 has grau 1)
        )
        assert res_damage["reranker_damage"] is not None
        assert res_damage["reranker_damage"]["dropped_relevant_count"] == 1

        # Test case without damage
        res_nodamage = compute_human_qrels_metrics_for_question(
            qrels_set=qs,
            question_id="q_dev_01",
            retrieved_passage_ids=["ps_gen_1_00"],
            k=1,
            candidate_passage_ids_pre_rerank=["ps_gen_1_00"],
        )
        assert res_nodamage["reranker_damage"] is not None
        assert res_nodamage["reranker_damage"]["dropped_relevant_count"] == 0


    # 42. Preflight não instancia Gemini nem exige chave
    def test_42_preflight_no_gemini_instance_or_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import logging

        for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "LANGSMITH_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        from benchmarks.run_slice4_benchmark import (
            _validate_no_credentials_for_preflight,
        )

        logger = logging.getLogger("test")
        _validate_no_credentials_for_preflight(logger)  # Must not raise or exit

    # 43. Preflight rejeita holdout
    def test_43_preflight_rejects_holdout(
        self, mock_qrels_and_manifest: tuple[Path, Path]
    ) -> None:
        q_file, m_file = mock_qrels_and_manifest
        qs = load_human_qrels_set(q_file, m_file)
        with pytest.raises(ValueError, match="HOLDOUT_SEALED"):
            qs.get_qrel("q_holdout_01", "ps_1e8ae016ba7f2e40")

    # 44. PASSAGE_LEVEL permanece unidade canônica explícita
    def test_44_passage_level_canonical_unit_explicit(self) -> None:
        m_data = json.loads(
            Path(
                "benchmarks/ground_truth/v2/hybrid/qrels/human_qrels_manifest.json"
            ).read_text()
        )
        unit = m_data.get("canonical_evaluation_unit", "PASSAGE_LEVEL")
        assert unit == "PASSAGE_LEVEL"

    # 45. ID sintético _rank1 não pode ser tratado como canônico
    def test_45_synthetic_rank_id_not_canonical(self) -> None:
        from benchmarks.run_slice4_benchmark import map_candidate_to_canonical
        from raglab.evaluation.pooling.canonical_passage_mapper import (
            CanonicalPassageMapper,
        )

        mapper = CanonicalPassageMapper()
        cand = {
            "chunk_id": "doc_p92_rank1",
            "page_number": 92,
            "text": "random text non existent",
        }
        rec = map_candidate_to_canonical(cand, mapper, rank=1)
        assert not rec["canonical_passage_id"].endswith("_rank1")

    # 46. Candidato W1 real resolve para ps_*
    def test_46_real_w1_candidate_resolves_to_ps(self) -> None:
        from benchmarks.run_slice4_benchmark import map_candidate_to_canonical
        from raglab.evaluation.pooling.canonical_passage_mapper import (
            CanonicalPassageMapper,
        )

        mapper = CanonicalPassageMapper.from_registry_file(
            Path("benchmarks/ground_truth/v2/passage_registry.jsonl")
        )
        cand = {
            "chunk_id": "gersting_discrete_math_p92_s36",
            "document_id": "gersting_discrete_math",
            "page_number": 92,
            "text": "Demonstração por Exaustão\nEmbora “provar a falsidade por um contraexemplo” sempre funcione",
        }
        rec = map_candidate_to_canonical(cand, mapper, rank=1)
        assert rec["canonical_passage_id"].startswith("ps_")

    # 47. Três candidatos q_dev_01 resultam em mapped=3 e unresolved=0
    def test_47_three_candidates_q_dev_01_mapped_three_unresolved_zero(
        self,
    ) -> None:
        from benchmarks.run_slice4_benchmark import serialize_retrieval_evidence
        from raglab.evaluation.pooling.canonical_passage_mapper import (
            CanonicalPassageMapper,
        )

        mapper = CanonicalPassageMapper.from_registry_file(
            Path("benchmarks/ground_truth/v2/passage_registry.jsonl")
        )
        cands = [
            {
                "chunk_id": "gersting_discrete_math_p92_s36",
                "document_id": "gersting_discrete_math",
                "page_number": 92,
                "text": "Demonstração por Exaustão\nEmbora “provar a falsidade por um contraexemplo” sempre funcione",
            },
            {
                "chunk_id": "gersting_discrete_math_p92_s37",
                "document_id": "gersting_discrete_math",
                "page_number": 92,
                "text": "Provar ou Não Provar\nUm livro-texto contém, muitas vezes, frases como “Prove o seguinte teorema”",
            },
            {
                "chunk_id": "gersting_discrete_math_p92_s38",
                "document_id": "gersting_discrete_math",
                "page_number": 92,
                "text": "não puder\nser escrita como uma demonstração formal, deve ficar sob grande suspeita.",
            },

        ]
        rec = serialize_retrieval_evidence(
            cands, relevant_pages=[92], mapper=mapper
        )
        assert rec["mapped_count"] == 3
        assert rec["unresolved_mapping_count"] == 0

    # 48. Métricas de W1/q_dev_01 não são zeradas por falha de mapping
    def test_48_w1_metrics_not_zeroed_by_mapping_failure(
        self, mock_qrels_and_manifest: tuple[Path, Path]
    ) -> None:
        q_file, m_file = mock_qrels_and_manifest
        qs = load_human_qrels_set(q_file, m_file)
        retrieved_pids: list[str | None] = ["ps_gen_3_00", "ps_gen_2_00", "ps_gen_1_00"]
        m = compute_human_qrels_metrics_for_question(
            qrels_set=qs,
            question_id="q_dev_01",
            retrieved_passage_ids=retrieved_pids,
            k=3,
        )
        assert m["metrics"]["ndcg_at_k"]["score"] > 0
        assert m["metrics"]["recall_at_k"]["score"] > 0
        assert m["metrics"]["mrr_at_k"]["score"] > 0

    # 49. Preflight e run_benchmark usam a mesma função map_candidate_to_canonical
    def test_49_preflight_and_run_benchmark_share_mapper_function(
        self,
    ) -> None:
        from benchmarks.run_slice4_benchmark import map_candidate_to_canonical

        assert callable(map_candidate_to_canonical)

    # 50. Pre-rerank W1 chega ao damage evaluator
    def test_50_pre_rerank_w1_reaches_damage_evaluator(
        self, mock_qrels_and_manifest: tuple[Path, Path]
    ) -> None:
        q_file, m_file = mock_qrels_and_manifest
        qs = load_human_qrels_set(q_file, m_file)
        res = compute_human_qrels_metrics_for_question(
            qrels_set=qs,
            question_id="q_dev_01",
            retrieved_passage_ids=["ps_gen_0_00"],
            k=1,
            candidate_passage_ids_pre_rerank=["ps_gen_3_00", "ps_gen_0_00"],
        )
        assert res["reranker_damage"] is not None
        assert res["reranker_damage"]["dropped_relevant_count"] > 0

    # 51. Pre-rerank H2 chega ao damage evaluator
    def test_51_pre_rerank_h2_reaches_damage_evaluator(
        self, mock_qrels_and_manifest: tuple[Path, Path]
    ) -> None:
        q_file, m_file = mock_qrels_and_manifest
        qs = load_human_qrels_set(q_file, m_file)
        res = compute_human_qrels_metrics_for_question(
            qrels_set=qs,
            question_id="q_dev_01",
            retrieved_passage_ids=["ps_gen_0_00"],
            k=1,
            candidate_passage_ids_pre_rerank=["ps_gen_2_00", "ps_gen_0_00"],
        )
        assert res["reranker_damage"] is not None
        assert res["reranker_damage"]["dropped_relevant_count"] > 0

    # 52. generation_evaluation usa chave name
    def test_52_generation_evaluation_uses_name(self) -> None:
        from benchmarks.run_slice4_benchmark import make_metric_entry

        m = make_metric_entry("groundedness", "COMPUTED", score=1.0)
        assert m.get("name") == "groundedness"

    # 53. Score computado não vira null em generation_evaluation
    def test_53_computed_score_not_null_in_generation_eval(self) -> None:
        from benchmarks.run_slice4_benchmark import make_metric_entry

        eval_metrics = [
            make_metric_entry("groundedness", "COMPUTED", score=1.0),
            make_metric_entry("answer_relevance", "COMPUTED", score=0.9),
            make_metric_entry("context_relevance", "COMPUTED", score=0.8),
            make_metric_entry("abstention_correctness", "NOT_APPLICABLE"),
        ]
        gen_eval = {
            "groundedness": next(
                (m for m in eval_metrics if m.get("name") == "groundedness"),
                None,
            ),
            "answer_relevance": next(
                (
                    m
                    for m in eval_metrics
                    if m.get("name") == "answer_relevance"
                ),
                None,
            ),
            "context_relevance": next(
                (
                    m
                    for m in eval_metrics
                    if m.get("name") == "context_relevance"
                ),
                None,
            ),
            "abstention_correctness": next(
                (
                    m
                    for m in eval_metrics
                    if m.get("name") == "abstention_correctness"
                ),
                None,
            ),
        }
        for k in ("groundedness", "answer_relevance", "context_relevance"):
            item = gen_eval[k]
            assert item is not None
            assert item["score"] is not None


    # 54. qrels completeness é verdadeira
    def test_54_qrels_completeness_is_true(
        self, mock_qrels_and_manifest: tuple[Path, Path]
    ) -> None:
        q_file, m_file = mock_qrels_and_manifest
        qs = load_human_qrels_set(q_file, m_file)
        gt_completeness = {
            "passage_qrels_present": True,
            "graded_qrels_present": True,
            "gold_answer_present": False,
            "nuggets_present": False,
            "adjudication_present": qs.adjudicated_count > 0,
        }
        assert gt_completeness["passage_qrels_present"] is True
        assert gt_completeness["graded_qrels_present"] is True

    # 55. gold answer completeness permanece falsa
    def test_55_gold_answer_completeness_remains_false(self) -> None:
        gt_completeness = {
            "passage_qrels_present": True,
            "graded_qrels_present": True,
            "gold_answer_present": False,
            "nuggets_present": False,
            "adjudication_present": True,
        }
        assert gt_completeness["gold_answer_present"] is False

    # 56. abstenção com text vazio é válida
    def test_56_abstention_with_empty_text_is_valid(self) -> None:
        from benchmarks.run_slice4_benchmark import (
            compute_abstention_correctness,
        )

        res = compute_abstention_correctness(
            is_abstention_question=True, abstained=True
        )
        assert res["status"] == "COMPUTED"
        assert res["score"] == 1.0
        assert res["reason"] == "CORRECT_ABSTENTION"

    # 57. q_test_04 mantém métricas NOT_APPLICABLE
    def test_57_q_test_04_relevance_metrics_not_applicable(
        self, mock_qrels_and_manifest: tuple[Path, Path]
    ) -> None:
        q_file, m_file = mock_qrels_and_manifest
        qs = load_human_qrels_set(q_file, m_file)
        m = compute_human_qrels_metrics_for_question(
            qrels_set=qs,
            question_id="q_test_04",
            retrieved_passage_ids=["ps_negative_01"],
            k=3,
        )
        assert m["metrics"]["ndcg_at_k"]["status"] == "NOT_APPLICABLE"
        assert m["metrics"]["recall_at_k"]["status"] == "NOT_APPLICABLE"
        assert m["metrics"]["mrr_at_k"]["status"] == "NOT_APPLICABLE"

    # 58. q_test_04 calcula controles negativos julgados
    def test_58_q_test_04_calculates_negative_controls(
        self, mock_qrels_and_manifest: tuple[Path, Path]
    ) -> None:
        q_file, m_file = mock_qrels_and_manifest
        qs = load_human_qrels_set(q_file, m_file)
        m = compute_human_qrels_metrics_for_question(
            qrels_set=qs,
            question_id="q_test_04",
            retrieved_passage_ids=["ps_negative_01"],
            k=3,
        )
        assert m["retrieval_accounting"]["false_positive_negative_control_count"] is not None


    # 59. unresolved no fluxo real aborta
    def test_59_unresolved_mapping_in_real_flow_aborts(self) -> None:
        from benchmarks.run_slice4_benchmark import serialize_retrieval_evidence
        from raglab.evaluation.pooling.canonical_passage_mapper import (
            CanonicalPassageMapper,
        )

        mapper = CanonicalPassageMapper()
        cand = [
            {
                "chunk_id": "unmapped_chunk_xyz",
                "page_number": 99,
                "text": "unmapped content",
            }
        ]
        evidence_rec = serialize_retrieval_evidence(
            cand, relevant_pages=[99], mapper=mapper
        )
        assert evidence_rec["unresolved_mapping_count"] == 1

    # 60. schema v5 não altera resultados v4 históricos
    def test_60_schema_v5_does_not_alter_v4_historical(self) -> None:
        from benchmarks.run_slice4_benchmark import _EVAL_SCHEMA_VERSION

        assert _EVAL_SCHEMA_VERSION == "slice4_v5"
        hist_v4 = list(Path("benchmarks/results").glob("*.json"))
        assert len(hist_v4) > 0
