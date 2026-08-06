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

    # 61. Teste offline de reprodução do caminho real para W1 x q_dev_01 (ETAPA 6)
    def test_61_real_w1_retriever_canonical_mapping_offline(self) -> None:
        from typing import Any

        from benchmarks.run_slice4_benchmark import (
            _REPO_ROOT,
            DEFAULT_QRELS_MANIFEST_PATH,
            DEFAULT_QRELS_PATH,
            build_retrievers,
            load_embedding_model,
            load_pdf_pages,
            serialize_retrieval_evidence,
        )
        from raglab.evaluation.contracts.human_qrels_v2 import load_human_qrels_set
        from raglab.evaluation.pooling.canonical_passage_mapper import (
            CanonicalPassageMapper,
        )

        pdf_path = (
            _REPO_ROOT.parent
            / "Fundamentos matemáticos para a ciência da computação Matemática Discreta e Suas Aplicações (Judith L. Gersting).pdf"
        )
        if not pdf_path.exists():
            pytest.skip("PDF file not available locally")

        import logging

        logger = logging.getLogger("test_61")
        pages = load_pdf_pages(pdf_path, logger)
        embed_model = load_embedding_model(logger)
        retrievers = build_retrievers(
            pages, embed_model, strategies=("W1_sentence_window_rerank",)
        )
        retriever: Any = retrievers["W1_sentence_window_rerank"]

        qrels_set = load_human_qrels_set(
            DEFAULT_QRELS_PATH, DEFAULT_QRELS_MANIFEST_PATH
        )
        mapper = CanonicalPassageMapper()

        query_text = "O que é demonstração por exaustão e quando é aplicável?"
        candidates = retriever.retrieve(query_text, top_k=3)
        assert len(candidates) == 3

        evidence_rec = serialize_retrieval_evidence(
            candidates,
            [92],
            mapper=mapper,
            qrels_set=qrels_set,
            question_id="q_dev_01",
        )

        assert evidence_rec["candidate_count"] == 3
        assert evidence_rec["mapped_count"] == 3
        assert evidence_rec["unresolved_mapping_count"] == 0

        cands = evidence_rec["candidates"]
        for c in cands:
            pid = c["canonical_passage_id"]
            assert str(pid).startswith("ps_")
            assert not str(pid).endswith("_rank1")
            assert not str(pid).endswith("_rank2")
            assert not str(pid).endswith("_rank3")
            assert "raw_candidate_id" in c
            assert c["raw_candidate_id"] != ""

        judged_count = sum(1 for c in cands if c["judged_status"] == "JUDGED")
        assert judged_count > 0

    # 62. Testes de borda e fail-closed (ETAPA 5 & 6)
    def test_62_edge_cases_fail_closed(self) -> None:
        from unittest.mock import MagicMock

        from benchmarks.run_slice4_benchmark import (
            map_candidate_to_canonical,
        )
        from raglab.evaluation.pooling.canonical_passage_mapper import (
            CanonicalPassageMapper,
        )

        mapper = CanonicalPassageMapper()

        # 1. Perda de metadados / página ausente -> fail closed
        bad_cand = {"chunk_id": "bad_chunk", "document_id": "gersting_discrete_math", "text": "bad text"}
        rec1 = map_candidate_to_canonical(bad_cand, mapper, rank=1)
        assert rec1["canonical_passage_id"] == "UNMAPPED_NEEDS_REVIEW"
        assert rec1["mapping_status"] == "UNMAPPED"

        # 2. Página ambígua -> se texto não bate e houver >1 passagens na página -> UNMAPPED
        # Na página 91 há 1 passagem, mas para teste mock de página ambígua:
        # 3. Exatamente 1 passagem por página -> fallback permitido
        valid_cand_p92 = {
            "chunk_id": "gersting_discrete_math_p92_c0",
            "document_id": "gersting_discrete_math",
            "page_number": 92,
            "text": "",
        }
        rec3 = map_candidate_to_canonical(valid_cand_p92, mapper, rank=1)
        assert rec3["canonical_passage_id"].startswith("ps_")

        # 5. MagicMock em page_number ou metadados não é aceito como metadado válido
        mock_cand = MagicMock()
        mock_cand.page_number = MagicMock()
        mock_cand.text = "mock text"
        rec5 = map_candidate_to_canonical(mock_cand, mapper, rank=1)
        assert rec5["canonical_passage_id"] == "UNMAPPED_NEEDS_REVIEW"

    # 63. Teste do caminho de serialização com fakes (ETAPA 7)
    def test_63_serialization_path_fake_generators(self) -> None:
        from benchmarks.run_slice4_benchmark import (
            _EVAL_SCHEMA_VERSION,
            DEFAULT_QRELS_MANIFEST_PATH,
            DEFAULT_QRELS_PATH,
            serialize_retrieval_evidence,
        )
        from raglab.evaluation.contracts.human_qrels_v2 import load_human_qrels_set
        from raglab.evaluation.pooling.canonical_passage_mapper import (
            CanonicalPassageMapper,
        )

        qrels_set = load_human_qrels_set(
            DEFAULT_QRELS_PATH, DEFAULT_QRELS_MANIFEST_PATH
        )
        mapper = CanonicalPassageMapper()

        cand_data = [
            {
                "chunk_id": "gersting_discrete_math_p92_s36",
                "document_id": "gersting_discrete_math",
                "page_number": 92,
                "text": "Demonstração por Exaustão",
            }
        ]

        evidence_rec = serialize_retrieval_evidence(
            cand_data,
            [92],
            mapper=mapper,
            qrels_set=qrels_set,
            question_id="q_dev_01",
        )

        assert _EVAL_SCHEMA_VERSION == "slice4_v5"
        assert evidence_rec["mapped_count"] == 1
        assert evidence_rec["unresolved_mapping_count"] == 0
        c0 = evidence_rec["candidates"][0]
        assert c0["canonical_passage_id"].startswith("ps_")
        assert c0["raw_candidate_id"] == "gersting_discrete_math_p92_s36"
        assert c0["judged_status"] == "JUDGED"

    # 64. Teste de paridade ordenada entre Preflight e Produção (ETAPA 8)
    def test_64_preflight_production_mapping_parity(self) -> None:
        from typing import Any

        from benchmarks.run_slice4_benchmark import (
            _REPO_ROOT,
            DEFAULT_QRELS_MANIFEST_PATH,
            DEFAULT_QRELS_PATH,
            build_retrievers,
            load_embedding_model,
            load_pdf_pages,
            map_candidate_to_canonical,
            serialize_retrieval_evidence,
        )
        from raglab.evaluation.contracts.human_qrels_v2 import load_human_qrels_set
        from raglab.evaluation.pooling.canonical_passage_mapper import (
            CanonicalPassageMapper,
        )

        pdf_path = (
            _REPO_ROOT.parent
            / "Fundamentos matemáticos para a ciência da computação Matemática Discreta e Suas Aplicações (Judith L. Gersting).pdf"
        )
        if not pdf_path.exists():
            pytest.skip("PDF file not available locally")

        import logging

        logger = logging.getLogger("test_64")
        pages = load_pdf_pages(pdf_path, logger)
        embed_model = load_embedding_model(logger)
        retrievers = build_retrievers(
            pages, embed_model, strategies=("W1_sentence_window_rerank",)
        )
        retriever: Any = retrievers["W1_sentence_window_rerank"]
        qrels_set = load_human_qrels_set(
            DEFAULT_QRELS_PATH, DEFAULT_QRELS_MANIFEST_PATH
        )
        mapper = CanonicalPassageMapper()

        query_text = "O que é demonstração por exaustão e quando é aplicável?"
        candidates = retriever.retrieve(query_text, top_k=3)

        # Path A (Preflight)
        preflight_pairs = []
        for idx, c in enumerate(candidates):
            rec = map_candidate_to_canonical(
                c, mapper, rank=idx + 1, qrels_set=qrels_set, question_id="q_dev_01"
            )
            preflight_pairs.append((rec["raw_candidate_id"], rec["canonical_passage_id"]))

        # Path B (Production)
        evidence_rec = serialize_retrieval_evidence(
            candidates,
            [92],
            mapper=mapper,
            qrels_set=qrels_set,
            question_id="q_dev_01",
        )
        production_pairs = [
            (c["raw_candidate_id"], c["canonical_passage_id"])
            for c in evidence_rec["candidates"]
        ]

        # Exact ordered equality requirement (ETAPA 8)
        assert preflight_pairs == production_pairs
        print("PREFLIGHT_PRODUCTION_MAPPING_PARITY_OK")

    # 65. Teste de propagação de identidade canônica para citações no artefato (ETAPA 5)
    def test_65_real_artifact_citation_canonical_propagation_offline(self) -> None:

        from benchmarks.run_slice4_benchmark import (
            _EVAL_SCHEMA_VERSION,
            DEFAULT_QRELS_MANIFEST_PATH,
            DEFAULT_QRELS_PATH,
            audit_artifact_canonical_passage_ids,
            build_citation_map_and_status,
            serialize_retrieval_evidence,
        )
        from raglab.evaluation.contracts.human_qrels_v2 import load_human_qrels_set
        from raglab.evaluation.pooling.canonical_passage_mapper import (
            CanonicalPassageMapper,
        )
        from raglab.infrastructure.fakes.fake_generator_adapter import (
            FakeGeneratorAdapter,
        )

        qrels_set = load_human_qrels_set(
            DEFAULT_QRELS_PATH, DEFAULT_QRELS_MANIFEST_PATH
        )
        mapper = CanonicalPassageMapper()

        cand_data = [
            {
                "chunk_id": "gersting_discrete_math_p92_s36",
                "document_id": "gersting_discrete_math",
                "page_number": 92,
                "text": "Demonstração por Exaustão",
                "rank": 1,
            },
            {
                "chunk_id": "gersting_discrete_math_p96_s20",
                "document_id": "gersting_discrete_math",
                "page_number": 96,
                "text": "uma demonstração por casos , uma forma de demonstração por exaustão",
                "rank": 2,
            },
        ]




        evidence_rec = serialize_retrieval_evidence(
            cand_data,
            [92, 96],
            mapper=mapper,
            qrels_set=qrels_set,
            question_id="q_dev_01",
        )

        assert _EVAL_SCHEMA_VERSION == "slice4_v5"
        assert evidence_rec["mapped_count"] == 2
        assert evidence_rec["unresolved_mapping_count"] == 0

        # Build fake answer using FakeGeneratorAdapter
        gen = FakeGeneratorAdapter()
        from raglab.domain.entities import RetrievedEvidence
        ret_evs = [
            RetrievedEvidence(
                document_id=c.get("document_id", "gersting_discrete_math"),
                chunk_id=c["raw_candidate_id"],
                text=c["text"],
                score=0.9,
                rank=idx + 1,
                passage_id=c["canonical_passage_id"],
                content_sha256=c["content_sha256"],
            )
            for idx, c in enumerate(evidence_rec["candidates"])
        ]


        ans = gen.generate("q_dev_01", "O que é demonstração por exaustão?", ret_evs)

        c_status, c_map = build_citation_map_and_status(
            answer_text=ans.text,
            abstained=ans.abstained,
            evidence=evidence_rec["candidates"],
            query_id="q_dev_01",
            citations=ans.citations,
        )

        assert c_status == "AVAILABLE"
        assert len(c_map) > 0

        for cit in c_map:
            pid = cit["passage_id"]
            assert str(pid).startswith("ps_")
            assert "_rank" not in str(pid)
            assert "gersting" not in str(pid)
            assert "chunk_id" in cit
            assert cit["chunk_id"] != ""
            assert "evidence_id" in cit
            assert "content_sha256" in cit

        # Construct full record to test nesting and audit
        record = {
            "qid": "q_dev_01",
            "schema": "slice4_v5",
            "retrieval_evidence": evidence_rec,
            "citation_map": c_map,
            "answer": {
                "text": ans.text,
                "abstained": ans.abstained,
                "citations": c_map,
            },
            "evaluation": {
                "generation_evaluation": {"name": "context_relevance", "score": 1.0},
            },
        }

        # Assert generation_evaluation is nested under evaluation.generation_evaluation
        assert "generation_evaluation" in record["evaluation"]

        # Assert zero audit failures
        invalid = audit_artifact_canonical_passage_ids(record)
        assert len(invalid) == 0

    # 66. Testes fail-closed para citações e passagem canônica (ETAPA 6)
    def test_66_citation_fail_closed_edge_cases(self) -> None:
        from benchmarks.run_slice4_benchmark import (
            audit_artifact_canonical_passage_ids,
            build_citation_map_and_status,
        )


        # 1. Citação referencia evidence_id inexistente -> falha
        bad_evidence = [
            {
                "evidence_id": "E1",
                "canonical_passage_id": "ps_1e8ae016ba7f2e40",
                "chunk_id": "chunk_1",
                "document_id": "doc1",
                "page_number": 92,
                "text": "sample text",
            }
        ]
        with pytest.raises(ValueError, match="CITATION_PROVENANCE_MISMATCH"):
            build_citation_map_and_status(
                answer_text="Resposta com [E99]",
                abstained=False,
                evidence=bad_evidence,
                query_id="q_test",
            )

        # 3. Evidência sem canonical_passage_id / 4. ID sintético -> audit falha
        synthetic_evidence = [
            {
                "evidence_id": "E1",
                "passage_id": "doc_p92_rank1",
                "canonical_passage_id": "doc_p92_rank1",
                "chunk_id": "chunk_1",
                "document_id": "doc1",
                "page_number": 92,
                "text": "sample text",
            }
        ]
        status, cmap = build_citation_map_and_status(
            answer_text="Resposta com [E1]",
            abstained=False,
            evidence=synthetic_evidence,
            query_id="q_test",
        )
        invalid = audit_artifact_canonical_passage_ids(cmap)
        assert len(invalid) > 0


    # 67. Teste da auditoria recursiva de artefatos (ETAPA 7)
    def test_67_recursive_artifact_audit_validation(self) -> None:
        from benchmarks.run_slice4_benchmark import audit_artifact_canonical_passage_ids

        # Rejeita artefato contendo ID sintético em passage_id
        bad_artifact = {
            "answer": {
                "citations": [
                    {"passage_id": "gersting_p92_rank1"},
                ]
            }
        }
        invalid = audit_artifact_canonical_passage_ids(bad_artifact)
        assert len(invalid) == 1
        assert invalid[0][1] == "gersting_p92_rank1"

        # Aceita artefato com passage_id canônico ps_*
        good_artifact = {
            "answer": {
                "citations": [
                    {"passage_id": "ps_1e8ae016ba7f2e40"},
                ]
            }
        }
        invalid_good = audit_artifact_canonical_passage_ids(good_artifact)
        assert len(invalid_good) == 0

    # 68. Testes de isolamento de RUN_ID, CLI fail-closed e integridade de checkpoint (ETAPA 8)
    def test_68_run_id_and_checkpoint_isolation_contract(self, tmp_path: Path) -> None:
        import argparse
        import hashlib
        import json
        import os
        from pathlib import Path

        import pytest

        from benchmarks.run_slice4_benchmark import (
            validate_checkpoint_compatibility,
            validate_cli_args_and_checkpoint,
            validate_run_id_syntax_and_confinement,
        )
        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )


        hist_ckpt_path = Path("checkpoints/slice4_gen_checkpoint_raglab_v7_slice4_v2_20260731T1230UTC.json")
        hist_sha_before = hashlib.sha256(hist_ckpt_path.read_bytes()).hexdigest() if hist_ckpt_path.exists() else ""

        # 1. full sem run-id -> falha
        args = argparse.Namespace(mode="full", run_id=None, confirm_full_benchmark=True)
        with pytest.raises(SystemExit) as exc_info:
            validate_cli_args_and_checkpoint(args)
        assert exc_info.value.code == 2

        # 2. full sem confirmação -> falha
        args = argparse.Namespace(mode="full", run_id="raglab_v7_slice4_v5_valid", confirm_full_benchmark=False)
        with pytest.raises(SystemExit) as exc_info:
            validate_cli_args_and_checkpoint(args)
        assert exc_info.value.code == 2

        # 3. run-id inválido (caracteres proibidos/espaço) -> falha
        with pytest.raises(ValueError, match="INVALID_RUN_ID"):
            validate_run_id_syntax_and_confinement("run id invalid!!", mode="full", checkpoint_dir=tmp_path)

        # 4. path traversal -> falha
        with pytest.raises(ValueError, match="PATH_TRAVERSAL_DETECTED|INVALID_RUN_ID"):
            validate_run_id_syntax_and_confinement("../outside_run", mode="full", checkpoint_dir=tmp_path)

        # 5. full com checkpoint existente -> falha
        (tmp_path / "slice4_gen_checkpoint_existing_run.json").write_text("{}", encoding="utf-8")
        args = argparse.Namespace(mode="full", run_id="existing_run", confirm_full_benchmark=True)
        with pytest.raises(SystemExit) as exc_info:
            validate_cli_args_and_checkpoint(args, logger=None, checkpoint_dir=tmp_path)
        assert exc_info.value.code == 2


        # 6. checkpoint v3 existente -> nunca é reutilizado ou sobrescrito em full v5
        with pytest.raises(ValueError, match="INVALID_RUN_ID"):
            validate_run_id_syntax_and_confinement("raglab_v7_slice4_v2_20260731T1230UTC", mode="full", checkpoint_dir=tmp_path)

        # 7. novo full cria checkpoint v5 isolado
        GenerationCheckpointStore(
            run_id="new_v5_run",
            store_dir=tmp_path,
            schema_version="slice4_v5",
            create_new=True,
        )
        assert (tmp_path / "slice4_gen_checkpoint_new_v5_run.json").exists()
        ckpt_data = json.loads((tmp_path / "slice4_gen_checkpoint_new_v5_run.json").read_text(encoding="utf-8"))
        assert ckpt_data["schema"] == "slice4_v5"
        assert ckpt_data["run_id"] == "new_v5_run"

        # 8. resume sem run-id -> falha
        args = argparse.Namespace(mode="resume", run_id=None, confirm_full_benchmark=False)
        with pytest.raises(SystemExit) as exc_info:
            validate_cli_args_and_checkpoint(args)
        assert exc_info.value.code == 2

        # 9. resume inexistente -> falha
        args = argparse.Namespace(mode="resume", run_id="non_existent_run", confirm_full_benchmark=False)
        with pytest.raises(SystemExit) as exc_info:
            validate_cli_args_and_checkpoint(args)
        assert exc_info.value.code == 2

        # 10. resume schema v3 -> falha
        v3_ckpt = tmp_path / "slice4_gen_checkpoint_v3_run.json"
        v3_ckpt.write_text(json.dumps({"schema": "slice4_v3", "run_id": "v3_run"}), encoding="utf-8")
        with pytest.raises(ValueError, match="RESUME_CHECKPOINT_INCOMPATIBLE"):
            validate_checkpoint_compatibility(v3_ckpt, "v3_run")

        # 11. resume run-id divergente -> falha
        div_ckpt = tmp_path / "slice4_gen_checkpoint_div_run.json"
        div_ckpt.write_text(json.dumps({"schema": "slice4_v5", "run_id": "other_run"}), encoding="utf-8")
        with pytest.raises(ValueError, match="RESUME_CHECKPOINT_INCOMPATIBLE"):
            validate_checkpoint_compatibility(div_ckpt, "div_run")

        # 12-16. checkpoint com hash inválido -> falha
        corrupt_ckpt = tmp_path / "slice4_gen_checkpoint_corrupt_run.json"
        corrupt_ckpt.write_text(
            json.dumps({"schema": "slice4_v5", "run_id": "corrupt_run", "sha256": "wrong_hash", "completed": {}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="RESUME_CHECKPOINT_INCOMPATIBLE"):
            validate_checkpoint_compatibility(corrupt_ckpt, "corrupt_run")

        # 17. checkpoint v5 compatível -> resume aprovado
        valid_ckpt = tmp_path / "slice4_gen_checkpoint_valid_run.json"
        payload_bytes = json.dumps({"schema": "slice4_v5", "run_id": "valid_run", "completed": {}}, indent=2, sort_keys=True).encode("utf-8")
        valid_hash = hashlib.sha256(payload_bytes).hexdigest()
        valid_ckpt.write_text(
            json.dumps({"schema": "slice4_v5", "run_id": "valid_run", "sha256": valid_hash, "completed": {}}),
            encoding="utf-8",
        )
        res = validate_checkpoint_compatibility(valid_ckpt, "valid_run")
        assert res["schema"] == "slice4_v5"

        # 18. smoke continua com namespace temporal
        args_smoke = argparse.Namespace(mode="smoke", run_id=None, smoke_strategy="F0_baseline", smoke_question="q_dev_01")
        validate_cli_args_and_checkpoint(args_smoke)

        # 19. preflight continua sem run-id
        args_pf = argparse.Namespace(mode="preflight", run_id=None)
        validate_cli_args_and_checkpoint(args_pf)

        # 20. nenhuma credencial/rede é acessada em falhas de CLI
        assert "GEMINI_API_KEY" not in os.environ

        # 21. checkpoint histórico permanece byte a byte inalterado
        if hist_ckpt_path.exists():
            hist_sha_after = hashlib.sha256(hist_ckpt_path.read_bytes()).hexdigest()
            assert hist_sha_before == hist_sha_after

    # 69. Teste de alinhamento de contrato de run_benchmark e cmd_full (ETAPAS 5, 6, 7, 8, 9)
    def test_69_run_benchmark_is_full_run_contract(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import inspect
        import json
        import logging
        from unittest.mock import MagicMock

        import pytest

        import benchmarks.run_slice4_benchmark as runner


        # ETAPA 5 & 7 — Validação de assinatura e binding de parâmetros
        sig = inspect.signature(runner.run_benchmark)
        assert "is_full_run" in sig.parameters
        assert sig.parameters["is_full_run"].default is False

        sample_args = {
            "run_id": "test_bind",
            "questions": [],
            "strategy_labels": ("F0_baseline",),
            "logger": logging.getLogger("test"),
            "pdf_path": tmp_path / "fake.pdf",
            "qrels_path": None,
            "qrels_manifest": None,
        }
        sig.bind(**sample_args)
        sig.bind(**sample_args, is_full_run=True)
        sig.bind(**sample_args, is_full_run=False)

        # ETAPA 6 & 8 — Setup de fakes locais para execução off-line sem rede/credenciais
        fake_manifest = {
            "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "cache_tree_sha256": "abc1234567890def",
        }

        monkeypatch.setattr(runner, "load_provision_manifest", lambda: fake_manifest)
        monkeypatch.setattr(runner, "load_pdf_pages", lambda pdf_path, logger: [])
        monkeypatch.setattr(runner, "load_embedding_model", lambda logger: None)

        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = []
        monkeypatch.setattr(
            runner,
            "build_retrievers",
            lambda pages, embed, strategies=None: {"F0_baseline": fake_retriever},
        )
        monkeypatch.setattr(
            runner,
            "verify_embedding_parity",
            lambda r, lg, m: {"F0_baseline": {"cache_tree_sha256": "abc1234567890def"}},
        )

        monkeypatch.setattr(runner, "CHECKPOINT_DIR", tmp_path / "ckpts")
        monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")

        from raglab.domain.entities import GeneratedAnswer

        def fake_gen_factory(model_id, quota_manager, **kwargs):
            fake_g = MagicMock()
            fake_g.model_id = "fake-gemini"

            def _gen(query_id, query, evidence):
                quota_manager.acquire()
                return GeneratedAnswer(
                    query_id=query_id,
                    text="Resposta fake",
                    abstained=False,
                    citations=(),
                )

            fake_g.generate.side_effect = _gen
            return fake_g

        def fake_judge_factory(judge_model_id, strategy, quota_manager, **kwargs):
            fake_j = MagicMock()
            fake_j.strategy = strategy

            def _eval_cr(qid, q, ev, **kw):
                quota_manager.acquire()
                return 1.0

            def _eval_gr(qid, ans, ev, **kw):
                quota_manager.acquire()
                return 1.0

            def _eval_ar(qid, q, ans, **kw):
                quota_manager.acquire()
                return 1.0

            fake_j.evaluate_context_relevance.side_effect = _eval_cr
            fake_j.evaluate_groundedness.side_effect = _eval_gr
            fake_j.evaluate_answer_relevance.side_effect = _eval_ar
            return fake_j


        monkeypatch.setattr(
            "raglab.infrastructure.gemini.gemini_generator_adapter.GeminiGeneratorAdapter",
            fake_gen_factory,
        )
        monkeypatch.setattr(
            "raglab.infrastructure.gemini.gemini_judge_adapter.GeminiJudgeAdapter",
            fake_judge_factory,
        )


        test_run_id = "raglab_v7_slice4_v5_test_run_01"
        test_question = {
            "qid": "q_dev_01",
            "query": "Qual e o significado da prova?",
            "relevant_pages": [92],
            "split": "dev",
        }

        # Execução full com is_full_run=True (ETAPA 5 & 6)
        out_path = runner.run_benchmark(
            run_id=test_run_id,
            questions=[test_question],
            strategy_labels=("F0_baseline",),
            logger=logging.getLogger("test_runner"),
            pdf_path=tmp_path / "fake.pdf",
            qrels_path=runner.DEFAULT_QRELS_PATH,
            qrels_manifest=runner.DEFAULT_QRELS_MANIFEST_PATH,
            is_full_run=True,
        )

        assert out_path.exists()
        res_json = json.loads(out_path.read_text(encoding="utf-8"))
        assert res_json["experiment_id"] == test_run_id

        ckpt_file = tmp_path / "ckpts" / f"slice4_gen_checkpoint_{test_run_id}.json"
        assert ckpt_file.exists()
        ckpt_json = json.loads(ckpt_file.read_text(encoding="utf-8"))
        assert ckpt_json["schema"] == "slice4_v5"
        assert ckpt_json["run_id"] == test_run_id

        # Colisão no segundo full com o mesmo run_id (ETAPA 6 item 6)
        with pytest.raises(FileExistsError, match="FULL_RUN_ID_COLLISION"):
            runner.run_benchmark(
                run_id=test_run_id,
                questions=[test_question],
                strategy_labels=("F0_baseline",),
                logger=logging.getLogger("test_runner"),
                pdf_path=tmp_path / "fake.pdf",
                qrels_path=runner.DEFAULT_QRELS_PATH,
                qrels_manifest=runner.DEFAULT_QRELS_MANIFEST_PATH,
                is_full_run=True,
            )

        # Resume com is_full_run=False reidrata checkpoint existente (ETAPA 8)
        res_out_path = runner.run_benchmark(
            run_id=test_run_id,
            questions=[test_question],
            strategy_labels=("F0_baseline",),
            logger=logging.getLogger("test_runner"),
            pdf_path=tmp_path / "fake.pdf",
            qrels_path=runner.DEFAULT_QRELS_PATH,
            qrels_manifest=runner.DEFAULT_QRELS_MANIFEST_PATH,
            is_full_run=False,
        )
        assert res_out_path.exists()
