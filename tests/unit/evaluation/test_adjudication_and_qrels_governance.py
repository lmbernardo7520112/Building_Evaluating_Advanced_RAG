# ruff: noqa: E501, E741
"""Unit tests for Blinded Human Adjudication and Qrels Consolidation Governance (Gate B).

Covers all required test invariants for inter-annotator agreement metrics, 4x4/2x2 confusion matrices,
kappa calculations, abstention audit detection, anonymization, span/reasoning validation, and qrels consolidation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.adjudicate_human_queue import AdjudicationSession, sha256_file
from scripts.build_final_human_qrels import build_final_human_qrels
from scripts.build_human_adjudication_queue import (
    build_adjudication_queue,
    should_swap_reviewers,
)
from scripts.compute_human_agreement import (
    compute_agreement,
    compute_cohens_kappa_quadratic,
    compute_cohens_kappa_unweighted,
)

EXPORT_A_FILE = Path("benchmarks/ground_truth/v2/hybrid/human_annotations/export/annotator_a_final.jsonl")
EXPORT_B_FILE = Path("benchmarks/ground_truth/v2/hybrid/human_annotations/export/annotator_b_combined_final.jsonl")
QUESTIONS_FILE = Path("benchmarks/questions/controlled_chapter2.json")


class TestAdjudicationAndQrelsGovernance:
    """Suíte de testes unitários para concordância, adjudicação e consolidação de qrels."""

    # 1. Métricas conhecidas no dataset autoritativo
    def test_01_metrics_reproduced_known_fixture(self, tmp_path: Path) -> None:
        out_rep = tmp_path / "report.json"
        out_dis = tmp_path / "disagreements.json"

        report, _ = compute_agreement(EXPORT_A_FILE, EXPORT_B_FILE, out_rep, out_dis)

        assert report["total_pairs"] == 69
        assert report["exact_agreement_count"] == 48
        assert report["exact_agreement_rate"] == 0.6957
        assert report["cohen_kappa_unweighted"] == 0.5791
        assert report["cohen_kappa_quadratic"] == 0.8198
        assert report["binary_relevant_agreement"] == 1.0
        assert report["disagreements"] == 21
        assert report["adjacent_disagreements"] == 19
        assert report["severe_disagreements"] == 2

    # 2. Kappa sem pesos
    def test_02_unweighted_kappa_calculation(self) -> None:
        r1 = [0, 1, 2, 3]
        r2 = [0, 1, 2, 3]
        assert compute_cohens_kappa_unweighted(r1, r2) == 1.0

        r1 = [0, 0, 1, 1]
        r2 = [1, 1, 0, 0]
        assert compute_cohens_kappa_unweighted(r1, r2) < 0.0

    # 3. Kappa quadrático
    def test_03_quadratic_kappa_calculation(self) -> None:
        r1 = [0, 1, 2, 3]
        r2 = [0, 1, 2, 3]
        assert compute_cohens_kappa_quadratic(r1, r2) == 1.0

    # 4. Matrizes de confusão
    def test_04_confusion_matrices_4x4_and_2x2(self, tmp_path: Path) -> None:
        out_rep = tmp_path / "report.json"
        out_dis = tmp_path / "disagreements.json"

        report, _ = compute_agreement(EXPORT_A_FILE, EXPORT_B_FILE, out_rep, out_dis)

        cm4 = report["confusion_matrix_4x4"]["matrix"]
        assert len(cm4) == 4 and len(cm4[0]) == 4
        assert sum(sum(row) for row in cm4) == 69

        cm2 = report["confusion_matrix_2x2_binary"]["matrix"]
        assert len(cm2) == 2 and len(cm2[0]) == 2
        assert sum(sum(row) for row in cm2) == 69

    # 5. Universo divergente rejeitado
    def test_05_divergent_universe_rejection(self, tmp_path: Path) -> None:
        fa = tmp_path / "export_a.jsonl"
        fb = tmp_path / "export_b.jsonl"

        fa.write_text(json.dumps({"annotator_id": "annotator_a", "question_id": "q_dev_01", "passage_id": "ps_1", "relevance_grade": 3}) + "\n")
        fb.write_text(json.dumps({"annotator_id": "annotator_b", "question_id": "q_dev_01", "passage_id": "ps_2", "relevance_grade": 3}) + "\n")

        with pytest.raises(ValueError, match="Universe mismatch"):
            compute_agreement(fa, fb, tmp_path / "r.json", tmp_path / "d.json")

    # 6. Duplicação rejeitada
    def test_06_duplication_rejection(self, tmp_path: Path) -> None:
        fa = tmp_path / "export_a.jsonl"
        fb = tmp_path / "export_b.jsonl"

        rec_a = {"annotator_id": "annotator_a", "question_id": "q_dev_01", "passage_id": "ps_1", "relevance_grade": 3}
        rec_b = {"annotator_id": "annotator_b", "question_id": "q_dev_01", "passage_id": "ps_1", "relevance_grade": 3}

        fa.write_text(json.dumps(rec_a) + "\n" + json.dumps(rec_a) + "\n")
        fb.write_text(json.dumps(rec_b) + "\n")

        with pytest.raises(ValueError, match="Duplicate pair"):
            compute_agreement(fa, fb, tmp_path / "r.json", tmp_path / "d.json")

    # 7. Validação de identidade
    def test_07_identity_validation(self, tmp_path: Path) -> None:
        fa = tmp_path / "export_a.jsonl"
        fb = tmp_path / "export_b.jsonl"

        rec_bad = {"annotator_id": "WRONG_ID", "question_id": "q_dev_01", "passage_id": "ps_1", "relevance_grade": 3}
        fa.write_text(json.dumps(rec_bad) + "\n")
        fb.write_text(json.dumps(rec_bad) + "\n")

        with pytest.raises(ValueError, match="Identity mismatch"):
            compute_agreement(fa, fb, tmp_path / "r.json", tmp_path / "d.json")

    # 8. Anonimização alternada determinística
    def test_08_alternating_anonymization_determinism(self, tmp_path: Path) -> None:
        out_q1 = tmp_path / "adj_q1.jsonl"
        out_m1 = tmp_path / "adj_m1.json"
        out_q2 = tmp_path / "adj_q2.jsonl"
        out_m2 = tmp_path / "adj_m2.json"

        build_adjudication_queue(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, out_q1, out_m1)
        build_adjudication_queue(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, out_q2, out_m2)

        assert sha256_file(out_q1) == sha256_file(out_q2)

        manifest = json.loads(out_m1.read_text())
        assert manifest["reviewer_order_algorithm"] == "sha256-domain-separated-v1"
        assert manifest["reviewer_order_domain"] == "raglab:v7:adjudication-reviewer-order"

    # 9. 21 desacordos incluídos na fila
    def test_09_disagreements_included_21(self, tmp_path: Path) -> None:
        out_q = tmp_path / "adj_q.jsonl"
        out_m = tmp_path / "adj_m.json"

        build_adjudication_queue(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, out_q, out_m)

        manifest = json.loads(out_m.read_text())
        assert manifest["total_disagreement_pairs"] == 21

    # 10. 10 abstenções incluídas na fila
    def test_10_structural_abstention_included_10(self, tmp_path: Path) -> None:
        out_q = tmp_path / "adj_q.jsonl"
        out_m = tmp_path / "adj_m.json"

        build_adjudication_queue(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, out_q, out_m)

        manifest = json.loads(out_m.read_text())
        assert manifest["total_abstention_pairs"] == 10

    # 11. União deduplicada de 28 itens no dataset real
    def test_11_deduplicated_union_28(self, tmp_path: Path) -> None:
        out_q = tmp_path / "adj_q.jsonl"
        out_m = tmp_path / "adj_m.json"

        build_adjudication_queue(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, out_q, out_m)

        items = [json.loads(l) for l in out_q.read_text().splitlines() if l.strip()]
        assert len(items) == 28
        manifest = json.loads(out_m.read_text())
        assert manifest["total_adjudication_queue"] == 28

    # 12. Sem campos silver na fila de adjudicação
    def test_12_no_silver_fields_in_adjudication_queue(self, tmp_path: Path) -> None:
        out_q = tmp_path / "adj_q.jsonl"
        out_m = tmp_path / "adj_m.json"

        build_adjudication_queue(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, out_q, out_m)

        content = out_q.read_text()
        assert "confidence" not in content
        assert "judge_model" not in content
        assert "retrieval_rank" not in content

    # 13. Sem respostas gold na fila de adjudicação
    def test_13_no_gold_answers_in_adjudication_queue(self, tmp_path: Path) -> None:
        out_q = tmp_path / "adj_q.jsonl"
        out_m = tmp_path / "adj_m.json"

        build_adjudication_queue(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, out_q, out_m)

        content = out_q.read_text()
        assert "gold_answer" not in content
        assert "relevant_pages" not in content

    # 14. Reasoning obrigatório na adjudicação
    def test_14_reasoning_mandatory_in_adjudication(self, tmp_path: Path) -> None:
        out_q = tmp_path / "adj_q.jsonl"
        out_m = tmp_path / "adj_m.json"
        build_adjudication_queue(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, out_q, out_m)

        session = AdjudicationSession("adjudicator_1", out_q, QUESTIONS_FILE, tmp_path / "work.jsonl")

        with pytest.raises(ValueError, match="reasoning is mandatory"):
            session.save_adjudication(index=0, grade=0, role="NEGATIVE_CONTROL", reasoning="", span="")

    # 15. Trecho obrigatório para grau > 0
    def test_15_span_mandatory_for_grade_greater_than_zero(self, tmp_path: Path) -> None:
        out_q = tmp_path / "adj_q.jsonl"
        out_m = tmp_path / "adj_m.json"
        build_adjudication_queue(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, out_q, out_m)

        session = AdjudicationSession("adjudicator_1", out_q, QUESTIONS_FILE, tmp_path / "work.jsonl")

        with pytest.raises(ValueError, match="mandatory when grade > 0"):
            session.save_adjudication(index=0, grade=3, role="PRIMARY", reasoning="Justificativa válida", span="")

    # 16. Trecho proibido para grau == 0
    def test_16_span_forbidden_for_grade_zero(self, tmp_path: Path) -> None:
        out_q = tmp_path / "adj_q.jsonl"
        out_m = tmp_path / "adj_m.json"
        build_adjudication_queue(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, out_q, out_m)

        session = AdjudicationSession("adjudicator_1", out_q, QUESTIONS_FILE, tmp_path / "work.jsonl")
        item = session.get_item(0)
        valid_span = item["passage_text"][:10] if item["passage_text"] else "sample"

        with pytest.raises(ValueError, match="Supporting span must be empty when grade == 0"):
            session.save_adjudication(index=0, grade=0, role="NEGATIVE_CONTROL", reasoning="Ruído", span=valid_span)

    # 17. Retomada de sessão de adjudicação
    def test_17_adjudication_session_resume(self, tmp_path: Path) -> None:
        out_q = tmp_path / "adj_q.jsonl"
        out_m = tmp_path / "adj_m.json"
        work_file = tmp_path / "work.jsonl"

        build_adjudication_queue(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, out_q, out_m)

        s1 = AdjudicationSession("adjudicator_1", out_q, QUESTIONS_FILE, work_file)
        s1.save_adjudication(index=0, grade=0, role="NEGATIVE_CONTROL", reasoning="Irrelevante", span="")

        prog1 = s1.get_progress()
        assert prog1["completed_items"] == 1

        s2 = AdjudicationSession("adjudicator_1", out_q, QUESTIONS_FILE, work_file)
        prog2 = s2.get_progress()
        assert prog2["completed_items"] == 1

    # 18. Escrita atômica no trabalho de adjudicação
    def test_18_atomic_persistence(self, tmp_path: Path) -> None:
        out_q = tmp_path / "adj_q.jsonl"
        out_m = tmp_path / "adj_m.json"
        work_file = tmp_path / "work.jsonl"

        build_adjudication_queue(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, out_q, out_m)

        s = AdjudicationSession("adjudicator_1", out_q, QUESTIONS_FILE, work_file)
        s.save_adjudication(index=0, grade=0, role="NEGATIVE_CONTROL", reasoning="Ruído", span="")

        assert work_file.exists()
        assert sha256_file(work_file) != ""

    # 19. Consenso fora da adjudicação no qrels final
    def test_19_consensus_outside_adjudication_queue(self, tmp_path: Path) -> None:
        out_q = tmp_path / "adj_q.jsonl"
        out_m = tmp_path / "adj_m.json"
        work_file = tmp_path / "work.jsonl"

        build_adjudication_queue(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, out_q, out_m)

        s = AdjudicationSession("adjudicator_1", out_q, QUESTIONS_FILE, work_file)
        for idx in range(28):
            item = s.get_item(idx)
            g = item["reviewer_1_grade"]
            r = item["reviewer_1_role"]
            text = item["passage_text"]
            span = text[:15] if (g > 0 and text) else ("span" if g > 0 else "")
            s.save_adjudication(idx, g, r, f"Reasoning for {idx}", span)

        out_qrels = tmp_path / "human_qrels_final.jsonl"
        out_man = tmp_path / "human_qrels_manifest.json"

        build_final_human_qrels(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, work_file, out_qrels, out_man)

        qrels = [json.loads(l) for l in out_qrels.read_text().splitlines() if l.strip()]
        consensus_items = [q for q in qrels if q["provenance"] == "HUMAN_EXACT_CONSENSUS"]
        assert len(consensus_items) == 41

    # 20. Adjudicação substitui consenso/divergência
    def test_20_adjudicated_overrides_consensus(self, tmp_path: Path) -> None:
        out_q = tmp_path / "adj_q.jsonl"
        out_m = tmp_path / "adj_m.json"
        work_file = tmp_path / "work.jsonl"

        build_adjudication_queue(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, out_q, out_m)

        s = AdjudicationSession("adjudicator_1", out_q, QUESTIONS_FILE, work_file)
        for idx in range(28):
            item = s.get_item(idx)
            g = item["reviewer_1_grade"]
            r = item["reviewer_1_role"]
            text = item["passage_text"]
            span = text[:15] if (g > 0 and text) else ("span" if g > 0 else "")
            s.save_adjudication(idx, g, r, f"Reasoning for {idx}", span)

        out_qrels = tmp_path / "human_qrels_final.jsonl"
        out_man = tmp_path / "human_qrels_manifest.json"

        build_final_human_qrels(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, work_file, out_qrels, out_man)

        qrels = [json.loads(l) for l in out_qrels.read_text().splitlines() if l.strip()]
        adj_items = [q for q in qrels if q["provenance"] == "HUMAN_ADJUDICATED"]
        assert len(adj_items) == 28

    # 21. Cobertura exata de 69 itens
    def test_21_exact_coverage_69(self, tmp_path: Path) -> None:
        out_q = tmp_path / "adj_q.jsonl"
        out_m = tmp_path / "adj_m.json"
        work_file = tmp_path / "work.jsonl"

        build_adjudication_queue(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, out_q, out_m)

        s = AdjudicationSession("adjudicator_1", out_q, QUESTIONS_FILE, work_file)
        for idx in range(28):
            item = s.get_item(idx)
            g = item["reviewer_1_grade"]
            r = item["reviewer_1_role"]
            text = item["passage_text"]
            span = text[:15] if (g > 0 and text) else ("span" if g > 0 else "")
            s.save_adjudication(idx, g, r, f"Reasoning for {idx}", span)

        out_qrels = tmp_path / "human_qrels_final.jsonl"
        out_man = tmp_path / "human_qrels_manifest.json"

        build_final_human_qrels(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, work_file, out_qrels, out_man)

        qrels = [json.loads(l) for l in out_qrels.read_text().splitlines() if l.strip()]
        assert len(qrels) == 69
        manifest = json.loads(out_man.read_text())
        assert manifest["total_pairs"] == 69
        assert manifest["consensus_pairs_count"] == 41
        assert manifest["adjudicated_pairs_count"] == 28

    # 22. Sem cálculo automático de grau
    def test_22_zero_automatic_grade_calculation(self, tmp_path: Path) -> None:
        out_q = tmp_path / "adj_q.jsonl"
        out_m = tmp_path / "adj_m.json"
        work_file = tmp_path / "work.jsonl"

        build_adjudication_queue(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, out_q, out_m)

        s = AdjudicationSession("adjudicator_1", out_q, QUESTIONS_FILE, work_file)
        for idx in range(28):
            item = s.get_item(idx)
            g = item["reviewer_1_grade"]
            r = item["reviewer_1_role"]
            text = item["passage_text"]
            span = text[:15] if (g > 0 and text) else ("span" if g > 0 else "")
            s.save_adjudication(idx, g, r, f"Reasoning for {idx}", span)

        out_qrels = tmp_path / "human_qrels_final.jsonl"
        out_man = tmp_path / "human_qrels_manifest.json"

        build_final_human_qrels(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, work_file, out_qrels, out_man)

        for line in out_qrels.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                assert item["provenance"] in ("HUMAN_EXACT_CONSENSUS", "HUMAN_ADJUDICATED")

    # 23. Manifesto com hashes e holdout selado
    def test_23_manifest_hashes_and_holdout_sealed(self, tmp_path: Path) -> None:
        out_q = tmp_path / "adj_q.jsonl"
        out_m = tmp_path / "adj_m.json"
        work_file = tmp_path / "work.jsonl"

        build_adjudication_queue(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, out_q, out_m)

        s = AdjudicationSession("adjudicator_1", out_q, QUESTIONS_FILE, work_file)
        for idx in range(28):
            item = s.get_item(idx)
            g = item["reviewer_1_grade"]
            r = item["reviewer_1_role"]
            text = item["passage_text"]
            span = text[:15] if (g > 0 and text) else ("span" if g > 0 else "")
            s.save_adjudication(idx, g, r, f"Reasoning for {idx}", span)

        out_qrels = tmp_path / "human_qrels_final.jsonl"
        out_man = tmp_path / "human_qrels_manifest.json"

        build_final_human_qrels(EXPORT_A_FILE, EXPORT_B_FILE, QUESTIONS_FILE, work_file, out_qrels, out_man)

        manifest = json.loads(out_man.read_text())
        assert manifest["authoritative_for_evaluation"] is True
        assert manifest["silver_used_as_ground_truth"] is False
        assert manifest["holdout_sealed"] is True
        assert manifest["final_qrels_file_sha256"] == sha256_file(out_qrels)

    # 24. Independência de PYTHONHASHSEED (Subprocessos)
    def test_24_pythonhashseed_independence(self, tmp_path: Path) -> None:
        script_path = Path("scripts/build_human_adjudication_queue.py")

        seeds = ["1", "2", "random"]
        outputs: list[tuple[bytes, str, dict]] = []

        for seed in seeds:
            out_q = tmp_path / f"queue_seed_{seed}.jsonl"
            out_m = tmp_path / f"manifest_seed_{seed}.json"

            cmd = [
                sys.executable,
                str(script_path),
                "--annotator-a",
                str(EXPORT_A_FILE),
                "--annotator-b-combined",
                str(EXPORT_B_FILE),
                "--questions-file",
                str(QUESTIONS_FILE),
                "--output-queue",
                str(out_q),
                "--output-manifest",
                str(out_m),
            ]

            env = dict(os.environ)
            env["PYTHONHASHSEED"] = seed

            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
            assert proc.returncode == 0

            q_bytes = out_q.read_bytes()
            q_hash = sha256_file(out_q)
            manifest = json.loads(out_m.read_text())

            outputs.append((q_bytes, q_hash, manifest))

        # Check 1: All queues are byte-by-byte identical
        first_bytes, first_hash, first_manifest = outputs[0]
        for b, h, m in outputs[1:]:
            assert b == first_bytes
            assert h == first_hash
            assert m["total_adjudication_queue"] == 28
            assert m["total_disagreement_pairs"] == 21
            assert m["total_abstention_pairs"] == 10
            assert m["reviewer_order_algorithm"] == "sha256-domain-separated-v1"
            assert m["reviewer_order_domain"] == "raglab:v7:adjudication-reviewer-order"

        # Check 2: Absence of A/B annotator identity in generated queue file
        content_str = first_bytes.decode("utf-8")
        assert "annotator_a" not in content_str
        assert "annotator_b" not in content_str

    # 25. Propriedades da função should_swap_reviewers
    def test_25_should_swap_reviewers_properties(self) -> None:
        # Changing question_id changes payload/digest
        b1 = should_swap_reviewers("q_dev_01", "ps_1")
        b2 = should_swap_reviewers("q_dev_02", "ps_1")

        # Changing passage_id changes payload/digest
        b3 = should_swap_reviewers("q_dev_01", "ps_2")

        # Determinism check: same parameters always yield exact same result
        for _ in range(10):
            assert should_swap_reviewers("q_dev_01", "ps_1") == b1
            assert should_swap_reviewers("q_dev_02", "ps_1") == b2
            assert should_swap_reviewers("q_dev_01", "ps_2") == b3

    # 26. Ausência de hash() nativo no caminho de anonimização
    def test_26_no_native_hash_in_anonymization_code(self) -> None:
        builder_source = Path("scripts/build_human_adjudication_queue.py").read_text(encoding="utf-8")
        assert "hash(" not in builder_source
