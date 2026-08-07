# ruff: noqa: E501
"""Unit tests for Annotator B Supplemental Queue Builder and Merger Governance (Gate B).

Covers all 20 required test points for A - B set difference, zero leakage, blinding,
holdout rejection, deterministic hashes, and atomic merge validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.annotate_human_queue import sha256_file
from scripts.build_supplemental_b_queue import (
    build_supplemental_b_queue,
    load_blinded_queue_items,
)
from scripts.merge_human_annotations_b import merge_annotator_b_exports

QUEUE_A_FILE = Path("benchmarks/ground_truth/v2/hybrid/human_queues/annotator_a.jsonl")
QUEUE_B_FILE = Path("benchmarks/ground_truth/v2/hybrid/human_queues/annotator_b.jsonl")
QUESTIONS_FILE = Path("benchmarks/questions/controlled_chapter2.json")

EXPORT_A_FILE = Path(
    "benchmarks/ground_truth/v2/hybrid/human_annotations/export/annotator_a_final.jsonl"
)
EXPORT_B_FILE = Path(
    "benchmarks/ground_truth/v2/hybrid/human_annotations/export/annotator_b_final.jsonl"
)


class TestSupplementalBQueueGovernance:
    """Suíte de 20 testes unitários de governança do fluxo suplementar de B."""

    # 1. Diferença A-B correta
    def test_01_correct_set_difference(self, tmp_path: Path) -> None:
        qa = tmp_path / "qa.jsonl"
        qb = tmp_path / "qb.jsonl"

        item1 = {
            "question_id": "q_dev_01",
            "passage_id": "ps_1",
            "text": "txt1",
            "page_number": 92,
            "annotator_id": "annotator_a",
        }
        item2 = {
            "question_id": "q_dev_01",
            "passage_id": "ps_2",
            "text": "txt2",
            "page_number": 92,
            "annotator_id": "annotator_a",
        }
        item3 = {
            "question_id": "q_dev_02",
            "passage_id": "ps_3",
            "text": "txt3",
            "page_number": 95,
            "annotator_id": "annotator_a",
        }

        qa.write_text("\n".join(json.dumps(i) for i in [item1, item2, item3]) + "\n")
        qb.write_text(json.dumps({**item1, "annotator_id": "annotator_b"}) + "\n")

        out_q = tmp_path / "supp_q.jsonl"
        out_m = tmp_path / "supp_m.json"

        build_supplemental_b_queue(qa, qb, out_q, out_m, expected_supplemental_count=2)

        supp_items = [
            json.loads(line) for line in out_q.read_text().splitlines() if line.strip()
        ]
        assert len(supp_items) == 2
        pairs = {(i["question_id"], i["passage_id"]) for i in supp_items}
        assert pairs == {("q_dev_01", "ps_2"), ("q_dev_02", "ps_3")}

    # 2. Exatamente 16 pares no dataset real
    def test_02_real_dataset_has_exactly_16_pairs(self, tmp_path: Path) -> None:
        out_q = tmp_path / "supp_q.jsonl"
        out_m = tmp_path / "supp_m.json"

        build_supplemental_b_queue(
            QUEUE_A_FILE, QUEUE_B_FILE, out_q, out_m, expected_supplemental_count=16
        )

        supp_items = [
            json.loads(line) for line in out_q.read_text().splitlines() if line.strip()
        ]
        assert len(supp_items) == 16
        manifest = json.loads(out_m.read_text())
        assert manifest["supplemental_count"] == 16
        assert manifest["total_queue_a"] == 69
        assert manifest["total_queue_b_original"] == 53

    # 3. Nenhuma leitura de exports humanos no builder
    def test_03_zero_reading_of_human_exports(self) -> None:
        with pytest.raises(ValueError, match="SECURITY VIOLATION"):
            load_blinded_queue_items(EXPORT_A_FILE)

    # 4. Nenhum campo do Anotador A nos itens suplementares
    def test_04_no_fields_from_annotator_a(self, tmp_path: Path) -> None:
        out_q = tmp_path / "supp_q.jsonl"
        out_m = tmp_path / "supp_m.json"
        build_supplemental_b_queue(QUEUE_A_FILE, QUEUE_B_FILE, out_q, out_m)

        for line in out_q.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                assert item["annotator_id"] == "annotator_b"
                assert item["relevance_grade"] is None
                assert item["evidence_role"] is None
                assert item["annotation_notes"] == ""

    # 5. Nenhuma informação silver
    def test_05_no_silver_information(self, tmp_path: Path) -> None:
        out_q = tmp_path / "supp_q.jsonl"
        out_m = tmp_path / "supp_m.json"
        build_supplemental_b_queue(QUEUE_A_FILE, QUEUE_B_FILE, out_q, out_m)

        content = out_q.read_text()
        assert "confidence" not in content
        assert "reasoning" not in content
        assert "judge_model" not in content
        assert "retrieval_rank" not in content

    # 6. annotator_id=B
    def test_06_annotator_id_b(self, tmp_path: Path) -> None:
        out_q = tmp_path / "supp_q.jsonl"
        out_m = tmp_path / "supp_m.json"
        build_supplemental_b_queue(QUEUE_A_FILE, QUEUE_B_FILE, out_q, out_m)

        for line in out_q.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                assert item["annotator_id"] == "annotator_b"

    # 7. Campos humanos vazios
    def test_07_empty_human_fields(self, tmp_path: Path) -> None:
        out_q = tmp_path / "supp_q.jsonl"
        out_m = tmp_path / "supp_m.json"
        build_supplemental_b_queue(QUEUE_A_FILE, QUEUE_B_FILE, out_q, out_m)

        for line in out_q.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                assert item["status"] == "PENDING"
                assert item["supporting_span_human"] == ""

    # 8. Holdout rejeitado
    def test_08_holdout_rejection(self, tmp_path: Path) -> None:
        bad_qa = tmp_path / "bad_qa.jsonl"
        bad_qa.write_text(
            json.dumps(
                {
                    "question_id": "q_holdout_01",
                    "passage_id": "ps_1",
                    "text": "txt",
                    "annotator_id": "annotator_a",
                }
            )
            + "\n"
        )
        qb = tmp_path / "qb.jsonl"
        qb.write_text("")

        out_q = tmp_path / "supp_q.jsonl"
        out_m = tmp_path / "supp_m.json"

        with pytest.raises(ValueError, match="HOLDOUT VIOLATION"):
            build_supplemental_b_queue(bad_qa, qb, out_q, out_m)

    # 9. Duplicação rejeitada
    def test_09_duplication_rejection(self, tmp_path: Path) -> None:
        dup_qa = tmp_path / "dup_qa.jsonl"
        item = {
            "question_id": "q_dev_01",
            "passage_id": "ps_1",
            "text": "txt",
            "annotator_id": "annotator_a",
        }
        dup_qa.write_text(json.dumps(item) + "\n" + json.dumps(item) + "\n")
        qb = tmp_path / "qb.jsonl"
        qb.write_text("")

        out_q = tmp_path / "supp_q.jsonl"
        out_m = tmp_path / "supp_m.json"

        with pytest.raises(ValueError, match="Duplicate pair"):
            build_supplemental_b_queue(dup_qa, qb, out_q, out_m)

    # 10. Ordenação determinística
    def test_10_deterministic_sorting(self, tmp_path: Path) -> None:
        out_q = tmp_path / "supp_q.jsonl"
        out_m = tmp_path / "supp_m.json"
        build_supplemental_b_queue(QUEUE_A_FILE, QUEUE_B_FILE, out_q, out_m)

        items = [
            json.loads(line) for line in out_q.read_text().splitlines() if line.strip()
        ]
        keys = [(i["question_id"], i["passage_id"]) for i in items]
        assert keys == sorted(keys)

    # 11. Hashes determinísticos
    def test_11_deterministic_hashes(self, tmp_path: Path) -> None:
        out_q1 = tmp_path / "supp_q1.jsonl"
        out_m1 = tmp_path / "supp_m1.json"
        out_q2 = tmp_path / "supp_q2.jsonl"
        out_m2 = tmp_path / "supp_m2.json"

        build_supplemental_b_queue(QUEUE_A_FILE, QUEUE_B_FILE, out_q1, out_m1)
        build_supplemental_b_queue(QUEUE_A_FILE, QUEUE_B_FILE, out_q2, out_m2)

        assert sha256_file(out_q1) == sha256_file(out_q2)

    # 12. Fila original A imutável
    def test_12_queue_a_immutability(self, tmp_path: Path) -> None:
        hash_before = sha256_file(QUEUE_A_FILE)
        out_q = tmp_path / "supp_q.jsonl"
        out_m = tmp_path / "supp_m.json"
        build_supplemental_b_queue(QUEUE_A_FILE, QUEUE_B_FILE, out_q, out_m)
        assert sha256_file(QUEUE_A_FILE) == hash_before

    # 13. Fila original B imutável
    def test_13_queue_b_immutability(self, tmp_path: Path) -> None:
        hash_before = sha256_file(QUEUE_B_FILE)
        out_q = tmp_path / "supp_q.jsonl"
        out_m = tmp_path / "supp_m.json"
        build_supplemental_b_queue(QUEUE_A_FILE, QUEUE_B_FILE, out_q, out_m)
        assert sha256_file(QUEUE_B_FILE) == hash_before

    # 14. Merge 53 + 16 = 69
    def test_14_merger_combines_53_plus_16_to_69(self, tmp_path: Path) -> None:
        # Construct synthetic fixture exports for 53 and 16 items matching Queue A
        qa_items = [
            json.loads(line)
            for line in QUEUE_A_FILE.read_text().splitlines()
            if line.strip()
        ]
        qb_items = [
            json.loads(line)
            for line in QUEUE_B_FILE.read_text().splitlines()
            if line.strip()
        ]

        qb_pairs = {(i["question_id"], i["passage_id"]) for i in qb_items}
        qa_diff_pairs = [
            i for i in qa_items if (i["question_id"], i["passage_id"]) not in qb_pairs
        ]

        orig_exp = tmp_path / "orig_b_export.jsonl"
        supp_exp = tmp_path / "supp_b_export.jsonl"

        # Create valid 53 export records
        recs_orig = [
            {
                "schema_version": "3.0.0",
                "protocol_version": "raglab_v7_slice4_v3",
                "annotator_id": "annotator_b",
                "question_id": i["question_id"],
                "passage_id": i["passage_id"],
                "page_number": i["page_number"],
                "relevance_grade": 3,
                "evidence_role": "PRIMARY",
                "supporting_span_human": "",
                "annotation_notes": "",
                "status": "COMPLETED",
                "export_status": "VALIDATED_HUMAN_QRELS",
            }
            for i in qb_items
        ]
        orig_exp.write_text("\n".join(json.dumps(r) for r in recs_orig) + "\n")

        # Create valid 16 export records
        recs_supp = [
            {
                "schema_version": "3.0.0",
                "protocol_version": "raglab_v7_slice4_v3",
                "annotator_id": "annotator_b",
                "question_id": i["question_id"],
                "passage_id": i["passage_id"],
                "page_number": i["page_number"],
                "relevance_grade": 1,
                "evidence_role": "CONTEXTUAL",
                "supporting_span_human": "",
                "annotation_notes": "",
                "status": "COMPLETED",
                "export_status": "VALIDATED_HUMAN_QRELS",
            }
            for i in qa_diff_pairs
        ]
        supp_exp.write_text("\n".join(json.dumps(r) for r in recs_supp) + "\n")

        out_comb = tmp_path / "combined_b.jsonl"
        out_man = tmp_path / "combined_manifest.json"

        merge_annotator_b_exports(orig_exp, supp_exp, QUEUE_A_FILE, out_comb, out_man)

        combined = [
            json.loads(line)
            for line in out_comb.read_text().splitlines()
            if line.strip()
        ]
        assert len(combined) == 69
        manifest = json.loads(out_man.read_text())
        assert manifest["total_combined_count"] == 69
        assert manifest["original_count"] == 53
        assert manifest["supplemental_count"] == 16

    # 15. Overlap no merge rejeitado
    def test_15_merger_overlap_rejection(self, tmp_path: Path) -> None:
        rec = {
            "annotator_id": "annotator_b",
            "question_id": "q_dev_01",
            "passage_id": "ps_1",
            "relevance_grade": 3,
            "evidence_role": "PRIMARY",
        }
        orig_exp = tmp_path / "orig.jsonl"
        supp_exp = tmp_path / "supp.jsonl"

        orig_exp.write_text(json.dumps(rec) + "\n")
        supp_exp.write_text(json.dumps(rec) + "\n")

        out_comb = tmp_path / "comb.jsonl"
        out_man = tmp_path / "man.json"

        with pytest.raises(ValueError, match="Overlap detected"):
            merge_annotator_b_exports(
                orig_exp, supp_exp, QUEUE_A_FILE, out_comb, out_man
            )

    # 16. Item ausente rejeitado
    def test_16_merger_missing_item_rejection(self, tmp_path: Path) -> None:
        orig_exp = tmp_path / "orig.jsonl"
        supp_exp = tmp_path / "supp.jsonl"

        rec1 = {
            "annotator_id": "annotator_b",
            "question_id": "q_dev_01",
            "passage_id": "ps_1",
            "relevance_grade": 3,
            "evidence_role": "PRIMARY",
        }
        rec2 = {
            "annotator_id": "annotator_b",
            "question_id": "q_dev_01",
            "passage_id": "ps_2",
            "relevance_grade": 2,
            "evidence_role": "SUPPORTING",
        }
        orig_exp.write_text(json.dumps(rec1) + "\n")
        supp_exp.write_text(json.dumps(rec2) + "\n")

        out_comb = tmp_path / "comb.jsonl"
        out_man = tmp_path / "man.json"

        with pytest.raises(ValueError, match="Combined count mismatch"):
            merge_annotator_b_exports(
                orig_exp, supp_exp, QUEUE_A_FILE, out_comb, out_man
            )

    # 17. Item inesperado rejeitado
    def test_17_merger_unexpected_item_rejection(self, tmp_path: Path) -> None:
        orig_exp = tmp_path / "orig.jsonl"
        supp_exp = tmp_path / "supp.jsonl"

        rec = {
            "annotator_id": "annotator_b",
            "question_id": "q_dev_UNEXPECTED",
            "passage_id": "ps_999",
            "relevance_grade": 3,
            "evidence_role": "PRIMARY",
        }
        orig_exp.write_text(json.dumps(rec) + "\n")
        supp_exp.write_text(
            json.dumps({**rec, "question_id": "q_dev_UNEXPECTED_2"}) + "\n"
        )

        out_comb = tmp_path / "comb.jsonl"
        out_man = tmp_path / "man.json"

        with pytest.raises(ValueError, match="Combined count mismatch"):
            merge_annotator_b_exports(
                orig_exp, supp_exp, QUEUE_A_FILE, out_comb, out_man
            )

    # 18. Identidade incorreta rejeitada
    def test_18_merger_incorrect_identity_rejection(self, tmp_path: Path) -> None:
        orig_exp = tmp_path / "orig.jsonl"
        supp_exp = tmp_path / "supp.jsonl"

        rec = {
            "annotator_id": "annotator_a",  # Erro: deveria ser annotator_b
            "question_id": "q_dev_01",
            "passage_id": "ps_1",
            "relevance_grade": 3,
            "evidence_role": "PRIMARY",
        }
        orig_exp.write_text(json.dumps(rec) + "\n")
        supp_exp.write_text("")

        out_comb = tmp_path / "comb.jsonl"
        out_man = tmp_path / "man.json"

        with pytest.raises(ValueError, match="Identity mismatch"):
            merge_annotator_b_exports(
                orig_exp, supp_exp, QUEUE_A_FILE, out_comb, out_man
            )

    # 19. Corrupção rejeitada
    def test_19_merger_corrupted_file_rejection(self, tmp_path: Path) -> None:
        orig_exp = tmp_path / "orig.jsonl"
        supp_exp = tmp_path / "supp.jsonl"

        orig_exp.write_text("CORRUPTED_JSON_LINE\n")
        supp_exp.write_text("")

        out_comb = tmp_path / "comb.jsonl"
        out_man = tmp_path / "man.json"

        with pytest.raises(ValueError, match="Invalid JSON"):
            merge_annotator_b_exports(
                orig_exp, supp_exp, QUEUE_A_FILE, out_comb, out_man
            )

    # 20. Arquivos originais não sobrescritos
    def test_20_original_files_not_overwritten(self, tmp_path: Path) -> None:
        hash_exp_a = sha256_file(EXPORT_A_FILE)
        hash_exp_b = sha256_file(EXPORT_B_FILE)

        out_q = tmp_path / "supp_q.jsonl"
        out_m = tmp_path / "supp_m.json"
        build_supplemental_b_queue(QUEUE_A_FILE, QUEUE_B_FILE, out_q, out_m)

        assert sha256_file(EXPORT_A_FILE) == hash_exp_a
        assert sha256_file(EXPORT_B_FILE) == hash_exp_b
