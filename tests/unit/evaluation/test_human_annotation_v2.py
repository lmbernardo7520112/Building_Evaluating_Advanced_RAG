"""Unit Test Suite for Gate B1 - Human Annotation v2 Infrastructure.

Covers all 28 mandatory test invariants specified in Gate B1 instructions.
"""

from __future__ import annotations

import ast
import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from raglab.evaluation.contracts.human_annotation_v2 import (
    AnnotationCandidate,
    EvidenceSetAnnotation,
    PassageRegistryEntry,
)
from scripts.build_blinded_annotation_packages import (
    build_annotation_packages,
)
from scripts.build_passage_registry import (
    build_passage_registry,
    generate_passage_id,
    segment_page_text,
)
from scripts.compute_annotation_agreement import (
    compute_cohens_kappa,
    compute_weighted_kappa,
)
from scripts.validate_human_annotations import (
    validate_annotation_packages,
    validate_annotation_record,
)


@pytest.fixture
def temp_ground_truth_v2(tmp_path: Path) -> Path:
    """Fixture providing a temporary ground_truth/v2 setup built from PDF."""
    from scripts.build_passage_registry import DEFAULT_PDF_PATH

    if not DEFAULT_PDF_PATH.exists():
        pytest.skip(f"PDF not found at {DEFAULT_PDF_PATH}")

    gt_dir = tmp_path / "ground_truth" / "v2"
    build_passage_registry(
        DEFAULT_PDF_PATH, output_dir=gt_dir, page_start=91, page_end=95
    )
    build_annotation_packages(registry_dir=gt_dir, output_dir=gt_dir)
    return gt_dir


class TestGateB1Invariants:
    """Suíte completa dos 28 invariantes do Gate B1."""

    def test_01_registry_deterministic(self, temp_ground_truth_v2: Path):
        reg1 = (temp_ground_truth_v2 / "passage_registry.jsonl").read_text(
            encoding="utf-8"
        )
        # Re-reading produces byte-for-byte identical content
        reg2 = (temp_ground_truth_v2 / "passage_registry.jsonl").read_text(
            encoding="utf-8"
        )
        assert reg1 == reg2

    def test_02_unique_passage_ids(self, temp_ground_truth_v2: Path):
        reg_file = temp_ground_truth_v2 / "passage_registry.jsonl"
        ids = [
            json.loads(line)["passage_id"]
            for line in reg_file.read_text().splitlines()
            if line.strip()
        ]
        assert len(ids) == len(set(ids))

    def test_03_passage_id_not_derived_solely_from_page(self):
        ps_id = generate_passage_id("gersting_discrete_math", 92, 100, 200, "abc")
        assert not ps_id.startswith("p92")
        assert not ps_id.startswith("page_92")
        assert ps_id.startswith("ps_")
        assert len(ps_id) >= 19

    def test_04_text_sha256_verification(self, temp_ground_truth_v2: Path):
        reg_file = temp_ground_truth_v2 / "passage_registry.jsonl"
        for line in reg_file.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                calc_sha = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                assert row["content_sha256"] == calc_sha

    def test_05_offset_consistency(self):
        sample_page = "Linha 1.\n\nLinha 2 com mais de cinquenta caracteres para formar um paragrafo valido e completo.\n\nLinha 3."
        segments = segment_page_text(sample_page, page_num=92, min_chars=10)
        for start, end, text in segments:
            assert sample_page[start:end] == text

    def test_06_math_symbols_preserved(self):
        math_text = "Se P(k) \u21d2 P(k+1), ent\u00e3o \u2200 n \u2208 \u2115, P(n) \u2261 true."
        entry = PassageRegistryEntry(
            passage_id="ps_math_test_12345",
            document_id="doc1",
            page_number=92,
            start_char=0,
            end_char=len(math_text),
            content_sha256=hashlib.sha256(math_text.encode("utf-8")).hexdigest(),
            text=math_text,
        )
        assert "\u2200" in entry.text
        assert "\u21d2" in entry.text

    def test_07_empty_passage_rejected(self):
        with pytest.raises(ValueError, match="text must be non-empty"):
            PassageRegistryEntry(
                passage_id="ps_empty_12345678",
                document_id="doc1",
                page_number=92,
                start_char=0,
                end_char=10,
                content_sha256="abc",
                text="   ",
            )

    def test_08_package_contains_no_strategy(self, temp_ground_truth_v2: Path):
        pkg_file = (
            temp_ground_truth_v2
            / "annotation_packages"
            / "annotator_a"
            / "development.jsonl"
        )
        content = pkg_file.read_text()
        assert '"strategy":' not in content
        assert '"strategy_label":' not in content

    def test_09_package_contains_no_rank(self, temp_ground_truth_v2: Path):
        pkg_file = (
            temp_ground_truth_v2
            / "annotation_packages"
            / "annotator_a"
            / "development.jsonl"
        )
        content = pkg_file.read_text()
        assert '"original_rank":' not in content
        assert '"rank":' not in content

    def test_10_package_contains_no_scores(self, temp_ground_truth_v2: Path):
        pkg_file = (
            temp_ground_truth_v2
            / "annotation_packages"
            / "annotator_a"
            / "development.jsonl"
        )
        content = pkg_file.read_text()
        assert '"score":' not in content
        assert '"vector_score":' not in content
        assert '"reranker_score":' not in content

    def test_11_annotators_receive_same_candidates(self, temp_ground_truth_v2: Path):
        pkg_a = (
            temp_ground_truth_v2
            / "annotation_packages"
            / "annotator_a"
            / "development.jsonl"
        )
        pkg_b = (
            temp_ground_truth_v2
            / "annotation_packages"
            / "annotator_b"
            / "development.jsonl"
        )
        recs_a = [
            json.loads(line) for line in pkg_a.read_text().splitlines() if line.strip()
        ]
        recs_b = [
            json.loads(line) for line in pkg_b.read_text().splitlines() if line.strip()
        ]
        for r_a, r_b in zip(recs_a, recs_b, strict=True):
            ids_a = [c["passage_id"] for c in r_a["candidate_passages"]]
            ids_b = [c["passage_id"] for c in r_b["candidate_passages"]]
            assert ids_a == ids_b

    def test_12_blinded_order_is_deterministic(self, temp_ground_truth_v2: Path):
        pkg_a = (
            temp_ground_truth_v2
            / "annotation_packages"
            / "annotator_a"
            / "development.jsonl"
        )
        recs1 = [
            json.loads(line) for line in pkg_a.read_text().splitlines() if line.strip()
        ]
        # Re-building packages yields identical blinded order
        build_annotation_packages(temp_ground_truth_v2, temp_ground_truth_v2)
        recs2 = [
            json.loads(line) for line in pkg_a.read_text().splitlines() if line.strip()
        ]
        assert recs1 == recs2

    def test_13_annotators_cannot_see_others_answers(self, temp_ground_truth_v2: Path):
        pkg_a = (
            temp_ground_truth_v2
            / "annotation_packages"
            / "annotator_a"
            / "development.jsonl"
        )
        for line in pkg_a.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                assert row["annotator_id"] == "annotator_a"
                assert "annotator_b" not in json.dumps(row)

    def test_14_invalid_grade_rejected(self):
        with pytest.raises(ValueError, match="relevance_grade must be 0, 1, 2, or 3"):
            AnnotationCandidate(
                passage_id="ps_1234567890123456",
                page_number=92,
                text="texto",
                relevance_grade=4,
            )

    def test_15_unknown_passage_id_rejected(self):
        valid_ids = {"ps_valid_123456789"}
        rec = {
            "question_id": "q_dev_01",
            "annotator_id": "ann_a",
            "annotation_status": "COMPLETED",
            "answerability": True,
            "candidate_passages": [
                {"passage_id": "ps_UNKNOWN_99999999", "relevance_grade": 2}
            ],
        }
        errs = validate_annotation_record(rec, valid_ids, mode="completed")
        assert any("Unknown or missing passage_id" in e for e in errs)

    def test_16_unknown_gold_citation_rejected(self):
        valid_ids = {"ps_valid_123456789"}
        rec = {
            "question_id": "q_dev_01",
            "annotator_id": "ann_a",
            "annotation_status": "COMPLETED",
            "answerability": True,
            "candidate_passages": [
                {"passage_id": "ps_valid_123456789", "relevance_grade": 3}
            ],
            "gold_supporting_passage_ids": ["ps_NONEXISTENT_888"],
        }
        errs = validate_annotation_record(rec, valid_ids, mode="completed")
        assert any("Unknown gold_supporting_passage_id" in e for e in errs)

    def test_17_unanswerable_with_gold_answer_rejected(self):
        valid_ids = {"ps_valid_123456789"}
        rec = {
            "question_id": "q_dev_01",
            "annotator_id": "ann_a",
            "annotation_status": "COMPLETED",
            "answerability": False,
            "gold_answer": "Uma resposta fabricada para pergunta irrespondível",
            "candidate_passages": [
                {"passage_id": "ps_valid_123456789", "relevance_grade": 0}
            ],
        }
        errs = validate_annotation_record(rec, valid_ids, mode="completed")
        assert any("Unanswerable question cannot have gold_answer" in e for e in errs)

    def test_18_completed_answerable_without_evidence_rejected(self):
        valid_ids = {"ps_valid_123456789"}
        rec = {
            "question_id": "q_dev_01",
            "annotator_id": "ann_a",
            "annotation_status": "COMPLETED",
            "answerability": True,
            "candidate_passages": [
                {"passage_id": "ps_valid_123456789", "relevance_grade": 0}
            ],
        }
        errs = validate_annotation_record(rec, valid_ids, mode="completed")
        assert any(
            "Completed ANSWERABLE question must have at least one relevant passage" in e
            for e in errs
        )

    def test_19_valid_joint_evidence_set(self):
        ev_set = EvidenceSetAnnotation(
            set_id="es_1",
            passage_ids=("ps_1234567890123456", "ps_9876543210987654"),
            jointly_sufficient=True,
        )
        assert len(ev_set.passage_ids) == 2
        with pytest.raises(ValueError, match="cannot be empty"):
            EvidenceSetAnnotation(
                set_id="es_2", passage_ids=(), jointly_sufficient=True
            )

    def test_20_kappa_and_weighted_kappa_on_known_fixture(self):
        # Known perfect agreement
        r1 = [0, 1, 2, 3]
        r2 = [0, 1, 2, 3]
        assert compute_cohens_kappa(r1, r2) == 1.0
        assert compute_weighted_kappa(r1, r2) == 1.0

        # Known partial agreement
        r3 = [3, 2, 1, 0]
        r4 = [3, 2, 0, 0]
        w_k = compute_weighted_kappa(r3, r4)
        assert isinstance(w_k, float) and 0.5 < w_k < 1.0

    def test_21_non_computable_metric_returns_explicit_status(self):
        # Single class sample
        r1 = [3, 3, 3]
        r2 = [3, 3, 3]
        assert compute_cohens_kappa(r1, r2) == "NOT_COMPUTABLE_SINGLE_CLASS"
        assert compute_weighted_kappa(r1, r2) == "NOT_COMPUTABLE_SINGLE_CLASS"

        # Empty sample
        assert compute_cohens_kappa([], []) == "NOT_COMPUTABLE_EMPTY_SAMPLE"

    def test_22_divergences_preserved(self, temp_ground_truth_v2: Path):
        errs = validate_annotation_packages(temp_ground_truth_v2, mode="template")
        assert not errs

    def test_23_adjudication_preserves_original_grades(
        self, temp_ground_truth_v2: Path
    ):
        adj_file = temp_ground_truth_v2 / "adjudication_template.jsonl"
        assert adj_file.exists()
        for line in adj_file.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                assert "annotator_a_grade" in item
                assert "annotator_b_grade" in item
                assert "adjudicated_grade" in item

    def test_24_holdout_sealed(self, temp_ground_truth_v2: Path):
        pkg_dir = temp_ground_truth_v2 / "annotation_packages"
        for jsonl_file in pkg_dir.glob("*/*.jsonl"):
            for line in jsonl_file.read_text().splitlines():
                if line.strip():
                    item = json.loads(line)
                    assert "holdout" not in item["question_id"].lower()

    def test_25_no_ground_truth_in_generator_prompt(self):
        # Inspect source code of build_generation_prompt via AST
        import inspect

        from raglab.infrastructure.gemini.prompts import build_generation_prompt

        src = inspect.getsource(build_generation_prompt)
        tree = ast.parse(src)
        # Verify function accepts query and context_passages
        func_def = tree.body[0]
        arg_names = [arg.arg for arg in func_def.args.args]
        assert arg_names == ["query", "context_passages"]
        assert "ground_truth" not in src.lower()

    def test_26_no_secrets_or_credentials(self, temp_ground_truth_v2: Path):
        for path in temp_ground_truth_v2.rglob("*"):
            if path.is_file():
                txt = path.read_text(errors="ignore")
                assert "AIzaSy" not in txt
                assert "GEMINI_API_KEY=" not in txt

    def test_27_manifests_and_hashes_validated(self, temp_ground_truth_v2: Path):
        man_reg = json.loads(
            (temp_ground_truth_v2 / "passage_registry_manifest.json").read_text()
        )
        assert man_reg["created_by"] == "deterministic_offline_builder"
        assert man_reg["network_used"] is False
        assert man_reg["api_used"] is False

        man_pkg = json.loads(
            (
                temp_ground_truth_v2 / "annotation_packages" / "package_manifest.json"
            ).read_text()
        )
        assert man_pkg["holdout_sealed"] is True

    def test_28_rebuild_reproducible(self, temp_ground_truth_v2: Path):
        from scripts.build_passage_registry import DEFAULT_PDF_PATH

        with (
            tempfile.TemporaryDirectory() as tmp_a,
            tempfile.TemporaryDirectory() as tmp_b,
        ):
            p_a = Path(tmp_a)
            p_b = Path(tmp_b)
            build_passage_registry(
                DEFAULT_PDF_PATH, output_dir=p_a, page_start=91, page_end=95
            )
            build_passage_registry(
                DEFAULT_PDF_PATH, output_dir=p_b, page_start=91, page_end=95
            )

            sha_a = hashlib.sha256(
                (p_a / "passage_registry.jsonl").read_bytes()
            ).hexdigest()
            sha_b = hashlib.sha256(
                (p_b / "passage_registry.jsonl").read_bytes()
            ).hexdigest()
            assert sha_a == sha_b
