"""Unit tests for Silver-to-Human Queue Routing Governance (Gate B2).

Validates all 18 fail-closed requirements for building blinded human review queues
from real Machine Silver triage execution.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_human_review_queues import build_human_queues

INPUT_ROOT = Path("benchmarks/ground_truth/v2/hybrid")


class TestSilverQueueRoutingGovernance:
    """Test suite covering the 18 governance invariants for real silver queue routing."""

    @pytest.fixture
    def synthetic_silver_file(self, tmp_path: Path) -> Path:
        pool_file = INPUT_ROOT / "candidate_pool" / "pool.jsonl"
        pool_lines = [
            json.loads(line)
            for line in pool_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        run_dir = tmp_path / "synthetic_silver_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        sil_file = run_dir / "silver_annotations.jsonl"

        records = []
        for i, item in enumerate(pool_lines):
            needs_review = i < 23
            grade = 2 if i < 24 else 0
            records.append({
                "question_id": item["question_id"],
                "passage_id": item["passage_id"],
                "label_source": "MACHINE_SILVER",
                "silver_grade": grade,
                "relevance_grade": grade,
                "needs_human_review": needs_review,
                "confidence_score": 0.85,
                "reasoning": "Synthetic test record",
            })


        content = "\n".join(json.dumps(r) for r in records) + "\n"
        sil_file.write_text(content, encoding="utf-8")
        sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()

        man_file = run_dir / "silver_manifest.json"
        man_file.write_text(
            json.dumps({
                "schema_version": "2.0.0",
                "mode": "full",
                "execution_mode": "FULL_REAL",
                "execution_authenticity": "REAL_MODEL_CALL",
                "status": "COMPLETED",
                "pending_count": 0,
                "authoritative_for_human_qrels": False,
                "holdout_sealed": True,
                "total_records": len(records),
                "silver_file_sha256": sha256,
            }),
            encoding="utf-8",
        )

        ckpt_file = run_dir / "checkpoint.json"
        ckpt_file.write_text(
            json.dumps({
                "mode": "full",
                "status": "COMPLETED",
                "execution_mode": "FULL_REAL",
                "execution_authenticity": "REAL_MODEL_CALL",
                "pending_count": 0,
                "processed_count": len(records),
                "total_pairs": len(records),
                "authoritative_for_human_qrels": False,
                "holdout_sealed": True,
                "records_sha256": sha256,
            }),
            encoding="utf-8",
        )

        return sil_file

    # 1. sem opção de silver e sem --without-silver-execution → falha
    def test_01_no_options_fails(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        with pytest.raises(ValueError, match="Must provide either --silver-file"):
            build_human_queues(INPUT_ROOT, out)
        assert not (out / "annotator_a.jsonl").exists()

    # 2. ambas as opções → falha
    def test_02_both_options_fails(
        self, tmp_path: Path, synthetic_silver_file: Path
    ) -> None:
        out = tmp_path / "out"
        with pytest.raises(ValueError, match="Cannot specify both"):
            build_human_queues(
                INPUT_ROOT,
                out,
                silver_file=synthetic_silver_file,
                without_silver_execution=True,
            )
        assert not (out / "annotator_a.jsonl").exists()

    # 3. silver inexistente → falha
    def test_03_nonexistent_silver_file_fails(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        fake_silver = tmp_path / "missing_silver.jsonl"
        with pytest.raises(FileNotFoundError):
            build_human_queues(INPUT_ROOT, out, silver_file=fake_silver)
        assert not (out / "annotator_a.jsonl").exists()

    # 4. JSONL inválido → falha sem saída parcial
    def test_04_invalid_jsonl_fails_without_partial_output(
        self, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        bad_silver = run_dir / "silver_annotations.jsonl"
        bad_silver.write_text(
            '{"question_id": "q1"}\nINVALID_JSON_LINE\n', encoding="utf-8"
        )

        out = tmp_path / "out"
        with pytest.raises(ValueError, match="Invalid JSON in silver file"):
            build_human_queues(INPUT_ROOT, out, silver_file=bad_silver)
        assert not (out / "annotator_a.jsonl").exists()

    # 5. par duplicado → falha
    def test_05_duplicate_pair_fails(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        dup_silver = run_dir / "silver_annotations.jsonl"
        line = json.dumps(
            {
                "question_id": "q_dev_01",
                "passage_id": "ps_01",
                "label_source": "MACHINE_SILVER",
            }
        )
        dup_silver.write_text(f"{line}\n{line}\n", encoding="utf-8")

        out = tmp_path / "out"
        with pytest.raises(ValueError, match="Duplicate pair"):
            build_human_queues(INPUT_ROOT, out, silver_file=dup_silver)
        assert not (out / "annotator_a.jsonl").exists()

    # 6. holdout → falha
    def test_06_holdout_fails(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        holdout_silver = run_dir / "silver_annotations.jsonl"
        holdout_silver.write_text(
            json.dumps(
                {
                    "question_id": "q_holdout_01",
                    "passage_id": "ps_01",
                    "label_source": "MACHINE_SILVER",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        out = tmp_path / "out"
        with pytest.raises(ValueError, match="HOLDOUT VIOLATION"):
            build_human_queues(INPUT_ROOT, out, silver_file=holdout_silver)
        assert not (out / "annotator_a.jsonl").exists()

    # 7. par ausente ou inesperado → falha
    def test_07_missing_or_unexpected_pair_fails(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        incomplete_silver = run_dir / "silver_annotations.jsonl"
        incomplete_silver.write_text(
            json.dumps(
                {
                    "question_id": "q_dev_01",
                    "passage_id": "ps_01",
                    "label_source": "MACHINE_SILVER",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        out = tmp_path / "out"
        with pytest.raises(
            ValueError, match="Silver records do not match candidate pool"
        ):
            build_human_queues(INPUT_ROOT, out, silver_file=incomplete_silver)
        assert not (out / "annotator_a.jsonl").exists()

    # 8. manifesto não real/incompleto → falha
    def test_08_invalid_manifest_fails(
        self, tmp_path: Path, synthetic_silver_file: Path
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        sil_file = run_dir / "silver_annotations.jsonl"
        sil_file.write_bytes(synthetic_silver_file.read_bytes())

        man_file = run_dir / "silver_manifest.json"
        man_file.write_text(
            json.dumps(
                {
                    "mode": "smoke",
                    "execution_mode": "FULL_REAL",
                    "execution_authenticity": "REAL_MODEL_CALL",
                    "status": "COMPLETED",
                    "pending_count": 0,
                    "authoritative_for_human_qrels": False,
                    "holdout_sealed": True,
                }
            ),
            encoding="utf-8",
        )

        out = tmp_path / "out"
        with pytest.raises(ValueError, match="Manifest mode must be 'full'"):
            build_human_queues(INPUT_ROOT, out, silver_file=sil_file)

    # 9. hash divergente → falha
    def test_09_divergent_hash_fails(
        self, tmp_path: Path, synthetic_silver_file: Path
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        sil_file = run_dir / "silver_annotations.jsonl"
        sil_file.write_text(
            synthetic_silver_file.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )

        real_run = synthetic_silver_file.parent
        (run_dir / "silver_manifest.json").write_bytes(
            (real_run / "silver_manifest.json").read_bytes()
        )
        (run_dir / "checkpoint.json").write_bytes(
            (real_run / "checkpoint.json").read_bytes()
        )

        out = tmp_path / "out"
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            build_human_queues(INPUT_ROOT, out, silver_file=sil_file)

    # 10. 69 pares válidos → sucesso
    def test_10_valid_real_silver_succeeds(
        self, tmp_path: Path, synthetic_silver_file: Path
    ) -> None:
        out = tmp_path / "out"
        fa, fb, fadj, fman = build_human_queues(
            INPUT_ROOT, out, silver_file=synthetic_silver_file
        )
        assert fa.exists()
        assert fb.exists()
        assert fadj.exists()
        assert fman.exists()

        man = json.loads(fman.read_text(encoding="utf-8"))
        assert man["queue_status"] == "DEFINITIVE_HUMAN_REVIEW"
        assert man["silver_used_in_routing"] is True
        assert man["silver_records_count"] == 69

    # 11. fila A contém exatamente os 69 pares
    def test_11_queue_a_contains_69_pairs(
        self, tmp_path: Path, synthetic_silver_file: Path
    ) -> None:
        out = tmp_path / "out"
        fa, _, _, _ = build_human_queues(
            INPUT_ROOT, out, silver_file=synthetic_silver_file
        )
        lines = [
            json.loads(line)
            for line in fa.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(lines) == 69
        keys = {(item["question_id"], item["passage_id"]) for item in lines}
        assert len(keys) == 69

    # 12. todos os 23 needs_human_review=true aparecem na fila B
    def test_12_all_23_needs_review_in_queue_b(
        self, tmp_path: Path, synthetic_silver_file: Path
    ) -> None:
        out = tmp_path / "out"
        _, fb, _, _ = build_human_queues(
            INPUT_ROOT, out, silver_file=synthetic_silver_file
        )
        lines_b = [
            json.loads(line)
            for line in fb.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        keys_b = {(item["question_id"], item["passage_id"]) for item in lines_b}

        silver_records = [
            json.loads(line)
            for line in synthetic_silver_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        review_keys = {
            (r["question_id"], r["passage_id"])
            for r in silver_records
            if r.get("needs_human_review") is True
        }
        assert len(review_keys) == 23
        assert review_keys.issubset(keys_b)

    # 13. todos os 24 registros com grade maior que zero aparecem na fila B
    def test_13_all_24_positives_in_queue_b(
        self, tmp_path: Path, synthetic_silver_file: Path
    ) -> None:
        out = tmp_path / "out"
        _, fb, _, _ = build_human_queues(
            INPUT_ROOT, out, silver_file=synthetic_silver_file
        )
        lines_b = [
            json.loads(line)
            for line in fb.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        keys_b = {(item["question_id"], item["passage_id"]) for item in lines_b}

        silver_records = [
            json.loads(line)
            for line in synthetic_silver_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        positive_keys = {
            (r["question_id"], r["passage_id"])
            for r in silver_records
            if r.get("silver_grade", 0) > 0
        }
        assert len(positive_keys) == 24
        assert positive_keys.issubset(keys_b)

    # 14. fila B não tem duplicatas
    def test_14_queue_b_no_duplicates(
        self, tmp_path: Path, synthetic_silver_file: Path
    ) -> None:
        out = tmp_path / "out"
        _, fb, _, _ = build_human_queues(
            INPUT_ROOT, out, silver_file=synthetic_silver_file
        )
        lines_b = [
            json.loads(line)
            for line in fb.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        keys_b = [(item["question_id"], item["passage_id"]) for item in lines_b]
        assert len(keys_b) == len(set(keys_b))

    # 15. campos silver_grade e reasoning ausentes das filas cegas
    def test_15_silver_fields_absent_from_blinded_queues(
        self, tmp_path: Path, synthetic_silver_file: Path
    ) -> None:
        out = tmp_path / "out"
        fa, fb, _, _ = build_human_queues(
            INPUT_ROOT, out, silver_file=synthetic_silver_file
        )
        for f in (fa, fb):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                assert "silver_grade" not in item
                assert "reasoning" not in item

    # 16. nenhum valor silver preenche anotação humana
    def test_16_human_fields_unfilled(
        self, tmp_path: Path, synthetic_silver_file: Path
    ) -> None:
        out = tmp_path / "out"
        fa, fb, fadj, _ = build_human_queues(
            INPUT_ROOT, out, silver_file=synthetic_silver_file
        )
        for path in (fa, fb):
            items = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for item in items:
                assert item["relevance_grade"] is None
                assert item["evidence_role"] is None
                assert item["annotation_notes"] == ""
                assert item["status"] == "PENDING"

        adj_items = [
            json.loads(line)
            for line in fadj.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for item in adj_items:
            assert item["annotator_a_grade"] is None
            assert item["annotator_b_grade"] is None
            assert item["adjudicated_grade"] is None
            assert item["adjudicated_role"] is None
            assert item["status"] == "PENDING_HUMAN_ANNOTATIONS"

    # 17. execução provisória continua possível somente com --without-silver-execution
    def test_17_provisional_execution_works(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        fa, fb, _, fman = build_human_queues(
            INPUT_ROOT, out, without_silver_execution=True
        )
        assert fa.exists()
        assert fb.exists()

        man = json.loads(fman.read_text(encoding="utf-8"))
        assert man["queue_status"] == "PROVISIONAL_WITHOUT_SILVER"
        assert man["silver_used_in_routing"] is False


    # 18. rebuild reproduz exatamente o mesmo conteúdo
    def test_18_reproducible_rebuild(
        self, tmp_path: Path, synthetic_silver_file: Path
    ) -> None:
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"

        fa1, fb1, fadj1, fman1 = build_human_queues(
            INPUT_ROOT, out1, silver_file=synthetic_silver_file
        )
        fa2, fb2, fadj2, fman2 = build_human_queues(
            INPUT_ROOT, out2, silver_file=synthetic_silver_file
        )

        assert fa1.read_bytes() == fa2.read_bytes()
        assert fb1.read_bytes() == fb2.read_bytes()
        assert fadj1.read_bytes() == fadj2.read_bytes()
        assert fman1.read_bytes() == fman2.read_bytes()
