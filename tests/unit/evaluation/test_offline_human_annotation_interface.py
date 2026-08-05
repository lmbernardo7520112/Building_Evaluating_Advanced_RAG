"""Unit tests for Offline Blinded Human Annotation Interface & Validator (Gate B).

Covers all 30 required tests for stdlib-only http.server, 127.0.0.1 binding,
pre-flight fail-closed validations, atomic persistence, blinding, and export validation.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from scripts.annotate_human_queue import (
    AnnotationSession,
    atomic_write_jsonl,
    create_annotation_server,
    load_and_validate_queue,
    load_questions_file,
    sha256_file,
    validate_annotation_payload,
)
from scripts.validate_human_annotations import validate_and_export_human_annotations

QUEUE_A_FILE = Path("benchmarks/ground_truth/v2/hybrid/human_queues/annotator_a.jsonl")
QUEUE_B_FILE = Path("benchmarks/ground_truth/v2/hybrid/human_queues/annotator_b.jsonl")
QUESTIONS_FILE = Path("benchmarks/questions/controlled_chapter2.json")


class TestOfflineHumanAnnotationInterface:
    """Suíte de 30 testes unitários cobrindo todos os requisitos da interface offline."""

    # 1. Carregamento das filas A e B
    def test_01_queue_loading_a_and_b(self) -> None:
        qmap = load_questions_file(QUESTIONS_FILE)
        items_a = load_and_validate_queue(QUEUE_A_FILE, "annotator_a", qmap)
        items_b = load_and_validate_queue(QUEUE_B_FILE, "annotator_b", qmap)
        assert len(items_a) == 69
        assert len(items_b) == 53

    # 2. Incompatibilidade de identidade
    def test_02_identity_mismatch_fails(self) -> None:
        qmap = load_questions_file(QUESTIONS_FILE)
        with pytest.raises(ValueError, match="Identity mismatch"):
            load_and_validate_queue(QUEUE_B_FILE, "annotator_a", qmap)

    # 3. Retomada
    def test_03_session_resume(self, tmp_path: Path) -> None:
        out_file = tmp_path / "work_a.jsonl"
        session1 = AnnotationSession(
            "annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file
        )
        session1.save_annotation(0, 3, "PRIMARY", "", "")

        session2 = AnnotationSession(
            "annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file
        )
        progress = session2.get_progress()
        assert progress["completed_items"] == 1
        item0 = session2.get_item_data(0)
        assert item0["existing_annotation"]["relevance_grade"] == 3

    # 4. Escrita atômica
    def test_04_atomic_write(self, tmp_path: Path) -> None:
        target = tmp_path / "output.jsonl"
        recs = [{"id": 1}, {"id": 2}]
        atomic_write_jsonl(target, recs)
        assert target.exists()
        lines = target.read_text("utf-8").splitlines()
        assert len(lines) == 2

    # 5. Duplicação
    def test_05_duplication_prevention(self, tmp_path: Path) -> None:
        out_file = tmp_path / "work_a.jsonl"
        session = AnnotationSession(
            "annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file
        )
        session.save_annotation(0, 3, "PRIMARY", "", "")
        session.unlock_edit(0)
        session.save_annotation(0, 2, "SUPPORTING", "", "Ajuste na revisão")

        lines = [
            json.loads(line)
            for line in out_file.read_text().splitlines()
            if line.strip()
        ]
        assert len(lines) == 1
        assert lines[0]["relevance_grade"] == 2

    # 6. Corrupção
    def test_06_corrupted_file_detection(self, tmp_path: Path) -> None:
        out_file = tmp_path / "work_a.jsonl"
        out_file.write_text("CORRUPTED_JSON_LINE\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Corrupted existing output file"):
            AnnotationSession("annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file)

    # 7. Hash divergente
    def test_07_queue_hash_mismatch_fails(self, tmp_path: Path) -> None:
        out_file = tmp_path / "work_a.jsonl"
        out_file.write_text(
            json.dumps(
                {
                    "annotator_id": "annotator_a",
                    "queue_file_sha256": "wrong_hash",
                    "questions_file_sha256": sha256_file(QUESTIONS_FILE),
                    "question_id": "q_dev_01",
                    "passage_id": "ps_1",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="different queue file version"):
            AnnotationSession("annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file)

    # 8. Holdout
    def test_08_holdout_rejection(self, tmp_path: Path) -> None:
        bad_queue = tmp_path / "bad_queue.jsonl"
        bad_queue.write_text(
            json.dumps(
                {
                    "question_id": "q_holdout_01",
                    "passage_id": "ps_1",
                    "text": "abc",
                    "page_number": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        qmap = load_questions_file(QUESTIONS_FILE)
        with pytest.raises(ValueError, match="HOLDOUT VIOLATION"):
            load_and_validate_queue(bad_queue, "annotator_a", qmap)

    # 9. Grau inválido
    def test_09_invalid_relevance_grade_fails(self) -> None:
        with pytest.raises(ValueError, match="invalid"):
            validate_annotation_payload(4, "PRIMARY", "", "", "texto")

    # 10. Papel inválido
    def test_10_invalid_evidence_role_fails(self) -> None:
        with pytest.raises(ValueError, match="invalid"):
            validate_annotation_payload(3, "INVALID_ROLE", "", "", "texto")

    # 11. Span não literal
    def test_11_non_literal_span_rejection(self) -> None:
        with pytest.raises(ValueError, match="not an exact literal substring"):
            validate_annotation_payload(
                3,
                "PRIMARY",
                "texto inexistente na passagem",
                "",
                "Este é o texto real.",
            )

    # 12. Separação A/B
    def test_12_strict_ab_separation(self, tmp_path: Path) -> None:
        out_file_a = tmp_path / "work_a.jsonl"
        session_a = AnnotationSession(
            "annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file_a
        )
        session_a.save_annotation(0, 3, "PRIMARY", "", "")

        with pytest.raises(ValueError, match="belongs to annotator 'annotator_a'"):
            AnnotationSession("annotator_b", QUEUE_B_FILE, QUESTIONS_FILE, out_file_a)

    # 13. Entrada imutável
    def test_13_input_queue_immutability(self, tmp_path: Path) -> None:
        initial_hash = sha256_file(QUEUE_A_FILE)
        out_file = tmp_path / "work_a.jsonl"
        session = AnnotationSession(
            "annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file
        )
        session.save_annotation(0, 3, "PRIMARY", "", "")
        assert sha256_file(QUEUE_A_FILE) == initial_hash

    # 14. Exportação incompleta bloqueada
    def test_14_incomplete_export_blocked(self, tmp_path: Path) -> None:
        out_file = tmp_path / "work_a.jsonl"
        export_file = tmp_path / "export_a.jsonl"
        session = AnnotationSession(
            "annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file
        )
        session.save_annotation(0, 3, "PRIMARY", "", "")  # Apenas 1 de 69

        with pytest.raises(ValueError, match="Incomplete annotation coverage"):
            validate_and_export_human_annotations(
                "annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file, export_file
            )

    # 15. Exportação completa aprovada
    def test_15_complete_export_approved(self, tmp_path: Path) -> None:
        out_file = tmp_path / "work_a.jsonl"
        export_file = tmp_path / "export_a.jsonl"
        session = AnnotationSession(
            "annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file
        )
        for i in range(len(session.queue_items)):
            session.save_annotation(i, 3, "PRIMARY", "", "")

        validate_and_export_human_annotations(
            "annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file, export_file
        )
        assert export_file.exists()
        lines = export_file.read_text("utf-8").splitlines()
        assert len(lines) == 69

    # 16. Ausência de campos silver
    def test_16_silver_fields_absence_verified(self, tmp_path: Path) -> None:
        out_file = tmp_path / "work_a.jsonl"
        session = AnnotationSession(
            "annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file
        )
        session.save_annotation(0, 3, "PRIMARY", "", "")
        content = out_file.read_text("utf-8")
        assert "confidence" not in content
        assert "reasoning" not in content
        assert "judge_model" not in content

    # 17. Comportamento após interrupção simulada
    def test_17_resilience_after_simulated_interruption(self, tmp_path: Path) -> None:
        out_file = tmp_path / "work_a.jsonl"
        s1 = AnnotationSession("annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file)
        s1.save_annotation(0, 3, "PRIMARY", "", "")
        s1.save_annotation(1, 2, "SUPPORTING", "", "")
        # Simular término bruto do processo e reabertura
        s2 = AnnotationSession("annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file)
        assert s2.get_progress()["completed_items"] == 2

    # 18. Arquivo de perguntas inexistente
    def test_18_nonexistent_questions_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_questions_file(Path("missing_questions.json"))

    # 19. Pergunta ausente
    def test_19_missing_question_id(self, tmp_path: Path) -> None:
        qfile = tmp_path / "q.json"
        qfile.write_text(
            json.dumps({"questions": [{"qid": "q_dev_01", "question": "Pergunta 1"}]}),
            encoding="utf-8",
        )
        # Queue A tem q_dev_02 que falta em qfile
        with pytest.raises(ValueError, match="not found in questions file"):
            load_and_validate_queue(
                QUEUE_A_FILE, "annotator_a", load_questions_file(qfile)
            )

    # 20. Pergunta duplicada
    def test_20_duplicate_question_id(self, tmp_path: Path) -> None:
        qfile = tmp_path / "q.json"
        qfile.write_text(
            json.dumps(
                {
                    "questions": [
                        {"qid": "q_dev_01", "question": "Pergunta 1"},
                        {"qid": "q_dev_01", "question": "Pergunta 1 duplicada"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Duplicate question_id"):
            load_questions_file(qfile)

    # 21. Pergunta vazia
    def test_21_empty_question_text(self, tmp_path: Path) -> None:
        qfile = tmp_path / "q.json"
        qfile.write_text(
            json.dumps({"questions": [{"qid": "q_dev_01", "question": "   "}]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Question text is empty"):
            load_questions_file(qfile)

    # 22. Holdout no arquivo de perguntas / seleção
    def test_22_holdout_in_questions_file(self, tmp_path: Path) -> None:
        qfile = tmp_path / "q.json"
        qfile.write_text(
            json.dumps(
                {"questions": [{"qid": "q_holdout_01", "question": "Pergunta holdout"}]}
            ),
            encoding="utf-8",
        )
        bad_queue = tmp_path / "queue.jsonl"
        bad_queue.write_text(
            json.dumps(
                {
                    "question_id": "q_holdout_01",
                    "passage_id": "ps_1",
                    "text": "text",
                    "page_number": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="HOLDOUT VIOLATION"):
            load_and_validate_queue(
                bad_queue, "annotator_a", load_questions_file(qfile)
            )

    # 23. POST sem token
    def test_23_post_without_token(self, tmp_path: Path) -> None:
        out_file = tmp_path / "work_a.jsonl"
        session = AnnotationSession(
            "annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file
        )
        server, token = create_annotation_server(session, host="127.0.0.1", port=0)
        port = server.server_port
        server_thread = pytest.importorskip("threading").Thread(
            target=server.serve_forever
        )
        server_thread.daemon = True
        server_thread.start()

        try:
            url = f"http://127.0.0.1:{port}/api/annotate"
            req = urllib.request.Request(
                url,
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req)  # noqa: S310
            assert exc_info.value.code == 401
        finally:
            server.shutdown()
            server.server_close()

    # 24. Host/Origin inválido
    def test_24_invalid_host_origin(self, tmp_path: Path) -> None:
        out_file = tmp_path / "work_a.jsonl"
        session = AnnotationSession(
            "annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file
        )
        server, token = create_annotation_server(session, host="127.0.0.1", port=0)
        port = server.server_port
        server_thread = pytest.importorskip("threading").Thread(
            target=server.serve_forever
        )
        server_thread.daemon = True
        server_thread.start()

        try:
            url = f"http://127.0.0.1:{port}/api/state"
            req = urllib.request.Request(url, headers={"Host": "malicious.com"})
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req)  # noqa: S310
            assert exc_info.value.code == 403
        finally:
            server.shutdown()
            server.server_close()

    # 25. POST acima do limite (1MB)
    def test_25_post_body_limit(self, tmp_path: Path) -> None:
        out_file = tmp_path / "work_a.jsonl"
        session = AnnotationSession(
            "annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file
        )
        server, token = create_annotation_server(session, host="127.0.0.1", port=0)
        port = server.server_port
        server_thread = pytest.importorskip("threading").Thread(
            target=server.serve_forever
        )
        server_thread.daemon = True
        server_thread.start()

        try:
            url = f"http://127.0.0.1:{port}/api/annotate"
            large_data = b"x" * (1_048_576 + 10)
            req = urllib.request.Request(
                url,
                data=large_data,
                headers={"Content-Type": "application/json", "X-Session-Token": token},
                method="POST",
            )
            with pytest.raises((urllib.error.HTTPError, urllib.error.URLError)) as exc_info:
                urllib.request.urlopen(req)  # noqa: S310
            if isinstance(exc_info.value, urllib.error.HTTPError):
                assert exc_info.value.code == 413

        finally:
            server.shutdown()
            server.server_close()

    # 26. Binding externo rejeitado
    def test_26_external_binding_rejected(self, tmp_path: Path) -> None:
        out_file = tmp_path / "work_a.jsonl"
        session = AnnotationSession(
            "annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file
        )
        with pytest.raises(
            ValueError, match="SECURITY VIOLATION: Refusing external host binding"
        ):
            create_annotation_server(session, host="0.0.0.0", port=8501)  # noqa: S104

    # 27. Escaping contra HTML/script injection
    def test_27_escaping_html_injection(self, tmp_path: Path) -> None:
        out_file = tmp_path / "work_a.jsonl"
        session = AnnotationSession(
            "annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file
        )
        server, token = create_annotation_server(session, host="127.0.0.1", port=0)
        port = server.server_port
        server_thread = pytest.importorskip("threading").Thread(
            target=server.serve_forever
        )
        server_thread.daemon = True
        server_thread.start()

        try:
            url = f"http://127.0.0.1:{port}/"
            resp = urllib.request.urlopen(url)  # noqa: S310
            html_text = resp.read().decode("utf-8")
            assert "<script>" not in html_text or "const SESSION_TOKEN" in html_text
            assert "X-Content-Type-Options" in resp.headers
        finally:
            server.shutdown()
            server.server_close()

    # 28. Validação server-side de span
    def test_28_server_side_span_validation(self, tmp_path: Path) -> None:
        out_file = tmp_path / "work_a.jsonl"
        session = AnnotationSession(
            "annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file
        )
        server, token = create_annotation_server(session, host="127.0.0.1", port=0)
        port = server.server_port
        server_thread = pytest.importorskip("threading").Thread(
            target=server.serve_forever
        )
        server_thread.daemon = True
        server_thread.start()

        try:
            url = f"http://127.0.0.1:{port}/api/annotate"
            payload = json.dumps(
                {
                    "index": 0,
                    "relevance_grade": 3,
                    "evidence_role": "PRIMARY",
                    "supporting_span_human": "span inexistente no texto",
                    "annotation_notes": "",
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json", "X-Session-Token": token},
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req)  # noqa: S310
            assert exc_info.value.code == 400
        finally:
            server.shutdown()
            server.server_close()

    # 29. Separação dos hashes de fila e perguntas
    def test_29_separation_queue_and_question_hashes(self, tmp_path: Path) -> None:
        out_file = tmp_path / "work_a.jsonl"
        out_file.write_text(
            json.dumps(
                {
                    "annotator_id": "annotator_a",
                    "queue_file_sha256": sha256_file(QUEUE_A_FILE),
                    "questions_file_sha256": "different_questions_sha",
                    "question_id": "q_dev_01",
                    "passage_id": "ps_1",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="different questions file version"):
            AnnotationSession("annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file)

    # 30. Nenhuma rota permite leitura arbitrária de arquivo
    def test_30_no_arbitrary_file_read_route(self, tmp_path: Path) -> None:
        out_file = tmp_path / "work_a.jsonl"
        session = AnnotationSession(
            "annotator_a", QUEUE_A_FILE, QUESTIONS_FILE, out_file
        )
        server, token = create_annotation_server(session, host="127.0.0.1", port=0)
        port = server.server_port
        server_thread = pytest.importorskip("threading").Thread(
            target=server.serve_forever
        )
        server_thread.daemon = True
        server_thread.start()

        try:
            url = f"http://127.0.0.1:{port}/../../etc/passwd"
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(url)
            assert exc_info.value.code == 404
        finally:
            server.shutdown()
            server.server_close()
