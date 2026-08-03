"""Regression test suite for Citation Provenance & Artifact Integration (Smoke Finding Fix).

Verifies mandatory TDD requirements:
1. content_sha256 missing + real text -> digest recalculated from text.
2. content_sha256 missing + text missing -> fail closed (CITATION_PROVENANCE_MISMATCH).
3. MagicMock is not needed to test provenance chain.
4. Unknown E1 -> CitationProvenanceMismatchError.
5. Page declared in model text does not override evidence page.
6. Legacy marker [92] does not get AVAILABLE in protocol v2.
7. citation_map survives checkpointing and rehydration.
8. ABSTAIN response remains without map and without citations.
9. gold_answer, relevant_pages and qrels do not enter generation prompt.
10. Complete W0/q_dev_01 fixture passes smoke validator.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from benchmarks.run_slice4_benchmark import (
    build_citation_map_and_status,
    serialize_retrieval_evidence,
    validate_smoke_result,
)
from raglab.domain.entities import GeneratedAnswer, RetrievedEvidence
from raglab.domain.errors import CitationProvenanceMismatchError
from raglab.domain.value_objects import ChunkId, Citation
from raglab.infrastructure.gemini.prompts import (
    PromptEvidence,
    build_generation_prompt,
)


@pytest.fixture
def regression_fixture_w0_q_dev_01():
    """Regression fixture matching W0_sentence_window x q_dev_01 smoke failure."""
    ev1 = RetrievedEvidence(
        chunk_id=ChunkId("gersting_p92_c0"),
        document_id="gersting_doc_p92",
        text="A indução matemática consiste em caso base e passo indutivo.",
        rank=1,
        score=0.95,
        passage_id="gersting_doc_p92_pass1",
    )
    ev2 = RetrievedEvidence(
        chunk_id=ChunkId("gersting_p96_c0"),
        document_id="gersting_doc_p96",
        text="Exemplo de prova indutiva para somatório.",
        rank=2,
        score=0.85,
        passage_id="gersting_doc_p96_pass1",
    )
    ev3 = RetrievedEvidence(
        chunk_id=ChunkId("gersting_p101_c0"),
        document_id="gersting_doc_p101",
        text="Indução forte e princípio da boa ordenação.",
        rank=3,
        score=0.75,
        passage_id="gersting_doc_p101_pass1",
    )

    retrieved = [ev1, ev2, ev3]
    prompt_evidences = PromptEvidence.from_retrieved_sequence(retrieved)
    evidence_record = serialize_retrieval_evidence(retrieved, relevant_pages=[92])

    citations_list = (
        Citation(
            document_id=ev1.document_id,
            page_number=92,
            chunk_id=ev1.chunk_id,
            text_span=ev1.text[:40],
            evidence_id="E1",
            passage_id="gersting_doc_p92_pass1",
            content_sha256=hashlib.sha256(ev1.text.encode("utf-8")).hexdigest(),
            retrieval_rank=1,
        ),
    )

    answer = GeneratedAnswer(
        query_id="q_dev_01",
        text="A demonstração por indução matemática exige a verificação do caso base e a hipótese indutiva.",
        abstained=False,
        citations=citations_list,
    )

    return {
        "strategy": "W0_sentence_window",
        "qid": "q_dev_01",
        "retrieved": retrieved,
        "prompt_evidences": prompt_evidences,
        "evidence_record": evidence_record,
        "answer": answer,
    }


class TestCitationProvenanceIntegration:
    """TDD test cases for citation map preservation and provenance auditability."""

    def test_1_content_sha256_missing_recalculated_from_text(self):
        """1. content_sha256 missing + real text -> digest recalculated from text."""
        cand = {
            "chunk_id": "c1",
            "page_number": 10,
            "text": "Texto de teste sem content_sha256 explicito",
            "evidence_id": "E1",
        }
        status, cit_map = build_citation_map_and_status(
            answer_text="Resposta [E1]",
            abstained=False,
            evidence=[cand],
            query_id="q1",
        )
        assert status == "AVAILABLE"
        expected_sha = hashlib.sha256(cand["text"].encode("utf-8")).hexdigest()
        assert cit_map[0]["content_sha256"] == expected_sha

    def test_2_content_sha256_missing_and_text_missing_fails_closed(self):
        """2. content_sha256 missing + text missing -> fail closed."""
        cand = {
            "chunk_id": "c1",
            "page_number": 10,
            "evidence_id": "E1",
        }
        with pytest.raises(ValueError, match="CITATION_PROVENANCE_MISMATCH"):
            build_citation_map_and_status(
                answer_text="Resposta [E1]",
                abstained=False,
                evidence=[cand],
                query_id="q1",
            )

    def test_3_magicmock_not_needed_for_provenance_chain(
        self, regression_fixture_w0_q_dev_01
    ):
        """3. MagicMock is not needed to test the provenance chain."""
        fix = regression_fixture_w0_q_dev_01
        status, cit_map = build_citation_map_and_status(
            answer_text=fix["answer"].text,
            abstained=fix["answer"].abstained,
            evidence=fix["retrieved"],
            query_id=fix["qid"],
            citations=fix["answer"].citations,
        )
        assert status == "AVAILABLE"
        assert len(cit_map) == 1
        assert cit_map[0]["evidence_id"] == "E1"
        assert cit_map[0]["page_number"] == 92

    def test_4_unknown_e1_fails_closed(self, regression_fixture_w0_q_dev_01):
        """4. Unknown E1 -> CitationProvenanceMismatchError / ValueError."""
        fix = regression_fixture_w0_q_dev_01
        with pytest.raises(
            (ValueError, CitationProvenanceMismatchError),
            match="CITATION_PROVENANCE_MISMATCH",
        ):
            build_citation_map_and_status(
                answer_text="Resposta [E99]",
                abstained=False,
                evidence=fix["evidence_record"]["candidates"],
                query_id=fix["qid"],
            )

    def test_5_model_text_page_number_does_not_override_evidence_page(self):
        """5. Page declared in model text does not override evidence page."""
        cand = {
            "chunk_id": "c1",
            "page_number": 42,
            "text": "Evidencia real da pagina 42",
            "evidence_id": "E1",
        }
        status, cit_map = build_citation_map_and_status(
            answer_text="Segundo a pagina 999 no texto do modelo [E1]",
            abstained=False,
            evidence=[cand],
            query_id="q1",
        )
        assert status == "AVAILABLE"
        assert cit_map[0]["page_number"] == 42

    def test_6_legacy_marker_92_does_not_get_v2_available_status(self):
        """6. Legacy marker [92] does not get AVAILABLE status in protocol v2."""
        cand = {
            "chunk_id": "c1",
            "page_number": 92,
            "text": "Evidencia da pagina 92",
        }
        # Since candidate list has length 1, marker [92] exceeds evidence list length
        with pytest.raises(ValueError, match="CITATION_PROVENANCE_MISMATCH"):
            build_citation_map_and_status(
                answer_text="Texto citando apenas marcador legado [92]",
                abstained=False,
                evidence=[cand],
                query_id="q1",
            )

    def test_6b_legacy_numeric_marker_returns_status_legacy_never_available(self):
        """Legacy numeric marker [1] returns status LEGACY, never AVAILABLE."""
        cand = {
            "chunk_id": "c1",
            "page_number": 92,
            "text": "Evidencia da pagina 92",
        }
        status, cit_map = build_citation_map_and_status(
            answer_text="Segundo a regra [1]",
            abstained=False,
            evidence=[cand],
            query_id="q1",
        )
        assert status == "LEGACY"
        assert status != "AVAILABLE"
        assert len(cit_map) == 1

    def test_7_citation_map_survives_checkpointing_and_rehydration(
        self, regression_fixture_w0_q_dev_01
    ):
        """7. citation_map survives checkpointing and rehydration."""
        fix = regression_fixture_w0_q_dev_01
        status, cit_map = build_citation_map_and_status(
            answer_text=fix["answer"].text,
            abstained=fix["answer"].abstained,
            evidence=fix["evidence_record"]["candidates"],
            query_id=fix["qid"],
            citations=fix["answer"].citations,
        )

        result_entry = {
            "qid": fix["qid"],
            "strategy": fix["strategy"],
            "abstained": False,
            "citation_mapping_status": status,
            "citation_map": cit_map,
            "citation_pages": [c["page_number"] for c in cit_map],
        }

        serialized = json.dumps(result_entry)
        rehydrated = json.loads(serialized)

        assert rehydrated["citation_mapping_status"] == "AVAILABLE"
        assert len(rehydrated["citation_map"]) == 1
        assert rehydrated["citation_map"][0]["evidence_id"] == "E1"
        assert rehydrated["citation_pages"] == [92]

    def test_8_abstain_response_remains_without_map_and_without_citations(
        self, regression_fixture_w0_q_dev_01
    ):
        """8. ABSTAIN response remains without map and without citations."""
        fix = regression_fixture_w0_q_dev_01
        status, cit_map = build_citation_map_and_status(
            answer_text="ABSTAIN",
            abstained=True,
            evidence=fix["evidence_record"]["candidates"],
            query_id=fix["qid"],
        )
        assert status == "NOT_APPLICABLE"
        assert cit_map == []

    def test_9_gold_answer_relevant_pages_qrels_not_in_prompt(
        self, regression_fixture_w0_q_dev_01
    ):
        """9. gold_answer, relevant_pages and qrels do not enter generation prompt."""
        fix = regression_fixture_w0_q_dev_01
        prompt = build_generation_prompt(
            "O que é indução?", fix["prompt_evidences"]
        )
        assert "gold_answer" not in prompt
        assert "relevant_pages" not in prompt
        assert "qrels" not in prompt
        assert "holdout" not in prompt

    def test_10_w0_q_dev_01_complete_fixture_passes_smoke_validator(
        self, regression_fixture_w0_q_dev_01
    ):
        """10. Complete W0/q_dev_01 fixture passes smoke validator."""
        import logging

        fix = regression_fixture_w0_q_dev_01
        status, cit_map = build_citation_map_and_status(
            answer_text=fix["answer"].text,
            abstained=fix["answer"].abstained,
            evidence=fix["evidence_record"]["candidates"],
            query_id=fix["qid"],
            citations=fix["answer"].citations,
        )

        full_result = {
            "qid": fix["qid"],
            "strategy": fix["strategy"],
            "relevant_pages": [92],
            "abstained": False,
            "is_abstention_question": False,
            "citation_mapping_status": status,
            "citation_map": cit_map,
            "citation_pages": [c["page_number"] for c in cit_map],
            "retrieval_evidence": fix["evidence_record"],
            "answer": {
                "text": fix["answer"].text,
                "text_sha256": hashlib.sha256(
                    fix["answer"].text.encode("utf-8")
                ).hexdigest(),
                "truncated": False,
            },
            "evaluation": {
                "schema_version": "slice4_v3",
                "metrics": [
                    {"name": "abstention_correctness", "status": "COMPUTED", "score": 1.0},
                    {"name": "context_relevance", "status": "COMPUTED", "score": 1.0},
                    {"name": "groundedness", "status": "COMPUTED", "score": 1.0},
                    {"name": "answer_relevance", "status": "COMPUTED", "score": 1.0},
                ],
            },
        }

        data_wrapper = {
            "embedding_fingerprints": {
                "fastembed": {"cache_tree_sha256": "0" * 64}
            },
            "manifest_fingerprint": "0" * 64,
            "results": {fix["strategy"]: [full_result]},
        }

        res = validate_smoke_result(
            data=data_wrapper,
            strategy=fix["strategy"],
            qid=fix["qid"],
            is_abstention_question=False,
            logger=logging.getLogger("test"),
        )
        assert res == "SMOKE_POSITIVE_OK"
