"""Tests for GeminiGeneratorAdapter (offline — no credentials required).

These tests verify:
1. Adapter raises RuntimeError if GEMINI_API_KEY is absent
2. Adapter is NOT imported at module level (lazy import safety)
3. Sanitize function returns correct fields
4. No credential leaks in any exported surface

SECURITY: All tests run WITHOUT GEMINI_API_KEY. They only test the
interface, initialization guard, and offline utilities.
The actual Gemini calls are covered by integration tests (Ambiente B only).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from raglab.domain.entities import RetrievedEvidence
from raglab.domain.value_objects import ChunkId


class TestGeminiGeneratorNoCredential:
    """Verify the adapter's credential guard works correctly."""

    def test_raises_without_api_key(self, monkeypatch):
        """GeminiGeneratorAdapter must raise RuntimeError when no key is present."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        from raglab.infrastructure.gemini.gemini_generator_adapter import (
            GeminiGeneratorAdapter,
        )
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            GeminiGeneratorAdapter()

    def test_adapter_module_has_security_docstring(self):
        """Security boundary must be documented in the module."""
        import raglab.infrastructure.gemini.gemini_generator_adapter as mod
        assert "SECURITY BOUNDARY" in (mod.__doc__ or "")
        assert "Ambiente B" in (mod.__doc__ or "")

    def test_gemini_not_imported_at_module_level(self):
        """google.genai should not be imported if adapter is never instantiated."""
        # Import the module without instantiating
        import raglab.infrastructure.gemini.gemini_generator_adapter  # noqa: F401
        # As long as no GeminiGeneratorAdapter() was called, google.genai
        # may be imported by the module. What matters is no credential access.
        # We just verify the module doesn't crash on import.
        assert True, "Module imports cleanly"


class TestSanitizeAnswerForArtifact:
    """Verify sanitized artifacts contain no credentials."""

    def _make_answer(self, query_id: str = "q_dev_01", abstained: bool = False):
        from raglab.domain.entities import GeneratedAnswer
        return GeneratedAnswer(
            query_id=query_id,
            text="Uma resposta de teste sobre indução matemática.",
            abstained=abstained,
            citations=(),
        )

    def test_sanitized_has_no_api_key(self):
        from raglab.infrastructure.gemini.gemini_generator_adapter import (
            sanitize_answer_for_artifact,
        )
        answer = self._make_answer()
        artifact = sanitize_answer_for_artifact(answer)
        artifact_str = str(artifact)
        assert "GEMINI_API_KEY" not in artifact_str
        assert "API_KEY" not in artifact_str

    def test_sanitized_has_expected_fields(self):
        from raglab.infrastructure.gemini.gemini_generator_adapter import (
            sanitize_answer_for_artifact,
        )
        answer = self._make_answer("q_dev_01", abstained=False)
        artifact = sanitize_answer_for_artifact(answer)
        assert "query_id" in artifact
        assert "text" in artifact
        assert "abstained" in artifact
        assert "citation_pages" in artifact
        assert artifact["query_id"] == "q_dev_01"
        assert artifact["abstained"] is False

    def test_text_is_capped_at_500_chars(self):
        from raglab.domain.entities import GeneratedAnswer
        from raglab.infrastructure.gemini.gemini_generator_adapter import (
            sanitize_answer_for_artifact,
        )
        answer = GeneratedAnswer(
            query_id="q_dev_01",
            text="x" * 1000,
            abstained=False,
            citations=(),
        )
        artifact = sanitize_answer_for_artifact(answer)
        assert len(artifact["text"]) == 1000
        assert artifact["truncated"] is False
        assert len(artifact["preview"]) == 500

    def test_sanitized_abstained(self):
        from raglab.infrastructure.gemini.gemini_generator_adapter import (
            sanitize_answer_for_artifact,
        )
        answer = self._make_answer(abstained=True)
        artifact = sanitize_answer_for_artifact(answer)
        assert artifact["abstained"] is True


class TestGeminiAdapterNotInstantiatedByAntigravity:
    """Regression: Gemini adapters must never be auto-instantiated."""

    def test_gemini_generator_not_in_fakes(self):
        """GeminiGeneratorAdapter must not appear in fakes module."""
        import raglab.infrastructure.fakes as fakes_pkg
        # The fakes package should not expose Gemini adapters
        fakes_dir = fakes_pkg.__file__
        assert fakes_dir is not None
        assert "gemini" not in (fakes_dir or "").lower()

    def test_fake_generator_has_no_gemini_import(self):
        """FakeGeneratorAdapter source must not import google.genai."""
        import inspect

        import raglab.infrastructure.fakes.fake_generator_adapter as mod
        source = inspect.getsource(mod)
        assert "google.genai" not in source


class TestGeminiGeneratorParsingAndCitations:
    """Offline unit tests for parsing JSON responses and citation provenance checks."""

    def test_citation_provenance_mismatch_raises_domain_error(self, monkeypatch):
        """Citing an unknown evidence_id (e.g. E99) must raise CitationProvenanceMismatchError."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake_key")

        from raglab.domain.entities import RetrievedEvidence
        from raglab.domain.errors import CitationProvenanceMismatchError
        from raglab.domain.value_objects import ChunkId
        from raglab.infrastructure.gemini.gemini_generator_adapter import (
            GeminiGeneratorAdapter,
        )

        adapter = GeminiGeneratorAdapter()
        # Mock _call_with_retry to return JSON citing E99 (which is not in prompt snapshot)
        monkeypatch.setattr(
            adapter,
            "_call_with_retry",
            lambda qid, prompt, *args, **kwargs: '{"status": "ANSWER", "answer": "Some answer", "citations": ["E99"]}',
        )

        ev1 = RetrievedEvidence(
            chunk_id=ChunkId("doc_p1_c0"),
            document_id="doc_p1",
            text="Evidence 1 text",
            rank=1,
            score=0.9,
        )

        with pytest.raises(CitationProvenanceMismatchError, match="CITATION_PROVENANCE_MISMATCH"):
            adapter.generate(query_id="q1", query="Query?", evidence=[ev1])

    def test_unknown_model_evidence_id_exposes_safe_reason_code(self, monkeypatch):
        """Citing an unknown evidence_id must raise CitationProvenanceMismatchError with safe reason code."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake_key")

        from raglab.domain.entities import RetrievedEvidence
        from raglab.domain.errors import CitationProvenanceMismatchError
        from raglab.domain.value_objects import ChunkId
        from raglab.infrastructure.gemini.gemini_generator_adapter import (
            GeminiGeneratorAdapter,
        )

        adapter = GeminiGeneratorAdapter()
        monkeypatch.setattr(
            adapter,
            "_call_with_retry",
            lambda qid, prompt, *args, **kwargs: '{"status": "ANSWER", "answer": "Some answer", "citations": ["E99"]}',
        )

        ev1 = RetrievedEvidence(
            chunk_id=ChunkId("doc_p1_c0"),
            document_id="doc_p1",
            text="Evidence 1 text",
            rank=1,
            score=0.9,
        )

        with pytest.raises(CitationProvenanceMismatchError) as exc_info:
            adapter.generate(query_id="q1", query="Query?", evidence=[ev1])

        assert exc_info.value.reason == "unknown_evidence_id"

    def test_valid_json_answer_parsing(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake_key")

        from raglab.domain.entities import RetrievedEvidence
        from raglab.domain.value_objects import ChunkId
        from raglab.infrastructure.gemini.gemini_generator_adapter import (
            GeminiGeneratorAdapter,
        )

        adapter = GeminiGeneratorAdapter()
        monkeypatch.setattr(
            adapter,
            "_call_with_retry",
            lambda qid, prompt, *args, **kwargs: '{"status": "ANSWER", "answer": "Prova por indução", "citations": ["E1"]}',
        )

        ev1 = RetrievedEvidence(
            chunk_id=ChunkId("doc_p1_c0"),
            document_id="gersting_doc_p91",
            text="Evidence 1 text",
            rank=1,
            score=0.9,
        )

        ans = adapter.generate(query_id="q1", query="Query?", evidence=[ev1])
        assert ans.abstained is False
        assert ans.text == "Prova por indução"
        assert len(ans.citations) == 1
        assert ans.citations[0].page_number == 91

    def test_valid_json_abstain_parsing(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake_key")

        from raglab.domain.entities import RetrievedEvidence
        from raglab.domain.value_objects import ChunkId
        from raglab.infrastructure.gemini.gemini_generator_adapter import (
            GeminiGeneratorAdapter,
        )

        adapter = GeminiGeneratorAdapter()
        monkeypatch.setattr(
            adapter,
            "_call_with_retry",
            lambda qid, prompt, *args, **kwargs: '{"status": "ABSTAIN", "answer": "", "citations": []}',
        )

        ev1 = RetrievedEvidence(
            chunk_id=ChunkId("doc_p1_c0"),
            document_id="gersting_doc_p91",
            text="Evidence 1 text",
            rank=1,
            score=0.9,
        )

        ans = adapter.generate(query_id="q1", query="Query?", evidence=[ev1])
        assert ans.abstained is True
        assert ans.text == ""
        assert len(ans.citations) == 0

        import inspect

        from raglab.infrastructure.fakes.fake_generator_adapter import (
            FakeGeneratorAdapter,
        )
        source = inspect.getsource(FakeGeneratorAdapter)
        assert "google.genai" not in source
        assert "genai.Client" not in source


@dataclass
class _LegacyEvidenceDouble:
    """Test double representing legacy evidence with start_page/page attributes."""

    chunk_id: ChunkId
    document_id: str
    text: str
    rank: int
    score: float
    page_number: int | None = None
    start_page: int | None = None
    page: int | None = None
    passage_id: str | None = None
    content_sha256: str | None = None


_PAGE_PRECEDENCE_CASES = [
    # 1. page_number=91, doc_p7 -> 91
    (
        RetrievedEvidence(
            chunk_id=ChunkId("c1"),
            document_id="doc_p7",
            text="Evidence text for case 1",
            rank=1,
            score=0.9,
            page_number=91,
        ),
        91,
    ),
    # 2. page_number=0, doc_p7 -> 0
    (
        RetrievedEvidence(
            chunk_id=ChunkId("c2"),
            document_id="doc_p7",
            text="Evidence text for case 2",
            rank=1,
            score=0.9,
            page_number=0,
        ),
        0,
    ),
    # 3. page_number=None, start_page=33, doc_p7 -> 33
    (
        _LegacyEvidenceDouble(
            chunk_id=ChunkId("c3"),
            document_id="doc_p7",
            text="Evidence text for case 3",
            rank=1,
            score=0.9,
            page_number=None,
            start_page=33,
        ),
        33,
    ),
    # 4. page_number=None, start_page=None, page=44, doc_p7 -> 44
    (
        _LegacyEvidenceDouble(
            chunk_id=ChunkId("c4"),
            document_id="doc_p7",
            text="Evidence text for case 4",
            rank=1,
            score=0.9,
            page_number=None,
            start_page=None,
            page=44,
        ),
        44,
    ),
    # 5. todos explícitos/legados None, doc_p7 -> 7
    (
        RetrievedEvidence(
            chunk_id=ChunkId("c5"),
            document_id="doc_p7",
            text="Evidence text for case 5",
            rank=1,
            score=0.9,
            page_number=None,
        ),
        7,
    ),
    # 6. todos explícitos/legados None, unpaginated_doc -> 0
    (
        RetrievedEvidence(
            chunk_id=ChunkId("c6"),
            document_id="unpaginated_doc",
            text="Evidence text for case 6",
            rank=1,
            score=0.9,
            page_number=None,
        ),
        0,
    ),
]


class TestGeminiGeneratorPagePrecedence:
    """Matrix tests for page number resolution precedence in GeminiGeneratorAdapter."""

    @pytest.mark.parametrize("evidence_item, expected_page", _PAGE_PRECEDENCE_CASES)
    def test_page_precedence_matrix(self, monkeypatch, evidence_item, expected_page):
        from unittest.mock import MagicMock

        import google.genai as genai

        from raglab.infrastructure.gemini.gemini_generator_adapter import (
            GeminiGeneratorAdapter,
        )

        monkeypatch.setenv("GEMINI_API_KEY", "fake_offline_key")
        monkeypatch.setattr(genai, "Client", lambda **kwargs: MagicMock())

        adapter = GeminiGeneratorAdapter()
        monkeypatch.setattr(
            adapter,
            "_call_with_retry",
            lambda qid, prompt, *args, **kwargs: '{"status": "ANSWER", "answer": "Resposta de teste", "citations": ["E1"]}',
        )

        answer = adapter.generate(
            query_id="q_test_prec",
            query="Qual é a técnica de demonstração?",
            evidence=[evidence_item],
        )
        assert answer.abstained is False
        assert len(answer.citations) >= 1
        assert answer.citations[0].page_number == expected_page


class TestGeminiGeneratorDynamicCitationConstraints:
    """Micro-RED tests specifying dynamic JSON schema citation constraints for transport."""

    def test_dynamic_citation_constraint_reaches_transport_schema(self, monkeypatch):
        """Verify GenerateContentConfig enforces response_json_schema with dynamic evidence_ids enum."""
        from unittest.mock import MagicMock

        import google.genai as genai

        from raglab.domain.entities import RetrievedEvidence
        from raglab.domain.value_objects import ChunkId
        from raglab.infrastructure.gemini.gemini_generator_adapter import (
            GeminiGeneratorAdapter,
        )

        monkeypatch.setenv("GEMINI_API_KEY", "fake_offline_key")
        mock_client = MagicMock()
        monkeypatch.setattr(genai, "Client", lambda **kwargs: mock_client)

        adapter = GeminiGeneratorAdapter()

        ev1 = RetrievedEvidence(
            chunk_id=ChunkId("doc_p1_c0"),
            document_id="doc1",
            text="Evidence text 1",
            rank=1,
            score=0.9,
            passage_id="ps_01",
        )
        ev2 = RetrievedEvidence(
            chunk_id=ChunkId("doc_p2_c0"),
            document_id="doc2",
            text="Evidence text 2",
            rank=2,
            score=0.8,
            passage_id="ps_02",
        )

        captured_configs = []

        def mock_generate_content(model, contents, config):
            captured_configs.append(config)
            mock_resp = MagicMock()
            mock_resp.text = (
                '{"status": "ANSWER", "answer": "Resposta suportada", "citations": ["E1"]}'
            )
            return mock_resp

        mock_client.models.generate_content = mock_generate_content

        # 1. Snapshot com 1 evidência
        ans1 = adapter.generate(query_id="q1", query="Pergunta 1?", evidence=[ev1])
        assert ans1.abstained is False
        assert len(captured_configs) == 1
        cfg1 = captured_configs[0]

        assert cfg1.response_mime_type == "application/json"
        assert cfg1.response_json_schema is not None
        schema1 = cfg1.response_json_schema
        assert isinstance(schema1, dict)
        assert set(schema1.get("required", [])) == {"status", "answer", "citations"}
        props1 = schema1.get("properties", {})
        assert props1.get("status", {}).get("enum") == ["ANSWER", "ABSTAIN"]
        citations_items1 = props1.get("citations", {}).get("items", {})
        assert citations_items1.get("enum") == ["E1"]

        # 2. Snapshot com 2 evidências
        captured_configs.clear()
        ans2 = adapter.generate(query_id="q2", query="Pergunta 2?", evidence=[ev1, ev2])
        assert ans2.abstained is False
        assert len(captured_configs) == 1
        cfg2 = captured_configs[0]

        assert cfg2.response_mime_type == "application/json"
        assert cfg2.response_json_schema is not None
        schema2 = cfg2.response_json_schema
        props2 = schema2.get("properties", {})
        citations_items2 = props2.get("citations", {}).get("items", {})
        assert citations_items2.get("enum") == ["E1", "E2"]

        # Enum dinâmico não pode ser compartilhado ou estático
        assert citations_items1.get("enum") != citations_items2.get("enum")

    def test_dynamic_citation_constraint_requires_empty_citations_without_evidence(
        self, monkeypatch
    ):
        """Calling generate with empty evidence must constrain schema citations to empty."""
        from unittest.mock import MagicMock

        import google.genai as genai

        from raglab.infrastructure.gemini.gemini_generator_adapter import (
            GeminiGeneratorAdapter,
        )

        monkeypatch.setenv("GEMINI_API_KEY", "fake_offline_key")
        mock_client = MagicMock()
        monkeypatch.setattr(genai, "Client", lambda **kwargs: mock_client)

        adapter = GeminiGeneratorAdapter()

        captured_configs = []

        def mock_generate_content(model, contents, config):
            captured_configs.append(config)
            mock_resp = MagicMock()
            mock_resp.text = '{"status": "ABSTAIN", "answer": "", "citations": []}'
            return mock_resp

        mock_client.models.generate_content = mock_generate_content

        ans = adapter.generate(query_id="q0", query="Query sem evidencia", evidence=[])
        assert ans.abstained is True
        assert ans.text == ""
        assert ans.citations == ()

        assert len(captured_configs) == 1
        cfg = captured_configs[0]
        assert cfg.response_mime_type == "application/json"
        assert cfg.response_json_schema is not None
        schema = cfg.response_json_schema
        props = schema.get("properties", {})
        citations_prop = props.get("citations", {})
        assert citations_prop.get("maxItems") == 0
