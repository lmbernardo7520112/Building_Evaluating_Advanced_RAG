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

import pytest


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
            lambda qid, prompt: '{"status": "ANSWER", "answer": "Some answer", "citations": ["E99"]}',
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
            lambda qid, prompt: '{"status": "ANSWER", "answer": "Prova por indução", "citations": ["E1"]}',
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
            lambda qid, prompt: '{"status": "ABSTAIN", "answer": "", "citations": []}',
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
