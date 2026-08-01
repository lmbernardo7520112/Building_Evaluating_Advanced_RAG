"""Tests for GeminiJudgeAdapter (offline — no credentials required).

Verifies:
1. Adapter raises RuntimeError if GEMINI_API_KEY is absent
2. Score JSON parser handles valid JSON, markdown-fenced JSON, and fallback
3. Sanitized evaluation artifacts contain no credentials
4. Judge model is independently configurable from generator
"""

from __future__ import annotations

import pytest


class TestGeminiJudgeNoCredential:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from raglab.infrastructure.gemini.gemini_judge_adapter import GeminiJudgeAdapter
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            GeminiJudgeAdapter()

    def test_adapter_module_has_security_docstring(self):
        import raglab.infrastructure.gemini.gemini_judge_adapter as mod
        assert "SECURITY BOUNDARY" in (mod.__doc__ or "")

    def test_judge_model_independently_configurable(self, monkeypatch):
        """Judge model_id must not be hardcoded to generator model_id."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        # Just verify the class accepts a different model_id parameter
        import inspect

        from raglab.infrastructure.gemini.gemini_judge_adapter import GeminiJudgeAdapter
        sig = inspect.signature(GeminiJudgeAdapter.__init__)
        assert "judge_model_id" in sig.parameters


class TestScoreJsonParser:
    """Test _parse_score_json in isolation."""

    def _parse(self, raw: str) -> tuple[float, str]:
        from raglab.infrastructure.gemini.gemini_judge_adapter import _parse_score_json
        return _parse_score_json(raw, "test_dim", "q_dev_01")

    def test_valid_json(self):
        score, reasoning = self._parse('{"reasoning": "good", "score": 0.85}')
        assert score == pytest.approx(0.85)
        assert "good" in reasoning

    def test_markdown_fenced_json(self):
        score, _ = self._parse('```json\n{"score": 0.7, "reasoning": "ok"}\n```')
        assert score == pytest.approx(0.7)

    def test_score_clamped_to_0_1(self):
        score, _ = self._parse('{"score": 1.5, "reasoning": "over"}')
        assert score == pytest.approx(1.0)
        score2, _ = self._parse('{"score": -0.5, "reasoning": "negative"}')
        assert score2 == pytest.approx(0.0)

    def test_fallback_regex(self):
        score, _ = self._parse('some text with "score": 0.6 in it')
        assert score == pytest.approx(0.6)

    def test_unparseable_returns_zero(self):
        score, _ = self._parse("totally invalid response without scores")
        assert score == pytest.approx(0.0)


class TestSanitizeEvaluationForArtifact:
    def _make_result(self):
        from raglab.domain.entities import EvaluationResult
        from raglab.domain.enums import PipelineStrategy
        from raglab.domain.value_objects import MetricResult
        return EvaluationResult(
            query_id="q_dev_01",
            strategy=PipelineStrategy.BASELINE,
            metrics=(
                MetricResult(name="context_relevance", value=0.8, normalized=True),
                MetricResult(name="groundedness", value=0.9, normalized=True),
                MetricResult(name="answer_relevance", value=0.75, normalized=True),
            ),
        )

    def test_sanitized_has_expected_fields(self):
        from raglab.infrastructure.gemini.gemini_judge_adapter import (
            sanitize_evaluation_for_artifact,
        )
        result = self._make_result()
        artifact = sanitize_evaluation_for_artifact(result)
        assert "query_id" in artifact
        assert "strategy" in artifact
        assert "metrics" in artifact
        metrics = artifact["metrics"]
        assert "context_relevance" in metrics
        assert "groundedness" in metrics
        assert "answer_relevance" in metrics

    def test_sanitized_no_credentials(self):
        from raglab.infrastructure.gemini.gemini_judge_adapter import (
            sanitize_evaluation_for_artifact,
        )
        result = self._make_result()
        artifact_str = str(sanitize_evaluation_for_artifact(result))
        assert "GEMINI_API_KEY" not in artifact_str
        assert "API_KEY" not in artifact_str

    def test_metric_values_preserved(self):
        from raglab.infrastructure.gemini.gemini_judge_adapter import (
            sanitize_evaluation_for_artifact,
        )
        result = self._make_result()
        artifact = sanitize_evaluation_for_artifact(result)
        assert artifact["metrics"]["context_relevance"] == pytest.approx(0.8)
        assert artifact["metrics"]["groundedness"] == pytest.approx(0.9)
