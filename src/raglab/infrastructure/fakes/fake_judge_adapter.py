"""Fake judge adapter — RAG Triad + Factual Correctness (offline, no credentials).

Implements the domain's EvaluationResult contract WITHOUT any network access.

SECURITY CONTRACT:
- Never imports google.generativeai or any Gemini SDK
- Never reads any credential variable
- Never makes network calls
- Never logs credential values
- LangSmith tracing: DISABLED

The real GeminiJudgeAdapter (Slice 4) exists in:
  src/raglab/infrastructure/gemini/gemini_judge_adapter.py

It MUST NOT be instantiated by Antigravity — only by the human operator
in an isolated terminal with GEMINI_API_KEY exported.
"""

from __future__ import annotations

from collections.abc import Sequence

from raglab.domain.entities import (
    EvaluationResult,
    GeneratedAnswer,
    RetrievedEvidence,
)
from raglab.domain.enums import PipelineStrategy
from raglab.domain.value_objects import MetricResult

_FAKE_JUDGE_ID = "fake-judge-v1-no-network"
_LANGSMITH_STATUS = "DISABLED"


class FakeJudgeAdapter:
    """Deterministic, network-free judge for offline testing.

    Conforms to EvaluationPort (structural typing via Protocol).

    RAG Triad scoring (deterministic, illustrative only):
    - context_relevance:   query words found in evidence (shallow)
    - groundedness:        answer tokens found in evidence
    - answer_relevance:    answer directly references key query terms
    - factual_correctness: rough token overlap with gold_answer (if provided)

    All scores are FAKE PLACEHOLDERS — NOT real LLM evaluations.
    LangSmith: permanently disabled in this adapter.
    """

    def __init__(self, strategy: PipelineStrategy | str = PipelineStrategy.BASELINE) -> None:
        if isinstance(strategy, str):
            self._strategy = PipelineStrategy.from_label(strategy)
        else:
            self._strategy = strategy

    @property
    def judge_model_id(self) -> str:
        return _FAKE_JUDGE_ID

    @property
    def strategy(self) -> PipelineStrategy:
        return getattr(self, "_strategy", PipelineStrategy.BASELINE)

    def evaluate_context_relevance(
        self, query_id: str, query: str, evidence: Sequence[RetrievedEvidence]
    ) -> float:
        query_tokens = {t for t in query.lower().split() if len(t) > 3}
        all_evidence_text = " ".join(ev.text.lower() for ev in evidence)
        evidence_tokens = {t for t in all_evidence_text.split() if len(t) > 3}
        if query_tokens:
            cr_hits = sum(1 for t in query_tokens if t in evidence_tokens)
            return round(min(1.0, cr_hits / len(query_tokens)), 4)
        return 0.0

    def evaluate_groundedness(
        self, query_id: str, query: str, answer: GeneratedAnswer, evidence: Sequence[RetrievedEvidence]
    ) -> float:
        answer_tokens = {t for t in answer.text.lower().split() if len(t) > 3}
        all_evidence_text = " ".join(ev.text.lower() for ev in evidence)
        evidence_tokens = {t for t in all_evidence_text.split() if len(t) > 3}
        if answer_tokens:
            g_hits = sum(1 for t in answer_tokens if t in evidence_tokens)
            return round(min(1.0, g_hits / len(answer_tokens)), 4)
        return 0.0

    def evaluate_answer_relevance(
        self, query_id: str, query: str, answer: GeneratedAnswer
    ) -> float:
        query_tokens = {t for t in query.lower().split() if len(t) > 3}
        answer_lower = answer.text.lower()
        if query_tokens:
            ar_hits = sum(1 for t in query_tokens if t in answer_lower)
            return round(min(1.0, ar_hits / len(query_tokens)), 4)
        return 0.0

    def evaluate(
        self,
        query_id: str,
        query: str,
        answer: GeneratedAnswer,
        evidence: Sequence[RetrievedEvidence],
        *,
        gold_answer: str | None = None,
    ) -> EvaluationResult:
        """Return deterministic fake RAG Triad evaluation (no network call)."""
        strategy = _infer_strategy_from_query_id(query_id)
        answer_lower = answer.text.lower()
        query_tokens = {t for t in query.lower().split() if len(t) > 3}
        all_evidence_text = " ".join(ev.text.lower() for ev in evidence)
        evidence_tokens = {t for t in all_evidence_text.split() if len(t) > 3}

        # Context Relevance: fraction of query tokens found in evidence
        if query_tokens:
            cr_hits = sum(1 for t in query_tokens if t in evidence_tokens)
            context_relevance = min(1.0, cr_hits / len(query_tokens))
        else:
            context_relevance = 0.0

        # Groundedness: fraction of answer tokens found in evidence
        answer_tokens = {t for t in answer_lower.split() if len(t) > 3}
        if answer_tokens:
            g_hits = sum(1 for t in answer_tokens if t in evidence_tokens)
            groundedness = min(1.0, g_hits / len(answer_tokens))
        else:
            groundedness = 0.0

        # Answer Relevance: fraction of query tokens found in answer
        if query_tokens:
            ar_hits = sum(1 for t in query_tokens if t in answer_lower)
            answer_relevance = min(1.0, ar_hits / len(query_tokens))
        else:
            answer_relevance = 0.0

        metrics: list[MetricResult] = [
            MetricResult(
                name="context_relevance",
                value=round(context_relevance, 4),
                normalized=True,
            ),
            MetricResult(
                name="groundedness",
                value=round(groundedness, 4),
                normalized=True,
            ),
            MetricResult(
                name="answer_relevance",
                value=round(answer_relevance, 4),
                normalized=True,
            ),
            MetricResult(name="fake_judge", value=1.0),
            MetricResult(name="langsmith_disabled", value=1.0),
            MetricResult(name="gemini_judge_planned", value=1.0),
        ]

        # Factual Correctness (only when gold_answer is provided)
        if gold_answer is not None:
            gold_tokens = {t for t in gold_answer.lower().split() if len(t) > 3}
            if gold_tokens:
                fc_hits = sum(1 for t in gold_tokens if t in answer_lower)
                factual_correctness = min(1.0, fc_hits / len(gold_tokens))
            else:
                factual_correctness = 0.0
            metrics.append(
                MetricResult(
                    name="factual_correctness",
                    value=round(factual_correctness, 4),
                    normalized=True,
                )
            )

        return EvaluationResult(
            query_id=query_id,
            strategy=strategy,
            metrics=tuple(metrics),
        )


def _infer_strategy_from_query_id(query_id: str) -> PipelineStrategy:
    """Infer pipeline strategy from query_id prefix for fake evaluation."""
    try:
        prefix = query_id.split("::")[0]
        return PipelineStrategy.from_label(prefix)
    except (ValueError, KeyError, IndexError):
        return PipelineStrategy.BASELINE
