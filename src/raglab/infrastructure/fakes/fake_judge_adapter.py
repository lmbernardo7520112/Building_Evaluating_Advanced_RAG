"""Fake judge adapter for offline testing — no network, no credentials.

Implements the domain's EvaluationResult contract without any network access.

SECURITY CONTRACT:
- Never imports LangSmith, TruLens, or any remote evaluation SDK
- Never reads any credential variable
- Never makes network calls
- Never logs credential values
- LangSmith tracing: DISABLED

The real GeminiJudgeAdapter DOES NOT exist in this codebase yet.
It will be implemented in a future slice with explicit Gate authorization.
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

    Scoring logic (deterministic, illustrative only):
    - faithfulness: 0.5 if answer tokens overlap with evidence tokens
    - answer_relevance: overlap between query words and answer words
    - context_precision: 0.0 (no real LLM grounding check)

    All scores are placeholders — NOT real LLM evaluations.
    LangSmith: permanently disabled in this adapter.
    """

    @property
    def judge_model_id(self) -> str:
        return _FAKE_JUDGE_ID

    def evaluate(
        self,
        query_id: str,
        strategy: PipelineStrategy,
        answer: GeneratedAnswer,
        evidence: Sequence[RetrievedEvidence],
    ) -> EvaluationResult:
        """Return deterministic fake evaluation without any network call."""
        answer_lower = answer.text.lower()
        query_words = set(answer_lower.split()[:20])  # use answer as proxy

        # Faithfulness: fake overlap
        evidence_refs = sum(
            1
            for ev in evidence
            if any(w in answer_lower for w in ev.text.lower().split()[:10])
        )
        faithfulness = min(1.0, evidence_refs / max(1, len(evidence)))

        # Answer relevance: fraction of answer words found in evidence
        all_evidence_text = " ".join(ev.text.lower() for ev in evidence)
        relevance_hits = sum(
            1 for w in query_words if len(w) > 3 and w in all_evidence_text
        )
        answer_relevance = min(
            1.0, relevance_hits / max(1, len([w for w in query_words if len(w) > 3]))
        )

        metrics = (
            MetricResult(name="faithfulness", value=round(faithfulness, 4)),
            MetricResult(name="answer_relevance", value=round(answer_relevance, 4)),
            MetricResult(name="context_precision", value=0.0),  # placeholder
            MetricResult(name="fake_judge", value=1.0),  # marks as fake
            MetricResult(name="langsmith_disabled", value=1.0),
            MetricResult(name="gemini_planned", value=1.0),
        )

        return EvaluationResult(
            query_id=query_id,
            strategy=strategy,
            metrics=metrics,
        )
