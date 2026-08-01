"""Port: evaluation (judge) — RAG Triad + Factual Correctness.

The judge evaluates three orthogonal dimensions:

  Context Relevance:  Is the retrieved context relevant to the query?
  Groundedness:       Is the answer grounded in the retrieved context?
  Answer Relevance:   Does the answer address the query?

Plus an optional fourth dimension:
  Factual Correctness: Does the answer match known gold facts? (requires
                       gold_answer; skipped when absent)

The query_id is required for:
- checkpoint idempotency;
- per-question quota tracking;
- sanitized artifact provenance.

Judge and generator MUST use independently configurable model IDs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from raglab.domain.entities import (
    EvaluationResult,
    GeneratedAnswer,
    RetrievedEvidence,
)


class EvaluationPort(Protocol):
    """Evaluate RAG pipeline outputs (RAG Triad + Factual Correctness).

    The judge is logically separate from the generator to prevent
    contamination between answer generation and evaluation.
    """

    def evaluate(
        self,
        query_id: str,
        query: str,
        answer: GeneratedAnswer,
        evidence: Sequence[RetrievedEvidence],
        *,
        gold_answer: str | None = None,
    ) -> EvaluationResult:
        """Evaluate an answer against evidence for a query.

        Args:
            query_id: Traceability identifier (mirrors GenerationPort).
            query: The original user query.
            answer: The generated answer to evaluate.
            evidence: Retrieved evidence used for generation.
            gold_answer: Optional gold answer for factual correctness.
                         When None, factual_correctness is skipped.

        Returns:
            EvaluationResult with at least:
              context_relevance, groundedness, answer_relevance metrics.
            If gold_answer is provided, also includes factual_correctness.
        """
        ...

    @property
    def judge_model_id(self) -> str:
        """Return the judge model identifier.

        Must be independently configurable from the generator.
        """
        ...
