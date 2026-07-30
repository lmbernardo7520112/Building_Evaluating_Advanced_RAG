"""Port: evaluation (judge)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from raglab.domain.entities import (
    EvaluationResult,
    GeneratedAnswer,
    RetrievedEvidence,
)


class EvaluationPort(Protocol):
    """Evaluate RAG pipeline outputs.

    The judge is logically separate from the generator to prevent
    contamination between answer generation and evaluation.
    """

    def evaluate(
        self,
        query: str,
        answer: GeneratedAnswer,
        evidence: Sequence[RetrievedEvidence],
    ) -> EvaluationResult:
        """Evaluate an answer against evidence for a query."""
        ...

    @property
    def judge_model_id(self) -> str:
        """Return the judge model identifier.

        Must be independently configurable from the generator.
        """
        ...
