"""Port: answer generation (LLM)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from raglab.domain.entities import GeneratedAnswer, RetrievedEvidence


class GenerationPort(Protocol):
    """Generate answers from retrieved evidence.

    Generator and judge are logically separate — this port is
    exclusively for answer generation, not evaluation.

    The query_id is required for:
    - checkpoint idempotency;
    - per-question quota tracking;
    - sanitized artifact provenance.
    """

    def generate(
        self,
        query_id: str,
        query: str,
        evidence: Sequence[RetrievedEvidence],
    ) -> GeneratedAnswer:
        """Generate an answer with citations from evidence."""
        ...

    @property
    def model_id(self) -> str:
        """Return the generator model identifier for reproducibility."""
        ...
