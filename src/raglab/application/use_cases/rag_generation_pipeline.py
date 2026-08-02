"""Use case: RAG generation pipeline.

Orchestrates the full retrieval → context assembly → generation loop.

This use case is intentionally thin and port-driven:
  - It does NOT know which retriever is used
  - It does NOT know which generator is used
  - It depends only on ports (Protocols) — safe for dependency injection
  - It is fully testable with fakes (no Gemini required)

SECURITY:
  - Does not handle credentials
  - Does not log query text
  - Does not persist raw LLM responses (only sanitized artifacts)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from raglab.domain.entities import GeneratedAnswer, RetrievedEvidence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Output of the RAG generation pipeline for one query."""

    query_id: str
    query: str
    answer: GeneratedAnswer
    evidence: tuple[RetrievedEvidence, ...]
    retrieval_strategy: str
    generator_model_id: str


class RagGenerationPipeline:
    """Thin orchestrator: retrieval → context → generation.

    Designed for offline testing (with fakes) and live execution
    (with GeminiGeneratorAdapter) via the same interface.

    Usage:
        pipeline = RagGenerationPipeline(retriever=..., generator=...)
        result = pipeline.run(query_id="q_dev_01", query="...")
    """

    def __init__(
        self,
        retriever: object,  # RetrievalPort — structural typing
        generator: object,  # GenerationPort — structural typing
        retrieval_strategy: str = "unknown",
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._retrieval_strategy = retrieval_strategy

    def run(
        self,
        query_id: str,
        query: str,
    ) -> GenerationResult:
        """Execute full retrieval → generation for one query.

        Args:
            query_id: Unique identifier for this query (for checkpointing).
            query: The user's question text.

        Returns:
            GenerationResult with answer, evidence, provenance.
        """
        logger.info(
            "query_id=%s strategy=%s: starting generation",
            query_id, self._retrieval_strategy,
        )

        # Retrieve evidence
        evidence: Sequence[RetrievedEvidence] = self._retriever.retrieve(query)  # type: ignore[attr-defined]

        logger.info(
            "query_id=%s: retrieved %d evidence passages",
            query_id, len(evidence),
        )

        # Generate answer
        answer: GeneratedAnswer = self._generator.generate(  # type: ignore[attr-defined]
            query_id=query_id,
            query=query,
            evidence=evidence,
        )

        logger.info(
            "query_id=%s: generation complete (abstained=%s)",
            query_id, answer.abstained,
        )

        return GenerationResult(
            query_id=query_id,
            query=query,
            answer=answer,
            evidence=tuple(evidence),
            retrieval_strategy=self._retrieval_strategy,
            generator_model_id=self._generator.model_id,  # type: ignore[attr-defined]
        )
