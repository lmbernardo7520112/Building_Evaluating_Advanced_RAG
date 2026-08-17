"""Agentic generation bridge — coordinates passage resolution with LLM generation."""

from __future__ import annotations

from collections.abc import Sequence

from raglab.agentic.contracts import EvidenceItem
from raglab.agentic.runtime.passage_resolver import VerifiedPassageResolver
from raglab.application.ports.generation import GenerationPort
from raglab.domain.entities import GeneratedAnswer
from raglab.domain.errors import (
    CitationProvenanceMismatchError,
    InvalidIdentifierError,
)


class AgenticGenerationBridge:
    """Coordinates passage resolution and generation for the agentic track.

    Invariants:
    1. Validates query_id before invoking any dependency.
    2. Empty evidence_items returns immediate abstention without dependency calls.
    3. Resolves EvidenceItem to RetrievedEvidence preserving input order.
    4. Delegates generation to GenerationPort.
    5. Validates that GeneratedAnswer.query_id matches the input query_id.
    6. Validates that all citations strictly match the resolved evidence provenance.
    """

    def __init__(
        self,
        resolver: VerifiedPassageResolver,
        generator: GenerationPort,
    ) -> None:
        self._resolver = resolver
        self._generator = generator

    @property
    def model_id(self) -> str:
        """Return the underlying generator model identifier."""
        return self._generator.model_id

    def generate(
        self,
        query_id: str,
        query: str,
        evidence_items: Sequence[EvidenceItem],
    ) -> GeneratedAnswer:
        """Generate answer with verified passage resolution and citation provenance."""
        if not isinstance(query_id, str) or not query_id.strip():
            raise InvalidIdentifierError("query_id")

        if not evidence_items:
            return GeneratedAnswer(
                query_id=query_id,
                text="",
                abstained=True,
                citations=(),
            )

        resolved = self._resolver.resolve(evidence_items)
        answer = self._generator.generate(query_id, query, resolved)

        if answer.query_id != query_id:
            raise InvalidIdentifierError("GeneratedAnswer.query_id")

        evidence_by_passage = {
            ev.passage_id: ev for ev in resolved if ev.passage_id is not None
        }

        for citation in answer.citations:
            cite_id = citation.evidence_id or citation.passage_id or "unknown"

            if citation.passage_id is None:
                raise CitationProvenanceMismatchError(
                    cite_id,
                    reason="missing_passage_id",
                )

            if citation.passage_id not in evidence_by_passage:
                raise CitationProvenanceMismatchError(
                    cite_id,
                    reason="unknown_passage_id",
                )

            ev = evidence_by_passage[citation.passage_id]

            if citation.document_id != ev.document_id:
                raise CitationProvenanceMismatchError(
                    cite_id,
                    reason="document_id_mismatch",
                )

            if citation.chunk_id != ev.chunk_id:
                raise CitationProvenanceMismatchError(
                    cite_id,
                    reason="chunk_id_mismatch",
                )

            if citation.content_sha256 != ev.content_sha256:
                raise CitationProvenanceMismatchError(
                    cite_id,
                    reason="content_sha256_mismatch",
                )

            if citation.retrieval_rank != ev.rank:
                raise CitationProvenanceMismatchError(
                    cite_id,
                    reason="retrieval_rank_mismatch",
                )

            if ev.page_number is not None and citation.page_number != ev.page_number:
                raise CitationProvenanceMismatchError(
                    cite_id,
                    reason="page_number_mismatch",
                )

        return answer
