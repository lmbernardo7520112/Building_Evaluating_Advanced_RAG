"""Ground Truth v2 Contract — Architectural Isolation & Strict Provenance.

GROUND TRUTH IS EXCLUSIVELY FOR POST-GENERATION EVALUATION.
IT MUST NEVER ENTER THE RAG INFERENCE PIPELINE (QUERY -> RETRIEVER -> GENERATOR).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class UnanswerableReason(str, Enum):  # noqa: UP042
    ABSENT_FROM_CORPUS = "ABSENT_FROM_CORPUS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONTRADICTORY_EVIDENCE = "CONTRADICTORY_EVIDENCE"
    AMBIGUOUS_QUERY = "AMBIGUOUS_QUERY"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


ProvenanceStatus = Literal[
    "ADJUDICATED",
    "SINGLE_ANNOTATOR",
    "LEGACY_MIGRATED",
    "LEGACY_METADATA_UNAVAILABLE",
]


@dataclass(frozen=True, slots=True)
class CanonicalEvidence:
    """Canonical representation of ground-truth evidence passage."""

    passage_id: str
    document_id: str
    start_page: int
    text_span: str
    content_sha256: str
    relevance_grade: int | None = None  # None for unannotated/legacy binary qrels

    def __post_init__(self) -> None:
        if not self.passage_id or not self.passage_id.strip():
            raise ValueError("passage_id must be non-empty")
        if not self.document_id or not self.document_id.strip():
            raise ValueError("document_id must be non-empty")
        if self.start_page < 0:
            raise ValueError("start_page must be non-negative")
        if self.relevance_grade is not None and self.relevance_grade < 0:
            raise ValueError("relevance_grade must be non-negative if provided")


@dataclass(frozen=True, slots=True)
class GroundTruthItemV2:
    """Ground Truth v2 Item.

    Invariants:
    - query_id must be non-empty.
    - If answerable is False, unanswerable_reason must be provided.
    - If provenance_status == 'ADJUDICATED', requires >= 2 annotation records.
    """

    query_id: str
    query_text: str
    answerable: bool
    unanswerable_reason: UnanswerableReason | None
    gold_answer: str | None
    relevant_evidences: tuple[CanonicalEvidence, ...]
    provenance_status: ProvenanceStatus
    annotation_completeness: dict[str, Any] = field(default_factory=dict)
    annotation_records: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.query_id or not self.query_id.strip():
            raise ValueError("query_id must be non-empty")

        if not self.answerable and self.unanswerable_reason is None:
            raise ValueError(
                "unanswerable_reason is required when answerable is False"
            )

        if (
            self.provenance_status == "ADJUDICATED"
            and len(self.annotation_records) < 2
        ):
            raise ValueError(
                "ADJUDICATED provenance status requires at least 2 "
                "annotation records"
            )
