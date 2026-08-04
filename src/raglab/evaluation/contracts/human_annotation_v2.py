"""Contracts and Dataclasses for Human Annotation v2 Infrastructure (Gate B1).

Defines structured schemas for:
- PassageRegistryEntry & PassageRegistryManifest
- AnnotationCandidate
- EvidenceSetAnnotation
- HumanAnnotationRecord
- AdjudicationRecord
- AnnotationManifest
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EvidenceRole(StrEnum):
    """Semantic role of a passage as evidence."""

    PRIMARY = "PRIMARY"
    SUPPORTING = "SUPPORTING"
    CONTEXTUAL = "CONTEXTUAL"
    NEGATIVE_CONTROL = "NEGATIVE_CONTROL"


class AnnotationStatus(StrEnum):
    """Status of an annotation package or question item."""

    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class AdjudicationStatus(StrEnum):
    """Status of an adjudication record."""

    PENDING = "PENDING"
    ADJUDICATED = "ADJUDICATED"
    RESOLVED = "RESOLVED"


@dataclass(frozen=True)
class PassageRegistryEntry:
    """Canonical registry entry for a text passage in the corpus."""

    passage_id: str
    document_id: str
    page_number: int
    start_char: int
    end_char: int
    content_sha256: str
    text: str
    registry_version: str = "2.0.0"

    def __post_init__(self) -> None:
        if not self.passage_id or not self.passage_id.startswith("ps_"):
            raise ValueError(f"passage_id must start with 'ps_': {self.passage_id}")
        if self.page_number < 1:
            raise ValueError(f"page_number must be >= 1: {self.page_number}")
        if self.start_char < 0 or self.end_char <= self.start_char:
            raise ValueError(
                f"Invalid offsets: start_char={self.start_char}, "
                f"end_char={self.end_char}"
            )
        if not self.text or not self.text.strip():
            raise ValueError("text must be non-empty")


@dataclass(frozen=True)
class PassageRegistryManifest:
    """Manifest describing a passage registry build."""

    schema_version: str
    registry_version: str
    corpus_filename: str
    corpus_sha256: str
    page_range: tuple[int, int]
    extraction_adapter: str
    segmentation_policy: str
    segmentation_parameters: dict[str, Any]
    passage_count: int
    registry_sha256: str
    created_by: str = "deterministic_offline_builder"
    network_used: bool = False
    api_used: bool = False


@dataclass
class AnnotationCandidate:
    """Blinded candidate passage presented to a human annotator."""

    passage_id: str
    page_number: int
    text: str
    relevance_grade: int | None = None  # 0, 1, 2, 3
    evidence_role: EvidenceRole | str | None = None
    annotation_notes: str = ""

    def __post_init__(self) -> None:
        if (
            self.relevance_grade is not None
            and not isinstance(self.relevance_grade, int)
            or (
                self.relevance_grade is not None
                and not (0 <= self.relevance_grade <= 3)
            )
        ):
            raise ValueError(
                f"relevance_grade must be 0, 1, 2, or 3: {self.relevance_grade}"
            )


@dataclass
class EvidenceSetAnnotation:
    """A set of passages that are jointly sufficient to answer a question."""

    set_id: str
    passage_ids: tuple[str, ...]
    jointly_sufficient: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if self.jointly_sufficient and not self.passage_ids:
            raise ValueError("jointly_sufficient evidence set cannot be empty")


@dataclass
class HumanAnnotationRecord:
    """Full annotation record for a question by a single annotator."""

    question_id: str
    question_text: str
    annotator_id: str
    answerability: bool | None = None
    unanswerable_reason: str | None = None
    candidate_passages: list[AnnotationCandidate] = field(default_factory=list)
    evidence_sets: list[EvidenceSetAnnotation] = field(default_factory=list)
    gold_answer: str | None = None
    gold_supporting_passage_ids: list[str] = field(default_factory=list)
    annotation_status: AnnotationStatus = AnnotationStatus.PENDING


@dataclass
class AdjudicationRecord:
    """Adjudication item for resolving annotator disagreements."""

    question_id: str
    passage_id: str
    annotator_a_grade: int | None = None
    annotator_b_grade: int | None = None
    adjudicated_grade: int | None = None
    adjudication_reason: str = ""
    adjudicator_id: str = ""
    adjudication_status: AdjudicationStatus = AdjudicationStatus.PENDING


@dataclass(frozen=True)
class AnnotationManifest:
    """Manifest describing blinded annotation packages."""

    schema_version: str
    package_version: str
    annotators: tuple[str, ...]
    splits: tuple[str, ...]
    question_count: int
    passage_registry_sha256: str
    candidate_sources: tuple[str, ...]
    unavailable_sources: tuple[str, ...]
    created_by: str = "deterministic_offline_builder"
    network_used: bool = False
    api_used: bool = False
