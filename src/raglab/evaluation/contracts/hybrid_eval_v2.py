"""Contracts and Schemas for Hybrid Human-Validated Evaluation (Gate B2).

Defines structured schemas for:
- CanonicalMappingStatus & CanonicalMappingResult
- CandidatePoolItem & CandidatePoolManifest
- SilverAnnotationRecord & SilverManifest
- HumanQueueItem & HumanRoutingManifest
- CalibrationReport
- HybridEvalManifest
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CanonicalMappingStatus(StrEnum):
    """Status of mapping arbitrary retrieved text/chunks to canonical passages."""

    EXACT_PASSAGE_ID = "EXACT_PASSAGE_ID"
    EXACT_OFFSETS = "EXACT_OFFSETS"
    EXACT_CONTENT_SHA256 = "EXACT_CONTENT_SHA256"
    EXACT_SUBSTRING = "EXACT_SUBSTRING"
    AMBIGUOUS_NEEDS_REVIEW = "AMBIGUOUS_NEEDS_REVIEW"
    UNMAPPED_NEEDS_REVIEW = "UNMAPPED_NEEDS_REVIEW"


class EvidenceAuthority(StrEnum):
    """Authority states for qrels and annotations."""

    HUMAN_GOLD = "HUMAN_GOLD"
    HUMAN_ADJUDICATED = "HUMAN_ADJUDICATED"
    HUMAN_VALIDATED = "HUMAN_VALIDATED"
    MACHINE_SILVER = "MACHINE_SILVER"
    UNJUDGED = "UNJUDGED"
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"


@dataclass(frozen=True)
class CanonicalMappingResult:
    """Result of mapping a retrieved chunk/passage to a canonical passage."""

    source_chunk_id: str
    document_id: str
    page_number: int
    mapped_passage_id: str | None
    mapping_status: CanonicalMappingStatus
    confidence: float
    notes: str = ""

    def __post_init__(self) -> None:
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"confidence must be in [0, 1]: {self.confidence}")
        if (
            self.mapping_status == CanonicalMappingStatus.UNMAPPED_NEEDS_REVIEW
            and self.mapped_passage_id is not None
        ):
            raise ValueError("UNMAPPED status cannot have mapped_passage_id")


@dataclass
class CandidatePoolItem:
    """An item in the multisystem candidate pool."""

    question_id: str
    passage_id: str
    page_number: int
    text: str
    source_provenance: list[dict[str, Any]] = field(default_factory=list)
    is_neighbor: bool = False
    neighbor_policy: str | None = None
    is_outside_pool_audit: bool = False

    def __post_init__(self) -> None:
        if not self.passage_id or not self.passage_id.startswith("ps_"):
            raise ValueError(f"passage_id must start with 'ps_': {self.passage_id}")


@dataclass
class SilverAnnotationRecord:
    """Record produced by automated machine silver triage judge."""

    question_id: str
    passage_id: str
    label_source: str = "MACHINE_SILVER"
    judge_id: str = "gemini_flash_judge"
    judge_provider: str = "google_genai"
    judge_model: str = "gemini-3.1-flash-lite"
    judge_model_version: str = "v1"
    judge_prompt_sha256: str = ""
    rubric_version: str = "2.0.0"
    order_seed: str = ""
    relevance_grade: int = 0
    evidence_role: str = "CONTEXTUAL"
    confidence: float = 0.0
    supporting_span: str | None = None
    reasoning: str = ""
    needs_human_review: bool = True
    created_at_utc: str = ""
    call_id: str = ""
    retry_count: int = 0

    def __post_init__(self) -> None:
        if self.label_source != "MACHINE_SILVER":
            raise ValueError(
                f"label_source must be 'MACHINE_SILVER': {self.label_source}"
            )
        if not (0 <= self.relevance_grade <= 3):
            raise ValueError(f"relevance_grade must be in 0-3: {self.relevance_grade}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1]: {self.confidence}")


@dataclass
class HumanQueueItem:
    """An item in a human review queue (blinded view)."""

    question_id: str
    passage_id: str
    page_number: int
    text: str
    annotator_id: str
    priority_rank: int = 1
    routing_reasons: list[str] = field(default_factory=list)
    is_overlap_sample: bool = False
    relevance_grade: int | None = None
    evidence_role: str | None = None
    annotation_notes: str = ""
    status: str = "PENDING"


@dataclass
class CalibrationReportData:
    """Data payload for silver vs human calibration."""

    confusion_matrix: list[list[int]]
    precision_by_class: dict[str, float | str]
    recall_by_class: dict[str, float | str]
    f1_by_class: dict[str, float | str]
    weighted_kappa: float | str
    false_negative_rate: float | str
    calibration_error: float | str
    human_silver_agreement: float | str
    sample_size: int
    status: str = "CALIBRATION_NOT_EXECUTED"


@dataclass(frozen=True)
class HybridEvalManifest:
    """Master manifest for the hybrid evaluation dataset."""

    schema_version: str = "2.0.0"
    protocol_version: str = "raglab_v7_slice4_v3"
    evidence_level: str = "E3_CONTROLLED_COMPARISON_IN_SLICE"
    policy_name: str = "HUMAN_VALIDATED_HYBRID_EVAL_SET"
    corpus_sha256: str = ""
    passage_registry_sha256: str = ""
    pool_size: int = 0
    mapped_exact_count: int = 0
    unmapped_count: int = 0
    outside_pool_audit_count: int = 0
    holdout_sealed: bool = True
    created_by: str = "deterministic_offline_builder"
