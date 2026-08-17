"""Domain contracts for Slice 5A.3 Canonical Coverage & Retrieval Parity Gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ParityOutcomeCategory(StrEnum):
    """Scientific evaluation status categories."""

    SCIENTIFICALLY_EVALUABLE = "SCIENTIFICALLY_EVALUABLE"
    NOT_EVALUABLE_CANONICAL_COVERAGE = "NOT_EVALUABLE_CANONICAL_COVERAGE"
    NOT_EVALUABLE_AMBIGUOUS_PROJECTION = "NOT_EVALUABLE_AMBIGUOUS_PROJECTION"
    NOT_EVALUABLE_JUDGED_COVERAGE = "NOT_EVALUABLE_JUDGED_COVERAGE"
    NOT_EVALUABLE_REPEATABILITY = "NOT_EVALUABLE_REPEATABILITY"
    NOT_EVALUABLE_RETRIEVAL_NONDETERMINISM = "NOT_EVALUABLE_RETRIEVAL_NONDETERMINISM"
    NOT_EVALUABLE_CONFIGURATION_DIVERGENCE = "NOT_EVALUABLE_CONFIGURATION_DIVERGENCE"
    INVALID_INPUT = "INVALID_INPUT"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    OPERATIONAL_FAILURE = "OPERATIONAL_FAILURE"


class JudgmentStatus(StrEnum):
    """Human evaluation status for a retrieved passage item."""

    JUDGED = "JUDGED"
    UNJUDGED = "UNJUDGED"
    UNMAPPED = "UNMAPPED"


class MappingStatus(StrEnum):
    """Mapping status of a retrieved item against the canonical corpus."""

    EXACT_PASSAGE_ID = "EXACT_PASSAGE_ID"
    EXACT_CONTENT_SHA256 = "EXACT_CONTENT_SHA256"
    EXACT_OFFSETS = "EXACT_OFFSETS"
    EXACT_SUBSTRING = "EXACT_SUBSTRING"
    DIRECT_MATCH = "DIRECT_MATCH"
    AMBIGUOUS = "AMBIGUOUS"
    UNMAPPED = "UNMAPPED"


@dataclass(frozen=True, slots=True)
class CanonicalCorpusSnapshot:
    """Immutable snapshot of the canonical corpus for consistent indexing."""

    snapshot_id: str
    source_artifact_hashes: dict[str, str]
    passage_count: int
    ordered_canonical_passage_ids: list[str]
    page_bounds: tuple[int, int]
    corpus_sha256: str
    construction_algorithm: str = "passage_registry_snapshot_v1"
    construction_algorithm_version: str = "1.0.0"


@dataclass(frozen=True, slots=True)
class ArmConfiguration:
    """Configuration for a specific retrieval strategy arm."""

    arm_id: str
    strategy: str
    retrieval_config: dict[str, Any]
    retrieval_config_sha256: str
    embedding_config_sha256: str
    corpus_sha256: str
    top_k: int = 3


@dataclass(frozen=True, slots=True)
class CanonicalRetrievedItem:
    """Retrieved evidence item linked to canonical passage ID and human qrel.

    Strict separation:
    - technical_chunk_id: raw retriever occurrence ID (e.g. doc_p91_s0)
    - anchor_passage_id: authoritative ps_* passage ID from registry if mapped,
      else None
    """

    qid: str
    arm_id: str
    rank: int
    technical_chunk_id: str = ""
    technical_node_id: str | None = None
    anchor_passage_id: str | None = None
    supporting_passage_ids: list[str] = field(default_factory=list)
    source_offsets: tuple[int, int] | None = None
    projection_method: str = "EXACT_SUBSTRING"
    page_number: int = 0
    content_sha256: str = ""
    score: float = 0.0
    mapping_status: MappingStatus = MappingStatus.DIRECT_MATCH
    judgment_status: JudgmentStatus = JudgmentStatus.JUDGED
    human_grade: float | None = None
    passage_id: str | None = None

    def __post_init__(self) -> None:
        """Ensure anchor_passage_id and passage_id backwards compatibility."""
        if self.passage_id and not self.anchor_passage_id:
            object.__setattr__(self, "anchor_passage_id", self.passage_id)
        if self.anchor_passage_id:
            object.__setattr__(self, "passage_id", self.anchor_passage_id)
        elif self.passage_id is None:
            object.__setattr__(self, "passage_id", "")

    @property
    def canonical_passage_id(self) -> str | None:
        """Alias for anchor_passage_id."""
        return self.anchor_passage_id


@dataclass(frozen=True, slots=True)
class ArmRun:
    """Materialized run for a single QID and arm."""

    qid: str
    arm_id: str
    retrieved_items: list[CanonicalRetrievedItem]
    latency: float
    configuration_hashes: dict[str, str]
    corpus_hash: str
    deterministic_content_hash: str


@dataclass(frozen=True, slots=True)
class ParityComparison:
    """Comparison of retrieval results across two independent runs."""

    qid: str
    arm_id: str
    first_run_ids: list[str]
    second_run_ids: list[str]
    exact_topk_match: bool
    rank_match: bool
    configuration_match: bool
    corpus_match: bool


@dataclass(frozen=True, slots=True)
class ParityGateResult:
    """Final output object of the Slice 5A.3 Parity Gate evaluation."""

    protocol_hash: str
    input_hashes: dict[str, str]
    canonical_coverage: float
    judged_coverage: float
    unmapped_count: int
    unjudged_count: int
    synthetic_fallback_count: int
    repeatability_topk_identity: float
    repeatability_rank_match: bool
    same_run_arm_completeness: bool
    status: ParityOutcomeCategory
    failure_reasons: list[str] = field(default_factory=list)
    metrics_status: str = "NOT_APPLICABLE"
    metrics: dict[str, Any] | None = None
