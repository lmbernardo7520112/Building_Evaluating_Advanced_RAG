"""Domain entities for hierarchical retrieval — Auto-merging (Slice 3).

Pure domain layer — zero infrastructure dependencies.
These entities model the node hierarchy used by H0 (hierarchical leaf
retrieval) and H1 (auto-merging retrieval).

Hierarchy levels:
  LEAF   (level 0): 128–256 tokens — units actually indexed
  MIDDLE (level 1): ~512 tokens    — intermediate grouping
  PARENT (level 2): ~1024 tokens   — top-level context block

Auto-merging rule:
  When ≥ merge_threshold fraction of a parent's leaf children are
  retrieved, replace them all with the parent node in the context.

Observability (Section 8 spec):
  All merge decisions — realised and refused — are recorded in
  MergeEvent so they can be audited after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique


@unique
class HierarchyLevel(Enum):
    """Node level in the document hierarchy."""

    LEAF = 0
    MIDDLE = 1
    PARENT = 2


@dataclass(frozen=True, slots=True)
class HierarchicalNode:
    """A node in the document hierarchy with full provenance.

    Invariants:
    - node_id must be non-empty
    - document_id must be non-empty (provenance required)
    - level must be a valid HierarchyLevel
    - char_start <= char_end
    - children_ids is empty for leaf nodes
    - parent_id is None for root (PARENT) nodes
    """

    node_id: str
    document_id: str
    level: HierarchyLevel
    text: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    fingerprint: str          # SHA-256 of text content
    parent_id: str | None     # None for PARENT-level nodes
    children_ids: tuple[str, ...] = field(default_factory=tuple)
    token_count: int = 0

    def __post_init__(self) -> None:
        if not self.node_id or not self.node_id.strip():
            raise ValueError("node_id must be non-empty")
        if not self.document_id or not self.document_id.strip():
            raise ValueError("document_id must be non-empty")
        if self.char_start < 0:
            raise ValueError("char_start must be non-negative")
        if self.char_end < self.char_start:
            raise ValueError("char_end must be >= char_start")
        if self.page_start < 0 or self.page_end < 0:
            raise ValueError("page numbers must be non-negative")
        if self.page_end < self.page_start:
            raise ValueError("page_end must be >= page_start")
        if self.level == HierarchyLevel.LEAF and self.children_ids:
            raise ValueError("LEAF nodes must not have children_ids")


@dataclass(frozen=True, slots=True)
class HierarchyStats:
    """Summary statistics for a built hierarchy."""

    total_nodes: int
    leaf_count: int
    middle_count: int
    parent_count: int
    avg_leaf_tokens: float
    avg_middle_tokens: float
    avg_parent_tokens: float


@dataclass(frozen=True, slots=True)
class MergeDecision:
    """Record of a single merge decision during auto-merging retrieval.

    Captures whether a parent was promoted or refused, and the
    coverage ratio that triggered (or rejected) the merge.
    """

    parent_id: str
    children_retrieved: int
    children_total: int
    coverage_ratio: float
    threshold: float
    merged: bool                     # True = promotion happened
    tokens_before: int               # sum of children tokens
    tokens_after: int                # parent tokens (if merged) or same
    relevant_evidence_before: int    # relevant passages in children
    relevant_evidence_after: int     # relevant passages in promoted parent
    noise_introduced: bool           # True if merge added irrelevant context


@dataclass(frozen=True, slots=True)
class AutoMergingTrace:
    """Full observability trace for one query's auto-merging pass.

    Fields match the observability spec (Section 8):
      - leaves_retrieved
      - parent_candidates
      - merge_decisions (one per candidate parent)
      - tokens_before / tokens_after
      - relevant_evidence_before / after
      - merge_rate, parent_promotion_rate, context_expansion_ratio
      - irrelevant_context_ratio, duplicate_context_ratio
      - latency_ms
    """

    query_id: str
    leaves_retrieved: int
    parent_candidates: int
    merge_decisions: tuple[MergeDecision, ...]
    tokens_before: int
    tokens_after: int
    relevant_evidence_before: int
    relevant_evidence_after: int
    latency_ms: float

    @property
    def merges_performed(self) -> int:
        return sum(1 for d in self.merge_decisions if d.merged)

    @property
    def merges_refused(self) -> int:
        return sum(1 for d in self.merge_decisions if not d.merged)

    @property
    def merge_rate(self) -> float:
        """Fraction of candidate parents that were actually promoted."""
        if not self.parent_candidates:
            return 0.0
        return self.merges_performed / self.parent_candidates

    @property
    def parent_promotion_rate(self) -> float:
        """Fraction of retrieved leaves replaced by parent nodes."""
        if not self.leaves_retrieved:
            return 0.0
        promoted_leaves = sum(
            d.children_retrieved for d in self.merge_decisions if d.merged
        )
        return promoted_leaves / self.leaves_retrieved

    @property
    def context_expansion_ratio(self) -> float:
        """Ratio of context size after vs before merging."""
        if not self.tokens_before:
            return 1.0
        return self.tokens_after / self.tokens_before

    @property
    def relevant_evidence_preservation(self) -> float:
        """Fraction of pre-merge relevant evidence preserved post-merge."""
        if not self.relevant_evidence_before:
            return 1.0
        return min(1.0, self.relevant_evidence_after / self.relevant_evidence_before)

    @property
    def relevant_evidence_loss(self) -> float:
        """Fraction of pre-merge relevant evidence lost post-merge."""
        return 1.0 - self.relevant_evidence_preservation
