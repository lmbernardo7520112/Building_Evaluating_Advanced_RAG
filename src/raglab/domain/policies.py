"""Domain policies — holdout protection and abstention."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from raglab.domain.enums import DatasetSplit
from raglab.domain.errors import HoldoutAccessViolationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HoldoutPolicy:
    """Policy that prevents accidental use of holdout data.

    The holdout set is reserved for confirmatory evaluation only.
    Any tuning after observing holdout invalidates its confirmatory character.

    This policy requires explicit authorization (a deliberate call with
    authorization_token set) to access holdout splits. It produces an
    auditable log event on every access attempt.

    This is NOT cryptographic security — it is a programmatic guardrail.
    """

    authorized_splits: frozenset[DatasetSplit]

    def check_access(self, split: DatasetSplit) -> None:
        """Verify that access to the given split is authorized.

        Raises HoldoutAccessViolationError if the split is a holdout
        and has not been explicitly authorized.
        """
        if split.is_holdout and split not in self.authorized_splits:
            logger.warning(
                "HOLDOUT_ACCESS_DENIED: attempted access to %s without authorization",
                split.value,
            )
            raise HoldoutAccessViolationError(split.value)

        if split.is_holdout:
            logger.info(
                "HOLDOUT_ACCESS_GRANTED: authorized access to %s (auditable event)",
                split.value,
            )

    @classmethod
    def development_only(cls) -> HoldoutPolicy:
        """Create a policy allowing only development and test access."""
        return cls(authorized_splits=frozenset())

    @classmethod
    def with_holdout_authorization(
        cls, *splits: DatasetSplit
    ) -> HoldoutPolicy:
        """Create a policy with explicit holdout authorization.

        Each authorized split produces an auditable log event on access.
        """
        holdout_splits = frozenset(s for s in splits if s.is_holdout)
        non_holdout = [s for s in splits if not s.is_holdout]
        if non_holdout:
            logger.info(
                "Non-holdout splits in authorization are redundant: %s",
                [s.value for s in non_holdout],
            )
        return cls(authorized_splits=holdout_splits)


@dataclass(frozen=True, slots=True)
class AbstentionPolicy:
    """Policy that determines when the system should abstain from answering.

    Threshold semantics: if all retrieval scores are below the threshold,
    the system should abstain rather than produce a low-confidence answer.
    """

    min_retrieval_score: float
    min_evidence_count: int

    def __post_init__(self) -> None:
        import math

        if not math.isfinite(self.min_retrieval_score):
            raise ValueError("min_retrieval_score must be finite")
        if self.min_evidence_count < 0:
            raise ValueError("min_evidence_count must be non-negative")

    def should_abstain(
        self, scores: list[float], evidence_count: int
    ) -> bool:
        """Determine if the system should abstain.

        Returns True if:
        - evidence_count is below threshold, OR
        - all scores are below the minimum threshold
        """
        if evidence_count < self.min_evidence_count:
            return True
        if not scores:
            return True
        return all(s < self.min_retrieval_score for s in scores)
