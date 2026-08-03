"""Domain value objects — immutable, self-validating, no infrastructure deps."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from raglab.domain.errors import (
    InvalidFingerprintError,
    InvalidIdentifierError,
    InvalidScoreError,
    NegativePositionError,
    NormalizedScoreOutOfRangeError,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ChunkId:
    """Unique identifier for a chunk within a corpus version."""

    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise InvalidIdentifierError("ChunkId")


@dataclass(frozen=True, slots=True)
class RunId:
    """Unique identifier for an experiment run."""

    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise InvalidIdentifierError("RunId")


@dataclass(frozen=True, slots=True)
class IntegrityDigest:
    """SHA-256 fingerprint for integrity verification.

    Must be a 64-character lowercase hex string.
    No cryptographic authentication or confidentiality is implied —
    this is an integrity envelope only.
    """

    hex_digest: str

    def __post_init__(self) -> None:
        if not _SHA256_PATTERN.match(self.hex_digest):
            raise InvalidFingerprintError()


@dataclass(frozen=True, slots=True)
class MetricResult:
    """A single metric measurement.

    Distinguishes absent metrics (value=None) from zero-valued metrics.
    When normalized=True, value must be in [0.0, 1.0].
    """

    name: str
    value: float | None
    normalized: bool = False
    k: int | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise InvalidIdentifierError("MetricResult.name")
        if self.value is not None:
            if not math.isfinite(self.value):
                raise InvalidScoreError(self.name, self.value)
            if self.normalized and not (0.0 <= self.value <= 1.0):
                raise NormalizedScoreOutOfRangeError(self.name)
        if self.k is not None and self.k < 1:
            raise NegativePositionError("MetricResult.k")

    @property
    def is_absent(self) -> bool:
        return self.value is None


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """A page from a source document with provenance."""

    document_id: str
    page_number: int
    text: str

    def __post_init__(self) -> None:
        if not self.document_id or not self.document_id.strip():
            raise InvalidIdentifierError("DocumentPage.document_id")
        if self.page_number < 0:
            raise NegativePositionError("page_number")


@dataclass(frozen=True, slots=True)
class Citation:
    """A citation linking an answer claim to source evidence."""

    document_id: str
    page_number: int
    chunk_id: ChunkId
    text_span: str
    evidence_id: str | None = None
    passage_id: str | None = None
    content_sha256: str | None = None
    retrieval_rank: int | None = None

    def __post_init__(self) -> None:
        if not self.document_id or not self.document_id.strip():
            raise InvalidIdentifierError("Citation.document_id")
        if self.page_number < 0:
            raise NegativePositionError("Citation.page_number")
