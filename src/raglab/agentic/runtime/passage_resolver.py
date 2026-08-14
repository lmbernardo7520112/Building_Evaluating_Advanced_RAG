"""Verified passage resolver runtime — resolves EvidenceItem to RetrievedEvidence."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from raglab.agentic.contracts import EvidenceItem, is_canonical_passage_id
from raglab.agentic.errors import (
    NonCanonicalIdError,
    PassageIntegrityError,
    PassageNotFoundError,
)
from raglab.domain.entities import RetrievedEvidence
from raglab.domain.value_objects import ChunkId


@dataclass(frozen=True, slots=True)
class PassagePayload:
    """Immutable payload returned by PassageLookupPort.

    Carries the real runtime ChunkId and physical text content.
    Derivation of ChunkId from passage_id is strictly forbidden.
    """

    passage_id: str
    chunk_id: ChunkId
    document_id: str
    text: str
    content_sha256: str


class PassageLookupPort(Protocol):
    """Pure runtime port for batch looking up passage payloads.

    Invariants:
    - Receives only canonical passage_ids (no filesystem paths).
    - Zero knowledge of evaluation, qrels, benchmarks or ground truth.
    - Returns payloads containing the real ChunkId and text.
    """

    def lookup_passages(
        self,
        passage_ids: Sequence[str],
    ) -> Sequence[PassagePayload]:
        """Fetch passage payloads for the requested canonical passage IDs."""
        ...


class VerifiedPassageResolver:
    """Verifies and resolves EvidenceItem instances into domain RetrievedEvidence.

    Coordinates with PassageLookupPort, enforcing cryptographic invariants.
    Fails closed before any GenerationPort is reached.
    """

    def __init__(self, lookup_port: PassageLookupPort) -> None:
        self._lookup_port = lookup_port

    def resolve(
        self,
        evidence_items: Sequence[EvidenceItem],
    ) -> tuple[RetrievedEvidence, ...]:
        """Batch-resolve EvidenceItem instances into RetrievedEvidence entities.

        Invariants enforced:
        1. Empty input returns () without calling lookup.
        2. Input passage_ids are canonical (ps_*) and unique.
        3. Lookup returns exactly the requested set of passage_ids.
        4. Output reconstructs the exact input cardinality and order.
        5. document_id and passage_id match between EvidenceItem and payload.
        6. Text is non-empty.
        7. SHA256(text) == EvidenceItem.content_sha256 == payload.content_sha256.
        8. rank and score originate from EvidenceItem.
        9. chunk_id (real domain ChunkId) and text originate from payload.
        """
        if not evidence_items:
            return ()

        # Validate input canonicality and uniqueness
        seen_input_ids: set[str] = set()
        requested_ids: list[str] = []

        for item in evidence_items:
            if not is_canonical_passage_id(item.passage_id):
                raise NonCanonicalIdError("passage_id", item.passage_id)

            if item.passage_id in seen_input_ids:
                raise PassageIntegrityError(
                    f"Duplicate input passage_id detected: '{item.passage_id}'"
                )
            seen_input_ids.add(item.passage_id)
            requested_ids.append(item.passage_id)

        # Batch lookup
        raw_payloads = self._lookup_port.lookup_passages(requested_ids)

        # Validate payloads
        payload_map: dict[str, PassagePayload] = {}
        for payload in raw_payloads:
            if not is_canonical_passage_id(payload.passage_id):
                raise NonCanonicalIdError("passage_id", payload.passage_id)

            if payload.passage_id in payload_map:
                raise PassageIntegrityError(
                    f"Duplicate lookup payload returned for '{payload.passage_id}'"
                )

            if payload.passage_id not in seen_input_ids:
                raise PassageIntegrityError(
                    f"Unexpected lookup payload returned for '{payload.passage_id}'"
                )

            payload_map[payload.passage_id] = payload

        # Check for missing IDs
        for req_id in requested_ids:
            if req_id not in payload_map:
                raise PassageNotFoundError(req_id)

        # Reconstruct in exact input order and validate invariants
        resolved: list[RetrievedEvidence] = []
        for item in evidence_items:
            payload = payload_map[item.passage_id]

            if payload.document_id != item.document_id:
                raise PassageIntegrityError(
                    f"Document ID mismatch for passage '{item.passage_id}': "
                    f"payload '{payload.document_id}' != item '{item.document_id}'"
                )

            if not payload.text or not payload.text.strip():
                raise PassageIntegrityError(
                    f"Empty text for passage '{item.passage_id}'"
                )

            computed_sha = hashlib.sha256(payload.text.encode("utf-8")).hexdigest()

            if computed_sha != item.content_sha256:
                raise PassageIntegrityError(
                    f"SHA-256 text mismatch for passage '{item.passage_id}': "
                    f"computed '{computed_sha}' != item '{item.content_sha256}'"
                )

            if payload.content_sha256 != item.content_sha256:
                raise PassageIntegrityError(
                    f"Payload hash mismatch for passage '{item.passage_id}': "
                    f"'{payload.content_sha256}' != '{item.content_sha256}'"
                )

            evidence = RetrievedEvidence(
                chunk_id=payload.chunk_id,
                document_id=item.document_id,
                text=payload.text,
                rank=item.rank,
                score=item.score,
                passage_id=item.passage_id,
                content_sha256=item.content_sha256,
            )
            resolved.append(evidence)

        return tuple(resolved)
