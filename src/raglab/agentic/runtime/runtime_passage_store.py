"""Runtime volatile passage store — in-memory implementation of PassageLookupPort."""

from __future__ import annotations

from collections.abc import Sequence

from raglab.agentic.contracts import is_canonical_passage_id
from raglab.agentic.errors import NonCanonicalIdError, PassageIntegrityError
from raglab.agentic.runtime.passage_resolver import PassagePayload


class RuntimePassageStore:
    """In-memory volatile passage lookup store.

    Invariants:
    1. Validates canonical ps_* prefix for passage IDs.
    2. Batch recording is strictly atomic (all-or-nothing).
    3. Idempotent on identical PassagePayload instances.
    4. Fails with PassageIntegrityError on conflicting payload for existing ID.
    5. Returns payloads as tuple matching requested order, omitting absent IDs.
    6. Pure in-memory dictionary state per instance with no global/class state.
    """

    def __init__(self) -> None:
        self._payloads: dict[str, PassagePayload] = {}

    def record_passages(
        self,
        payloads: Sequence[PassagePayload],
    ) -> None:
        """Atomically record a batch of PassagePayload instances.

        Validates all payloads against canonicality and conflict invariants
        before updating internal state.
        """
        candidate = dict(self._payloads)

        for payload in payloads:
            if not is_canonical_passage_id(payload.passage_id):
                raise NonCanonicalIdError("passage_id", payload.passage_id)

            existing = candidate.get(payload.passage_id)
            if existing is not None and existing != payload:
                raise PassageIntegrityError(
                    f"Conflicting payload for passage_id '{payload.passage_id}'"
                )

            candidate[payload.passage_id] = payload

        self._payloads = candidate

    def lookup_passages(
        self,
        passage_ids: Sequence[str],
    ) -> tuple[PassagePayload, ...]:
        """Fetch passage payloads for requested IDs in exact requested order.

        Omits unknown passage IDs without error (delegating absence handling
        to VerifiedPassageResolver).
        """
        results: list[PassagePayload] = []
        for pid in passage_ids:
            payload = self._payloads.get(pid)
            if payload is not None:
                results.append(payload)
        return tuple(results)
