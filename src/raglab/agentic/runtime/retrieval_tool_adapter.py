"""Retrieval tool adapter — bridge from RetrievalBackend to RetrievalPort.

Implements the agentic RetrievalBackend protocol by delegating to a
concrete RetrievalPort (the existing fixed-strategy retrieval pipelines).

Design:
- Does NOT import infrastructure modules directly.
- Receives a RetrievalPort at construction time (dependency injection).
- Converts Sequence[RetrievedEvidence] → ToolObservation.
- Preserves ps_* passage IDs, document IDs, ranks, scores.
- Computes content hashes and retrieval config hashes.
- Records latency.
- Reports typed failures.
- Optionally captures resolved passage payloads into PassageCapturePort.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from typing import Any, Protocol

from raglab.agentic.contracts import ToolObservation, _canonical_json, _sha256
from raglab.agentic.enums import InvocationStatus
from raglab.agentic.errors import NonCanonicalIdError
from raglab.agentic.runtime.passage_resolver import PassagePayload
from raglab.domain.entities import RetrievedEvidence


class PassageCapturePort(Protocol):
    """Pure runtime port for capturing passage payloads during retrieval."""

    def record_passages(
        self,
        payloads: Sequence[PassagePayload],
    ) -> None:
        """Atomically record a batch of PassagePayload instances."""
        ...


class RetrievalToolAdapter:
    """Bridge: agentic RetrievalBackend ← RetrievalPort.

    Each instance wraps ONE strategy's RetrievalPort.
    Read-only, no network, no document filtering.
    """

    def __init__(
        self,
        strategy: str,
        retrieval_port: Any,  # RetrievalPort protocol — duck-typed
        *,
        version: str = "1.0.0",
        passage_store: PassageCapturePort | None = None,
    ) -> None:
        self._strategy = strategy
        self._port = retrieval_port
        self._version = version
        self._passage_store = passage_store
        self._config_hash = _sha256(
            _canonical_json(
                {
                    "type": "retrieval_tool_adapter",
                    "strategy": strategy,
                    "version": version,
                }
            )
        )

    @property
    def strategy(self) -> str:
        return self._strategy

    @property
    def config_hash(self) -> str:
        return self._config_hash

    def retrieve(
        self,
        query: str,
        strategy: str,
        top_k: int,
    ) -> ToolObservation:
        """Execute retrieval via the wrapped RetrievalPort.

        Satisfies the RetrievalBackend protocol from tool_executor.py.
        """
        if strategy != self._strategy:
            raise ValueError(
                f"Strategy mismatch: adapter is for '{self._strategy}', "
                f"got '{strategy}'"
            )

        start_ns = time.monotonic_ns()
        failure_code: str | None = None
        evidence: Sequence[RetrievedEvidence] = ()

        try:
            evidence = self._port.retrieve(query=query, top_k=top_k)
        except Exception as exc:
            failure_code = f"{type(exc).__name__}: {exc}"
            evidence = ()

        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000

        # Convert to ToolObservation and PassagePayload
        passage_ids: list[str] = []
        document_ids: list[str] = []
        ranks: list[int] = []
        scores: list[float] = []
        content_hashes: list[str] = []
        captured_payloads: list[PassagePayload] = []

        for ev in evidence:
            # Use passage_id if available, else generate from chunk_id
            pid = ev.passage_id if ev.passage_id else f"ps_{ev.chunk_id.value}"

            # Validate canonical prefix
            if not pid.startswith("ps_"):
                raise NonCanonicalIdError(
                    "passage_id",
                    f"{pid} (must start with 'ps_')",
                )

            passage_ids.append(pid)
            document_ids.append(ev.document_id)
            ranks.append(ev.rank)
            scores.append(ev.score)

            # Content hash
            ch = (
                ev.content_sha256
                if ev.content_sha256
                else hashlib.sha256(ev.text.encode("utf-8")).hexdigest()
            )
            content_hashes.append(ch)

            captured_payloads.append(
                PassagePayload(
                    passage_id=pid,
                    chunk_id=ev.chunk_id,
                    document_id=ev.document_id,
                    text=ev.text,
                    content_sha256=ch,
                    page_number=ev.page_number,
                )
            )

        if self._passage_store is not None and captured_payloads:
            self._passage_store.record_passages(tuple(captured_payloads))

        # Build a stable invocation_id from the query+strategy+top_k
        inv_hash = _sha256(
            _canonical_json({"query": query, "strategy": strategy, "top_k": top_k})
        )[:16]
        invocation_id = f"adapter_{inv_hash}"

        return ToolObservation(
            invocation_id=invocation_id,
            status=InvocationStatus.EXECUTED
            if failure_code is None
            else InvocationStatus.FAILED,
            passage_ids=tuple(passage_ids),
            document_ids=tuple(document_ids),
            ranks=tuple(ranks),
            scores=tuple(scores),
            content_hashes=tuple(content_hashes),
            retrieval_config_hash=self._config_hash,
            latency_ms=elapsed_ms,
            failure_code=failure_code,
        )
