"""Port: retrieval from vector store."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from raglab.domain.entities import RetrievedEvidence


class RetrievalPort(Protocol):
    """Retrieve evidence from indexed corpus."""

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> Sequence[RetrievedEvidence]:
        """Retrieve top-k evidence for a query."""
        ...
