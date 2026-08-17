"""Public dispatching retrieval backend — routes retrieve() calls by strategy.

Implements the agentic RetrievalBackend protocol by dispatching
to a concrete RetrievalToolAdapter matching the requested strategy.
"""

from __future__ import annotations

from collections.abc import Mapping

from raglab.agentic.contracts import ToolObservation
from raglab.agentic.runtime.retrieval_tool_adapter import RetrievalToolAdapter


class DispatchingRetrievalBackend:
    """Public retrieval backend that dispatches calls to registered adapters."""

    def __init__(self, adapters: Mapping[str, RetrievalToolAdapter]) -> None:
        """Store a private copy of the strategy/tool_id -> adapter mapping."""
        self._adapters: dict[str, RetrievalToolAdapter] = dict(adapters)

    def retrieve(
        self,
        query: str,
        strategy: str,
        top_k: int,
    ) -> ToolObservation:
        """Dispatch retrieval to the adapter matching the requested strategy.

        Satisfies the agentic RetrievalBackend protocol from tool_executor.py.
        Forwards query, strategy, and top_k without alteration.
        Rejects unknown strategies fail-closed before invoking any adapter.
        """
        tool_id = f"retrieve_{strategy}"
        adapter = self._adapters.get(tool_id) or self._adapters.get(strategy)
        if adapter is None:
            raise ValueError(
                f"No adapter registered for strategy '{strategy}' or tool '{tool_id}'. "
                f"Available: {sorted(self._adapters.keys())}"
            )
        return adapter.retrieve(query=query, strategy=strategy, top_k=top_k)
