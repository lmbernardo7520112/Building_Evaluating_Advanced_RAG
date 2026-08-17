"""Strategy tool factory — maps PipelineStrategy to ToolSpec + adapter.

Connects each of the seven fixed strategies to the agentic tool registry
by producing a ToolSpecification and a RetrievalToolAdapter pair.

Does NOT:
- consult qrels, results, or holdout
- choose "the best" strategy
- alter retrieval behaviour
- use Settings global
- initialise external models at import time
"""

from __future__ import annotations

from raglab.agentic.contracts import ToolSpecification, _canonical_json, _sha256
from raglab.agentic.runtime.retrieval_tool_adapter import (
    PassageCapturePort,
    RetrievalToolAdapter,
)
from raglab.agentic.tool_registry import ToolRegistry
from raglab.domain.enums import PipelineStrategy

# Canonical mapping: PipelineStrategy → tool_id suffix
_STRATEGY_TO_TOOL: dict[PipelineStrategy, str] = {
    PipelineStrategy.BASELINE: "baseline",
    PipelineStrategy.SENTENCE_ANCHOR: "sentence_anchor",
    PipelineStrategy.SENTENCE_WINDOW: "sentence_window",
    PipelineStrategy.SENTENCE_WINDOW_RERANK: "sentence_window_rerank",
    PipelineStrategy.HIERARCHICAL_LEAF: "hierarchical_leaf",
    PipelineStrategy.AUTO_MERGING: "auto_merging",
    PipelineStrategy.AUTO_MERGING_RERANK: "auto_merging_rerank",
}

ALL_STRATEGIES: frozenset[PipelineStrategy] = frozenset(_STRATEGY_TO_TOOL.keys())


def tool_id_for_strategy(strategy: PipelineStrategy) -> str:
    """Return the canonical tool_id for a strategy."""
    return f"retrieve_{_STRATEGY_TO_TOOL[strategy]}"


def make_tool_spec(strategy: PipelineStrategy) -> ToolSpecification:
    """Create a governed ToolSpecification for a strategy."""
    suffix = _STRATEGY_TO_TOOL[strategy]
    impl_hash = _sha256(
        _canonical_json({"type": "retrieval", "strategy": suffix, "version": "1.0.0"})
    )
    return ToolSpecification(
        tool_id=f"retrieve_{suffix}",
        version="1.0.0",
        description=f"Retrieve evidence using the {suffix} strategy",
        read_only=True,
        network_access=False,
        deterministic=True,
        max_top_k=10,
        timeout_seconds=30.0,
        allowed_strategies=(suffix,),
        implementation_sha256=impl_hash,
    )


def build_adapter(
    strategy: PipelineStrategy,
    retrieval_port: object,
    *,
    passage_store: PassageCapturePort | None = None,
) -> RetrievalToolAdapter:
    """Build a RetrievalToolAdapter for a strategy + port pair."""
    suffix = _STRATEGY_TO_TOOL[strategy]
    return RetrievalToolAdapter(
        strategy=suffix,
        retrieval_port=retrieval_port,
        passage_store=passage_store,
    )


def build_registry_with_adapters(
    ports: dict[PipelineStrategy, object],
    *,
    passage_store: PassageCapturePort | None = None,
) -> tuple[ToolRegistry, dict[str, RetrievalToolAdapter]]:
    """Build a frozen registry and adapter map from strategy→port mapping.

    Returns (registry, {tool_id: adapter}).
    """
    registry = ToolRegistry()
    adapters: dict[str, RetrievalToolAdapter] = {}

    for strategy in sorted(ports.keys(), key=lambda s: s.value):
        spec = make_tool_spec(strategy)
        registry.register(spec)
        adapter = build_adapter(
            strategy,
            ports[strategy],
            passage_store=passage_store,
        )
        adapters[spec.tool_id] = adapter

    registry.freeze()
    return registry, adapters
