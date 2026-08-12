"""Public bounded loop coordinator factory — composes 2-step L3A agentic pipeline.

Connects:
    ports -> strategy_tool_factory -> ToolRegistry + adapters
    -> DispatchingRetrievalBackend -> Budget -> BoundedLoopRunner
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from raglab.agentic.budget import Budget
from raglab.agentic.runtime.bounded_loop_runner import BoundedLoopRunner
from raglab.agentic.runtime.dispatching_backend import (
    DispatchingRetrievalBackend,
)
from raglab.agentic.runtime.strategy_tool_factory import (
    build_registry_with_adapters,
)
from raglab.domain.enums import PipelineStrategy

# Required strategies for full L3A bounded loop pipeline
_REQUIRED_STRATEGIES: tuple[PipelineStrategy, ...] = (
    PipelineStrategy.BASELINE,
    PipelineStrategy.SENTENCE_WINDOW_RERANK,
)


def build_bounded_loop_coordinator(
    ports: Mapping[PipelineStrategy, Any],
    run_id: str,
    *,
    clock: Any = None,
    invocation_id_gen: Any = None,
) -> BoundedLoopRunner:
    """Build a BoundedLoopRunner wired with DispatchingRetrievalBackend.

    Validates required strategies exist in ports, creates new registry, adapters,
    dispatcher, and budget per call, and returns a fresh BoundedLoopRunner instance.
    """
    if not run_id:
        raise ValueError("run_id must be non-empty")

    # Validate required strategies
    for req in _REQUIRED_STRATEGIES:
        if req not in ports or ports[req] is None:
            raise ValueError(
                f"Missing required strategy port: '{req.value}' ({req}). "
                f"Required strategies: {[s.value for s in _REQUIRED_STRATEGIES]}"
            )

    # 1. Build registry and adapters using existing strategy_tool_factory
    ports_dict = {s: ports[s] for s in ports}
    registry, adapters = build_registry_with_adapters(ports_dict)

    # 2. Build public DispatchingRetrievalBackend
    dispatcher = DispatchingRetrievalBackend(adapters)

    # 3. Build a fresh Budget instance frozen at 2/2/0
    budget = Budget(
        max_logical_calls=2,
        max_physical_attempts=2,
        max_retries=0,
    )

    # 4. Build and return BoundedLoopRunner
    runner_kwargs: dict[str, Any] = {
        "registry": registry,
        "budget": budget,
        "backend": dispatcher,
        "run_id": run_id,
    }
    if clock is not None:
        runner_kwargs["clock"] = clock
    if invocation_id_gen is not None:
        runner_kwargs["invocation_id_gen"] = invocation_id_gen

    return BoundedLoopRunner(**runner_kwargs)
