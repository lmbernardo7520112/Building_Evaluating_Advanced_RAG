"""Public one-shot coordinator factory — composes L1/L2 agentic pipeline.

Connects:
    ports -> strategy_tool_factory -> ToolRegistry + adapters
    -> DispatchingRetrievalBackend -> Budget -> OneShotRunner
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from raglab.agentic.budget import Budget
from raglab.agentic.one_shot_runner import OneShotRunner
from raglab.agentic.runtime.dispatching_backend import (
    DispatchingRetrievalBackend,
)
from raglab.agentic.runtime.strategy_tool_factory import (
    build_registry_with_adapters,
)
from raglab.domain.enums import PipelineStrategy

# Required strategies for full L1/L2 one-shot pipeline
_REQUIRED_STRATEGIES: tuple[PipelineStrategy, ...] = (
    PipelineStrategy.BASELINE,
    PipelineStrategy.SENTENCE_WINDOW_RERANK,
)


def build_one_shot_coordinator(
    ports: Mapping[PipelineStrategy, Any],
    run_id: str,
    *,
    max_logical_calls: int = 1,
    max_physical_attempts: int = 1,
    max_retries: int = 0,
    clock: Any = None,
    invocation_id_gen: Any = None,
) -> OneShotRunner:
    """Build a OneShotRunner wired with DispatchingRetrievalBackend.

    Validates required strategies exist in ports, creates new registry, adapters,
    dispatcher, and budget per call, and returns a fresh OneShotRunner instance.
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

    # 3. Build a fresh Budget instance
    budget = Budget(
        max_logical_calls=max_logical_calls,
        max_physical_attempts=max_physical_attempts,
        max_retries=max_retries,
    )

    # 4. Build and return OneShotRunner
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

    return OneShotRunner(**runner_kwargs)
