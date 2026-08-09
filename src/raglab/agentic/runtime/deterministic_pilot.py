"""Deterministic pilot — end-to-end one-shot retrieval per question.

Composes OneShotRunner with real RetrievalToolAdapters (via factory)
to produce a complete pilot run with trajectory, manifest, and results.

Constraints:
- Exactly one routing decision and one logical call per QID.
- No LLM generation (NOT_EXECUTED).
- No credential required.
- No qrels access during execution (analysis only).
- Deterministic: same inputs → same outputs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from raglab.agentic.budget import Budget
from raglab.agentic.contracts import SCHEMA_VERSION
from raglab.agentic.one_shot_runner import OneShotResult, OneShotRunner
from raglab.agentic.router import get_deterministic_policy_metadata
from raglab.agentic.runtime.retrieval_tool_adapter import RetrievalToolAdapter
from raglab.agentic.tool_registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class PilotQuestion:
    """A question for the deterministic pilot."""

    qid: str
    text: str
    split: str


@dataclass(frozen=True, slots=True)
class PilotRunConfig:
    """Configuration for a pilot run."""

    run_id: str
    top_k: int = 3
    max_logical_calls: int = 1
    max_physical_attempts: int = 1
    max_retries: int = 0


@dataclass
class PilotRunResult:
    """Aggregate result of a pilot run."""

    run_id: str
    config: PilotRunConfig
    policy_metadata: dict[str, str]
    questions_processed: int
    results: list[dict[str, Any]] = field(default_factory=list)
    routing_decisions: list[dict[str, Any]] = field(default_factory=list)
    trajectories: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)


class DeterministicPilot:
    """Execute a deterministic pilot run across questions.

    Uses OneShotRunner with RetrievalToolAdapter backends.
    One routing decision + one retrieval call per QID.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        adapters: dict[str, RetrievalToolAdapter],
        config: PilotRunConfig,
        *,
        clock: Any = None,
        invocation_id_gen: Any = None,
    ) -> None:
        self._registry = registry
        self._adapters = adapters
        self._config = config
        self._clock = clock
        self._inv_gen = invocation_id_gen

    def _make_backend(self) -> _DispatchingBackend:
        """Create a dispatching backend that routes to the correct adapter."""
        return _DispatchingBackend(self._adapters)

    def run(
        self,
        questions: list[PilotQuestion],
    ) -> PilotRunResult:
        """Execute the pilot for all questions.

        Returns aggregate results. No qrels access here.
        """
        policy_meta = get_deterministic_policy_metadata()
        result = PilotRunResult(
            run_id=self._config.run_id,
            config=self._config,
            policy_metadata={
                "policy_id": policy_meta.policy_id,
                "policy_version": policy_meta.policy_version,
                "policy_sha256": policy_meta.policy_sha256,
            },
            questions_processed=0,
        )

        for q in questions:
            budget = Budget(
                max_logical_calls=self._config.max_logical_calls,
                max_physical_attempts=self._config.max_physical_attempts,
                max_retries=self._config.max_retries,
            )
            backend = self._make_backend()

            runner_kwargs: dict[str, Any] = {
                "registry": self._registry,
                "budget": budget,
                "backend": backend,
                "run_id": self._config.run_id,
            }
            if self._clock is not None:
                runner_kwargs["clock"] = self._clock
            if self._inv_gen is not None:
                runner_kwargs["invocation_id_gen"] = self._inv_gen

            runner = OneShotRunner(**runner_kwargs)
            one_result: OneShotResult = runner.execute(
                query_id=q.qid,
                query_text=q.text,
                top_k=self._config.top_k,
            )

            # Record result
            result.results.append(
                {
                    "qid": q.qid,
                    "split": q.split,
                    "strategy_selected": one_result.routing_decision.selected_strategy,
                    "evidence_count": one_result.evidence_count,
                    "stop_reason": one_result.stop_decision.reason.value,
                    "error": one_result.error,
                    "generation": "NOT_EXECUTED",
                }
            )

            # Record routing decision
            rd = one_result.routing_decision
            result.routing_decisions.append(
                {
                    "qid": q.qid,
                    "selected_strategy": rd.selected_strategy,
                    "decision_code": rd.decision_code.value,
                    "fallback_used": rd.fallback_used,
                    "public_features_used": list(rd.public_features_used),
                    "policy_sha256": rd.policy_sha256,
                }
            )

            # Record trajectory
            result.trajectories.append(one_result.trajectory.to_dict())

            if one_result.error:
                result.errors.append({"qid": q.qid, "error": one_result.error})

            result.questions_processed += 1

        return result


class _DispatchingBackend:
    """Routes retrieve() calls to the correct adapter by strategy."""

    def __init__(self, adapters: dict[str, RetrievalToolAdapter]) -> None:
        self._adapters = adapters

    def retrieve(self, query: str, strategy: str, top_k: int) -> Any:
        """Dispatch to the adapter matching the strategy."""
        tool_id = f"retrieve_{strategy}"
        if tool_id not in self._adapters:
            raise ValueError(
                f"No adapter registered for tool '{tool_id}'. "
                f"Available: {sorted(self._adapters.keys())}"
            )
        return self._adapters[tool_id].retrieve(
            query=query, strategy=strategy, top_k=top_k
        )


def build_pilot_manifest(
    run_result: PilotRunResult,
    config_path: Path | None = None,
    questions_path: Path | None = None,
) -> dict[str, Any]:
    """Build a manifest for the pilot run."""

    def _file_hash(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "deterministic_pilot_v1",
        "run_id": run_result.run_id,
        "questions_processed": run_result.questions_processed,
        "policy_metadata": run_result.policy_metadata,
        "config": {
            "run_id": run_result.config.run_id,
            "top_k": run_result.config.top_k,
            "max_logical_calls": run_result.config.max_logical_calls,
        },
        "credentials_used": False,
        "gemini_used": False,
        "generation_status": "NOT_EXECUTED",
        "errors_count": len(run_result.errors),
    }

    if config_path and config_path.exists():
        manifest["config_sha256"] = _file_hash(config_path)
    if questions_path and questions_path.exists():
        manifest["questions_sha256"] = _file_hash(questions_path)

    return manifest
