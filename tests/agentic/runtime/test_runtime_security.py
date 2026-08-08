"""Security tests for runtime integration — anti-leakage and governance."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from raglab.agentic.budget import Budget
from raglab.agentic.contracts import ToolArguments, ToolInvocation
from raglab.agentic.enums import CallType, InvocationStatus
from raglab.agentic.errors import (
    BudgetExhaustedError,
    InvalidToolArgumentsError,
    LeakageDetectedError,
    NonCanonicalIdError,
    UnknownToolError,
)
from raglab.agentic.runtime.strategy_tool_factory import (
    ALL_STRATEGIES,
    build_registry_with_adapters,
)
from raglab.agentic.tool_executor import ToolExecutor
from raglab.domain.entities import RetrievedEvidence
from raglab.domain.value_objects import ChunkId


@dataclass
class SafePort:
    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedEvidence]:
        return [
            RetrievedEvidence(
                chunk_id=ChunkId(value="safe_001"),
                document_id="doc_01",
                text="safe evidence",
                rank=1,
                score=0.9,
                passage_id="ps_safe_001",
            )
        ]


def _build_executor() -> tuple[ToolExecutor, object]:
    ports = {s: SafePort() for s in ALL_STRATEGIES}
    registry, adapters = build_registry_with_adapters(ports)
    budget = Budget(max_logical_calls=1, max_physical_attempts=1, max_retries=0)
    executor = ToolExecutor(registry, budget)
    backend = _DispatchBackend(adapters)
    return executor, backend


class _DispatchBackend:
    def __init__(self, adapters: dict) -> None:
        self._adapters = adapters

    def retrieve(self, query: str, strategy: str, top_k: int) -> object:
        tool_id = f"retrieve_{strategy}"
        return self._adapters[tool_id].retrieve(query, strategy, top_k)


def _make_invocation(
    tool_id: str = "retrieve_baseline",
    query: str = "test query",
    strategy: str = "baseline",
    top_k: int = 3,
    allowed_doc_ids: tuple[str, ...] | None = None,
) -> ToolInvocation:
    args = ToolArguments(
        query=query,
        strategy=strategy,
        top_k=top_k,
        allowed_document_ids=allowed_doc_ids,
    )
    return ToolInvocation(
        invocation_id="inv_test",
        query_id="q_test",
        step_index=0,
        tool_id=tool_id,
        tool_version="1.0.0",
        arguments=args,
        arguments_sha256=args.sha256,
        authorization_status=InvocationStatus.AUTHORIZED,
        call_type=CallType.LOGICAL_CALL,
        logical_call_index=0,
        started_at="2026-01-01T00:00:00+00:00",
    )


class TestRuntimeSecurity:
    def test_unknown_tool_rejected(self) -> None:
        executor, backend = _build_executor()
        inv = _make_invocation(tool_id="retrieve_unknown_strategy")
        with pytest.raises(UnknownToolError):
            executor.validate_and_execute(inv, backend)

    def test_qrels_in_query_rejected(self) -> None:
        executor, backend = _build_executor()
        inv = _make_invocation(query="show me qrels for this question")
        with pytest.raises(LeakageDetectedError):
            executor.validate_and_execute(inv, backend)

    def test_gold_answer_in_query_rejected(self) -> None:
        executor, backend = _build_executor()
        inv = _make_invocation(query="the gold answer is X")
        with pytest.raises(LeakageDetectedError):
            executor.validate_and_execute(inv, backend)

    def test_holdout_in_query_rejected(self) -> None:
        executor, backend = _build_executor()
        inv = _make_invocation(query="use holdout data")
        with pytest.raises(LeakageDetectedError):
            executor.validate_and_execute(inv, backend)

    def test_document_filter_rejected(self) -> None:
        executor, backend = _build_executor()
        inv = _make_invocation(allowed_doc_ids=("doc_01",))
        with pytest.raises(LeakageDetectedError):
            executor.validate_and_execute(inv, backend)

    def test_empty_query_rejected(self) -> None:
        with pytest.raises(ValueError, match="query must be non-empty"):
            _make_invocation(query="")

    def test_invalid_top_k_rejected(self) -> None:
        executor, backend = _build_executor()
        inv = _make_invocation(top_k=999)
        with pytest.raises(InvalidToolArgumentsError):
            executor.validate_and_execute(inv, backend)

    def test_budget_exhaustion(self) -> None:
        """Second logical call must be rejected."""
        executor, backend = _build_executor()
        inv1 = _make_invocation()
        executor.validate_and_execute(inv1, backend)  # First call OK

        inv2 = _make_invocation()
        with pytest.raises(BudgetExhaustedError):
            executor.validate_and_execute(inv2, backend)

    def test_canonical_id_enforced(self) -> None:
        """Non-ps_ passage IDs must raise NonCanonicalIdError."""
        from raglab.agentic.contracts import ToolObservation

        # This is tested at the ToolObservation level
        with pytest.raises(NonCanonicalIdError):
            ToolObservation(
                invocation_id="inv_1",
                status=InvocationStatus.EXECUTED,
                passage_ids=("bad_no_prefix",),
                document_ids=("doc",),
                ranks=(1,),
                scores=(0.9,),
                content_hashes=("abc",),
                retrieval_config_hash="hash",
                latency_ms=1.0,
            )

    def test_leakage_rate_zero_on_clean_queries(self) -> None:
        """Clean queries must pass all leakage checks."""
        executor, backend = _build_executor()
        clean_queries = [
            "What is the main concept?",
            "How does retrieval work?",
            "Compare chunking strategies.",
        ]
        for q in clean_queries:
            # Must not raise — but budget allows only 1 call
            # so we rebuild each time
            executor2, backend2 = _build_executor()
            inv = _make_invocation(query=q)
            obs = executor2.validate_and_execute(inv, backend2)
            assert obs.passage_ids  # non-empty
