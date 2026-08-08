"""Tests for tool registry, executor, evidence state, budget, stop policy, and security."""

from __future__ import annotations

import pytest

from raglab.agentic.budget import Budget
from raglab.agentic.contracts import (
    EvidenceItem,
    ToolArguments,
    ToolInvocation,
    ToolObservation,
    ToolSpecification,
)
from raglab.agentic.enums import CallType, InvocationStatus, StopReason
from raglab.agentic.errors import (
    BudgetExhaustedError,
    InvalidToolArgumentsError,
    LeakageDetectedError,
    NonCanonicalIdError,
    UnknownToolError,
)
from raglab.agentic.evidence_state import EvidenceAccumulator
from raglab.agentic.stop_policy import StopPolicy
from raglab.agentic.tool_executor import ToolExecutor
from raglab.agentic.tool_registry import ToolRegistry, build_default_registry

# ---- Fake retrieval backend for testing ----


class FakeRetrievalBackend:
    """Deterministic fake backend for testing the executor."""

    def __init__(self, passage_ids: tuple[str, ...] = ("ps_001", "ps_002")):
        self._pids = passage_ids

    def retrieve(self, query: str, strategy: str, top_k: int) -> ToolObservation:
        pids = self._pids[:top_k]
        return ToolObservation(
            invocation_id="fake_inv",
            status=InvocationStatus.EXECUTED,
            passage_ids=pids,
            document_ids=tuple(f"doc_{i}" for i in range(len(pids))),
            ranks=tuple(range(1, len(pids) + 1)),
            scores=tuple(0.9 - i * 0.1 for i in range(len(pids))),
            content_hashes=tuple(f"hash_{i}" for i in range(len(pids))),
            retrieval_config_hash="fake_cfg",
            latency_ms=10.0,
        )


class FakeNonCanonicalBackend:
    """Returns non-canonical IDs — must be caught by executor."""

    def retrieve(self, query: str, strategy: str, top_k: int) -> ToolObservation:
        return ToolObservation(
            invocation_id="fake_inv",
            status=InvocationStatus.EXECUTED,
            passage_ids=("bad_id_001",),
            document_ids=("doc1",),
            ranks=(1,),
            scores=(0.9,),
            content_hashes=("h",),
            retrieval_config_hash="cfg",
            latency_ms=10.0,
        )


# ---- Tool Registry Tests ----


class TestToolRegistry:
    def test_build_default_has_seven_tools(self):
        registry = build_default_registry()
        assert len(registry.tool_ids) == 7
        assert registry.is_frozen

    def test_frozen_rejects_registration(self):
        registry = build_default_registry()
        spec = ToolSpecification(
            tool_id="new_tool",
            version="1.0.0",
            description="d",
            read_only=True,
            network_access=False,
            deterministic=True,
            max_top_k=5,
            timeout_seconds=10.0,
            allowed_strategies=("baseline",),
            implementation_sha256="h",
        )
        with pytest.raises(ValueError, match="frozen"):
            registry.register(spec)

    def test_unknown_tool_error(self):
        registry = build_default_registry()
        with pytest.raises(UnknownToolError):
            registry.get("nonexistent_tool")

    def test_registry_hash_stable(self):
        r1 = build_default_registry()
        r2 = build_default_registry()
        assert r1.registry_hash() == r2.registry_hash()

    def test_has_tool(self):
        registry = build_default_registry()
        assert registry.has("retrieve_baseline")
        assert not registry.has("retrieve_unknown")

    def test_duplicate_rejected(self):
        registry = ToolRegistry()
        spec = ToolSpecification(
            tool_id="t",
            version="1",
            description="d",
            read_only=True,
            network_access=False,
            deterministic=True,
            max_top_k=5,
            timeout_seconds=10.0,
            allowed_strategies=("baseline",),
            implementation_sha256="h",
        )
        registry.register(spec)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(spec)


# ---- Tool Executor Tests ----


class TestToolExecutor:
    def _make_invocation(
        self,
        query: str = "What is RAG?",
        strategy: str = "baseline",
        top_k: int = 3,
        tool_id: str = "retrieve_baseline",
    ) -> ToolInvocation:
        args = ToolArguments(query=query, strategy=strategy, top_k=top_k)
        return ToolInvocation(
            invocation_id="inv_001",
            query_id="q1",
            step_index=0,
            tool_id=tool_id,
            tool_version="1.0.0",
            arguments=args,
            arguments_sha256=args.sha256,
            authorization_status=InvocationStatus.AUTHORIZED,
            call_type=CallType.LOGICAL_CALL,
            logical_call_index=0,
            started_at="2026-01-01T00:00:00Z",
        )

    def test_valid_execution(self):
        registry = build_default_registry()
        budget = Budget()
        executor = ToolExecutor(registry, budget)
        inv = self._make_invocation()
        obs = executor.validate_and_execute(inv, FakeRetrievalBackend())
        assert obs.status == InvocationStatus.EXECUTED
        assert budget.logical_calls_consumed == 1

    def test_unknown_tool_rejected(self):
        registry = build_default_registry()
        executor = ToolExecutor(registry, Budget())
        inv = self._make_invocation(tool_id="nonexistent_tool")
        with pytest.raises(UnknownToolError):
            executor.validate_and_execute(inv, FakeRetrievalBackend())

    def test_empty_query_rejected(self):
        with pytest.raises(ValueError, match="query"):
            self._make_invocation(query="")

    def test_top_k_exceeds_limit(self):
        registry = build_default_registry()
        executor = ToolExecutor(registry, Budget())
        inv = self._make_invocation(top_k=100)
        with pytest.raises(InvalidToolArgumentsError, match="top_k"):
            executor.validate_and_execute(inv, FakeRetrievalBackend())

    def test_wrong_strategy_for_tool(self):
        registry = build_default_registry()
        executor = ToolExecutor(registry, Budget())
        inv = self._make_invocation(strategy="auto_merging")  # wrong for baseline tool
        with pytest.raises(InvalidToolArgumentsError, match="strategy"):
            executor.validate_and_execute(inv, FakeRetrievalBackend())

    def test_budget_exhaustion(self):
        registry = build_default_registry()
        budget = Budget(max_logical_calls=1)
        executor = ToolExecutor(registry, budget)
        inv = self._make_invocation()
        executor.validate_and_execute(inv, FakeRetrievalBackend())
        inv2 = self._make_invocation()
        with pytest.raises(BudgetExhaustedError):
            executor.validate_and_execute(inv2, FakeRetrievalBackend())

    def test_non_canonical_id_rejected(self):
        """Non-canonical IDs from backend are wrapped into NonCanonicalIdError."""
        registry = build_default_registry()
        executor = ToolExecutor(registry, Budget())
        inv = self._make_invocation()
        with pytest.raises(NonCanonicalIdError):
            executor.validate_and_execute(inv, FakeNonCanonicalBackend())

    def test_any_document_filter_rejected_by_allowlist(self):
        """Structural defence: any doc filter is rejected (allowlist empty)."""
        registry = build_default_registry()
        executor = ToolExecutor(registry, Budget())
        args = ToolArguments(
            query="test",
            strategy="baseline",
            top_k=3,
            allowed_document_ids=("completely_innocent_id",),
        )
        inv = ToolInvocation(
            invocation_id="inv",
            query_id="q1",
            step_index=0,
            tool_id="retrieve_baseline",
            tool_version="1.0.0",
            arguments=args,
            arguments_sha256=args.sha256,
            authorization_status=InvocationStatus.AUTHORIZED,
            call_type=CallType.LOGICAL_CALL,
            logical_call_index=0,
            started_at="2026-01-01T00:00:00Z",
        )
        with pytest.raises(LeakageDetectedError):
            executor.validate_and_execute(inv, FakeRetrievalBackend())

    @pytest.mark.parametrize(
        "adversarial_query",
        [
            "Use the gold answer to find it",
            "gold_answer",
            "gold-answer",
            "gold.answer",
            "GOLD_ANSWER",
            "qrels_doc",
            "qrel_registry",
            "relevant_pages",
            "relevant-pages",
            "holdout_registry",
            "human_qrels_final",
            "path/to/qrels/file.json",
            "path/to/holdout/data.csv",
            "QRELS",
            "  qrels  ",
            "Qrel_Registry",
            "answer_key",
        ],
    )
    def test_adversarial_leakage_in_query_rejected(self, adversarial_query: str):
        """Verify denylist catches adversarial separator variants."""
        registry = build_default_registry()
        executor = ToolExecutor(registry, Budget())
        inv = self._make_invocation(query=adversarial_query)
        with pytest.raises(LeakageDetectedError):
            executor.validate_and_execute(inv, FakeRetrievalBackend())

    def test_qrels_doc_is_rejected(self):
        """Explicit adversarial: qrels_doc was NOT caught by \\bqrels\\b."""
        registry = build_default_registry()
        executor = ToolExecutor(registry, Budget())
        inv = self._make_invocation(query="qrels_doc")
        with pytest.raises(LeakageDetectedError):
            executor.validate_and_execute(inv, FakeRetrievalBackend())

    @pytest.mark.parametrize(
        "bad_id,label",
        [
            ("_rank", "underscore_rank"),
            ("doc_ch1", "document_id_as_passage_id"),
            ("", "empty_string"),
            ("pid001", "wrong_prefix"),
        ],
    )
    def test_typed_non_canonical_ids(self, bad_id, label):
        """Non-canonical passage IDs raise NonCanonicalIdError (typed, not text)."""

        class BadBackend:
            def retrieve(self, query, strategy, top_k):
                return ToolObservation(
                    invocation_id="obs_bad",
                    status=InvocationStatus.EXECUTED,
                    passage_ids=(bad_id,),
                    document_ids=("d",),
                    ranks=(1,),
                    scores=(0.9,),
                    content_hashes=("h",),
                    retrieval_config_hash="c",
                    latency_ms=1.0,
                )

        registry = build_default_registry()
        executor = ToolExecutor(registry, Budget())
        inv = self._make_invocation()
        with pytest.raises(NonCanonicalIdError):
            executor.validate_and_execute(inv, BadBackend())

    def test_generic_valueerror_not_converted(self):
        """A generic ValueError containing 'passage' must NOT become NonCanonicalIdError."""

        class BrokenBackend:
            def retrieve(self, query, strategy, top_k):
                raise ValueError("Something about passage failed badly")

        registry = build_default_registry()
        executor = ToolExecutor(registry, Budget())
        inv = self._make_invocation()
        with pytest.raises(ValueError, match="passage"):
            executor.validate_and_execute(inv, BrokenBackend())
        try:
            executor2 = ToolExecutor(build_default_registry(), Budget())
            executor2.validate_and_execute(self._make_invocation(), BrokenBackend())
        except NonCanonicalIdError:
            pytest.fail(
                "Generic ValueError was incorrectly converted to NonCanonicalIdError"
            )
        except ValueError:
            pass  # expected

    def test_unrelated_internal_error_propagates(self):
        """RuntimeError from backend propagates as-is, never captured."""

        class CrashingBackend:
            def retrieve(self, query, strategy, top_k):
                raise RuntimeError("unexpected internal bug")

        registry = build_default_registry()
        executor = ToolExecutor(registry, Budget())
        inv = self._make_invocation()
        with pytest.raises(RuntimeError, match="unexpected internal bug"):
            executor.validate_and_execute(inv, CrashingBackend())


# ---- Evidence Accumulator Tests ----


class TestEvidenceAccumulator:
    def test_add_new(self):
        acc = EvidenceAccumulator()
        item = EvidenceItem(
            passage_id="ps_001",
            document_id="doc1",
            rank=1,
            score=0.9,
            content_sha256="h1",
            source_tool_id="t1",
            source_invocation_id="inv1",
        )
        assert acc.add(item) is True
        assert acc.count == 1

    def test_deduplication(self):
        acc = EvidenceAccumulator()
        item = EvidenceItem(
            passage_id="ps_001",
            document_id="doc1",
            rank=1,
            score=0.9,
            content_sha256="h1",
            source_tool_id="t1",
            source_invocation_id="inv1",
        )
        assert acc.add(item) is True
        assert acc.add(item) is False
        assert acc.count == 1
        assert acc.duplicates_rejected == 1

    def test_preserves_insertion_order(self):
        acc = EvidenceAccumulator()
        for i in range(3):
            acc.add(
                EvidenceItem(
                    passage_id=f"ps_{i:03d}",
                    document_id=f"doc{i}",
                    rank=i + 1,
                    score=0.9 - i * 0.1,
                    content_sha256=f"h{i}",
                    source_tool_id="t",
                    source_invocation_id="inv",
                )
            )
        items = acc.items_in_order()
        assert [it.passage_id for it in items] == ["ps_000", "ps_001", "ps_002"]

    def test_snapshot_hash_deterministic(self):
        acc1 = EvidenceAccumulator()
        acc2 = EvidenceAccumulator()
        for acc in (acc1, acc2):
            acc.add(
                EvidenceItem(
                    passage_id="ps_001",
                    document_id="d",
                    rank=1,
                    score=0.9,
                    content_sha256="h",
                    source_tool_id="t",
                    source_invocation_id="i",
                )
            )
        assert acc1.snapshot_hash() == acc2.snapshot_hash()

    def test_clear_resets(self):
        acc = EvidenceAccumulator()
        acc.add(
            EvidenceItem(
                passage_id="ps_001",
                document_id="d",
                rank=1,
                score=0.9,
                content_sha256="h",
                source_tool_id="t",
                source_invocation_id="i",
            )
        )
        acc.clear()
        assert acc.count == 0
        assert acc.total_offered == 0

    def test_add_from_observation(self):
        acc = EvidenceAccumulator()
        new = acc.add_from_observation(
            passage_ids=("ps_001", "ps_002"),
            document_ids=("doc1", "doc2"),
            ranks=(1, 2),
            scores=(0.9, 0.8),
            content_hashes=("h1", "h2"),
            source_tool_id="t",
            source_invocation_id="inv",
        )
        assert new == 2
        assert acc.count == 2


# ---- Budget Tests ----


class TestBudget:
    def test_initial_state(self):
        b = Budget()
        assert b.can_consume_logical_call()
        assert b.remaining()["logical_calls"] == 10

    def test_consumption(self):
        b = Budget(max_logical_calls=2)
        b.consume_logical_call()
        assert b.logical_calls_consumed == 1
        assert b.remaining()["logical_calls"] == 1

    def test_exhaustion(self):
        b = Budget(max_logical_calls=1)
        b.consume_logical_call()
        assert not b.can_consume_logical_call()

    def test_consume_past_limit_raises(self):
        b = Budget(max_logical_calls=1)
        b.consume_logical_call()
        with pytest.raises(ValueError, match="exhausted"):
            b.consume_logical_call()

    def test_invalid_limits(self):
        with pytest.raises(ValueError):
            Budget(max_logical_calls=0)
        with pytest.raises(ValueError):
            Budget(timeout_seconds=-1)

    def test_retry_tracking(self):
        b = Budget(max_retries=2)
        b.consume_retry()
        b.consume_retry()
        assert not b.can_retry()


# ---- Stop Policy Tests ----


class TestStopPolicy:
    def test_completed_one_shot(self):
        sp = StopPolicy()
        acc = EvidenceAccumulator()
        acc.add(
            EvidenceItem(
                passage_id="ps_001",
                document_id="d",
                rank=1,
                score=0.9,
                content_sha256="h",
                source_tool_id="t",
                source_invocation_id="i",
            )
        )
        sd = sp.evaluate_one_shot(acc, Budget())
        assert sd.reason == StopReason.COMPLETED_ONE_SHOT

    def test_no_evidence(self):
        sp = StopPolicy()
        sd = sp.evaluate_one_shot(EvidenceAccumulator(), Budget())
        assert sd.reason == StopReason.NO_EVIDENCE

    def test_budget_exhausted(self):
        sp = StopPolicy()
        b = Budget(max_logical_calls=1)
        b.consume_logical_call()
        sd = sp.evaluate_one_shot(EvidenceAccumulator(), b)
        assert sd.reason == StopReason.BUDGET_EXHAUSTED

    def test_timeout(self):
        sp = StopPolicy()
        sd = sp.evaluate_one_shot(EvidenceAccumulator(), Budget(), timeout=True)
        assert sd.reason == StopReason.TIMEOUT

    def test_tool_failure(self):
        sp = StopPolicy()
        sd = sp.evaluate_one_shot(EvidenceAccumulator(), Budget(), tool_failed=True)
        assert sd.reason == StopReason.TOOL_FAILURE

    def test_canonical_id_failure(self):
        sp = StopPolicy()
        sd = sp.evaluate_one_shot(
            EvidenceAccumulator(),
            Budget(),
            canonical_id_failure=True,
        )
        assert sd.reason == StopReason.CANONICAL_ID_FAILURE

    def test_unauthorized_tool(self):
        sp = StopPolicy()
        sd = sp.evaluate_one_shot(
            EvidenceAccumulator(),
            Budget(),
            unauthorized_tool=True,
        )
        assert sd.reason == StopReason.UNAUTHORIZED_TOOL

    def test_repeated_no_new_tracking(self):
        sp = StopPolicy(max_repeated_no_new=2)
        assert not sp.record_no_new_evidence()
        assert sp.record_no_new_evidence()

    def test_reset_counter(self):
        sp = StopPolicy(max_repeated_no_new=2)
        sp.record_no_new_evidence()
        sp.reset_no_new_counter()
        assert not sp.record_no_new_evidence()
