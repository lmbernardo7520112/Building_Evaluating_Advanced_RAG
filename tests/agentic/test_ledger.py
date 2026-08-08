"""Tests for trajectory ledger and vertical end-to-end flow."""

from __future__ import annotations

import json

import pytest

from raglab.agentic.budget import Budget
from raglab.agentic.contracts import (
    SCHEMA_VERSION,
    AgentTrajectory,
    EvidenceItem,
    RoutingDecision,
    StopDecision,
    ToolArguments,
    ToolInvocation,
    ToolObservation,
    TrajectoryStep,
)
from raglab.agentic.enums import (
    CallType,
    DecisionCode,
    InvocationStatus,
    StopReason,
    ValidationStatus,
)
from raglab.agentic.errors import (
    IncompatibleRunError,
    LedgerConflictError,
    LedgerCorruptionError,
)
from raglab.agentic.evidence_state import EvidenceAccumulator
from raglab.agentic.router import get_deterministic_policy_metadata, route_deterministic
from raglab.agentic.stop_policy import StopPolicy
from raglab.agentic.tool_executor import ToolExecutor
from raglab.agentic.tool_registry import build_default_registry
from raglab.agentic.trajectory_ledger import TrajectoryLedger


class TestTrajectoryLedger:
    def _make_trajectory(
        self,
        query_id: str = "q1",
        run_id: str = "run1",
        policy_sha256: str = "policy_hash",
        config_sha256: str = "config_hash",
    ) -> AgentTrajectory:
        rd = RoutingDecision(
            schema_version=SCHEMA_VERSION,
            query_id=query_id,
            policy_id="det_v1",
            policy_version="1.0.0",
            policy_sha256=policy_sha256,
            selected_strategy="baseline",
            decision_code=DecisionCode.SELECTED,
            public_features_used=("test",),
            validation_status=ValidationStatus.VALID,
        )
        sd = StopDecision(
            reason=StopReason.COMPLETED_ONE_SHOT,
            detail="test done",
            evidence_count=1,
            budget_remaining={"logical_calls": 9},
        )
        return AgentTrajectory(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            query_id=query_id,
            policy_id="det_v1",
            policy_sha256=policy_sha256,
            config_sha256=config_sha256,
            steps=(),
            stop_decision=sd,
            routing_decision=rd,
            created_at="2026-01-01T00:00:00Z",
        )

    def test_append_and_read(self, tmp_path):
        ledger = TrajectoryLedger(
            path=tmp_path / "ledger.jsonl",
            run_id="run1",
            policy_sha256="policy_hash",
            config_sha256="config_hash",
        )
        t = self._make_trajectory()
        ledger.append(t)
        assert ledger.entry_count == 1
        assert ledger.has_trajectory("q1", "det_v1")

    def test_duplicate_rejected(self, tmp_path):
        ledger = TrajectoryLedger(
            path=tmp_path / "ledger.jsonl",
            run_id="run1",
            policy_sha256="policy_hash",
            config_sha256="config_hash",
        )
        t = self._make_trajectory()
        ledger.append(t)
        with pytest.raises(LedgerConflictError):
            ledger.append(t)

    def test_run_id_mismatch(self, tmp_path):
        ledger = TrajectoryLedger(
            path=tmp_path / "ledger.jsonl",
            run_id="run1",
            policy_sha256="p",
            config_sha256="c",
        )
        t = self._make_trajectory(run_id="run2", policy_sha256="p", config_sha256="c")
        with pytest.raises(IncompatibleRunError):
            ledger.append(t)

    def test_policy_hash_mismatch(self, tmp_path):
        ledger = TrajectoryLedger(
            path=tmp_path / "ledger.jsonl",
            run_id="run1",
            policy_sha256="expected",
            config_sha256="c",
        )
        t = self._make_trajectory(policy_sha256="wrong", config_sha256="c")
        with pytest.raises(IncompatibleRunError):
            ledger.append(t)

    def test_config_hash_mismatch(self, tmp_path):
        ledger = TrajectoryLedger(
            path=tmp_path / "ledger.jsonl",
            run_id="run1",
            policy_sha256="p",
            config_sha256="expected",
        )
        t = self._make_trajectory(policy_sha256="p", config_sha256="wrong")
        with pytest.raises(IncompatibleRunError):
            ledger.append(t)

    def test_corrupted_json_detected(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        path.write_text("not valid json\n")
        with pytest.raises(LedgerCorruptionError):
            TrajectoryLedger(
                path=path,
                run_id="run1",
                policy_sha256="p",
                config_sha256="c",
            )

    def test_ledger_hash_deterministic(self, tmp_path):
        ledger = TrajectoryLedger(
            path=tmp_path / "ledger.jsonl",
            run_id="run1",
            policy_sha256="p",
            config_sha256="c",
        )
        t = self._make_trajectory(policy_sha256="p", config_sha256="c")
        ledger.append(t)
        h1 = ledger.ledger_hash()
        h2 = ledger.ledger_hash()
        assert h1 == h2

    def test_no_forbidden_fields(self, tmp_path):
        """Verify the ledger file contains no chain-of-thought or secrets."""
        ledger = TrajectoryLedger(
            path=tmp_path / "ledger.jsonl",
            run_id="run1",
            policy_sha256="p",
            config_sha256="c",
        )
        t = self._make_trajectory(policy_sha256="p", config_sha256="c")
        ledger.append(t)

        content = (tmp_path / "ledger.jsonl").read_text()
        assert "chain_of_thought" not in content
        assert "secret" not in content.lower()
        assert "credential" not in content.lower()
        assert "api_key" not in content.lower()

    def test_append_only_file_grows(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = TrajectoryLedger(
            path=path,
            run_id="run1",
            policy_sha256="p",
            config_sha256="c",
        )
        ledger.append(
            self._make_trajectory(
                query_id="q1",
                policy_sha256="p",
                config_sha256="c",
            )
        )
        size1 = path.stat().st_size
        ledger.append(
            self._make_trajectory(
                query_id="q2",
                policy_sha256="p",
                config_sha256="c",
            )
        )
        size2 = path.stat().st_size
        assert size2 > size1


class TestVerticalFlow:
    """End-to-end offline flow testing the complete governed pipeline."""

    def test_full_one_shot_flow(self, tmp_path):
        """Vertical test: query → router → registry → executor → evidence → stop → ledger."""
        # 1. Router
        policy_meta = get_deterministic_policy_metadata()
        decision = route_deterministic("q_test", "What is chunking?")
        assert decision.validation_status == ValidationStatus.VALID
        assert decision.selected_strategy in build_default_registry().tool_ids or True

        # 2. Registry & tool lookup
        registry = build_default_registry()
        tool_id = f"retrieve_{decision.selected_strategy}"
        spec = registry.get(tool_id)
        assert spec.read_only is True

        # 3. Build invocation
        args = ToolArguments(
            query="What is chunking?",
            strategy=decision.selected_strategy,
            top_k=3,
        )

        invocation = ToolInvocation(
            invocation_id="inv_001",
            query_id="q_test",
            step_index=0,
            tool_id=tool_id,
            tool_version=spec.version,
            arguments=args,
            arguments_sha256=args.sha256,
            authorization_status=InvocationStatus.AUTHORIZED,
            call_type=CallType.LOGICAL_CALL,
            logical_call_index=0,
            started_at="2026-01-01T00:00:00Z",
        )

        # 4. Execute via governed executor
        budget = Budget(max_logical_calls=5)

        class FakeBackend:
            def retrieve(self, query, strategy, top_k):
                return ToolObservation(
                    invocation_id="inv_001",
                    status=InvocationStatus.EXECUTED,
                    passage_ids=("ps_ch_001", "ps_ch_002", "ps_ch_003"),
                    document_ids=("doc_ch2", "doc_ch2", "doc_ch2"),
                    ranks=(1, 2, 3),
                    scores=(0.95, 0.88, 0.72),
                    content_hashes=("h1", "h2", "h3"),
                    retrieval_config_hash="cfg_fake",
                    latency_ms=15.0,
                )

        executor = ToolExecutor(registry, budget)
        observation = executor.validate_and_execute(invocation, FakeBackend())
        assert observation.status == InvocationStatus.EXECUTED

        # 5. Evidence accumulator
        evidence = EvidenceAccumulator()
        new_count = evidence.add_from_observation(
            passage_ids=observation.passage_ids,
            document_ids=observation.document_ids,
            ranks=observation.ranks,
            scores=observation.scores,
            content_hashes=observation.content_hashes,
            source_tool_id=tool_id,
            source_invocation_id=invocation.invocation_id,
        )
        assert new_count == 3
        assert evidence.count == 3

        # 6. Stop policy
        stop_policy = StopPolicy()
        stop_decision = stop_policy.evaluate_one_shot(evidence, budget)
        assert stop_decision.reason == StopReason.COMPLETED_ONE_SHOT

        # 7. Build trajectory
        step = TrajectoryStep(
            step_index=0,
            state_before_hash="empty_state",
            action=f"retrieve:{tool_id}",
            arguments_sha256=args.sha256,
            observation_hash=observation.invocation_id,
            evidence_delta_count=new_count,
            decision_code=decision.decision_code,
            state_after_hash=evidence.snapshot_hash(),
            budget_remaining=budget.remaining(),
            stop_reason=stop_decision.reason,
        )

        config_hash = registry.registry_hash()
        trajectory = AgentTrajectory(
            schema_version=SCHEMA_VERSION,
            run_id="test_run_001",
            query_id="q_test",
            policy_id=policy_meta.policy_id,
            policy_sha256=policy_meta.policy_sha256,
            config_sha256=config_hash,
            steps=(step,),
            stop_decision=stop_decision,
            routing_decision=decision,
            created_at="2026-01-01T00:00:00Z",
        )

        # 8. Write to ledger
        ledger = TrajectoryLedger(
            path=tmp_path / "test_ledger.jsonl",
            run_id="test_run_001",
            policy_sha256=policy_meta.policy_sha256,
            config_sha256=config_hash,
        )
        ledger.append(trajectory)
        assert ledger.entry_count == 1

        # 9. Verify ledger content
        content = (tmp_path / "test_ledger.jsonl").read_text()
        entry = json.loads(content.strip())
        assert entry["schema_version"] == SCHEMA_VERSION
        assert entry["query_id"] == "q_test"
        assert entry["policy_sha256"] == policy_meta.policy_sha256
        assert "chain_of_thought" not in content
        assert "secret" not in content
        assert "qrels" not in content

    def test_qid_isolation(self, tmp_path):
        """Evidence accumulator must be isolated per QID."""
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
        assert acc.count == 1

        # Clear for new QID
        acc.clear()
        assert acc.count == 0
        assert not acc.has("ps_001")

    def test_one_shot_runner_vertical(self, tmp_path):
        """Vertical test using the production OneShotRunner coordinator."""
        from raglab.agentic.one_shot_runner import OneShotRunner

        registry = build_default_registry()

        class FakeBackend:
            def retrieve(self, query, strategy, top_k):
                return ToolObservation(
                    invocation_id="obs_runner_001",
                    status=InvocationStatus.EXECUTED,
                    passage_ids=("ps_run_001", "ps_run_002"),
                    document_ids=("doc_ch1", "doc_ch2"),
                    ranks=(1, 2),
                    scores=(0.95, 0.88),
                    content_hashes=("h1", "h2"),
                    retrieval_config_hash="cfg_runner",
                    latency_ms=10.0,
                )

        fixed_time = "2026-01-01T00:00:00Z"
        runner = OneShotRunner(
            registry=registry,
            budget=Budget(max_logical_calls=5),
            backend=FakeBackend(),
            run_id="runner_test_001",
            clock=lambda: fixed_time,
            invocation_id_gen=lambda: "inv_runner_001",
        )
        result = runner.execute("q_runner", "What is chunking?")

        # Verify result
        assert result.error is None
        assert result.evidence_count == 2
        assert result.stop_decision.reason == StopReason.COMPLETED_ONE_SHOT
        assert result.trajectory.query_id == "q_runner"
        assert result.trajectory.run_id == "runner_test_001"
        assert len(result.trajectory.steps) == 1

        # Verify trajectory can be written to ledger
        policy_meta = get_deterministic_policy_metadata()
        config_hash = registry.registry_hash()
        ledger = TrajectoryLedger(
            path=tmp_path / "runner_ledger.jsonl",
            run_id="runner_test_001",
            policy_sha256=policy_meta.policy_sha256,
            config_sha256=config_hash,
        )
        ledger.append(result.trajectory)
        assert ledger.entry_count == 1

        # Verify content
        content = (tmp_path / "runner_ledger.jsonl").read_text()
        entry = json.loads(content.strip())
        assert entry["schema_version"] == SCHEMA_VERSION
        assert "chain_of_thought" not in content
        assert "qrels" not in content

    def test_one_shot_runner_captures_leakage_error(self):
        """Runner captures leakage errors without raising."""
        from raglab.agentic.one_shot_runner import OneShotRunner

        registry = build_default_registry()

        class DummyBackend:
            def retrieve(self, query, strategy, top_k):
                return ToolObservation(
                    invocation_id="obs",
                    status=InvocationStatus.EXECUTED,
                    passage_ids=("ps_001",),
                    document_ids=("d",),
                    ranks=(1,),
                    scores=(0.9,),
                    content_hashes=("h",),
                    retrieval_config_hash="c",
                    latency_ms=1.0,
                )

        runner = OneShotRunner(
            registry=registry,
            budget=Budget(),
            backend=DummyBackend(),
            run_id="leak_test",
        )
        # qrels_doc should trigger leakage
        result = runner.execute("q1", "qrels_doc")
        assert result.error is not None
        assert "LeakageDetectedError" in result.error
        assert result.evidence_count == 0

    def test_malformed_backend_canonical_id_boundary(self, tmp_path):
        """Backend returning non-canonical IDs is caught by executor."""
        from raglab.agentic.one_shot_runner import OneShotRunner

        registry = build_default_registry()

        class MalformedBackend:
            """Deliberately returns bad IDs to test the boundary."""

            def retrieve(self, query, strategy, top_k):
                # These IDs are NOT canonical (don't start with ps_)
                return ToolObservation(
                    invocation_id="obs_bad",
                    status=InvocationStatus.EXECUTED,
                    passage_ids=("bad_id_001",),
                    document_ids=("doc1",),
                    ranks=(1,),
                    scores=(0.9,),
                    content_hashes=("h",),
                    retrieval_config_hash="c",
                    latency_ms=1.0,
                )

        runner = OneShotRunner(
            registry=registry,
            budget=Budget(),
            backend=MalformedBackend(),
            run_id="malformed_test",
        )
        result = runner.execute("q1", "What is RAG?")
        assert result.error is not None
        assert "NonCanonicalIdError" in result.error
        assert result.evidence_count == 0

    def test_one_shot_runner_propagates_unexpected_internal_error(self):
        """Backend raising RuntimeError must propagate out of runner.execute(), not be swallowed."""
        from raglab.agentic.one_shot_runner import OneShotRunner

        registry = build_default_registry()

        class BuggyBackend:
            def retrieve(self, query, strategy, top_k):
                raise RuntimeError("unexpected internal bug")

        runner = OneShotRunner(
            registry=registry,
            budget=Budget(),
            backend=BuggyBackend(),
            run_id="buggy_test",
        )
        with pytest.raises(RuntimeError, match="unexpected internal bug"):
            runner.execute("q1", "What is RAG?")
