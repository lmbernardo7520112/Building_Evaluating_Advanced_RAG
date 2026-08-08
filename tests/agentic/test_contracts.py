"""Tests for agentic domain contracts — serialization, validation, immutability."""

from __future__ import annotations

import json

import pytest

from raglab.agentic.contracts import (
    SCHEMA_VERSION,
    AgentTrajectory,
    EvidenceItem,
    PolicyMetadata,
    RoutingDecision,
    StopDecision,
    ToolArguments,
    ToolObservation,
    ToolSpecification,
    TrajectoryStep,
    is_canonical_passage_id,
)
from raglab.agentic.enums import (
    DecisionCode,
    InvocationStatus,
    StopReason,
    ValidationStatus,
)
from raglab.agentic.errors import NonCanonicalIdError


class TestSchemaVersion:
    def test_schema_is_slice5a(self):
        assert SCHEMA_VERSION == "slice5a_agentic_v1"


class TestCanonicalPassageId:
    def test_valid_canonical(self):
        assert is_canonical_passage_id("ps_123")
        assert is_canonical_passage_id("ps_abc_def")

    def test_invalid_canonical(self):
        assert not is_canonical_passage_id("")
        assert not is_canonical_passage_id("chunk_123")
        assert not is_canonical_passage_id("123_ps_")
        assert not is_canonical_passage_id("passage_123")

    def test_rejects_rank_suffix(self):
        """IDs with _rank are not canonical ps_* IDs."""
        assert not is_canonical_passage_id("node_1_rank")
        assert not is_canonical_passage_id("evidence_rank_2")

    def test_rejects_non_string(self):
        assert not is_canonical_passage_id(123)  # type: ignore[arg-type]


class TestPolicyMetadata:
    def test_valid_creation(self):
        pm = PolicyMetadata(
            policy_id="test_v1",
            policy_version="1.0.0",
            policy_sha256="abc123",
        )
        assert pm.policy_id == "test_v1"
        assert pm.schema_version == SCHEMA_VERSION

    def test_empty_policy_id_rejected(self):
        with pytest.raises(ValueError, match="policy_id"):
            PolicyMetadata(
                policy_id="",
                policy_version="1.0.0",
                policy_sha256="abc123",
            )

    def test_empty_version_rejected(self):
        with pytest.raises(ValueError, match="policy_version"):
            PolicyMetadata(
                policy_id="test",
                policy_version="",
                policy_sha256="abc123",
            )

    def test_empty_hash_rejected(self):
        with pytest.raises(ValueError, match="policy_sha256"):
            PolicyMetadata(
                policy_id="test",
                policy_version="1.0.0",
                policy_sha256="",
            )


class TestRoutingDecision:
    def test_valid_creation(self):
        rd = RoutingDecision(
            schema_version=SCHEMA_VERSION,
            query_id="q1",
            policy_id="det_v1",
            policy_version="1.0.0",
            policy_sha256="hash123",
            selected_strategy="baseline",
            decision_code=DecisionCode.SELECTED,
            public_features_used=("query_class:LOCALIZED_FACT",),
            validation_status=ValidationStatus.VALID,
        )
        assert rd.selected_strategy == "baseline"
        assert rd.fallback_used is False

    def test_schema_mismatch_rejected(self):
        with pytest.raises(ValueError, match="Schema mismatch"):
            RoutingDecision(
                schema_version="wrong_schema",
                query_id="q1",
                policy_id="det_v1",
                policy_version="1.0.0",
                policy_sha256="hash",
                selected_strategy="baseline",
                decision_code=DecisionCode.SELECTED,
                public_features_used=(),
                validation_status=ValidationStatus.VALID,
            )

    def test_empty_query_id_rejected(self):
        with pytest.raises(ValueError, match="query_id"):
            RoutingDecision(
                schema_version=SCHEMA_VERSION,
                query_id="",
                policy_id="det_v1",
                policy_version="1.0.0",
                policy_sha256="hash",
                selected_strategy="baseline",
                decision_code=DecisionCode.SELECTED,
                public_features_used=(),
                validation_status=ValidationStatus.VALID,
            )

    def test_confidence_out_of_range(self):
        with pytest.raises(ValueError, match="confidence"):
            RoutingDecision(
                schema_version=SCHEMA_VERSION,
                query_id="q1",
                policy_id="det_v1",
                policy_version="1.0.0",
                policy_sha256="hash",
                selected_strategy="baseline",
                decision_code=DecisionCode.SELECTED,
                public_features_used=(),
                validation_status=ValidationStatus.VALID,
                confidence=1.5,
            )


class TestToolSpecification:
    def test_valid_read_only(self):
        spec = ToolSpecification(
            tool_id="retrieve_baseline",
            version="1.0.0",
            description="Retrieve using baseline",
            read_only=True,
            network_access=False,
            deterministic=True,
            max_top_k=10,
            timeout_seconds=30.0,
            allowed_strategies=("baseline",),
            implementation_sha256="hash",
        )
        assert spec.read_only is True

    def test_write_tool_rejected(self):
        with pytest.raises(ValueError, match="read-only"):
            ToolSpecification(
                tool_id="write_tool",
                version="1.0.0",
                description="writes",
                read_only=False,
                network_access=False,
                deterministic=True,
                max_top_k=10,
                timeout_seconds=30.0,
                allowed_strategies=("baseline",),
                implementation_sha256="hash",
            )

    def test_network_tool_rejected(self):
        with pytest.raises(ValueError, match="network"):
            ToolSpecification(
                tool_id="net_tool",
                version="1.0.0",
                description="needs net",
                read_only=True,
                network_access=True,
                deterministic=True,
                max_top_k=10,
                timeout_seconds=30.0,
                allowed_strategies=("baseline",),
                implementation_sha256="hash",
            )

    def test_invalid_top_k(self):
        with pytest.raises(ValueError, match="max_top_k"):
            ToolSpecification(
                tool_id="t",
                version="1.0.0",
                description="d",
                read_only=True,
                network_access=False,
                deterministic=True,
                max_top_k=0,
                timeout_seconds=30.0,
                allowed_strategies=("baseline",),
                implementation_sha256="hash",
            )


class TestToolArguments:
    def test_valid_args(self):
        args = ToolArguments(query="What is RAG?", strategy="baseline", top_k=3)
        assert args.sha256  # non-empty hash

    def test_empty_query_rejected(self):
        with pytest.raises(ValueError, match="query"):
            ToolArguments(query="", strategy="baseline", top_k=3)

    def test_whitespace_query_rejected(self):
        with pytest.raises(ValueError, match="query"):
            ToolArguments(query="   ", strategy="baseline", top_k=3)

    def test_query_too_long(self):
        with pytest.raises(ValueError, match="exceeds maximum"):
            ToolArguments(query="x" * 10001, strategy="baseline", top_k=3)

    def test_invalid_top_k(self):
        with pytest.raises(ValueError, match="top_k"):
            ToolArguments(query="test", strategy="baseline", top_k=0)

    def test_deterministic_serialization(self):
        a1 = ToolArguments(query="test", strategy="baseline", top_k=3)
        a2 = ToolArguments(query="test", strategy="baseline", top_k=3)
        assert a1.sha256 == a2.sha256

    def test_different_args_different_hash(self):
        a1 = ToolArguments(query="test", strategy="baseline", top_k=3)
        a2 = ToolArguments(query="test", strategy="baseline", top_k=5)
        assert a1.sha256 != a2.sha256


class TestToolObservation:
    def test_canonical_ids_required(self):
        with pytest.raises(NonCanonicalIdError):
            ToolObservation(
                invocation_id="inv1",
                status=InvocationStatus.EXECUTED,
                passage_ids=("chunk_123",),  # not canonical!
                document_ids=("doc1",),
                ranks=(1,),
                scores=(0.9,),
                content_hashes=("hash1",),
                retrieval_config_hash="cfg_hash",
                latency_ms=100.0,
            )

    def test_valid_canonical_ids(self):
        obs = ToolObservation(
            invocation_id="inv1",
            status=InvocationStatus.EXECUTED,
            passage_ids=("ps_001", "ps_002"),
            document_ids=("doc1", "doc2"),
            ranks=(1, 2),
            scores=(0.9, 0.8),
            content_hashes=("h1", "h2"),
            retrieval_config_hash="cfg",
            latency_ms=50.0,
        )
        assert len(obs.passage_ids) == 2


class TestEvidenceItem:
    def test_non_canonical_rejected(self):
        with pytest.raises(NonCanonicalIdError):
            EvidenceItem(
                passage_id="bad_id",
                document_id="doc1",
                rank=1,
                score=0.9,
                content_sha256="hash",
                source_tool_id="tool1",
                source_invocation_id="inv1",
            )

    def test_valid_canonical(self):
        item = EvidenceItem(
            passage_id="ps_001",
            document_id="doc1",
            rank=1,
            score=0.9,
            content_sha256="hash",
            source_tool_id="tool1",
            source_invocation_id="inv1",
        )
        assert item.is_new is True


class TestTrajectoryStep:
    def test_deterministic_serialization(self):
        step = TrajectoryStep(
            step_index=0,
            state_before_hash="before",
            action="retrieve",
            arguments_sha256="args_hash",
            observation_hash="obs_hash",
            evidence_delta_count=3,
            decision_code=DecisionCode.SELECTED,
            state_after_hash="after",
            budget_remaining={"logical_calls": 9},
        )
        d = step.to_dict()
        assert d["step_index"] == 0
        assert d["decision_code"] == "SELECTED"
        assert "stop_reason" not in d

    def test_with_stop_reason(self):
        step = TrajectoryStep(
            step_index=0,
            state_before_hash="b",
            action="stop",
            arguments_sha256="a",
            observation_hash="o",
            evidence_delta_count=0,
            decision_code=DecisionCode.ABSTAIN,
            state_after_hash="a",
            budget_remaining={},
            stop_reason=StopReason.COMPLETED_ONE_SHOT,
        )
        d = step.to_dict()
        assert d["stop_reason"] == "COMPLETED_ONE_SHOT"


class TestStopDecision:
    def test_empty_detail_rejected(self):
        with pytest.raises(ValueError, match="detail"):
            StopDecision(
                reason=StopReason.NO_EVIDENCE,
                detail="",
                evidence_count=0,
                budget_remaining={},
            )


class TestAgentTrajectory:
    def test_deterministic_hash(self):
        rd = RoutingDecision(
            schema_version=SCHEMA_VERSION,
            query_id="q1",
            policy_id="det_v1",
            policy_version="1.0.0",
            policy_sha256="hash",
            selected_strategy="baseline",
            decision_code=DecisionCode.SELECTED,
            public_features_used=("test",),
            validation_status=ValidationStatus.VALID,
        )
        sd = StopDecision(
            reason=StopReason.COMPLETED_ONE_SHOT,
            detail="done",
            evidence_count=1,
            budget_remaining={"logical_calls": 9},
        )
        t1 = AgentTrajectory(
            schema_version=SCHEMA_VERSION,
            run_id="run1",
            query_id="q1",
            policy_id="det_v1",
            policy_sha256="hash",
            config_sha256="cfg",
            steps=(),
            stop_decision=sd,
            routing_decision=rd,
            created_at="2026-01-01T00:00:00Z",
        )
        t2 = AgentTrajectory(
            schema_version=SCHEMA_VERSION,
            run_id="run1",
            query_id="q1",
            policy_id="det_v1",
            policy_sha256="hash",
            config_sha256="cfg",
            steps=(),
            stop_decision=sd,
            routing_decision=rd,
            created_at="2026-01-01T00:00:00Z",
        )
        assert t1.sha256 == t2.sha256

    def test_no_chain_of_thought_in_dict(self):
        """Verify serialized trajectory has no chain-of-thought fields."""
        rd = RoutingDecision(
            schema_version=SCHEMA_VERSION,
            query_id="q1",
            policy_id="p1",
            policy_version="1.0",
            policy_sha256="h",
            selected_strategy="baseline",
            decision_code=DecisionCode.SELECTED,
            public_features_used=(),
            validation_status=ValidationStatus.VALID,
        )
        sd = StopDecision(
            reason=StopReason.COMPLETED_ONE_SHOT,
            detail="ok",
            evidence_count=0,
            budget_remaining={},
        )
        t = AgentTrajectory(
            schema_version=SCHEMA_VERSION,
            run_id="r",
            query_id="q1",
            policy_id="p1",
            policy_sha256="h",
            config_sha256="c",
            steps=(),
            stop_decision=sd,
            routing_decision=rd,
            created_at="2026-01-01T00:00:00Z",
        )
        serialized = json.dumps(t.to_dict())
        assert "chain_of_thought" not in serialized
        assert "reasoning" not in serialized.lower()
        assert "secret" not in serialized
        assert "credential" not in serialized
        assert "api_key" not in serialized
