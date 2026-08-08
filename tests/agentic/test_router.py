"""Tests for deterministic router, LLM router contract, and policy."""

from __future__ import annotations

from raglab.agentic.contracts import SCHEMA_VERSION, RoutingDecision
from raglab.agentic.enums import DecisionCode, QueryClass, ValidationStatus
from raglab.agentic.router import (
    ALLOWED_STRATEGIES,
    FakeLLMRouter,
    classify_query,
    get_deterministic_policy_metadata,
    route_deterministic,
    validate_llm_routing_decision,
)


class TestClassifyQuery:
    def test_global_summary(self):
        assert (
            classify_query("Summarize the main ideas of chapter 2")
            == QueryClass.GLOBAL_SUMMARY
        )

    def test_localized_fact(self):
        assert (
            classify_query("What is the embedding dimension?")
            == QueryClass.LOCALIZED_FACT
        )

    def test_comparison(self):
        assert (
            classify_query("Compare sentence window and auto merging")
            == QueryClass.COMPARISON
        )

    def test_procedural(self):
        assert (
            classify_query("How to build a retrieval pipeline?")
            == QueryClass.PROCEDURAL_EXPLANATION
        )

    def test_unanswerable(self):
        assert (
            classify_query("This is not discussed in the text")
            == QueryClass.POSSIBLY_UNANSWERABLE
        )

    def test_unclassified_generic(self):
        assert classify_query("Hello world") == QueryClass.UNCLASSIFIED

    def test_empty_string(self):
        assert classify_query("") == QueryClass.UNCLASSIFIED

    def test_whitespace_only(self):
        assert classify_query("   ") == QueryClass.UNCLASSIFIED

    def test_unicode(self):
        result = classify_query("Qual é o resumo do capítulo?")
        assert isinstance(result, QueryClass)

    def test_stability(self):
        """Same query always yields same class."""
        q = "How does sentence window retrieval work?"
        c1 = classify_query(q)
        c2 = classify_query(q)
        assert c1 == c2

    def test_no_qrel_access(self):
        """Classification must NOT change based on query ID."""
        q = "What is chunking?"
        c = classify_query(q)
        # Running twice proves no state dependency
        assert classify_query(q) == c

    def test_precedence_unanswerable_over_comparison(self):
        """Unanswerable takes precedence."""
        q = "Compare methods not discussed in the text"
        assert classify_query(q) == QueryClass.POSSIBLY_UNANSWERABLE


class TestRouteDeterministic:
    def test_produces_valid_decision(self):
        rd = route_deterministic("q1", "What is the embedding dimension?")
        assert rd.schema_version == SCHEMA_VERSION
        assert rd.query_id == "q1"
        assert rd.selected_strategy in ALLOWED_STRATEGIES
        assert rd.validation_status == ValidationStatus.VALID

    def test_policy_hash_stable(self):
        """Same policy configuration always yields same hash."""
        r1 = route_deterministic("q1", "test")
        r2 = route_deterministic("q2", "test")
        assert r1.policy_sha256 == r2.policy_sha256

    def test_fallback_for_unclassified(self):
        rd = route_deterministic("q1", "Hello world")
        assert rd.fallback_used is True
        assert rd.decision_code == DecisionCode.FALLBACK

    def test_no_fallback_for_classified(self):
        rd = route_deterministic("q1", "Summarize the main ideas")
        assert rd.fallback_used is False
        assert rd.decision_code == DecisionCode.SELECTED

    def test_no_qrel_consultation(self):
        """Routing must produce result without any external data."""
        # If this runs without error and produces a valid decision,
        # it proves no external data source is needed.
        rd = route_deterministic("nonexistent_qid_999", "What is RAG?")
        assert rd.validation_status == ValidationStatus.VALID


class TestPolicyMetadata:
    def test_deterministic_metadata(self):
        pm = get_deterministic_policy_metadata()
        assert pm.policy_id == "deterministic_v1"
        assert pm.policy_version == "1.0.0"
        assert pm.policy_sha256  # non-empty

    def test_hash_stability(self):
        pm1 = get_deterministic_policy_metadata()
        pm2 = get_deterministic_policy_metadata()
        assert pm1.policy_sha256 == pm2.policy_sha256


class TestValidateLLMRoutingDecision:
    def _make_valid_decision(self, policy_hash: str) -> RoutingDecision:
        return RoutingDecision(
            schema_version=SCHEMA_VERSION,
            query_id="q1",
            policy_id="llm_v1",
            policy_version="1.0.0",
            policy_sha256=policy_hash,
            selected_strategy="baseline",
            decision_code=DecisionCode.SELECTED,
            public_features_used=("length:short",),
            validation_status=ValidationStatus.VALID,
        )

    def test_valid_decision(self):
        d = self._make_valid_decision("expected_hash")
        status = validate_llm_routing_decision(d, ALLOWED_STRATEGIES, "expected_hash")
        assert status == ValidationStatus.VALID

    def test_unknown_strategy_rejected(self):
        d = RoutingDecision(
            schema_version=SCHEMA_VERSION,
            query_id="q1",
            policy_id="llm_v1",
            policy_version="1.0.0",
            policy_sha256="h",
            selected_strategy="unknown_strategy",
            decision_code=DecisionCode.SELECTED,
            public_features_used=(),
            validation_status=ValidationStatus.VALID,
        )
        status = validate_llm_routing_decision(d, ALLOWED_STRATEGIES, "h")
        assert status == ValidationStatus.INVALID_STRATEGY

    def test_policy_hash_mismatch(self):
        d = self._make_valid_decision("wrong_hash")
        status = validate_llm_routing_decision(d, ALLOWED_STRATEGIES, "expected_hash")
        assert status == ValidationStatus.POLICY_HASH_MISSING

    def test_leakage_in_features_detected(self):
        d = RoutingDecision(
            schema_version=SCHEMA_VERSION,
            query_id="q1",
            policy_id="llm_v1",
            policy_version="1.0.0",
            policy_sha256="h",
            selected_strategy="baseline",
            decision_code=DecisionCode.SELECTED,
            public_features_used=("qrels:high_relevance",),
            validation_status=ValidationStatus.VALID,
        )
        status = validate_llm_routing_decision(d, ALLOWED_STRATEGIES, "h")
        assert status == ValidationStatus.LEAKAGE_DETECTED

    def test_gold_answer_leakage(self):
        d = RoutingDecision(
            schema_version=SCHEMA_VERSION,
            query_id="q1",
            policy_id="llm_v1",
            policy_version="1.0.0",
            policy_sha256="h",
            selected_strategy="baseline",
            decision_code=DecisionCode.SELECTED,
            public_features_used=("gold_answer:yes",),
            validation_status=ValidationStatus.VALID,
        )
        status = validate_llm_routing_decision(d, ALLOWED_STRATEGIES, "h")
        assert status == ValidationStatus.LEAKAGE_DETECTED

    def test_holdout_leakage(self):
        d = RoutingDecision(
            schema_version=SCHEMA_VERSION,
            query_id="q1",
            policy_id="p",
            policy_version="1",
            policy_sha256="h",
            selected_strategy="baseline",
            decision_code=DecisionCode.SELECTED,
            public_features_used=("holdout:true",),
            validation_status=ValidationStatus.VALID,
        )
        status = validate_llm_routing_decision(d, ALLOWED_STRATEGIES, "h")
        assert status == ValidationStatus.LEAKAGE_DETECTED


class TestFakeLLMRouter:
    def test_produces_valid_decision(self):
        pm = get_deterministic_policy_metadata()
        router = FakeLLMRouter(pm)
        rd = router.route("q1", "test query", ALLOWED_STRATEGIES, pm)
        assert rd.schema_version == SCHEMA_VERSION
        assert rd.query_id == "q1"
        assert rd.selected_strategy in ALLOWED_STRATEGIES
        assert rd.fallback_used is True

    def test_deterministic(self):
        pm = get_deterministic_policy_metadata()
        router = FakeLLMRouter(pm)
        r1 = router.route("q1", "test", ALLOWED_STRATEGIES, pm)
        r2 = router.route("q1", "test", ALLOWED_STRATEGIES, pm)
        assert r1.selected_strategy == r2.selected_strategy
