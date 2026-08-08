"""Routing — deterministic router and LLM router contract.

The deterministic router classifies queries using only public text features.
The LLM router contract defines validation for one-shot structured output,
but does NOT execute any external model in this Slice 5A.
"""

from __future__ import annotations

import re
from typing import Protocol

from raglab.agentic.contracts import (
    SCHEMA_VERSION,
    PolicyMetadata,
    RoutingDecision,
    _canonical_json,
    _sha256,
)
from raglab.agentic.enums import DecisionCode, QueryClass, ValidationStatus

# Allowed strategies for routing — mirrors PipelineStrategy enum values
ALLOWED_STRATEGIES: frozenset[str] = frozenset(
    {
        "baseline",
        "sentence_anchor",
        "sentence_window",
        "sentence_window_rerank",
        "hierarchical_leaf",
        "auto_merging",
        "auto_merging_rerank",
    }
)

# Leakage detection patterns — terms that must NEVER appear in routing input
_LEAKAGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bqrel[s]?\b", re.IGNORECASE),
    re.compile(r"\bgold[_ ]?answer\b", re.IGNORECASE),
    re.compile(r"\brelevant[_ ]?page[s]?\b", re.IGNORECASE),
    re.compile(r"\bholdout\b", re.IGNORECASE),
)

_POLICY_ID = "deterministic_v1"
_POLICY_VERSION = "1.0.0"

# Classification rules — order matters: first match wins.
# These operate ONLY on the public question text.
_COMPARISON_PATTERN = re.compile(
    r"\b(compar\w*|differ\w*|versus|vs\.?|contrast\w*|advantage|disadvantage)\b",
    re.IGNORECASE,
)
_GLOBAL_SUMMARY_PATTERN = re.compile(
    r"\b(summar\w*|overview|main\s+(idea|point|concept|theme)|"
    r"what\s+is\s+the\s+(book|chapter|text)\s+about|"
    r"explain\s+the\s+(overall|general|broad))\b",
    re.IGNORECASE,
)
_PROCEDURAL_PATTERN = re.compile(
    r"\b(how\s+(to|do|does|can|should|would)|step[s]?\s+(to|for|in)|"
    r"process\s+(of|for)|procedure|method\s+for|approach\s+to)\b",
    re.IGNORECASE,
)
_UNANSWERABLE_PATTERN = re.compile(
    r"\b(unanswerable|cannot\s+be\s+answered|not\s+(addressed|discussed|"
    r"mentioned|covered)\s+(in|by)\s+the\s+(text|book|chapter|document))\b",
    re.IGNORECASE,
)

# Strategy mapping per query class
_STRATEGY_MAP: dict[str, str] = {
    QueryClass.GLOBAL_SUMMARY.value: "auto_merging",
    QueryClass.LOCALIZED_FACT.value: "sentence_window_rerank",
    QueryClass.COMPARISON.value: "sentence_window_rerank",
    QueryClass.PROCEDURAL_EXPLANATION.value: "sentence_window",
    QueryClass.POSSIBLY_UNANSWERABLE.value: "baseline",
    QueryClass.UNCLASSIFIED.value: "sentence_window_rerank",
}


def _compute_deterministic_policy_hash() -> str:
    """Compute the frozen hash for the deterministic router."""
    payload = {
        "policy_id": _POLICY_ID,
        "policy_version": _POLICY_VERSION,
        "rules": _STRATEGY_MAP,
        "fallback_strategy": _STRATEGY_MAP[QueryClass.UNCLASSIFIED.value],
    }
    return _sha256(_canonical_json(payload))


_POLICY_HASH = _compute_deterministic_policy_hash()


def classify_query(query_text: str) -> QueryClass:
    """Classify a query using only its public text.

    Rules are applied in precedence order. No access to qrels,
    gold answers, evaluation results, or historical QID performance.
    """
    if not query_text or not query_text.strip():
        return QueryClass.UNCLASSIFIED

    text = query_text.strip()

    if _UNANSWERABLE_PATTERN.search(text):
        return QueryClass.POSSIBLY_UNANSWERABLE
    if _COMPARISON_PATTERN.search(text):
        return QueryClass.COMPARISON
    if _GLOBAL_SUMMARY_PATTERN.search(text):
        return QueryClass.GLOBAL_SUMMARY
    if _PROCEDURAL_PATTERN.search(text):
        return QueryClass.PROCEDURAL_EXPLANATION

    # Default: assume localized fact for specific questions
    if text.endswith("?") or text.lower().startswith(
        ("what", "where", "when", "who", "which")
    ):
        return QueryClass.LOCALIZED_FACT

    return QueryClass.UNCLASSIFIED


def get_deterministic_policy_metadata() -> PolicyMetadata:
    """Return frozen metadata for the deterministic router."""
    return PolicyMetadata(
        policy_id=_POLICY_ID,
        policy_version=_POLICY_VERSION,
        policy_sha256=_POLICY_HASH,
    )


def route_deterministic(query_id: str, query_text: str) -> RoutingDecision:
    """Route a query using the deterministic policy.

    This router:
    - uses ONLY the public query text
    - does NOT consult qrels, gold answers, or evaluation results
    - is reproducible and hashable
    - has an explicit fallback
    """
    query_class = classify_query(query_text)
    strategy = _STRATEGY_MAP[query_class.value]
    is_fallback = query_class == QueryClass.UNCLASSIFIED

    return RoutingDecision(
        schema_version=SCHEMA_VERSION,
        query_id=query_id,
        policy_id=_POLICY_ID,
        policy_version=_POLICY_VERSION,
        policy_sha256=_POLICY_HASH,
        selected_strategy=strategy,
        decision_code=DecisionCode.FALLBACK if is_fallback else DecisionCode.SELECTED,
        public_features_used=(f"query_class:{query_class.value}",),
        validation_status=ValidationStatus.VALID,
        fallback_used=is_fallback,
    )


# ---------------------------------------------------------------------------
# LLM Router — contract and validation only (no model execution in 5A)
# ---------------------------------------------------------------------------


class LLMRouterPort(Protocol):
    """Port for an LLM-based one-shot router.

    In Slice 5A, only the contract and validation are implemented.
    No external model is called.
    """

    def route(
        self,
        query_id: str,
        query_text: str,
        allowed_strategies: frozenset[str],
        policy_metadata: PolicyMetadata,
    ) -> RoutingDecision:
        """Produce a structured routing decision."""
        ...


def validate_llm_routing_decision(
    decision: RoutingDecision,
    allowed_strategies: frozenset[str],
    expected_policy_sha256: str,
) -> ValidationStatus:
    """Validate an LLM router's output against governance rules.

    Checks:
    - schema version
    - strategy in allowlist
    - required fields present
    - no forbidden fields
    - policy hash matches
    - confidence in valid range (if present)
    - no leakage terms in features
    """
    if decision.schema_version != SCHEMA_VERSION:
        return ValidationStatus.SCHEMA_MISMATCH

    if not decision.query_id:
        return ValidationStatus.MISSING_REQUIRED_FIELD

    if not decision.policy_sha256:
        return ValidationStatus.POLICY_HASH_MISSING

    if decision.policy_sha256 != expected_policy_sha256:
        return ValidationStatus.POLICY_HASH_MISSING

    if decision.selected_strategy not in allowed_strategies:
        return ValidationStatus.INVALID_STRATEGY

    if decision.confidence is not None and not (0.0 <= decision.confidence <= 1.0):
        return ValidationStatus.CONFIDENCE_OUT_OF_RANGE

    # Anti-leakage check on features
    for feat in decision.public_features_used:
        for pattern in _LEAKAGE_PATTERNS:
            if pattern.search(feat):
                return ValidationStatus.LEAKAGE_DETECTED

    return ValidationStatus.VALID


class FakeLLMRouter:
    """Deterministic fake for testing the LLM router contract.

    Always selects sentence_window_rerank with FALLBACK decision.
    Does NOT represent a real LLM capability.
    """

    def __init__(self, policy_metadata: PolicyMetadata) -> None:
        self._metadata = policy_metadata

    def route(
        self,
        query_id: str,
        query_text: str,
        allowed_strategies: frozenset[str],
        policy_metadata: PolicyMetadata,
    ) -> RoutingDecision:
        """Return a deterministic routing decision for testing."""
        fallback = "sentence_window_rerank"
        if fallback not in allowed_strategies:
            # Use first available strategy
            fallback = sorted(allowed_strategies)[0]

        return RoutingDecision(
            schema_version=SCHEMA_VERSION,
            query_id=query_id,
            policy_id=policy_metadata.policy_id,
            policy_version=policy_metadata.policy_version,
            policy_sha256=policy_metadata.policy_sha256,
            selected_strategy=fallback,
            decision_code=DecisionCode.FALLBACK,
            public_features_used=("fake_llm_router",),
            validation_status=ValidationStatus.VALID,
            fallback_used=True,
        )
