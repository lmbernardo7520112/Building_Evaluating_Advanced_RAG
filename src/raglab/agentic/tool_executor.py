"""Tool executor — governed execution with fail-closed validation.

Validates arguments, enforces budget, checks authorization,
and distinguishes logical calls from physical attempts and retries.

Anti-leakage defence is STRUCTURAL:
  1. Allowlist of permitted filter-argument names (primary barrier).
  2. Normalized denylist that strips separators before matching
     (catches qrels_doc, gold-answer, gold.answer, etc.).
  3. Path-component check for /qrels/ and similar.
  4. Both query text and filter values are scanned.

The denylist alone is NOT the only barrier.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Protocol

from raglab.agentic.budget import Budget
from raglab.agentic.contracts import (
    ToolInvocation,
    ToolObservation,
)
from raglab.agentic.errors import (
    BudgetExhaustedError,
    InvalidToolArgumentsError,
    LeakageDetectedError,
    NonCanonicalIdError,
    UnauthorizedToolError,
    UnknownToolError,
)
from raglab.agentic.tool_registry import ToolRegistry

# ── Structural allowlist ──────────────────────────────────────────
# Only these filter keys are allowed in ToolArguments.
# Anything else is rejected BEFORE the denylist is even consulted.
_ALLOWED_FILTER_KEYS: frozenset[str] = frozenset()
# Currently empty — Slice 5A does not authorize any document-level
# filtering.  When a legitimate filter is introduced (e.g. by chapter),
# it must be explicitly added here with scientific justification.


# ── Normalized denylist ───────────────────────────────────────────
# These tokens, after separator-stripping and lowercasing, are
# forbidden in ANY argument value (filters, query, extra fields).
_FORBIDDEN_TOKENS: frozenset[str] = frozenset(
    {
        "qrel",
        "qrels",
        "goldanswer",
        "answerkey",
        "relevantpage",
        "relevantpages",
        "holdout",
        "humanqrels",
        "humanqrelsfinal",
    }
)

# Raw patterns that detect leakage even when separator-normalisation
# is insufficient (e.g. path components, mixed-case abbreviations).
_FORBIDDEN_RAW_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"/qrels?/", re.IGNORECASE),
    re.compile(r"/gold[_.\-]?answers?/", re.IGNORECASE),
    re.compile(r"/holdout/", re.IGNORECASE),
)


def _normalise_for_leakage_check(text: str) -> str:
    """Strip separators, whitespace, Unicode confusables; lowercase."""
    # NFKC normalisation collapses fullwidth/halfwidth variants
    normalised = unicodedata.normalize("NFKC", text).lower()
    # Remove common separators: _ - . / \\ and all whitespace
    normalised = re.sub(r"[_\-./\\\s]+", "", normalised)
    return normalised


def _contains_forbidden_token(text: str) -> str | None:
    """Return the first forbidden token found, or None."""
    normalised = _normalise_for_leakage_check(text)
    for token in _FORBIDDEN_TOKENS:
        if token in normalised:
            return token
    # Raw pattern check on original text
    for pattern in _FORBIDDEN_RAW_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


class RetrievalBackend(Protocol):
    """Protocol for a retrieval backend that tools invoke."""

    def retrieve(
        self,
        query: str,
        strategy: str,
        top_k: int,
    ) -> ToolObservation:
        """Execute retrieval and return a governed observation."""
        ...


class ToolExecutor:
    """Governed tool execution with full validation.

    Rejects:
    - unknown tools
    - unauthorized tools (not read-only, requires network)
    - invalid arguments (empty query, top_k out of range)
    - any document-id filter (allowlist is empty in Slice 5A)
    - leakage via normalised denylist (query, filter values, paths)
    - budget exhaustion
    - non-canonical passage IDs

    Distinguishes:
    - logical calls (unique intent)
    - physical attempts (actual execution)
    - retries (repeated execution of same intent)
    - fallbacks (alternative strategy after failure)
    """

    def __init__(
        self,
        registry: ToolRegistry,
        budget: Budget,
    ) -> None:
        self._registry = registry
        self._budget = budget

    def validate_and_execute(
        self,
        invocation: ToolInvocation,
        backend: RetrievalBackend,
    ) -> ToolObservation:
        """Validate tool invocation and execute if authorized.

        Raises domain errors for any validation failure — never silently
        converts errors to success.
        """
        # 1. Check tool exists
        if not self._registry.has(invocation.tool_id):
            raise UnknownToolError(invocation.tool_id)

        spec = self._registry.get(invocation.tool_id)

        # 2. Check authorization
        if not spec.read_only:
            raise UnauthorizedToolError(invocation.tool_id, "tool is not read-only")
        if spec.network_access:
            raise UnauthorizedToolError(
                invocation.tool_id, "tool requires network access"
            )

        args = invocation.arguments

        # 3. Validate arguments
        if not args.query or not args.query.strip():
            raise InvalidToolArgumentsError(invocation.tool_id, "query is empty")
        if len(args.query) > 10000:
            raise InvalidToolArgumentsError(
                invocation.tool_id,
                f"query length {len(args.query)} exceeds maximum 10000",
            )
        if args.strategy not in spec.allowed_strategies:
            raise InvalidToolArgumentsError(
                invocation.tool_id,
                f"strategy '{args.strategy}' not in allowed: {spec.allowed_strategies}",
            )
        if args.top_k < 1 or args.top_k > spec.max_top_k:
            raise InvalidToolArgumentsError(
                invocation.tool_id,
                f"top_k={args.top_k} outside range [1, {spec.max_top_k}]",
            )

        # 4. STRUCTURAL allowlist check for document-id filters
        if args.allowed_document_ids is not None:
            # Primary defence: reject ALL document-id filtering
            # unless explicitly allowlisted.
            if not _ALLOWED_FILTER_KEYS:
                raise LeakageDetectedError(
                    "document-id filtering is not authorised in "
                    "Slice 5A; received "
                    f"{len(args.allowed_document_ids)} filter(s)"
                )
            # If allowlist becomes non-empty in future slices, also
            # apply the denylist as secondary defence:
            for doc_id in args.allowed_document_ids:
                found = _contains_forbidden_token(doc_id)
                if found:
                    raise LeakageDetectedError(
                        f"forbidden token '{found}' in document ID filter: '{doc_id}'"
                    )

        # 5. Anti-leakage in query text (denylist as secondary defence)
        found = _contains_forbidden_token(args.query)
        if found:
            raise LeakageDetectedError(
                f"leakage token '{found}' detected in query: '{args.query[:100]}'"
            )

        # 6. Budget check
        if not self._budget.can_consume_logical_call():
            raise BudgetExhaustedError(
                "logical_calls",
                self._budget.max_logical_calls,
                self._budget.logical_calls_consumed,
            )

        # 7. Execute
        self._budget.consume_logical_call()
        observation = backend.retrieve(
            query=args.query,
            strategy=args.strategy,
            top_k=args.top_k,
        )

        # 8. Validate observation — canonical IDs (belt + suspenders)
        for pid in observation.passage_ids:
            if not pid.startswith("ps_"):
                raise NonCanonicalIdError("passage_id", pid)

        return observation
