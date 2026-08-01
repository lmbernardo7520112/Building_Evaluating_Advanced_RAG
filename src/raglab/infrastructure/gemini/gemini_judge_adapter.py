"""Gemini judge adapter — RAG Triad + Factual Correctness evaluation.

SECURITY BOUNDARY:
    This adapter reads GEMINI_API_KEY from the environment.
    It MUST NOT be instantiated by Antigravity IDE.
    It MUST only be instantiated in the human-operated isolated terminal.
    See: docs/security/credential_boundary.md

Execution environment: Ambiente B (human terminal only).

This adapter implements EvaluationPort via structural subtyping (Protocol).

Provider:  google_gemini
Model:     gemini-3.1-flash-lite (independently configurable from generator)
SDK:       google-genai >= 1.0
Temp:      0.0 (deterministic evaluation)

RAG Triad dimensions evaluated:
  1. Context Relevance:    Is the context relevant to the query?
  2. Groundedness:         Is the answer grounded in the context?
  3. Answer Relevance:     Does the answer address the query?
  4. Factual Correctness:  Does the answer match a gold reference? (optional)
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Sequence
from typing import Final

from raglab.domain.entities import (
    EvaluationResult,
    GeneratedAnswer,
    RetrievedEvidence,
)
from raglab.domain.enums import PipelineStrategy
from raglab.domain.quota import QuotaManager
from raglab.domain.retry import NonRetryableError, RetryExhaustedError, RetryPolicy
from raglab.domain.value_objects import MetricResult
from raglab.infrastructure.gemini.prompts import (
    JUDGE_SYSTEM,
    build_answer_relevance_prompt,
    build_context_relevance_prompt,
    build_factual_correctness_prompt,
    build_groundedness_prompt,
)

logger = logging.getLogger(__name__)

_DEFAULT_JUDGE_MODEL: Final[str] = "gemini-3.1-flash-lite"
_CREDENTIAL_ENV: Final[str] = "GEMINI_API_KEY"
_SCORE_KEY: Final[str] = "score"
_REASONING_KEY: Final[str] = "reasoning"


class GeminiJudgeAdapter:
    """RAG Triad + Factual Correctness judge using Gemini API.

    SECURITY REQUIREMENTS:
    1. GEMINI_API_KEY must be present in environment.
    2. Key is NEVER logged, stored in checkpoints, or passed to other adapters.
    3. Quota limits are enforced before every call.
    4. Each RAG Triad dimension is a separate API call.
    5. 429 errors trigger exponential backoff via RetryPolicy.
    6. Non-retryable errors (400, 403) are raised immediately.
    7. LangSmith is DISABLED.
    8. The judge model MAY differ from the generator model.

    Judge MUST be independent from generator to prevent contamination.
    """

    def __init__(
        self,
        judge_model_id: str = _DEFAULT_JUDGE_MODEL,
        strategy: PipelineStrategy = PipelineStrategy.BASELINE,
        quota_manager: QuotaManager | None = None,
        retry_policy: RetryPolicy | None = None,
        temperature: float = 0.0,
    ) -> None:
        self._judge_model_id = judge_model_id
        self._strategy = strategy
        self._quota = quota_manager or QuotaManager()
        self._retry = retry_policy or RetryPolicy()
        self._temperature = temperature
        self._client = self._init_client()

    def _init_client(self) -> object:
        """Initialize Gemini client from environment credential."""
        api_key = os.environ.get(_CREDENTIAL_ENV)
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found in environment. "
                "This adapter must be executed in an isolated human terminal. "
                "See docs/security/credential_boundary.md"
            )
        import google.genai as genai
        return genai.Client(api_key=api_key)

    @property
    def judge_model_id(self) -> str:
        return self._judge_model_id

    def evaluate(
        self,
        query_id: str,
        query: str,
        answer: GeneratedAnswer,
        evidence: Sequence[RetrievedEvidence],
        *,
        gold_answer: str | None = None,
    ) -> EvaluationResult:
        """Evaluate one answer using the full RAG Triad.

        Makes 3 Gemini API calls (4 if gold_answer is provided).
        Each call is independently quota-managed and retry-wrapped.

        Args:
            query_id: Traceability identifier.
            query: The original user query.
            answer: The generated answer to evaluate.
            evidence: Retrieved evidence used for generation.
            gold_answer: Optional reference answer for factual correctness.

        Returns:
            EvaluationResult with metrics:
              context_relevance, groundedness, answer_relevance,
              [factual_correctness if gold_answer], judge_model, langsmith_disabled.
        """
        context_passages = [ev.text for ev in evidence]

        # 1. Context Relevance
        cr_score, cr_reasoning = self._score_dimension(
            query_id=query_id,
            dimension="context_relevance",
            prompt=build_context_relevance_prompt(query, context_passages),
        )

        # 2. Groundedness
        gr_score, gr_reasoning = self._score_dimension(
            query_id=query_id,
            dimension="groundedness",
            prompt=build_groundedness_prompt(query, context_passages, answer.text),
        )

        # 3. Answer Relevance
        ar_score, ar_reasoning = self._score_dimension(
            query_id=query_id,
            dimension="answer_relevance",
            prompt=build_answer_relevance_prompt(query, answer.text),
        )

        metrics: list[MetricResult] = [
            MetricResult(
                name="context_relevance",
                value=round(cr_score, 4),
                normalized=True,
            ),
            MetricResult(
                name="groundedness",
                value=round(gr_score, 4),
                normalized=True,
            ),
            MetricResult(
                name="answer_relevance",
                value=round(ar_score, 4),
                normalized=True,
            ),
            MetricResult(name="langsmith_disabled", value=1.0),
        ]

        # 4. Factual Correctness (optional)
        if gold_answer is not None:
            fc_score, fc_reasoning = self._score_dimension(
                query_id=query_id,
                dimension="factual_correctness",
                prompt=build_factual_correctness_prompt(
                    query, gold_answer, answer.text
                ),
            )
            metrics.append(
                MetricResult(
                    name="factual_correctness",
                    value=round(fc_score, 4),
                    normalized=True,
                )
            )

        logger.info(
            "query_id=%s strategy=%s "
            "CR=%.3f GR=%.3f AR=%.3f %s",
            query_id,
            self._strategy.value,
            cr_score,
            gr_score,
            ar_score,
            f"FC={metrics[-1].value:.3f}" if gold_answer else "",
        )

        return EvaluationResult(
            query_id=query_id,
            strategy=self._strategy,
            metrics=tuple(metrics),
        )

    def _score_dimension(
        self, query_id: str, dimension: str, prompt: str
    ) -> tuple[float, str]:
        """Call Gemini for a single evaluation dimension.

        Returns:
            (score, reasoning) where score ∈ [0.0, 1.0].
        """
        raw = self._call_with_retry(query_id, dimension, prompt)
        return _parse_score_json(raw, dimension, query_id)

    def _call_with_retry(
        self, query_id: str, dimension: str, prompt: str
    ) -> str:
        """Execute Gemini API call with quota management and retry logic."""
        import google.genai.types as types

        last_error: Exception = RuntimeError("No attempt made")
        for attempt in range(self._retry.max_attempts):
            try:
                wait = self._quota.acquire()
                if wait > 0:
                    logger.info(
                        "query_id=%s dimension=%s: quota wait %.1fs (attempt %d)",
                        query_id, dimension, wait, attempt + 1,
                    )

                response = self._client.models.generate_content(  # type: ignore[attr-defined]
                    model=self._judge_model_id,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part(
                                    text=JUDGE_SYSTEM + "\n\n" + prompt
                                )
                            ],
                        )
                    ],
                    config=types.GenerateContentConfig(
                        temperature=self._temperature,
                        candidate_count=1,
                    ),
                )
                text: str | None = response.text
                if text is None:
                    logger.warning(
                        "query_id=%s dimension=%s: empty response",
                        query_id, dimension,
                    )
                    return '{"reasoning": "empty response", "score": 0.0}'
                return str(text)

            except Exception as exc:
                exc_str = str(exc)
                status = _extract_status_code(exc_str)

                if status in (400, 403):
                    raise NonRetryableError(status, exc_str) from exc

                if status == 429 or "429" in exc_str:
                    backoff = self._retry.sleep_for_retry(attempt)
                    self._quota.record_retry(backoff)
                    logger.warning(
                        "query_id=%s dimension=%s: 429 (attempt %d/%d) waited %.1fs",
                        query_id, dimension,
                        attempt + 1, self._retry.max_attempts, backoff,
                    )
                    last_error = exc
                    continue

                if status and status >= 500:
                    backoff = self._retry.sleep_for_retry(attempt)
                    logger.warning(
                        "query_id=%s dimension=%s: %d (attempt %d/%d) waited %.1fs",
                        query_id, dimension, status,
                        attempt + 1, self._retry.max_attempts, backoff,
                    )
                    last_error = exc
                    continue

                raise

        raise RetryExhaustedError(self._retry.max_attempts, last_error)


def _parse_score_json(
    raw: str, dimension: str, query_id: str
) -> tuple[float, str]:
    """Parse JSON score from Gemini response.

    Tries strict JSON first; falls back to regex extraction.
    Returns (score, reasoning) with score ∈ [0.0, 1.0].
    """
    try:
        # Strip potential markdown fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
        data = json.loads(cleaned)
        score = float(data.get(_SCORE_KEY, 0.0))
        reasoning = str(data.get(_REASONING_KEY, ""))
        score = max(0.0, min(1.0, score))
        return score, reasoning
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Regex fallback
    match = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
    if match:
        score = max(0.0, min(1.0, float(match.group(1))))
        logger.warning(
            "query_id=%s dimension=%s: JSON parse failed, used regex fallback",
            query_id, dimension,
        )
        return score, ""

    logger.error(
        "query_id=%s dimension=%s: could not parse score from: %.100s",
        query_id, dimension, raw,
    )
    return 0.0, ""


def _extract_status_code(error_str: str) -> int | None:
    for code in (400, 403, 429, 500, 502, 503, 504):
        if str(code) in error_str:
            return code
    return None


# ─────────────────────────────────────────────────────────────────
# Sanitization helper
# ─────────────────────────────────────────────────────────────────

def sanitize_evaluation_for_artifact(result: EvaluationResult) -> dict[str, object]:
    """Return a JSON-safe dict with no credentials.

    Excludes: API keys, headers, HTTP responses, internal IDs.
    Includes: query_id, strategy, all metric names and values.
    """
    return {
        "query_id": result.query_id,
        "strategy": result.strategy.value,
        "metrics": {m.name: m.value for m in result.metrics},
    }
