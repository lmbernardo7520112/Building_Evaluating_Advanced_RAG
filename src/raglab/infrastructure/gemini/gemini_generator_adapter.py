"""Gemini generator adapter — wraps google-genai SDK for answer generation.

SECURITY BOUNDARY:
    This adapter reads GEMINI_API_KEY from the environment.
    It MUST NOT be instantiated by Antigravity IDE.
    It MUST only be instantiated in the human-operated isolated terminal.
    See: docs/security/credential_boundary.md

Execution environment: Ambiente B (human terminal only).

This adapter implements GenerationPort via structural subtyping (Protocol).

Provider:  google_gemini
Model:     gemini-3.1-flash-lite (configurable)
SDK:       google-genai >= 1.0
Temp:      0.0 (deterministic)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Sequence
from typing import Final

from raglab.domain.entities import GeneratedAnswer, RetrievedEvidence
from raglab.domain.errors import CitationProvenanceMismatchError
from raglab.domain.quota import QuotaManager
from raglab.domain.retry import NonRetryableError, RetryExhaustedError, RetryPolicy
from raglab.domain.value_objects import Citation
from raglab.infrastructure.gemini.prompts import (
    GENERATION_SYSTEM,
    PromptEvidence,
    build_generation_prompt,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL: Final[str] = "gemini-3.1-flash-lite"
_CREDENTIAL_ENV: Final[str] = "GEMINI_API_KEY"
_ABSTAIN_SIGNAL: Final[str] = "ABSTAIN"


def _extract_page_from_doc_id(document_id: str) -> int:
    """Extract page number from document_id like 'doc_p91'. Returns 0 on failure."""
    try:
        return int(document_id.split("_p")[-1])
    except (ValueError, IndexError):
        return 0


class GeminiGeneratorAdapter:
    """Answer generator using Gemini API.

    SECURITY REQUIREMENTS:
    1. GEMINI_API_KEY must be present in environment.
    2. Key is NEVER logged, stored in checkpoints, or passed to other adapters.
    3. Quota limits are enforced before every call.
    4. 429 errors trigger exponential backoff via RetryPolicy.
    5. Non-retryable errors (400, 403) are raised immediately.
    6. LangSmith is DISABLED.

    Usage (human terminal only):
        export GEMINI_API_KEY="$(your-key)"
        python benchmarks/run_slice4_benchmark.py
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        quota_manager: QuotaManager | None = None,
        retry_policy: RetryPolicy | None = None,
        temperature: float = 0.0,
    ) -> None:
        self._model_id = model_id
        self._quota = quota_manager or QuotaManager()
        self._retry = retry_policy or RetryPolicy()
        self._temperature = temperature
        self._client = self._init_client()

    def _init_client(self) -> object:
        """Initialize Gemini client from environment credential.

        Raises:
            RuntimeError: If GEMINI_API_KEY is not set.
        """
        api_key = os.environ.get(_CREDENTIAL_ENV)
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found in environment. "
                "This adapter must be executed in an isolated human terminal. "
                "See docs/security/credential_boundary.md"
            )
        # Key is used only for initialization — never logged
        import google.genai as genai

        return genai.Client(api_key=api_key)

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(
        self,
        query_id: str,
        query: str,
        evidence: Sequence[RetrievedEvidence],
    ) -> GeneratedAnswer:
        """Generate an answer using Gemini with quota control and retries.

        Args:
            query_id: Unique identifier for checkpoint idempotency.
            query: The user's question.
            evidence: Retrieved evidence passages.

        Returns:
            GeneratedAnswer with text, abstention flag, and citations.
        """
        if not evidence:
            logger.info("query_id=%s: no evidence — abstaining", query_id)
            return GeneratedAnswer(
                query_id=query_id,
                text="",
                abstained=True,
                citations=(),
            )

        prompt_evidences = PromptEvidence.from_retrieved_sequence(evidence)
        prompt = build_generation_prompt(query, prompt_evidences)

        raw_text = self._call_with_retry(query_id, prompt).strip()

        # Handle legacy raw "ABSTAIN"
        if raw_text.upper() == _ABSTAIN_SIGNAL:
            return GeneratedAnswer(
                query_id=query_id,
                text="",
                abstained=True,
                citations=(),
            )

        # Parse JSON output
        try:
            # Strip markdown fences if present
            clean_json = raw_text
            if clean_json.startswith("```"):
                clean_json = clean_json.strip("`")
                if clean_json.startswith("json"):
                    clean_json = clean_json[4:].strip()

            payload = json.loads(clean_json)
        except Exception as err:
            logger.warning(
                "query_id=%s: invalid JSON output from generator — "
                "fail closed abstaining: %s",
                query_id,
                err,
            )
            return GeneratedAnswer(
                query_id=query_id,
                text="",
                abstained=True,
                citations=(),
            )

        if not isinstance(payload, dict):
            return GeneratedAnswer(
                query_id=query_id, text="", abstained=True, citations=()
            )

        status = str(payload.get("status", "")).upper()
        if status == "ABSTAIN":
            return GeneratedAnswer(
                query_id=query_id,
                text="",
                abstained=True,
                citations=(),
            )

        if status != "ANSWER":
            logger.warning(
                "query_id=%s: unexpected status '%s' — fail closed abstaining",
                query_id,
                status,
            )
            return GeneratedAnswer(
                query_id=query_id, text="", abstained=True, citations=()
            )

        answer_text = str(payload.get("answer", "")).strip()
        raw_citations = payload.get("citations", [])

        if not answer_text or not raw_citations:
            logger.warning(
                "query_id=%s: ANSWER status missing text or citations — "
                "fail closed abstaining",
                query_id,
            )
            return GeneratedAnswer(
                query_id=query_id, text="", abstained=True, citations=()
            )

        # Map cited evidence IDs (E1, E2, ...) to persistent evidence items
        evidence_by_id = {
            pe.evidence_id: pe.retrieved_evidence for pe in prompt_evidences
        }
        citations_list = []

        for cite_id in raw_citations:
            cite_str = str(cite_id).strip()
            if cite_str not in evidence_by_id:
                raise CitationProvenanceMismatchError(cite_str)

            ev = evidence_by_id[cite_str]
            page_num = getattr(ev, "start_page", getattr(ev, "page", None))
            if page_num is None:
                page_num = _extract_page_from_doc_id(ev.document_id)

            citations_list.append(
                Citation(
                    document_id=ev.document_id,
                    page_number=int(page_num),
                    chunk_id=ev.chunk_id,
                    text_span=ev.text[:40],
                )
            )

        return GeneratedAnswer(
            query_id=query_id,
            text=answer_text,
            abstained=False,
            citations=tuple(citations_list),
        )

    def _call_with_retry(self, query_id: str, prompt: str) -> str:
        """Execute Gemini API call with quota management and retry logic."""
        import google.genai.types as types

        last_error: Exception = RuntimeError("No attempt made")
        for attempt in range(self._retry.max_attempts):
            try:
                wait = self._quota.acquire()
                if wait > 0:
                    logger.info(
                        "query_id=%s: quota wait %.1fs (attempt %d)",
                        query_id, wait, attempt + 1,
                    )

                response = self._client.models.generate_content(  # type: ignore[attr-defined]
                    model=self._model_id,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part(text=GENERATION_SYSTEM + "\n\n" + prompt)
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
                    logger.warning("query_id=%s: empty response from Gemini", query_id)
                    return _ABSTAIN_SIGNAL
                return str(text)

            except Exception as exc:
                exc_str = str(exc)
                status = _extract_status_code(exc_str)

                if status in (400, 403):
                    raise NonRetryableError(status, exc_str) from exc

                if status == 429 or "429" in exc_str:
                    backoff = self._retry.sleep_for_retry(attempt)
                    self._quota.record_retry(backoff, cause="429")
                    logger.warning(
                        "query_id=%s: 429 rate limit (attempt %d/%d), "
                        "waited %.1fs",
                        query_id, attempt + 1, self._retry.max_attempts, backoff,
                    )
                    last_error = exc
                    continue

                # Transient 5xx
                if status and status >= 500:
                    backoff = self._retry.sleep_for_retry(attempt)
                    self._quota.record_retry(backoff, cause="5xx")
                    logger.warning(
                        "query_id=%s: transient %d (attempt %d/%d), waited %.1fs",
                        query_id, status, attempt + 1, self._retry.max_attempts,
                        backoff,
                    )
                    last_error = exc
                    continue

                raise

        raise RetryExhaustedError(self._retry.max_attempts, last_error)


def _extract_status_code(error_str: str) -> int | None:
    """Extract HTTP status code from error string. Returns None if not found."""
    for code in (400, 403, 429, 500, 502, 503, 504):
        if str(code) in error_str:
            return code
    return None


# ─────────────────────────────────────────────────────────────────
# Sanitization helper
# ─────────────────────────────────────────────────────────────────

def sanitize_answer_for_artifact(answer: GeneratedAnswer) -> dict[str, object]:
    """Return a JSON-safe dict with no credentials and untruncated evaluated text.

    Excludes: API keys, headers, HTTP responses, internal IDs.
    Includes: query_id, text (full, untruncated), text_sha256, text_length_chars,
              truncated (False), preview, abstained, citation page numbers.
    """
    text_val = answer.text
    text_sha = hashlib.sha256(text_val.encode("utf-8")).hexdigest()
    return {
        "query_id": answer.query_id,
        "text": text_val,
        "text_sha256": text_sha,
        "text_length_chars": len(text_val),
        "truncated": False,
        "preview": text_val[:500] if len(text_val) > 500 else text_val,
        "abstained": answer.abstained,
        "citation_pages": [c.page_number for c in answer.citations],
    }
