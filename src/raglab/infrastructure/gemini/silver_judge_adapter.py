"""Gemini Silver Judge Adapter for Machine Silver Triage (Gate B2).

SECURITY BOUNDARY:
    Reads GEMINI_API_KEY ONLY from os.environ.
    Key value is NEVER logged, stored, or output in any artifact.
    Must ONLY be instantiated during human-operated execution with API key.

Model: gemini-3.1-flash-lite (default)
Prohibited: gemini-2.5-flash
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import UTC, datetime
from typing import Any, Final

from raglab.domain.retry import NonRetryableError, RetryExhaustedError
from raglab.infrastructure.gemini.prompts import (
    render_silver_judge_prompt,
)

logger = logging.getLogger(__name__)

DEFAULT_SILVER_JUDGE_MODEL: Final[str] = "gemini-3.1-flash-lite"
FORBIDDEN_JUDGE_MODELS: Final[set[str]] = {"gemini-2.5-flash", "gemini-1.5-flash"}
_CREDENTIAL_ENV: Final[str] = "GEMINI_API_KEY"


class SilverJudgeAdapter:
    """Automated Machine Silver triage judge adapter using Gemini API."""

    def __init__(
        self,
        model_id: str | None = None,
        client: Any | None = None,
        rpm_limit: int | None = None,
        max_retries: int | None = None,
        base_backoff: float | None = None,
    ) -> None:
        selected_model = model_id or os.environ.get(
            "RAGLAB_SILVER_JUDGE_MODEL", DEFAULT_SILVER_JUDGE_MODEL
        )
        if selected_model in FORBIDDEN_JUDGE_MODELS:
            raise ValueError(
                f"Forbidden judge model '{selected_model}'. "
                f"Must use authorized model '{DEFAULT_SILVER_JUDGE_MODEL}'."
            )

        self._model_id = selected_model
        self._rpm_limit = rpm_limit or int(
            os.environ.get("RAGLAB_GEMINI_RPM_LIMIT", "15")
        )
        self._max_retries = max_retries or int(
            os.environ.get("RAGLAB_GEMINI_MAX_RETRIES", "3")
        )
        self._base_backoff = base_backoff or float(
            os.environ.get("RAGLAB_GEMINI_BASE_BACKOFF_SECONDS", "2.0")
        )
        self._client = client
        self._last_call_time: float = 0.0

    @property
    def model_id(self) -> str:
        return self._model_id

    def _get_client(self) -> Any:
        """Initialize client on-demand if not injected."""
        if self._client is not None:
            return self._client

        api_key = os.environ.get(_CREDENTIAL_ENV)
        if not api_key:
            raise RuntimeError(
                f"Environment variable '{_CREDENTIAL_ENV}' is missing. "
                "API key required for real LLM execution."
            )
        try:
            import google.genai as genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai library is required for Gemini silver triage execution."
            ) from exc

        self._client = genai.Client(api_key=api_key)
        return self._client

    def _acquire_rate_limit_slot(self) -> None:
        """Enforce RPM limit with interval delay."""
        if self._rpm_limit <= 0:
            return
        interval = 60.0 / self._rpm_limit
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_call_time = time.monotonic()

    def evaluate_passage(
        self,
        question_id: str,
        question_text: str,
        passage_id: str,
        passage_text: str,
        rubric_version: str = "2.0.0",
    ) -> tuple[dict[str, Any], int, int]:
        """Evaluate a single passage for question relevance.

        Returns:
            (record_dict, logical_calls=1, physical_attempts)
        """
        prompt = render_silver_judge_prompt(
            question_text=question_text,
            passage_id=passage_id,
            passage_text=passage_text,
        )
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        raw_response, physical_attempts = self._call_model_with_retry(
            prompt=prompt,
            qid=question_id,
            ps_id=passage_id,
        )

        parsed_data = self._parse_and_validate_response(
            raw_text=raw_response,
            passage_text=passage_text,
            qid=question_id,
            ps_id=passage_id,
        )

        raw_seed = f"{question_id}:{passage_id}".encode()
        order_seed = hashlib.sha256(raw_seed).hexdigest()[:8]
        call_id = f"call_{question_id}_{passage_id}_{prompt_sha[:8]}"

        rec_dict = {
            "question_id": question_id,
            "passage_id": passage_id,
            "label_source": "MACHINE_SILVER",
            "judge_id": "gemini_3.1_flash_lite_silver_judge",
            "judge_provider": "google_genai",
            "judge_model": self._model_id,
            "judge_model_version": "v1",
            "judge_prompt_sha256": prompt_sha,
            "rubric_version": rubric_version,
            "order_seed": order_seed,
            "relevance_grade": parsed_data["relevance_grade"],
            "evidence_role": parsed_data["evidence_role"],
            "confidence": parsed_data["confidence"],
            "supporting_span": parsed_data["supporting_span"],
            "reasoning": parsed_data["reasoning"],
            "needs_human_review": parsed_data["needs_human_review"],
            "created_at_utc": datetime.now(UTC).isoformat(),
            "call_id": call_id,
            "retry_count": physical_attempts - 1,
        }

        return rec_dict, 1, physical_attempts

    def _call_model_with_retry(
        self, prompt: str, qid: str, ps_id: str
    ) -> tuple[str, int]:
        """Execute Gemini model call with rate limiting and retry logic."""

        attempts = 0
        last_exception: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            attempts = attempt
            self._acquire_rate_limit_slot()

            try:
                client = self._get_client()

                if hasattr(client, "models"):
                    import google.genai.types as types

                    config = types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                    )
                    response = client.models.generate_content(
                        model=self._model_id,
                        contents=prompt,
                        config=config,
                    )
                    text = getattr(response, "text", "") or ""
                    return str(text), attempts
                elif callable(getattr(client, "generate_content", None)):
                    response = client.generate_content(prompt)
                    text = getattr(response, "text", str(response))
                    return str(text), attempts
                else:
                    raise RuntimeError("Invalid client structure")

            except Exception as exc:
                last_exception = exc
                err_msg = str(exc)
                status_code = _extract_status_code(err_msg)

                if status_code in (400, 401, 403):
                    logger.error(
                        "Terminal API error %d for qid=%s ps_id=%s",
                        status_code,
                        qid,
                        ps_id,
                    )
                    raise NonRetryableError(
                        status_code,
                        f"Terminal API error {status_code} during silver triage",
                    ) from exc

                if (
                    status_code == 429
                    or "429" in err_msg
                    or "RESOURCE_EXHAUSTED" in err_msg
                ):
                    sleep_time = self._base_backoff * (2 ** (attempt - 1))
                    logger.warning(
                        "Rate limit 429 on attempt %d/%d for qid=%s ps_id=%s."
                        " Sleeping %.1fs",
                        attempt,
                        self._max_retries,
                        qid,
                        ps_id,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    continue

                if status_code and status_code >= 500:
                    sleep_time = self._base_backoff * (2 ** (attempt - 1))
                    logger.warning(
                        "Server error %d on attempt %d/%d for qid=%s ps_id=%s."
                        " Sleeping %.1fs",
                        status_code,
                        attempt,
                        self._max_retries,
                        qid,
                        ps_id,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    continue

                if attempt < self._max_retries:
                    sleep_time = self._base_backoff * (2 ** (attempt - 1))
                    time.sleep(sleep_time)
                    continue

        msg = f"Retry exhausted for qid={qid} ps_id={ps_id}"
        raise RetryExhaustedError(
            self._max_retries,
            last_exception or RuntimeError(msg),
        )

    def _parse_and_validate_response(
        self, raw_text: str, passage_text: str, qid: str, ps_id: str
    ) -> dict[str, Any]:
        """Parse model response JSON and validate fields."""
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned).strip()

        fallback = {
            "relevance_grade": 0,
            "evidence_role": "NEGATIVE_CONTROL",
            "confidence": 0.0,
            "supporting_span": "",
            "reasoning": "Falha na análise estruturada da resposta do modelo.",
            "needs_human_review": True,
        }

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("JSON Decode error for qid=%s ps_id=%s", qid, ps_id)
            return fallback

        if not isinstance(data, dict):
            return fallback

        grade = data.get("relevance_grade")
        if not isinstance(grade, int) or not (0 <= grade <= 3):
            return fallback

        role = str(data.get("evidence_role", "")).upper()
        if role not in {"PRIMARY", "SUPPORTING", "CONTEXTUAL", "NEGATIVE_CONTROL"}:
            if grade == 3:
                role = "PRIMARY"
            elif grade == 2:
                role = "SUPPORTING"
            elif grade == 1:
                role = "CONTEXTUAL"
            else:
                role = "NEGATIVE_CONTROL"

        conf = data.get("confidence", 0.0)
        try:
            conf_val = float(conf)
            conf_val = max(0.0, min(1.0, conf_val))
        except (ValueError, TypeError):
            conf_val = 0.0

        span = str(data.get("supporting_span", "") or "").strip()
        needs_review = bool(data.get("needs_human_review", False))

        if span and span not in passage_text:
            logger.warning(
                "Supporting span not found literally in passage text for"
                " qid=%s ps_id=%s",
                qid,
                ps_id,
            )
            span = ""
            needs_review = True

        reason = "Triagem via Gemini 3.1 Flash Lite."
        reasoning = str(data.get("reasoning", "") or reason).strip()

        return {
            "relevance_grade": grade,
            "evidence_role": role,
            "confidence": conf_val,
            "supporting_span": span,
            "reasoning": reasoning,
            "needs_human_review": needs_review,
        }


def _extract_status_code(error_str: str) -> int | None:
    for code in (400, 401, 403, 429, 500, 502, 503, 504):
        if str(code) in error_str:
            return code
    return None
