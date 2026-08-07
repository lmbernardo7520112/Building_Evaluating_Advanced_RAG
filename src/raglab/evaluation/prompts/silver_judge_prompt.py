"""Prompt Template for Machine Silver Triage Judge (Gate B2).

Isolated prompt template for evaluating passage relevance.
Re-exports from infrastructure layer for evaluation pipeline compatibility.
"""

from __future__ import annotations

from raglab.infrastructure.gemini.prompts import (
    SILVER_JUDGE_PROMPT_TEMPLATE,
    render_silver_judge_prompt,
)

__all__ = [
    "SILVER_JUDGE_PROMPT_TEMPLATE",
    "render_silver_judge_prompt",
]
