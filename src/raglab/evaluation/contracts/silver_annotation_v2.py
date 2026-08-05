"""Silver Annotation Contracts and Governance (Gate B2).

Defines schema, rules, and validators for automated Machine Silver triage annotations.
Enforces strict separation from Human Gold qrels and governs execution modes.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from raglab.evaluation.contracts.hybrid_eval_v2 import SilverAnnotationRecord

DEFAULT_SILVER_JUDGE_MODEL: Final[str] = "gemini-3.1-flash-lite"
FORBIDDEN_JUDGE_MODELS: Final[set[str]] = {"gemini-2.5-flash", "gemini-1.5-flash"}
ALLOWED_EVIDENCE_ROLES: Final[set[str]] = {
    "PRIMARY",
    "SUPPORTING",
    "CONTEXTUAL",
    "NEGATIVE_CONTROL",
}


class SilverExecutionMode(StrEnum):
    """Governed execution modes for Machine Silver triage."""

    VALIDATION_ONLY = "VALIDATION_ONLY"
    SMOKE_REAL = "SMOKE_REAL"
    FULL_REAL = "FULL_REAL"
    RESUME_REAL = "RESUME_REAL"
    TEST_FIXTURE = "TEST_FIXTURE"


def validate_silver_record(
    record: SilverAnnotationRecord | dict[str, Any], passage_text: str
) -> bool:
    """Validate a SilverAnnotationRecord against strict contract rules."""
    rec_dict = record.__dict__ if isinstance(record, SilverAnnotationRecord) else record

    label_source = rec_dict.get("label_source", "")
    if label_source != "MACHINE_SILVER":
        raise ValueError(f"label_source must be MACHINE_SILVER, got '{label_source}'")

    grade = rec_dict.get("relevance_grade")
    if grade is None or not (0 <= grade <= 3):
        raise ValueError(f"relevance_grade must be in 0..3, got {grade}")

    role = rec_dict.get("evidence_role", "")
    if role not in ALLOWED_EVIDENCE_ROLES:
        raise ValueError(
            f"evidence_role must be one of {ALLOWED_EVIDENCE_ROLES}, got '{role}'"
        )

    conf = rec_dict.get("confidence")
    if conf is None or not (0.0 <= conf <= 1.0):
        raise ValueError(f"confidence must be in [0, 1], got {conf}")

    span = rec_dict.get("supporting_span", "")
    if span and span not in passage_text:
        msg = f"supporting_span '{span}' not found literally in passage text"
        raise ValueError(msg)

    judge_model = rec_dict.get("judge_model", "")
    if judge_model in FORBIDDEN_JUDGE_MODELS:
        raise ValueError(
            f"Forbidden judge_model '{judge_model}' used. "
            f"Must use '{DEFAULT_SILVER_JUDGE_MODEL}'."
        )

    return True


def validate_human_qrels_exclusion(qrels_file: Path) -> bool:
    """Ensure no MACHINE_SILVER labels exist in human_qrels.jsonl."""
    if not qrels_file.exists():
        return True

    with qrels_file.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            data = json.loads(line)
            source = data.get("label_source", "")
            if source == "MACHINE_SILVER":
                raise ValueError(
                    f"Violation at line {line_num}: MACHINE_SILVER label"
                    " found in human_qrels"
                )
    return True
