"""Silver Annotation Contracts and Governance (Gate B2).

Defines schema, rules, and validators for automated Machine Silver triage annotations.
Enforces strict separation from Human Gold qrels.
"""

from __future__ import annotations

import json
from pathlib import Path

from raglab.evaluation.contracts.hybrid_eval_v2 import SilverAnnotationRecord


def validate_silver_record(record: SilverAnnotationRecord, passage_text: str) -> bool:
    """Validate a SilverAnnotationRecord against strict contract rules."""
    if record.label_source != "MACHINE_SILVER":
        raise ValueError(
            f"label_source must be MACHINE_SILVER, got {record.label_source}"
        )

    if not (0 <= record.relevance_grade <= 3):
        raise ValueError(
            f"relevance_grade must be in 0..3, got {record.relevance_grade}"
        )

    if not (0.0 <= record.confidence <= 1.0):
        raise ValueError(f"confidence must be in [0, 1], got {record.confidence}")

    if record.supporting_span and record.supporting_span not in passage_text:
        msg = (
            f"supporting_span '{record.supporting_span}' not found literally"
            " in passage text"
        )
        raise ValueError(msg)

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
            if data.get("relevance_grade") is None and data.get("status") == "UNJUDGED":
                # Unjudged is allowed as explicit status, but not converted to 0
                pass
    return True
