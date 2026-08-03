"""Legacy Ground Truth to Ground Truth v2 Migration Utility.

EXECUTION GUARD 2:
- Never manufacture missing metadata or convert binary qrels into numeric grades.
- Annotates unannotated/legacy fields with
  provenance_status="LEGACY_METADATA_UNAVAILABLE" and relevance_grade=None.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from raglab.evaluation.contracts.ground_truth_v2 import (
    CanonicalEvidence,
    GroundTruthItemV2,
    UnanswerableReason,
)


def migrate_legacy_qrel_item(legacy_item: dict[str, Any]) -> GroundTruthItemV2:
    """Migrate a single legacy qrel item without inventing metadata."""
    query_id = str(legacy_item.get("query_id", "")).strip()
    raw_query = legacy_item.get("query", legacy_item.get("query_text", ""))
    query_text = str(raw_query).strip()
    gold_answer = legacy_item.get("gold_answer")
    if gold_answer is not None:
        gold_answer = str(gold_answer).strip()

    is_answerable = bool(legacy_item.get("answerable", True))
    unanswerable_reason = None

    if not is_answerable:
        raw_reason = legacy_item.get("unanswerable_reason", "INSUFFICIENT_EVIDENCE")
        try:
            unanswerable_reason = UnanswerableReason(raw_reason)
        except ValueError:
            unanswerable_reason = UnanswerableReason.INSUFFICIENT_EVIDENCE

    # Relevant pages / passages to canonical evidence
    relevant_pages = legacy_item.get("relevant_pages", [])
    evidences: list[CanonicalEvidence] = []

    for item in relevant_pages:
        if isinstance(item, int):
            page_num = item
            doc_id = f"gersting_p{page_num}"
            passage_id = f"{doc_id}_legacy"
            text_span = f"Page {page_num} legacy content span"
        elif isinstance(item, dict):
            raw_page = item.get("page", item.get("start_page", 0)) or 0
            page_num = int(raw_page)
            doc_id = str(item.get("document_id", f"gersting_p{page_num}"))
            passage_id = str(item.get("passage_id", f"{doc_id}_p{page_num}"))
            text_span = str(item.get("text_span", item.get("text", "")))
        else:
            continue

        sha = hashlib.sha256(text_span.encode("utf-8")).hexdigest()

        evidences.append(
            CanonicalEvidence(
                passage_id=passage_id,
                document_id=doc_id,
                start_page=page_num,
                text_span=text_span,
                content_sha256=sha,
                relevance_grade=None,  # Binary legacy qrel: grade NOT manufactured
            )
        )

    return GroundTruthItemV2(
        query_id=query_id,
        query_text=query_text,
        answerable=is_answerable,
        unanswerable_reason=unanswerable_reason,
        gold_answer=gold_answer if is_answerable else None,
        relevant_evidences=tuple(evidences),
        provenance_status="LEGACY_METADATA_UNAVAILABLE",
        annotation_completeness={
            "relevance_grades_present": False,
            "nuggets_present": False,
            "adjudication_present": False,
        },
        annotation_records=(),
    )


def migrate_legacy_dataset(
    legacy_items: Sequence[dict[str, Any]],
) -> tuple[GroundTruthItemV2, ...]:
    """Migrate a full sequence of legacy items."""
    return tuple(migrate_legacy_qrel_item(item) for item in legacy_items)
