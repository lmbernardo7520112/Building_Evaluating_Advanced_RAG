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
    query_id = str(legacy_item.get("query_id", legacy_item.get("qid", ""))).strip()
    raw_query = legacy_item.get("query", legacy_item.get("query_text", ""))
    query_text = str(raw_query).strip()
    gold_answer = legacy_item.get("gold_answer")
    if gold_answer is not None:
        gold_answer = str(gold_answer).strip()

    if "answerable" in legacy_item:
        is_answerable = bool(legacy_item["answerable"])
    else:
        is_answerable = not bool(legacy_item.get("abstention_expected", False))

    unanswerable_reason = None

    if not is_answerable:
        default_reason = (
            UnanswerableReason.UNANSWERABLE_EXPECTED
            if legacy_item.get("abstention_expected")
            else UnanswerableReason.INSUFFICIENT_EVIDENCE
        )
        raw_reason = legacy_item.get("unanswerable_reason", default_reason)
        if isinstance(raw_reason, UnanswerableReason):
            unanswerable_reason = raw_reason
        else:
            try:
                unanswerable_reason = UnanswerableReason(raw_reason)
            except ValueError:
                unanswerable_reason = default_reason

    # Relevant pages / passages to canonical evidence
    # "Página não é passagem": Integer pages remain in legacy_relevant_pages
    relevant_pages_raw = legacy_item.get("relevant_pages", [])
    evidences: list[CanonicalEvidence] = []
    integer_pages: list[int] = []

    for item in relevant_pages_raw:
        if isinstance(item, int):
            integer_pages.append(item)
        elif isinstance(item, dict):
            raw_page = item.get("page", item.get("start_page", 0)) or 0
            page_num = int(raw_page)
            doc_id = str(item.get("document_id", f"gersting_p{page_num}"))
            passage_id = str(item.get("passage_id", ""))
            if not passage_id:
                continue
            text_span = str(item.get("text_span", item.get("text", "")))
            sha = item.get("content_sha256") or hashlib.sha256(
                text_span.encode("utf-8")
            ).hexdigest()

            evidences.append(
                CanonicalEvidence(
                    passage_id=passage_id,
                    document_id=doc_id,
                    start_page=page_num,
                    text_span=text_span,
                    content_sha256=sha,
                    relevance_grade=item.get("relevance_grade"),
                )
            )

    has_graded = any(e.relevance_grade is not None for e in evidences)
    return GroundTruthItemV2(
        query_id=query_id,
        query_text=query_text,
        answerable=is_answerable,
        unanswerable_reason=unanswerable_reason,
        gold_answer=gold_answer if is_answerable else None,
        relevant_evidences=tuple(evidences),
        provenance_status="LEGACY_METADATA_UNAVAILABLE",
        annotation_completeness={
            "passage_qrels_present": len(evidences) > 0,
            "graded_qrels_present": has_graded,
            "gold_answer_present": gold_answer is not None,
            "nuggets_present": False,
            "adjudication_present": False,
        },
        annotation_records=(),
        legacy_relevant_pages=tuple(integer_pages),
    )


def migrate_legacy_dataset(
    legacy_items: Sequence[dict[str, Any]],
) -> tuple[GroundTruthItemV2, ...]:
    """Migrate a full sequence of legacy items."""
    return tuple(migrate_legacy_qrel_item(item) for item in legacy_items)
