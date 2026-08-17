"""Canonical corpus snapshot loader and builder for Slice 5A.3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from raglab.agentic.evaluation.parity_contracts import CanonicalCorpusSnapshot
from raglab.domain.entities import Chunk
from raglab.domain.value_objects import ChunkId, DocumentPage


class CanonicalCorpusError(Exception):
    """Exception raised when canonical corpus validation fails."""


def load_canonical_corpus_snapshot(
    registry_path: str | Path,
    manifest_path: str | Path,
    expected_registry_sha256: str | None = None,
) -> tuple[CanonicalCorpusSnapshot, list[dict[str, Any]]]:
    """Load, validate, and hash the canonical passage registry snapshot.

    Args:
        registry_path: Path to passage_registry.jsonl.
        manifest_path: Path to passage_registry_manifest.json.
        expected_registry_sha256: Expected SHA-256 hash of passage_registry.jsonl.

    Returns:
        Tuple of (CanonicalCorpusSnapshot, list of raw passage dicts).

    Raises:
        CanonicalCorpusError: If validation rules are violated.
    """
    reg_p = Path(registry_path)
    man_p = Path(manifest_path)

    if not reg_p.exists():
        raise CanonicalCorpusError(f"Passage registry file missing: {reg_p}")
    if not man_p.exists():
        raise CanonicalCorpusError(f"Registry manifest file missing: {man_p}")

    reg_bytes = reg_p.read_bytes()
    actual_registry_sha256 = hashlib.sha256(reg_bytes).hexdigest()

    if (
        expected_registry_sha256
        and actual_registry_sha256 != expected_registry_sha256
    ):
        raise CanonicalCorpusError(
            f"Passage registry SHA-256 mismatch: "
            f"{actual_registry_sha256} != {expected_registry_sha256}"
        )

    man_bytes = man_p.read_bytes()
    man_sha256 = hashlib.sha256(man_bytes).hexdigest()

    passages: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    pages: list[int] = []

    for line_idx, line in enumerate(reg_bytes.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as err:
            raise CanonicalCorpusError(
                f"Invalid JSON at line {line_idx}: {err}"
            ) from err

        pid = item.get("passage_id")
        text = item.get("text", "")
        page_num = item.get("page_number")
        content_sha = item.get("content_sha256")

        if not pid or not isinstance(pid, str):
            raise CanonicalCorpusError(
                f"Missing or invalid passage_id at line {line_idx}"
            )

        if not pid.startswith("ps_"):
            raise CanonicalCorpusError(
                f"Passage ID '{pid}' does not have required 'ps_' prefix "
                f"at line {line_idx}"
            )

        if pid in seen_ids:
            raise CanonicalCorpusError(
                f"Duplicate passage_id '{pid}' detected at line {line_idx}"
            )
        seen_ids.add(pid)

        if not text or not text.strip():
            raise CanonicalCorpusError(
                f"Empty text for passage '{pid}' at line {line_idx}"
            )

        if not isinstance(page_num, int) or page_num < 1:
            raise CanonicalCorpusError(
                f"Invalid page_number '{page_num}' for passage '{pid}' "
                f"at line {line_idx}"
            )
        pages.append(page_num)

        if content_sha:
            computed_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if computed_sha != content_sha:
                raise CanonicalCorpusError(
                    f"Content SHA-256 mismatch for '{pid}': "
                    f"{computed_sha} != {content_sha}"
                )

        passages.append(item)

    if not passages:
        raise CanonicalCorpusError("Passage registry is empty")

    # Sort deterministically by passage_id
    passages.sort(key=lambda x: x["passage_id"])
    ordered_ids = [p["passage_id"] for p in passages]

    min_page = min(pages)
    max_page = max(pages)

    snapshot = CanonicalCorpusSnapshot(
        snapshot_id=f"corpus_snapshot_{actual_registry_sha256[:12]}",
        source_artifact_hashes={
            "passage_registry_sha256": actual_registry_sha256,
            "passage_registry_manifest_sha256": man_sha256,
        },
        passage_count=len(passages),
        ordered_canonical_passage_ids=ordered_ids,
        page_bounds=(min_page, max_page),
        corpus_sha256=actual_registry_sha256,
    )

    return snapshot, passages


def passages_to_chunks(passages: list[dict[str, Any]]) -> list[Chunk]:
    """Convert canonical passage dicts to domain Chunk entities."""
    chunks: list[Chunk] = []
    for item in passages:
        pid = item["passage_id"]
        doc_id = item.get("document_id", "gersting_discrete_math")
        page_num = item["page_number"]
        text = item["text"]
        chunks.append(
            Chunk(
                chunk_id=ChunkId(pid),
                document_id=doc_id,
                text=text,
                start_page=page_num,
                end_page=page_num,
            )
        )
    return chunks


def passages_to_document_pages(passages: list[dict[str, Any]]) -> list[DocumentPage]:
    """Convert canonical passage dicts to domain DocumentPage entities.

    Grouped by page number.
    """
    page_texts: dict[int, list[str]] = {}
    doc_id = "gersting_discrete_math"

    for item in passages:
        page_num = item["page_number"]
        doc_id = item.get("document_id", doc_id)
        text = item["text"]
        page_texts.setdefault(page_num, []).append(text)

    pages: list[DocumentPage] = []
    for page_num in sorted(page_texts.keys()):
        combined_text = "\n\n".join(page_texts[page_num])
        pages.append(
            DocumentPage(
                document_id=doc_id,
                page_number=page_num,
                text=combined_text,
            )
        )
    return pages
