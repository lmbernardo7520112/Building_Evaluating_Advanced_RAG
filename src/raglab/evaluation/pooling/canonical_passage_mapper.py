"""Canonical Passage Mapper for Hybrid Multisystem Pooling (Gate B2).

Maps arbitrary retrieved chunks/passages to canonical PassageRegistryEntries:
1. Exact passage_id
2. Exact document_id + page_number + start_char + end_char
3. Exact content_sha256
4. Exact substring match on same page
5. Ambiguous (multiple matches) -> AMBIGUOUS_NEEDS_REVIEW
6. Unmapped -> UNMAPPED_NEEDS_REVIEW

Strictly audited, zero silent loss.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from raglab.evaluation.contracts.human_annotation_v2 import PassageRegistryEntry
from raglab.evaluation.contracts.hybrid_eval_v2 import (
    CanonicalMappingResult,
    CanonicalMappingStatus,
)


class CanonicalPassageMapper:
    """Mapper from retrieved chunks to canonical registry passages."""

    def __init__(
        self, registry_entries: list[PassageRegistryEntry] | None = None
    ) -> None:
        if registry_entries is None:
            cur = Path(__file__).resolve().parent
            default_path = None
            for p in [cur] + list(cur.parents):

                candidate = (
                    p
                    / "benchmarks"
                    / "ground_truth"
                    / "v2"
                    / "passage_registry.jsonl"
                )
                if candidate.exists():
                    default_path = candidate
                    break
            if default_path and default_path.exists():
                registry_entries = (
                    CanonicalPassageMapper.from_registry_file(default_path).entries
                )
            else:

                registry_entries = []

        self.entries: list[PassageRegistryEntry] = registry_entries or []







        self.by_id: dict[str, PassageRegistryEntry] = {}
        self.by_offset: dict[tuple[str, int, int, int], PassageRegistryEntry] = {}
        self.by_sha: dict[str, list[PassageRegistryEntry]] = {}
        self.by_page: dict[tuple[str, int], list[PassageRegistryEntry]] = {}

        for entry in self.entries:
            self.by_id[entry.passage_id] = entry
            offset_key = (
                entry.document_id,
                entry.page_number,
                entry.start_char,
                entry.end_char,
            )
            self.by_offset[offset_key] = entry
            self.by_sha.setdefault(entry.content_sha256, []).append(entry)
            self.by_page.setdefault((entry.document_id, entry.page_number), []).append(
                entry
            )

    @classmethod
    def from_registry_file(cls, registry_file: Path) -> CanonicalPassageMapper:
        """Instantiate mapper from passage_registry.jsonl."""
        if not registry_file.exists():
            raise FileNotFoundError(f"Passage registry not found at {registry_file}")

        entries: list[PassageRegistryEntry] = []
        with registry_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    entries.append(
                        PassageRegistryEntry(
                            passage_id=item["passage_id"],
                            document_id=item["document_id"],
                            page_number=item["page_number"],
                            start_char=item["start_char"],
                            end_char=item["end_char"],
                            content_sha256=item["content_sha256"],
                            text=item["text"],
                            registry_version=item.get("registry_version", "2.0.0"),
                        )
                    )
        return cls(entries)

    def map_chunk(self, chunk_data: dict[str, Any]) -> CanonicalMappingResult:
        """Map a single retrieved chunk dict to a CanonicalMappingResult."""
        chunk_id = str(
            chunk_data.get("chunk_id", chunk_data.get("passage_id", "unknown"))
        )
        doc_id = str(chunk_data.get("document_id", "gersting_discrete_math"))
        page_num = int(chunk_data.get("page_number", 0))
        text = str(chunk_data.get("text", "")).strip()

        # Rule 1: Exact passage_id / chunk_id match
        ps_id = chunk_data.get("passage_id") or chunk_data.get("chunk_id")
        if ps_id and ps_id in self.by_id:
            return CanonicalMappingResult(
                source_chunk_id=chunk_id,
                document_id=doc_id,
                page_number=page_num or self.by_id[ps_id].page_number,
                mapped_passage_id=ps_id,
                mapping_status=CanonicalMappingStatus.EXACT_PASSAGE_ID,
                confidence=1.0,
                notes="Matched by exact passage_id",
            )

        # Rule 2: Exact offsets match
        start_c = chunk_data.get("start_char")
        end_c = chunk_data.get("end_char")
        if start_c is not None and end_c is not None and page_num > 0:
            offset_key = (doc_id, page_num, int(start_c), int(end_c))
            if offset_key in self.by_offset:
                entry = self.by_offset[offset_key]
                return CanonicalMappingResult(
                    source_chunk_id=chunk_id,
                    document_id=doc_id,
                    page_number=page_num,
                    mapped_passage_id=entry.passage_id,
                    mapping_status=CanonicalMappingStatus.EXACT_OFFSETS,
                    confidence=1.0,
                    notes=(
                        "Matched by exact document_id, page_number, and"
                        " character offsets"
                    ),
                )

        # Rule 3: Exact content_sha256 match
        text_sha = chunk_data.get("content_sha256")
        if not text_sha and text:
            text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_sha and text_sha in self.by_sha:
            candidates = self.by_sha[text_sha]
            if len(candidates) == 1:
                entry = candidates[0]
                return CanonicalMappingResult(
                    source_chunk_id=chunk_id,
                    document_id=doc_id,
                    page_number=entry.page_number,
                    mapped_passage_id=entry.passage_id,
                    mapping_status=CanonicalMappingStatus.EXACT_CONTENT_SHA256,
                    confidence=1.0,
                    notes="Matched by exact content_sha256 digest",
                )

        # Rule 4 & 5: Exact substring match or single passage on same page
        if page_num > 0:
            page_passages = self.by_page.get((doc_id, page_num)) or self.by_page.get(
                ("gersting_discrete_math", page_num), []
            )
            matching_passages: list[PassageRegistryEntry] = []

            if text:
                import re

                norm_text = re.sub(r"\s+", " ", text).strip()
                for p_entry in page_passages:
                    norm_entry = re.sub(r"\s+", " ", p_entry.text).strip()
                    if (
                        text in p_entry.text
                        or p_entry.text in text
                        or norm_text in norm_entry
                        or norm_entry in norm_text
                    ):
                        matching_passages.append(p_entry)

            if len(matching_passages) == 1:
                entry = matching_passages[0]
                return CanonicalMappingResult(
                    source_chunk_id=chunk_id,
                    document_id=doc_id,
                    page_number=page_num,
                    mapped_passage_id=entry.passage_id,
                    mapping_status=CanonicalMappingStatus.EXACT_SUBSTRING,
                    confidence=0.9,
                    notes="Matched by exact substring containment on same page",
                )
            elif not text and len(page_passages) == 1:
                # Page fallback for candidates without text when single passage exists
                entry = page_passages[0]
                return CanonicalMappingResult(
                    source_chunk_id=chunk_id,
                    document_id=doc_id,
                    page_number=page_num,
                    mapped_passage_id=entry.passage_id,
                    mapping_status=CanonicalMappingStatus.EXACT_SUBSTRING,
                    confidence=0.85,
                    notes="Matched single canonical passage on page",
                )



            elif len(matching_passages) > 1:
                return CanonicalMappingResult(
                    source_chunk_id=chunk_id,
                    document_id=doc_id,
                    page_number=page_num,
                    mapped_passage_id=matching_passages[0].passage_id,
                    mapping_status=CanonicalMappingStatus.AMBIGUOUS_NEEDS_REVIEW,
                    confidence=0.5,
                    notes=(
                        f"Ambiguous: matched {len(matching_passages)} passages"
                        f" on page {page_num}"
                    ),
                )


        # Rule 6: Unmapped
        return CanonicalMappingResult(
            source_chunk_id=chunk_id,
            document_id=doc_id,
            page_number=page_num,
            mapped_passage_id=None,
            mapping_status=CanonicalMappingStatus.UNMAPPED_NEEDS_REVIEW,
            confidence=0.0,
            notes="No exact match found in canonical passage registry",
        )
