"""Audited PDF Extractor Adapter using pypdf.

Preserves physical page numbers, page fingerprints, line-level provenance,
whitespace normalization without destroying mathematical formulas, and zero
verbatim copyright publishing.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from raglab.application.ports.corpus import CorpusReaderPort
from raglab.domain.value_objects import DocumentPage, IntegrityDigest

# Conservative hyphenation rejoining pattern
_HYPHEN_REJOIN = re.compile(r"([a-zà-ÿ])-\s*\n\s*([a-zà-ÿ])", re.IGNORECASE)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving lines and mathematical notation."""
    # Rejoin split hyphenated words conservatively
    text = _HYPHEN_REJOIN.sub(r"\1\2", text)
    # Collapse multiple horizontal spaces into single space per line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    # Remove empty trailing lines but preserve single linebreaks
    return "\n".join(line for line in lines if line)


class PyPdfExtractorAdapter(CorpusReaderPort):
    """Audited PDF reader implementing CorpusReaderPort using pypdf."""

    def __init__(self, extraction_method: str = "pypdf_v4") -> None:
        self.extraction_method = extraction_method

    def compute_fingerprint(self, path: str) -> IntegrityDigest:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"PDF path not found: {path}")
        raw_bytes = p.read_bytes()
        digest = hashlib.sha256(raw_bytes).hexdigest()
        return IntegrityDigest(digest)

    def read_document(
        self, path: str, page_start: int | None = None, page_end: int | None = None
    ) -> list[DocumentPage]:
        pages, _ = self.extract_pages_with_audit(
            path, page_start=page_start, page_end=page_end
        )
        return pages

    def extract_pages_with_audit(
        self, path: str, page_start: int | None = None, page_end: int | None = None
    ) -> tuple[list[DocumentPage], dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"PDF path not found: {path}")

        digest = self.compute_fingerprint(path)
        reader = PdfReader(p)
        total_pages = len(reader.pages)

        start_idx = (page_start - 1) if page_start is not None else 0
        end_idx = page_end if page_end is not None else total_pages

        start_idx = max(0, start_idx)
        end_idx = min(total_pages, end_idx)

        document_id = p.stem
        pages: list[DocumentPage] = []
        audit_details: list[dict[str, Any]] = []
        quality_warnings: list[str] = []

        cumulative_char_offset = 0

        for idx in range(start_idx, end_idx):
            page_num = idx + 1
            pypdf_page = reader.pages[idx]

            try:
                raw_text = pypdf_page.extract_text() or ""
            except Exception as exc:
                raw_text = ""
                quality_warnings.append(
                    f"Page {page_num}: extraction error - {exc}"
                )

            cleaned_text = normalize_whitespace(raw_text)
            char_count = len(cleaned_text)

            page_fingerprint = hashlib.sha256(
                cleaned_text.encode("utf-8")
            ).hexdigest()

            page_warnings: list[str] = []
            if char_count < 10:
                page_warnings.append("EMPTY_OR_NEAR_EMPTY_PAGE")
            elif char_count < 100:
                page_warnings.append("LOW_CHARACTER_COUNT")

            doc_page = DocumentPage(
                document_id=document_id,
                page_number=page_num,
                text=cleaned_text,
            )
            pages.append(doc_page)

            char_start = cumulative_char_offset
            char_end = cumulative_char_offset + char_count
            cumulative_char_offset = char_end + 1  # newline separator

            audit_details.append({
                "page_number": page_num,
                "char_offset_start": char_start,
                "char_offset_end": char_end,
                "char_count": char_count,
                "page_fingerprint": page_fingerprint,
                "warnings": page_warnings,
            })

        audit_report = {
            "document_id": document_id,
            "filename": p.name,
            "doc_fingerprint": digest.hex_digest,
            "extraction_method": self.extraction_method,
            "total_document_pages": total_pages,
            "extracted_page_range": [start_idx + 1, end_idx],
            "extracted_page_count": len(pages),
            "quality_warnings": quality_warnings,
            "pages_audit": audit_details,
        }

        return pages, audit_report
