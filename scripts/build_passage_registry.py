"""Passage Registry Builder (Gate B1 - Etapa 1).

Deterministically extracts text from physical pages 91-115 of the Discrete Math PDF,
segments into canonical passages with character offsets, and generates:
- benchmarks/ground_truth/v2/passage_registry.jsonl
- benchmarks/ground_truth/v2/passage_registry_manifest.json

Strictly offline, reproducible byte-by-byte. "Página não é passagem".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# Ensure src is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from raglab.evaluation.contracts.human_annotation_v2 import (  # noqa: E402
    PassageRegistryEntry,
)
from raglab.infrastructure.pdf_parsers.pdf_parser_adapter import (  # noqa: E402
    PyPdfExtractorAdapter,
)

PDF_SHA256_EXPECTED = "33e2e9f1e190158b3e99c19fced1acd050720247c7556780bad82b2f93bf1254"
DEFAULT_PDF_PATH = _REPO_ROOT.parent / (
    "Fundamentos matemáticos para a ciência da computação "
    "Matemática Discreta e Suas Aplicações (Judith L. Gersting).pdf"
)
DOCUMENT_ID = "gersting_discrete_math"
REGISTRY_VERSION = "2.0.0"
SCHEMA_VERSION = "2.0.0"


def generate_passage_id(
    doc_id: str, page_num: int, start_c: int, end_c: int, content_sha: str
) -> str:
    """Generate a deterministic passage_id.

    Formula: ps_<16 hex chars of sha256(doc_id:page_num:start_c:end_c:content_sha)>
    """
    raw_key = f"{doc_id}:{page_num}:{start_c}:{end_c}:{content_sha}"
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
    return f"ps_{digest}"


def segment_page_text(
    page_text: str, page_num: int, min_chars: int = 50
) -> list[tuple[int, int, str]]:
    """Segment a page's text into paragraphs with exact start_char and end_char offsets."""  # noqa: E501
    segments: list[tuple[int, int, str]] = []

    pattern = re.compile(r"(?:\S[^\n]*\n*)+")
    for match in pattern.finditer(page_text):
        start_c, end_c = match.span()
        text_content = match.group(0)
        if len(text_content.strip()) >= min_chars:
            segments.append((start_c, end_c, text_content))

    # Fallback if no segments found
    if not segments and page_text.strip():
        segments.append((0, len(page_text), page_text))

    return segments


def build_passage_registry(
    pdf_path: Path,
    output_dir: Path,
    page_start: int = 91,
    page_end: int = 115,
) -> tuple[Path, Path, str]:
    """Extract passages from PDF pages and save passage_registry.jsonl + manifest."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    actual_pdf_sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    if actual_pdf_sha != PDF_SHA256_EXPECTED:
        raise ValueError(
            f"PDF SHA-256 mismatch: expected {PDF_SHA256_EXPECTED}, "
            f"got {actual_pdf_sha}"
        )

    adapter = PyPdfExtractorAdapter()
    pages = adapter.read_document(
        str(pdf_path), page_start=page_start, page_end=page_end
    )

    entries: list[PassageRegistryEntry] = []
    seen_ids: set[str] = set()

    for page in pages:
        p_num = page.page_number
        p_text = page.text

        raw_segments = segment_page_text(p_text, p_num)
        for start_c, end_c, text_content in raw_segments:
            # Verification: text == page_text[start_char:end_char]
            extracted_sub = p_text[start_c:end_c]
            if extracted_sub != text_content:
                raise ValueError(
                    f"Offset mismatch on page {p_num}: "
                    f"extracted_sub={extracted_sub[:20]!r} != text_content={text_content[:20]!r}"  # noqa: E501
                )

            content_sha = hashlib.sha256(text_content.encode("utf-8")).hexdigest()
            ps_id = generate_passage_id(DOCUMENT_ID, p_num, start_c, end_c, content_sha)

            if ps_id in seen_ids:
                raise ValueError(f"Duplicate passage_id generated: {ps_id}")
            seen_ids.add(ps_id)

            entry = PassageRegistryEntry(
                passage_id=ps_id,
                document_id=DOCUMENT_ID,
                page_number=p_num,
                start_char=start_c,
                end_char=end_c,
                content_sha256=content_sha,
                text=text_content,
                registry_version=REGISTRY_VERSION,
            )
            entries.append(entry)

    output_dir.mkdir(parents=True, exist_ok=True)
    registry_file = output_dir / "passage_registry.jsonl"
    manifest_file = output_dir / "passage_registry_manifest.json"

    # Write registry.jsonl deterministically
    lines: list[str] = []
    for entry in entries:
        row = {
            "passage_id": entry.passage_id,
            "document_id": entry.document_id,
            "page_number": entry.page_number,
            "start_char": entry.start_char,
            "end_char": entry.end_char,
            "content_sha256": entry.content_sha256,
            "text": entry.text,
            "registry_version": entry.registry_version,
        }
        lines.append(json.dumps(row, ensure_ascii=False))

    jsonl_content = "\n".join(lines) + "\n"
    registry_file.write_text(jsonl_content, encoding="utf-8")

    registry_sha256 = hashlib.sha256(jsonl_content.encode("utf-8")).hexdigest()

    manifest_data = {
        "schema_version": SCHEMA_VERSION,
        "registry_version": REGISTRY_VERSION,
        "corpus_filename": pdf_path.name,
        "corpus_sha256": actual_pdf_sha,
        "page_range": [page_start, page_end],
        "extraction_adapter": "PyPdfExtractorAdapter",
        "segmentation_policy": "paragraph_split_with_min_50_chars",
        "segmentation_parameters": {"min_chars": 50, "split_regex": r"(\n\s*\n+)"},
        "passage_count": len(entries),
        "registry_sha256": registry_sha256,
        "created_by": "deterministic_offline_builder",
        "network_used": False,
        "api_used": False,
    }

    manifest_file.write_text(
        json.dumps(manifest_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    return registry_file, manifest_file, registry_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description="Build passage registry for Gate B1")
    parser.add_argument(
        "--pdf-path", type=Path, default=DEFAULT_PDF_PATH, help="Path to PDF"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "benchmarks" / "ground_truth" / "v2",
        help="Output directory for registry and manifest",
    )
    parser.add_argument(
        "--page-start", type=int, default=91, help="Start page (inclusive)"
    )
    parser.add_argument(
        "--page-end", type=int, default=115, help="End page (inclusive)"
    )
    args = parser.parse_args()

    reg_path, man_path, reg_sha = build_passage_registry(
        pdf_path=args.pdf_path,
        output_dir=args.output_dir,
        page_start=args.page_start,
        page_end=args.page_end,
    )
    print("PASSAGE REGISTRY BUILT SUCCESSFULLY")
    print(f"Registry Path: {reg_path}")
    print(f"Manifest Path: {man_path}")
    print(f"Registry SHA256: {reg_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
