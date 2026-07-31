"""Unit tests for PyPdfExtractorAdapter."""

import tempfile
import unittest
from pathlib import Path

from pypdf import PdfWriter

from raglab.infrastructure.pdf_parsers.pdf_parser_adapter import (
    PyPdfExtractorAdapter,
    normalize_whitespace,
)


class TestPyPdfExtractorAdapter(unittest.TestCase):

    def test_normalize_whitespace(self) -> None:
        raw = "Esta é uma demonstra-\nção por indução   matemática.\n\n"
        cleaned = normalize_whitespace(raw)
        self.assertEqual(
            cleaned, "Esta é uma demonstração por indução matemática."
        )

    def test_extract_synthetic_pdf(self) -> None:
        # Create minimal synthetic PDF in memory
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            writer.write(tmp_path)

        try:
            adapter = PyPdfExtractorAdapter()
            fingerprint = adapter.compute_fingerprint(str(tmp_path))
            self.assertEqual(len(fingerprint.hex_digest), 64)

            pages, audit = adapter.extract_pages_with_audit(str(tmp_path))
            self.assertEqual(len(pages), 1)
            self.assertEqual(audit["total_document_pages"], 1)
            self.assertEqual(pages[0].page_number, 1)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
