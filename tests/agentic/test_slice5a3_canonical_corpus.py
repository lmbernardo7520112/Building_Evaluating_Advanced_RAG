"""Hermetic unit tests for Canonical Corpus loader and snapshot validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from raglab.agentic.evaluation.canonical_corpus import (
    CanonicalCorpusError,
    load_canonical_corpus_snapshot,
    passages_to_chunks,
    passages_to_document_pages,
)


class TestCanonicalCorpusLoader(unittest.TestCase):
    """Test suite for canonical_corpus module."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_registry(self, items: list[dict]) -> tuple[Path, Path]:
        reg_path = self.dir_path / "passage_registry.jsonl"
        man_path = self.dir_path / "passage_registry_manifest.json"

        with reg_path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        man_path.write_text(json.dumps({"version": "2.0.0"}), encoding="utf-8")
        return reg_path, man_path

    def test_01_valid_canonical_corpus(self) -> None:
        """Valid canonical corpus passes validation and produces correct snapshot."""
        items = [
            {
                "passage_id": "ps_001",
                "document_id": "doc1",
                "page_number": 91,
                "text": "Valid text 1",
            },
            {
                "passage_id": "ps_002",
                "document_id": "doc1",
                "page_number": 92,
                "text": "Valid text 2",
            },
        ]
        reg_p, man_p = self._write_registry(items)
        snapshot, passages = load_canonical_corpus_snapshot(reg_p, man_p)

        self.assertEqual(snapshot.passage_count, 2)
        self.assertEqual(snapshot.ordered_canonical_passage_ids, ["ps_001", "ps_002"])
        self.assertEqual(snapshot.page_bounds, (91, 92))

    def test_02_duplicate_passage_id(self) -> None:
        """Duplicate passage ID raises CanonicalCorpusError."""
        items = [
            {"passage_id": "ps_001", "page_number": 91, "text": "Text 1"},
            {"passage_id": "ps_001", "page_number": 92, "text": "Text 2"},
        ]
        reg_p, man_p = self._write_registry(items)
        with self.assertRaises(CanonicalCorpusError) as ctx:
            load_canonical_corpus_snapshot(reg_p, man_p)
        self.assertIn("Duplicate passage_id", str(ctx.exception))

    def test_03_empty_text_rejected(self) -> None:
        """Empty text raises CanonicalCorpusError."""
        items = [
            {"passage_id": "ps_001", "page_number": 91, "text": "   "},
        ]
        reg_p, man_p = self._write_registry(items)
        with self.assertRaises(CanonicalCorpusError) as ctx:
            load_canonical_corpus_snapshot(reg_p, man_p)
        self.assertIn("Empty text", str(ctx.exception))

    def test_04_missing_ps_prefix(self) -> None:
        """Passage ID without ps_ prefix raises CanonicalCorpusError."""
        items = [
            {"passage_id": "invalid_id_001", "page_number": 91, "text": "Text 1"},
        ]
        reg_p, man_p = self._write_registry(items)
        with self.assertRaises(CanonicalCorpusError) as ctx:
            load_canonical_corpus_snapshot(reg_p, man_p)
        self.assertIn("prefix", str(ctx.exception))

    def test_05_passages_to_chunks_and_pages(self) -> None:
        """Conversion to domain Chunk and DocumentPage entities works properly."""
        items = [
            {"passage_id": "ps_001", "document_id": "doc1", "page_number": 91, "text": "Text 1"},
            {"passage_id": "ps_002", "document_id": "doc1", "page_number": 91, "text": "Text 2"},
        ]
        chunks = passages_to_chunks(items)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].chunk_id.value, "ps_001")

        pages = passages_to_document_pages(items)
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].page_number, 91)
        self.assertIn("Text 1", pages[0].text)
        self.assertIn("Text 2", pages[0].text)


if __name__ == "__main__":
    unittest.main()
