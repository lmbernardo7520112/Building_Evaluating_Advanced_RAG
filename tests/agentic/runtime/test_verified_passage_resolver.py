"""RED contract tests for VerifiedPassageResolver and PassageLookupPort."""

from __future__ import annotations

import hashlib
import unittest
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from raglab.agentic.contracts import EvidenceItem
from raglab.agentic.errors import NonCanonicalIdError
from raglab.domain.entities import RetrievedEvidence
from raglab.domain.value_objects import ChunkId


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class MockPassageLookupPort:
    """Mock implementation of PassageLookupPort for testing resolver behavior."""

    payload_map: dict[str, Any] = field(default_factory=dict)
    return_order: list[str] | None = None
    custom_returns: Sequence[Any] | None = None
    lookup_calls: list[list[str]] = field(default_factory=list)

    def lookup_passages(self, passage_ids: Sequence[str]) -> Sequence[Any]:
        self.lookup_calls.append(list(passage_ids))
        if self.custom_returns is not None:
            return self.custom_returns
        if self.return_order is not None:
            return [
                self.payload_map[pid]
                for pid in self.return_order
                if pid in self.payload_map
            ]
        return [
            self.payload_map[pid]
            for pid in passage_ids
            if pid in self.payload_map
        ]


class TestVerifiedPassageResolver(unittest.TestCase):
    """RED test suite defining contract invariants for VerifiedPassageResolver."""

    def _import_target_modules(self) -> None:
        """Dynamic runtime import helper to ensure tests are collectable in RED phase."""
        from raglab.agentic.runtime.passage_resolver import (  # noqa: F401
            PassageLookupPort,
            PassagePayload,
            VerifiedPassageResolver,
        )

        from raglab.agentic.errors import (  # noqa: F401
            PassageIntegrityError,
            PassageNotFoundError,
        )

    def test_01_empty_input_returns_empty_tuple_without_lookup(self) -> None:
        self._import_target_modules()
        from raglab.agentic.runtime.passage_resolver import (
            VerifiedPassageResolver,
        )

        port = MockPassageLookupPort()
        resolver = VerifiedPassageResolver(port)

        result = resolver.resolve(())

        self.assertEqual(result, ())
        self.assertEqual(len(port.lookup_calls), 0)

    def test_02_reconstructs_input_order_from_reversed_lookup(self) -> None:
        self._import_target_modules()
        from raglab.agentic.runtime.passage_resolver import (
            PassagePayload,
            VerifiedPassageResolver,
        )

        t1, t2, t3 = "Text one", "Text two", "Text three"
        sha1, sha2, sha3 = _sha256(t1), _sha256(t2), _sha256(t3)

        p1 = PassagePayload("ps_001", ChunkId("chunk_a"), "doc_1", t1, sha1)
        p2 = PassagePayload("ps_002", ChunkId("chunk_b"), "doc_1", t2, sha2)
        p3 = PassagePayload("ps_003", ChunkId("chunk_c"), "doc_1", t3, sha3)

        port = MockPassageLookupPort(
            payload_map={"ps_001": p1, "ps_002": p2, "ps_003": p3},
            return_order=["ps_003", "ps_001", "ps_002"],
        )
        resolver = VerifiedPassageResolver(port)

        items = (
            EvidenceItem("ps_001", "doc_1", 1, 0.9, sha1, "tool_1", "inv_1"),
            EvidenceItem("ps_002", "doc_1", 2, 0.8, sha2, "tool_1", "inv_1"),
            EvidenceItem("ps_003", "doc_1", 3, 0.7, sha3, "tool_1", "inv_1"),
        )

        resolved = resolver.resolve(items)

        self.assertEqual(len(resolved), 3)
        self.assertEqual(
            [r.passage_id for r in resolved],
            ["ps_001", "ps_002", "ps_003"],
        )
        self.assertEqual(
            [r.chunk_id.value for r in resolved],
            ["chunk_a", "chunk_b", "chunk_c"],
        )

    def test_03_maps_real_chunk_text_rank_and_score(self) -> None:
        self._import_target_modules()
        from raglab.agentic.runtime.passage_resolver import (
            PassagePayload,
            VerifiedPassageResolver,
        )

        text = "Verified passage text from store."
        sha = _sha256(text)
        real_chunk = ChunkId("real_chunk_guid_98765")

        payload = PassagePayload(
            passage_id="ps_chapter1_001",
            chunk_id=real_chunk,
            document_id="doc_stat_101",
            text=text,
            content_sha256=sha,
        )
        port = MockPassageLookupPort(payload_map={"ps_chapter1_001": payload})
        resolver = VerifiedPassageResolver(port)

        item = EvidenceItem(
            passage_id="ps_chapter1_001",
            document_id="doc_stat_101",
            rank=1,
            score=0.985,
            content_sha256=sha,
            source_tool_id="retrieve_baseline",
            source_invocation_id="inv_abc123",
        )

        result = resolver.resolve((item,))

        self.assertEqual(len(result), 1)
        ev = result[0]
        self.assertIsInstance(ev, RetrievedEvidence)
        self.assertEqual(ev.chunk_id, real_chunk)
        self.assertNotEqual(ev.chunk_id.value, "ps_chapter1_001")
        self.assertEqual(ev.text, text)
        self.assertEqual(ev.rank, 1)
        self.assertEqual(ev.score, 0.985)
        self.assertEqual(ev.document_id, "doc_stat_101")
        self.assertEqual(ev.passage_id, "ps_chapter1_001")
        self.assertEqual(ev.content_sha256, sha)

    def test_04_rejects_noncanonical_input_or_payload_id(self) -> None:
        self._import_target_modules()
        from raglab.agentic.runtime.passage_resolver import (
            PassagePayload,
            VerifiedPassageResolver,
        )

        text = "Some text"
        sha = _sha256(text)
        payload_bad = PassagePayload(
            "raw_bad_id_001", ChunkId("c1"), "doc_1", text, sha
        )
        port = MockPassageLookupPort(payload_map={"raw_bad_id_001": payload_bad})
        resolver = VerifiedPassageResolver(port)

        valid_item = EvidenceItem(
            "ps_001", "doc_1", 1, 0.9, sha, "tool_1", "inv_1"
        )
        with self.assertRaises(NonCanonicalIdError):
            resolver.resolve((valid_item,))

    def test_05_rejects_duplicate_input_ids(self) -> None:
        self._import_target_modules()
        from raglab.agentic.runtime.passage_resolver import (
            PassagePayload,
            VerifiedPassageResolver,
        )

        from raglab.agentic.errors import PassageIntegrityError

        text = "Unique text"
        sha = _sha256(text)
        p = PassagePayload("ps_dup_001", ChunkId("c1"), "doc_1", text, sha)
        port = MockPassageLookupPort(payload_map={"ps_dup_001": p})
        resolver = VerifiedPassageResolver(port)

        item1 = EvidenceItem("ps_dup_001", "doc_1", 1, 0.9, sha, "t1", "inv_1")
        item2 = EvidenceItem("ps_dup_001", "doc_1", 2, 0.8, sha, "t1", "inv_2")

        with self.assertRaises(PassageIntegrityError):
            resolver.resolve((item1, item2))

    def test_06_rejects_missing_unexpected_or_duplicate_lookup_ids(self) -> None:
        self._import_target_modules()
        from raglab.agentic.runtime.passage_resolver import (
            PassagePayload,
            VerifiedPassageResolver,
        )

        from raglab.agentic.errors import (
            PassageIntegrityError,
            PassageNotFoundError,
        )

        t1, t2 = "Text 1", "Text 2"
        sha1, sha2 = _sha256(t1), _sha256(t2)
        p1 = PassagePayload("ps_001", ChunkId("c1"), "doc_1", t1, sha1)
        p2 = PassagePayload("ps_002", ChunkId("c2"), "doc_1", t2, sha2)
        p_extra = PassagePayload("ps_unexpected", ChunkId("cx"), "doc_1", t1, sha1)

        item1 = EvidenceItem("ps_001", "doc_1", 1, 0.9, sha1, "t1", "inv_1")
        item2 = EvidenceItem("ps_002", "doc_1", 2, 0.8, sha2, "t1", "inv_1")

        # Missing lookup ID
        port_missing = MockPassageLookupPort(payload_map={"ps_001": p1})
        resolver_missing = VerifiedPassageResolver(port_missing)
        with self.assertRaises(PassageNotFoundError):
            resolver_missing.resolve((item1, item2))

        # Unexpected lookup ID
        port_unexpected = MockPassageLookupPort(
            custom_returns=[p1, p2, p_extra]
        )
        resolver_unexpected = VerifiedPassageResolver(port_unexpected)
        with self.assertRaises(PassageIntegrityError):
            resolver_unexpected.resolve((item1, item2))

        # Duplicate lookup payload
        port_dup_lookup = MockPassageLookupPort(custom_returns=[p1, p1])
        resolver_dup_lookup = VerifiedPassageResolver(port_dup_lookup)
        with self.assertRaises(PassageIntegrityError):
            resolver_dup_lookup.resolve((item1,))

    def test_07_rejects_document_id_mismatch(self) -> None:
        self._import_target_modules()
        from raglab.agentic.runtime.passage_resolver import (
            PassagePayload,
            VerifiedPassageResolver,
        )

        from raglab.agentic.errors import PassageIntegrityError

        text = "Mismatched document text"
        sha = _sha256(text)
        payload = PassagePayload(
            "ps_001", ChunkId("c1"), "doc_source_real", text, sha
        )
        port = MockPassageLookupPort(payload_map={"ps_001": payload})
        resolver = VerifiedPassageResolver(port)

        item = EvidenceItem(
            "ps_001", "doc_mismatched_claim", 1, 0.9, sha, "t1", "inv_1"
        )
        with self.assertRaises(PassageIntegrityError):
            resolver.resolve((item,))

    def test_08_rejects_empty_text_with_typed_error(self) -> None:
        self._import_target_modules()
        from raglab.agentic.runtime.passage_resolver import (
            PassagePayload,
            VerifiedPassageResolver,
        )

        from raglab.agentic.errors import PassageIntegrityError

        empty_text = "   "
        sha = _sha256(empty_text)
        payload = PassagePayload("ps_001", ChunkId("c1"), "doc_1", empty_text, sha)
        port = MockPassageLookupPort(payload_map={"ps_001": payload})
        resolver = VerifiedPassageResolver(port)

        item = EvidenceItem("ps_001", "doc_1", 1, 0.9, sha, "t1", "inv_1")
        with self.assertRaises(PassageIntegrityError):
            resolver.resolve((item,))

    def test_09_rejects_text_sha256_mismatch(self) -> None:
        self._import_target_modules()
        from raglab.agentic.runtime.passage_resolver import (
            PassagePayload,
            VerifiedPassageResolver,
        )

        from raglab.agentic.errors import PassageIntegrityError

        actual_text = "Actual physical text from disk/store"
        claimed_sha = _sha256("Tampered or different text")

        payload = PassagePayload(
            "ps_001", ChunkId("c1"), "doc_1", actual_text, claimed_sha
        )
        port = MockPassageLookupPort(payload_map={"ps_001": payload})
        resolver = VerifiedPassageResolver(port)

        item = EvidenceItem("ps_001", "doc_1", 1, 0.9, claimed_sha, "t1", "inv_1")
        with self.assertRaises(PassageIntegrityError):
            resolver.resolve((item,))

    def test_10_rejects_payload_hash_mismatch(self) -> None:
        self._import_target_modules()
        from raglab.agentic.runtime.passage_resolver import (
            PassagePayload,
            VerifiedPassageResolver,
        )

        from raglab.agentic.errors import PassageIntegrityError

        text = "Consistent text body"
        sha = _sha256(text)
        corrupted_payload_hash = _sha256("Corrupted internal metadata")

        payload = PassagePayload(
            "ps_001", ChunkId("c1"), "doc_1", text, corrupted_payload_hash
        )
        port = MockPassageLookupPort(payload_map={"ps_001": payload})
        resolver = VerifiedPassageResolver(port)

        item = EvidenceItem("ps_001", "doc_1", 1, 0.9, sha, "t1", "inv_1")
        with self.assertRaises(PassageIntegrityError):
            resolver.resolve((item,))
