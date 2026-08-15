"""RED contract tests for RuntimePassageStore."""

from __future__ import annotations

import hashlib

import pytest

from raglab.agentic.errors import NonCanonicalIdError, PassageIntegrityError
from raglab.agentic.runtime.passage_resolver import PassagePayload
from raglab.domain.value_objects import ChunkId


def _make_payload(
    passage_id: str,
    text: str = "sample text",
    document_id: str = "doc_001",
    chunk_id_val: str | None = None,
    page_number: int | None = None,
) -> PassagePayload:
    cid = ChunkId(chunk_id_val or f"chk_{passage_id}")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return PassagePayload(
        passage_id=passage_id,
        chunk_id=cid,
        document_id=document_id,
        text=text,
        content_sha256=sha,
        page_number=page_number,
    )


class TestRuntimePassageStore:
    """Contract tests for volatile in-memory passage store."""

    def _get_store_cls(self) -> type:
        from raglab.agentic.runtime.runtime_passage_store import (
            RuntimePassageStore,
        )

        return RuntimePassageStore

    def test_01_empty_record_and_lookup_return_empty_tuple(self) -> None:
        store_cls = self._get_store_cls()
        store = store_cls()

        store.record_passages([])
        res = store.lookup_passages([])
        assert res == ()
        assert isinstance(res, tuple)

        res_unrecorded = store.lookup_passages(["ps_nonexistent"])
        assert res_unrecorded == ()
        assert isinstance(res_unrecorded, tuple)

    def test_02_records_and_returns_requested_order(self) -> None:
        store_cls = self._get_store_cls()
        store = store_cls()

        p1 = _make_payload("ps_001", "Text 1")
        p2 = _make_payload("ps_002", "Text 2")
        p3 = _make_payload("ps_003", "Text 3")

        store.record_passages([p1, p2, p3])

        res = store.lookup_passages(["ps_003", "ps_001", "ps_002"])
        assert res == (p3, p1, p2)
        assert isinstance(res, tuple)

    def test_03_identical_duplicate_record_is_idempotent(self) -> None:
        store_cls = self._get_store_cls()
        store = store_cls()

        p1 = _make_payload("ps_001", "Text 1", page_number=5)

        store.record_passages([p1])
        store.record_passages([p1])

        res = store.lookup_passages(["ps_001"])
        assert res == (p1,)

    def test_04_conflicting_existing_payload_fails_without_mutation(
        self,
    ) -> None:
        store_cls = self._get_store_cls()
        store = store_cls()

        p1 = _make_payload("ps_001", "Text 1", page_number=5)
        store.record_passages([p1])

        p1_conflict = _make_payload(
            "ps_001", "Different Text", page_number=5
        )

        with pytest.raises(PassageIntegrityError):
            store.record_passages([p1_conflict])

        assert store.lookup_passages(["ps_001"]) == (p1,)

    def test_05_missing_ids_are_omitted(self) -> None:
        store_cls = self._get_store_cls()
        store = store_cls()

        p1 = _make_payload("ps_001", "Text 1")
        p2 = _make_payload("ps_002", "Text 2")
        store.record_passages([p1, p2])

        res = store.lookup_passages(
            ["ps_001", "ps_missing_999", "ps_002", "ps_missing_888"]
        )
        assert res == (p1, p2)
        assert isinstance(res, tuple)

    def test_06_preserves_none_and_explicit_page_numbers(self) -> None:
        store_cls = self._get_store_cls()
        store = store_cls()

        p_unpaginated = _make_payload(
            "ps_unp_001", "Unpaginated", page_number=None
        )
        p_paginated = _make_payload(
            "ps_pag_002", "Paginated", page_number=42
        )
        p_zero = _make_payload("ps_zero_003", "Zero page", page_number=0)

        store.record_passages([p_unpaginated, p_paginated, p_zero])

        res = store.lookup_passages(
            ["ps_unp_001", "ps_pag_002", "ps_zero_003"]
        )
        assert res[0].page_number is None
        assert res[1].page_number == 42
        assert res[2].page_number == 0

        p_pag_conflict = _make_payload(
            "ps_pag_002", "Paginated", page_number=43
        )
        with pytest.raises(PassageIntegrityError):
            store.record_passages([p_pag_conflict])

    def test_07_instances_have_strict_state_isolation(self) -> None:
        store_cls = self._get_store_cls()
        store1 = store_cls()
        store2 = store_cls()

        p1 = _make_payload("ps_001", "Text 1")
        p2 = _make_payload("ps_002", "Text 2")

        store1.record_passages([p1])
        store2.record_passages([p2])

        assert store1.lookup_passages(["ps_001", "ps_002"]) == (p1,)
        assert store2.lookup_passages(["ps_001", "ps_002"]) == (p2,)

    def test_08_mixed_batch_conflict_is_fully_atomic(self) -> None:
        store_cls = self._get_store_cls()
        store = store_cls()

        p1 = _make_payload("ps_001", "Original Text 1")
        store.record_passages([p1])

        p2 = _make_payload("ps_002", "New Text 2")
        p1_bad = _make_payload("ps_001", "Conflicting Text 1")

        with pytest.raises(PassageIntegrityError):
            store.record_passages([p2, p1_bad])

        assert store.lookup_passages(["ps_001"]) == (p1,)
        assert store.lookup_passages(["ps_002"]) == ()

    def test_09_noncanonical_record_fails_before_any_mutation(self) -> None:
        store_cls = self._get_store_cls()
        store = store_cls()

        p_valid = _make_payload("ps_valid_001", "Valid Text")
        p_invalid = _make_payload("invalid_prefix_002", "Invalid Text")

        with pytest.raises(NonCanonicalIdError):
            store.record_passages([p_valid, p_invalid])

        assert store.lookup_passages(["ps_valid_001"]) == ()

        with pytest.raises(NonCanonicalIdError):
            store.record_passages([p_invalid])
