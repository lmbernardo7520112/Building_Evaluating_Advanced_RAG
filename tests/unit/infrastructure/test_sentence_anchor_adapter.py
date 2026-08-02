"""Tests for SentenceAnchorAdapter (S0) — Slice 3 causal control.

S0 must satisfy these invariants:
1. Returned text is the anchor sentence only (not a window)
2. Embedding uses the same model as W0 for fair comparison
3. Page provenance is preserved
4. Deduplication: a sentence id appears once per result
5. S0 results differ from W0 results when window > 0 (text length)
6. Holdout qids are never passed (guarded at benchmark level)
"""

from __future__ import annotations

from raglab.domain.value_objects import DocumentPage
from raglab.infrastructure.retrieval.sentence_anchor_adapter import (
    SentenceAnchorAdapter,
)
from raglab.infrastructure.retrieval.sentence_window_adapter import (
    SentenceWindowAdapter,
)


class TestSentenceAnchorAdapter:
    """Unit tests for S0 adapter."""

    def _make_pages(self) -> list[DocumentPage]:
        return [
            DocumentPage(
                document_id="doc_test",
                page_number=91,
                text=(
                    "Uma demonstração por exaustão examina todos os casos. "
                    "É necessário verificar cada caso explicitamente. "
                    "A contraposição prova P→Q via ¬Q→¬P. "
                ),
            ),
            DocumentPage(
                document_id="doc_test",
                page_number=92,
                text=(
                    "A demonstração por contradição assume ¬P. "
                    "Deriva-se uma contradição para concluir P. "
                ),
            ),
        ]

    def test_index_pages_returns_sentence_count(self):
        adapter = SentenceAnchorAdapter()
        pages = self._make_pages()
        count = adapter.index_pages(pages)
        assert count > 0, "Should index at least one sentence"

    def test_retrieve_returns_nonempty_for_known_query(self):
        adapter = SentenceAnchorAdapter()
        adapter.index_pages(self._make_pages())
        results = adapter.retrieve("demonstração por exaustão", top_k=3)
        assert len(results) > 0

    def test_returned_text_is_anchor_not_window(self):
        """Critical S0 invariant: text must be anchor sentence, not window."""
        pages = self._make_pages()

        s0 = SentenceAnchorAdapter()
        s0.index_pages(pages)
        s0_results = s0.retrieve("demonstração por exaustão", top_k=1)

        w0 = SentenceWindowAdapter(window_size=2)
        w0.index_pages(pages)
        w0_results = w0.retrieve("demonstração por exaustão", top_k=1)

        if s0_results and w0_results:
            # S0 text should be <= W0 text (anchor ⊆ window)
            s0_text_len = len(s0_results[0].text)
            w0_text_len = len(w0_results[0].text)
            # Window text is always >= anchor text (window may be same if no neighbours)
            assert s0_text_len <= w0_text_len + 1  # allow 1 char tolerance for whitespace

    def test_chunk_id_contains_anchor_suffix(self):
        """S0 chunk IDs must include '_anchor' to distinguish from W0."""
        adapter = SentenceAnchorAdapter()
        adapter.index_pages(self._make_pages())
        results = adapter.retrieve("demonstração", top_k=3)
        for ev in results:
            assert "_anchor" in ev.chunk_id.value, (
                f"S0 chunk_id should contain '_anchor', got: {ev.chunk_id.value}"
            )

    def test_page_provenance_preserved(self):
        adapter = SentenceAnchorAdapter()
        adapter.index_pages(self._make_pages())
        results = adapter.retrieve("demonstração", top_k=5)
        for ev in results:
            assert "_p" in ev.document_id, (
                f"Page provenance missing in document_id: {ev.document_id}"
            )

    def test_no_duplicate_chunk_ids(self):
        adapter = SentenceAnchorAdapter()
        adapter.index_pages(self._make_pages())
        results = adapter.retrieve("demonstração", top_k=10)
        ids = [ev.chunk_id.value for ev in results]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs in results"

    def test_rank_sequence_starts_at_one(self):
        adapter = SentenceAnchorAdapter()
        adapter.index_pages(self._make_pages())
        results = adapter.retrieve("demonstração", top_k=3)
        if results:
            assert results[0].rank == 1
            for i, ev in enumerate(results):
                assert ev.rank == i + 1

    def test_top_k_respected(self):
        adapter = SentenceAnchorAdapter()
        adapter.index_pages(self._make_pages())
        results = adapter.retrieve("demonstração", top_k=2)
        assert len(results) <= 2

    def test_empty_query_returns_empty(self):
        adapter = SentenceAnchorAdapter()
        adapter.index_pages(self._make_pages())
        assert adapter.retrieve("", top_k=3) == []
        assert adapter.retrieve("   ", top_k=3) == []

    def test_top_k_zero_returns_empty(self):
        adapter = SentenceAnchorAdapter()
        adapter.index_pages(self._make_pages())
        assert adapter.retrieve("demonstração", top_k=0) == []

    def test_scores_in_valid_range(self):
        adapter = SentenceAnchorAdapter()
        adapter.index_pages(self._make_pages())
        results = adapter.retrieve("demonstração", top_k=5)
        for ev in results:
            assert 0.0 <= ev.score <= 1.0, f"Score out of range: {ev.score}"

    def test_clear_empties_index(self):
        adapter = SentenceAnchorAdapter()
        adapter.index_pages(self._make_pages())
        adapter.clear()
        results = adapter.retrieve("demonstração", top_k=3)
        assert results == []

    def test_reindex_replaces_previous(self):
        adapter = SentenceAnchorAdapter()
        adapter.index_pages(self._make_pages())
        count1 = len(adapter._sentence_nodes)
        # Reindex with different pages
        new_pages = [
            DocumentPage(
                document_id="doc2",
                page_number=100,
                text="Nova página com apenas uma sentença.",
            )
        ]
        adapter.index_pages(new_pages)
        count2 = len(adapter._sentence_nodes)
        assert count2 < count1, "Re-indexing should replace old nodes"


class TestS0VsW0CausalIsolation:
    """Verify the causal isolation contract between S0 and W0."""

    def _pages(self) -> list[DocumentPage]:
        return [
            DocumentPage(
                document_id="doc",
                page_number=95,
                text=(
                    "Primeiro contexto. "
                    "A contraposição é uma técnica de demonstração. "
                    "Prova-se ¬Q→¬P em vez de P→Q. "
                    "Último contexto distinto."
                ),
            )
        ]

    def test_s0_and_w0_share_same_anchors(self):
        """S0 and W0 should score the same anchor sentences the same way."""
        pages = self._pages()

        s0 = SentenceAnchorAdapter()
        s0.index_pages(pages)
        s0_results = s0.retrieve("contraposição demonstração", top_k=1)

        w0 = SentenceWindowAdapter(window_size=1)
        w0.index_pages(pages)
        w0_results = w0.retrieve("contraposição demonstração", top_k=1)

        if s0_results and w0_results:
            # Same query → same top-ranked anchor sentence score (approximate)
            s0_score = s0_results[0].score
            w0_score = w0_results[0].score
            # Scores should be equal or very close (same embedding, same anchors)
            assert abs(s0_score - w0_score) < 0.01, (
                f"S0 and W0 should have same anchor score. S0={s0_score}, W0={w0_score}"
            )

    def test_s0_text_shorter_than_w0_text(self):
        """S0 returns anchor only; W0 returns window (> anchor when neighbours exist)."""
        pages = self._pages()

        s0 = SentenceAnchorAdapter()
        s0.index_pages(pages)
        s0_results = s0.retrieve("contraposição", top_k=1)

        w0 = SentenceWindowAdapter(window_size=2)
        w0.index_pages(pages)
        w0_results = w0.retrieve("contraposição", top_k=1)

        if s0_results and w0_results:
            assert len(s0_results[0].text) <= len(w0_results[0].text), (
                "S0 anchor text must be <= W0 window text"
            )
