"""Tests for Slice 3 domain entities: HierarchicalNode and AutoMergingTrace."""

from __future__ import annotations

import dataclasses

import pytest

from raglab.domain.hierarchy import (
    AutoMergingTrace,
    HierarchicalNode,
    HierarchyLevel,
    HierarchyStats,
    MergeDecision,
)

# ---------------------------------------------------------------------------
# HierarchicalNode
# ---------------------------------------------------------------------------

class TestHierarchicalNode:
    def _make_node(self, **kwargs) -> HierarchicalNode:
        defaults = {
            "node_id": "n001",
            "document_id": "doc_gersting",
            "level": HierarchyLevel.LEAF,
            "text": "Sample leaf text for testing.",
            "page_start": 91,
            "page_end": 91,
            "char_start": 0,
            "char_end": 30,
            "fingerprint": "abcdef12",
            "parent_id": "parent_n001",
            "children_ids": (),
            "token_count": 8,
        }
        defaults.update(kwargs)
        return HierarchicalNode(**defaults)

    def test_leaf_node_created_ok(self):
        node = self._make_node(level=HierarchyLevel.LEAF, children_ids=(), parent_id="p1")
        assert node.level == HierarchyLevel.LEAF
        assert node.children_ids == ()
        assert node.parent_id == "p1"

    def test_parent_node_has_no_parent(self):
        node = self._make_node(
            level=HierarchyLevel.PARENT,
            children_ids=("c1", "c2"),
            parent_id=None,
        )
        assert node.parent_id is None
        assert "c1" in node.children_ids

    def test_leaf_with_children_raises(self):
        with pytest.raises(ValueError, match="LEAF nodes must not have children_ids"):
            self._make_node(level=HierarchyLevel.LEAF, children_ids=("child",))

    def test_empty_node_id_raises(self):
        with pytest.raises(ValueError, match="node_id"):
            self._make_node(node_id="")

    def test_empty_document_id_raises(self):
        with pytest.raises(ValueError, match="document_id"):
            self._make_node(document_id="")

    def test_negative_char_start_raises(self):
        with pytest.raises(ValueError, match="char_start"):
            self._make_node(char_start=-1)

    def test_char_end_before_char_start_raises(self):
        with pytest.raises(ValueError, match="char_end"):
            self._make_node(char_start=100, char_end=50)

    def test_page_end_before_page_start_raises(self):
        with pytest.raises(ValueError, match="page_end"):
            self._make_node(page_start=10, page_end=5)

    def test_middle_level_node(self):
        node = self._make_node(
            level=HierarchyLevel.MIDDLE,
            parent_id="root",
            children_ids=("l1", "l2"),
            token_count=64,
        )
        assert node.level == HierarchyLevel.MIDDLE
        assert len(node.children_ids) == 2

    def test_node_is_frozen(self):
        node = self._make_node()
        with pytest.raises(dataclasses.FrozenInstanceError):
            node.text = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MergeDecision
# ---------------------------------------------------------------------------

class TestMergeDecision:
    def _make(self, **kwargs) -> MergeDecision:
        defaults = {
            "parent_id": "p001",
            "children_retrieved": 3,
            "children_total": 4,
            "coverage_ratio": 0.75,
            "threshold": 0.5,
            "merged": True,
            "tokens_before": 150,
            "tokens_after": 300,
            "relevant_evidence_before": 2,
            "relevant_evidence_after": 2,
            "noise_introduced": False,
        }
        defaults.update(kwargs)
        return MergeDecision(**defaults)

    def test_merged_decision(self):
        d = self._make(merged=True)
        assert d.merged is True
        assert d.coverage_ratio == 0.75
        assert d.tokens_after > d.tokens_before

    def test_refused_decision(self):
        d = self._make(merged=False, coverage_ratio=0.25)
        assert d.merged is False

    def test_noise_introduced_flag(self):
        d = self._make(noise_introduced=True)
        assert d.noise_introduced is True


# ---------------------------------------------------------------------------
# AutoMergingTrace
# ---------------------------------------------------------------------------

class TestAutoMergingTrace:
    def _make_trace(self, **kwargs) -> AutoMergingTrace:
        merge1 = MergeDecision(
            parent_id="p1",
            children_retrieved=3,
            children_total=4,
            coverage_ratio=0.75,
            threshold=0.5,
            merged=True,
            tokens_before=200,
            tokens_after=400,
            relevant_evidence_before=1,
            relevant_evidence_after=1,
            noise_introduced=False,
        )
        merge2 = MergeDecision(
            parent_id="p2",
            children_retrieved=1,
            children_total=4,
            coverage_ratio=0.25,
            threshold=0.5,
            merged=False,
            tokens_before=50,
            tokens_after=50,
            relevant_evidence_before=0,
            relevant_evidence_after=0,
            noise_introduced=False,
        )
        defaults = {
            "query_id": "q_dev_01",
            "leaves_retrieved": 4,
            "parent_candidates": 2,
            "merge_decisions": (merge1, merge2),
            "tokens_before": 250,
            "tokens_after": 450,
            "relevant_evidence_before": 1,
            "relevant_evidence_after": 1,
            "latency_ms": 42.5,
        }
        defaults.update(kwargs)
        return AutoMergingTrace(**defaults)

    def test_merges_performed(self):
        trace = self._make_trace()
        assert trace.merges_performed == 1

    def test_merges_refused(self):
        trace = self._make_trace()
        assert trace.merges_refused == 1

    def test_merge_rate(self):
        trace = self._make_trace()
        # 1 merged / 2 candidates
        assert trace.merge_rate == pytest.approx(0.5)

    def test_parent_promotion_rate(self):
        trace = self._make_trace()
        # 3 leaves replaced by p1 / 4 total leaves retrieved
        assert trace.parent_promotion_rate == pytest.approx(0.75)

    def test_context_expansion_ratio(self):
        trace = self._make_trace()
        assert trace.context_expansion_ratio == pytest.approx(450 / 250)

    def test_relevant_evidence_preservation(self):
        trace = self._make_trace()
        assert trace.relevant_evidence_preservation == pytest.approx(1.0)

    def test_relevant_evidence_loss(self):
        trace = self._make_trace()
        assert trace.relevant_evidence_loss == pytest.approx(0.0)

    def test_zero_leaves_promotion_rate(self):
        trace = self._make_trace(leaves_retrieved=0)
        assert trace.parent_promotion_rate == 0.0

    def test_zero_tokens_before_expansion_ratio(self):
        trace = self._make_trace(tokens_before=0)
        assert trace.context_expansion_ratio == pytest.approx(1.0)

    def test_zero_relevant_before_preservation(self):
        trace = self._make_trace(relevant_evidence_before=0, relevant_evidence_after=0)
        assert trace.relevant_evidence_preservation == pytest.approx(1.0)
        assert trace.relevant_evidence_loss == pytest.approx(0.0)

    def test_evidence_loss_when_evidence_dropped(self):
        trace = self._make_trace(relevant_evidence_before=2, relevant_evidence_after=1)
        assert trace.relevant_evidence_preservation == pytest.approx(0.5)
        assert trace.relevant_evidence_loss == pytest.approx(0.5)

    def test_trace_is_frozen(self):
        trace = self._make_trace()
        with pytest.raises(dataclasses.FrozenInstanceError):
            trace.query_id = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HierarchyStats
# ---------------------------------------------------------------------------

class TestHierarchyStats:
    def test_stats_creation(self):
        stats = HierarchyStats(
            total_nodes=30,
            leaf_count=20,
            middle_count=7,
            parent_count=3,
            avg_leaf_tokens=64.0,
            avg_middle_tokens=128.0,
            avg_parent_tokens=256.0,
        )
        assert stats.total_nodes == 30
        assert stats.leaf_count + stats.middle_count + stats.parent_count == 30

    def test_stats_frozen(self):
        stats = HierarchyStats(
            total_nodes=1, leaf_count=1, middle_count=0, parent_count=0,
            avg_leaf_tokens=32.0, avg_middle_tokens=0.0, avg_parent_tokens=0.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            stats.leaf_count = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HierarchyLevel enum
# ---------------------------------------------------------------------------

class TestHierarchyLevel:
    def test_levels(self):
        assert HierarchyLevel.LEAF.value == 0
        assert HierarchyLevel.MIDDLE.value == 1
        assert HierarchyLevel.PARENT.value == 2

    def test_ordering(self):
        assert HierarchyLevel.LEAF.value < HierarchyLevel.MIDDLE.value
        assert HierarchyLevel.MIDDLE.value < HierarchyLevel.PARENT.value
