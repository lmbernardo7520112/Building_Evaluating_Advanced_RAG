"""Evidence accumulator — deduplication, provenance, and canonical ordering."""

from __future__ import annotations

import hashlib

from raglab.agentic.contracts import EvidenceItem, _canonical_json


class EvidenceAccumulator:
    """Accumulates evidence with deduplication by canonical passage ID.

    - Deduplicates by passage_id identity
    - Preserves original rank and provenance
    - Tracks new vs repeated evidence
    - Does NOT alter evidence content
    - Produces deterministic snapshots
    """

    def __init__(self) -> None:
        self._items: dict[str, EvidenceItem] = {}  # passage_id -> item
        self._insertion_order: list[str] = []
        self._total_offered: int = 0
        self._duplicates_rejected: int = 0

    def add(self, item: EvidenceItem) -> bool:
        """Add evidence. Returns True if new, False if duplicate.

        Duplicates are silently deduplicated — the first insertion wins.
        """
        self._total_offered += 1

        if item.passage_id in self._items:
            self._duplicates_rejected += 1
            return False

        self._items[item.passage_id] = item
        self._insertion_order.append(item.passage_id)
        return True

    def add_from_observation(
        self,
        passage_ids: tuple[str, ...],
        document_ids: tuple[str, ...],
        ranks: tuple[int, ...],
        scores: tuple[float, ...],
        content_hashes: tuple[str, ...],
        source_tool_id: str,
        source_invocation_id: str,
    ) -> int:
        """Add evidence items from a tool observation.

        Returns the count of NEW (non-duplicate) items added.
        """
        new_count = 0
        for i, pid in enumerate(passage_ids):
            item = EvidenceItem(
                passage_id=pid,
                document_id=document_ids[i] if i < len(document_ids) else "",
                rank=ranks[i] if i < len(ranks) else i + 1,
                score=scores[i] if i < len(scores) else 0.0,
                content_sha256=content_hashes[i] if i < len(content_hashes) else "",
                source_tool_id=source_tool_id,
                source_invocation_id=source_invocation_id,
            )
            if self.add(item):
                new_count += 1
        return new_count

    @property
    def count(self) -> int:
        """Number of unique evidence items."""
        return len(self._items)

    @property
    def total_offered(self) -> int:
        return self._total_offered

    @property
    def duplicates_rejected(self) -> int:
        return self._duplicates_rejected

    def has(self, passage_id: str) -> bool:
        """Check if a passage ID is already accumulated."""
        return passage_id in self._items

    def items_in_order(self) -> tuple[EvidenceItem, ...]:
        """Return items in insertion order."""
        return tuple(self._items[pid] for pid in self._insertion_order)

    def snapshot_hash(self) -> str:
        """Deterministic SHA-256 of the current evidence state."""
        payload = [
            {
                "passage_id": item.passage_id,
                "document_id": item.document_id,
                "rank": item.rank,
                "score": item.score,
                "content_sha256": item.content_sha256,
                "source_tool_id": item.source_tool_id,
                "source_invocation_id": item.source_invocation_id,
            }
            for item in self.items_in_order()
        ]
        canonical = _canonical_json({"evidence": payload})
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def clear(self) -> None:
        """Reset the accumulator for a new query (QID isolation)."""
        self._items.clear()
        self._insertion_order.clear()
        self._total_offered = 0
        self._duplicates_rejected = 0
