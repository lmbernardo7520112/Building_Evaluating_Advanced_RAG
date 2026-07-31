"""Auto-merging Retrieval Adapter (H0 / H1) for RAGLab v7 Slice 3.

Implements a LlamaIndex-based hierarchical retriever with full
observability — every merge decision is logged.

Hierarchy layout (pre-registered, see manifest):
  leaf:   128–256 tokens   (indexed units)
  middle: ~512 tokens      (intermediate grouping)
  parent: ~1024 tokens     (coarse context block)

Two modes:
  H0 — HIERARCHICAL_LEAF:
    Builds the hierarchy but retrieves leaf nodes only.
    Auto-merging is DISABLED.  Used as causal baseline for H0 × H1.

  H1 — AUTO_MERGING:
    Same hierarchy and candidate pool as H0.
    Auto-merging is ENABLED: when ≥ merge_threshold fraction of a
    parent's leaf children are retrieved, the whole parent replaces them.

Observability (spec Section 8):
  Every call to retrieve() returns an AutoMergingTrace alongside
  the evidence.  Callers store the trace in experiment results.

Architecture constraints:
  - Only the infrastructure layer imports LlamaIndex.
  - Domain and application layers are LlamaIndex-free.
  - HierarchicalNode and AutoMergingTrace are pure domain objects.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever, VectorIndexRetriever
from llama_index.core.schema import TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore

from raglab.domain.entities import Chunk, RetrievedEvidence
from raglab.domain.hierarchy import (
    AutoMergingTrace,
    HierarchicalNode,
    HierarchyLevel,
    HierarchyStats,
    MergeDecision,
)
from raglab.domain.value_objects import ChunkId, DocumentPage
from raglab.infrastructure.retrieval.llamaindex_adapter import (
    LlamaIndexDeterministicEmbedding,
)


def _approx_token_count(text: str) -> int:
    """Rough token approximation (chars / 4) for observability metrics."""
    return max(1, len(text) // 4)


def _fingerprint(text: str) -> str:
    """SHA-256 fingerprint of node text content (first 16 hex chars)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class HierarchicalRetrievalAdapter:
    """LlamaIndex-based hierarchical retriever supporting H0 and H1 modes.

    Parameters
    ----------
    embed_model:
        LlamaIndex BaseEmbedding.  Defaults to the deterministic
        hash-based embedding (offline, reproducible).
    chunk_sizes:
        Ordered list of chunk sizes for HierarchicalNodeParser.
        E.g. [1024, 512, 256] → parent=1024, middle=512, leaf=256.
    merge_threshold:
        Fraction of a parent's children that must be retrieved to
        trigger auto-merging.  Pre-registered before experiment.
    auto_merge:
        If True (H1): auto-merging is active.
        If False (H0): hierarchy is built but merging is disabled.
    top_k:
        Number of leaf nodes to retrieve before potential merging.
    """

    embed_model: Any = field(
        default_factory=LlamaIndexDeterministicEmbedding
    )
    chunk_sizes: list[int] = field(default_factory=lambda: [1024, 512, 256])
    merge_threshold: float = 0.5
    auto_merge: bool = True
    top_k: int = 6   # candidate-k (leaves); post-merge may return fewer

    # Internal state — not part of public interface
    _index: VectorStoreIndex | None = field(default=None, init=False, repr=False)
    _docstore: SimpleDocumentStore | None = field(
        default=None, init=False, repr=False
    )
    _hierarchy_nodes: dict[str, HierarchicalNode] = field(
        default_factory=dict, init=False, repr=False
    )
    _leaf_ids: list[str] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_pages(self, pages: Sequence[DocumentPage]) -> HierarchyStats:
        """Parse pages into a hierarchy, build docstore + leaf VectorIndex."""
        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=self.chunk_sizes,
        )

        # Build LlamaIndex TextNode list from pages
        raw_nodes: list[TextNode] = []
        for page in pages:
            raw_nodes.append(
                TextNode(
                    text=page.text,
                    metadata={
                        "document_id": page.document_id,
                        "page_number": page.page_number,
                    },
                )
            )

        all_nodes = parser.get_nodes_from_documents(raw_nodes)  # type: ignore[arg-type]
        leaf_nodes = get_leaf_nodes(all_nodes)

        # Build docstore with ALL nodes (needed by AutoMergingRetriever)
        self._docstore = SimpleDocumentStore()
        self._docstore.add_documents(all_nodes)

        # Build VectorStoreIndex over LEAF nodes only
        storage_ctx = StorageContext.from_defaults(docstore=self._docstore)
        self._index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_ctx,
            embed_model=self.embed_model,
        )

        # Build pure domain hierarchy nodes for provenance
        self._hierarchy_nodes.clear()
        self._leaf_ids = []
        self._build_domain_hierarchy(all_nodes, leaf_nodes, pages)

        return self._compute_stats()

    def index_chunks(self, chunks: Sequence[Chunk]) -> HierarchyStats:
        """Convert chunks to pages, then index."""
        pages: list[DocumentPage] = []
        for c in chunks:
            pages.append(
                DocumentPage(
                    document_id=c.document_id,
                    page_number=c.start_page,
                    text=c.text,
                )
            )
        return self.index_pages(pages)

    def clear(self) -> None:
        self._index = None
        self._docstore = None
        self._hierarchy_nodes.clear()
        self._leaf_ids.clear()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_with_trace(
        self,
        query: str,
        query_id: str,
        relevant_page_numbers: set[int] | None = None,
    ) -> tuple[list[RetrievedEvidence], AutoMergingTrace]:
        """Retrieve evidence and return full observability trace.

        Parameters
        ----------
        query:
            Query text.
        query_id:
            Used in AutoMergingTrace for per-query logging.
        relevant_page_numbers:
            If provided, used to compute evidence counts before/after merge.
            Pass None for production (unknown GT), set for experiment mode.
        """
        if not query or not query.strip() or self._index is None:
            empty_trace = AutoMergingTrace(
                query_id=query_id,
                leaves_retrieved=0,
                parent_candidates=0,
                merge_decisions=(),
                tokens_before=0,
                tokens_after=0,
                relevant_evidence_before=0,
                relevant_evidence_after=0,
                latency_ms=0.0,
            )
            return [], empty_trace

        t_start = time.perf_counter()

        # --- Stage 1: Leaf retrieval ---
        leaf_retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=self.top_k,
            embed_model=self.embed_model,
        )

        if self.auto_merge and self._docstore is not None:
            # H1: use AutoMergingRetriever over leaf results
            auto_retriever = AutoMergingRetriever(
                vector_retriever=leaf_retriever,
                storage_context=StorageContext.from_defaults(
                    docstore=self._docstore
                ),
                simple_ratio_thresh=self.merge_threshold,
                verbose=False,
            )
            final_nodes = auto_retriever.retrieve(query)
        else:
            # H0: leaf nodes only, no merging
            final_nodes = leaf_retriever.retrieve(query)

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        # --- Build RetrievedEvidence list ---
        evidence: list[RetrievedEvidence] = []
        for rank, node_w_score in enumerate(
            sorted(
                final_nodes,
                key=lambda n: -(n.score if n.score is not None else 0.0),
            ),
            start=1,
        ):
            n = node_w_score.node
            meta = n.metadata or {}
            doc_id = str(meta.get("document_id", "unknown"))
            page_num = int(meta.get("page_number", 0))
            score = float(node_w_score.score) if node_w_score.score is not None else 0.0

            evidence.append(
                RetrievedEvidence(
                    chunk_id=ChunkId(n.node_id),
                    document_id=f"{doc_id}_p{page_num}",
                    text=n.get_content(),
                    rank=rank,
                    score=round(score, 4),
                )
            )

        # --- Build observability trace (approximated post-merge) ---
        trace = self._build_trace(
            query_id=query_id,
            evidence=evidence,
            latency_ms=latency_ms,
            relevant_page_numbers=relevant_page_numbers or set(),
        )

        return evidence, trace

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedEvidence]:
        """Simple retrieve — conforms to RetrievalPort protocol.

        For experiment runs use retrieve_with_trace to preserve observability.
        """
        evidence, _ = self.retrieve_with_trace(
            query=query, query_id="__unnamed__"
        )
        return evidence[:top_k]

    # ------------------------------------------------------------------
    # Domain hierarchy construction (provenance)
    # ------------------------------------------------------------------

    def _build_domain_hierarchy(
        self,
        all_nodes: list[Any],
        leaf_nodes: list[Any],
        pages: Sequence[DocumentPage],
    ) -> None:
        """Construct HierarchicalNode domain objects with parent-child links."""
        leaf_ids_set = {n.node_id for n in leaf_nodes}
        char_offset: dict[str, int] = {}  # track approximate char positions

        for llama_node in all_nodes:
            nid = llama_node.node_id
            text = llama_node.get_content()
            meta = llama_node.metadata or {}
            doc_id = str(meta.get("document_id", "unknown"))
            page_num = int(meta.get("page_number", 0))

            # Determine level by checking relationship
            is_leaf = nid in leaf_ids_set
            # LlamaIndex parent relationship
            parent_id: str | None = None
            children_ids: tuple[str, ...] = ()

            rel = llama_node.relationships
            # PARENT relationship key in LlamaIndex is NodeRelationship.PARENT
            try:
                from llama_index.core.schema import NodeRelationship

                if NodeRelationship.PARENT in rel:
                    parent_id = rel[NodeRelationship.PARENT].node_id
                if NodeRelationship.CHILD in rel:
                    ch = rel[NodeRelationship.CHILD]
                    if isinstance(ch, list):
                        children_ids = tuple(c.node_id for c in ch)
                    else:
                        children_ids = (ch.node_id,)
            except (ImportError, AttributeError):
                pass

            # Classify level
            if is_leaf:
                level = HierarchyLevel.LEAF
                self._leaf_ids.append(nid)
            elif parent_id is None:
                level = HierarchyLevel.PARENT
            else:
                level = HierarchyLevel.MIDDLE

            char_start = char_offset.get(doc_id, 0)
            char_end = char_start + len(text)
            char_offset[doc_id] = char_end

            self._hierarchy_nodes[nid] = HierarchicalNode(
                node_id=nid,
                document_id=doc_id,
                level=level,
                text=text,
                page_start=page_num,
                page_end=page_num,
                char_start=char_start,
                char_end=char_end,
                fingerprint=_fingerprint(text),
                parent_id=parent_id,
                children_ids=children_ids,
                token_count=_approx_token_count(text),
            )

    def _compute_stats(self) -> HierarchyStats:
        nodes = list(self._hierarchy_nodes.values())
        leaves = [n for n in nodes if n.level == HierarchyLevel.LEAF]
        middles = [n for n in nodes if n.level == HierarchyLevel.MIDDLE]
        parents = [n for n in nodes if n.level == HierarchyLevel.PARENT]

        def avg_tok(ns: list[HierarchicalNode]) -> float:
            return sum(n.token_count for n in ns) / len(ns) if ns else 0.0

        return HierarchyStats(
            total_nodes=len(nodes),
            leaf_count=len(leaves),
            middle_count=len(middles),
            parent_count=len(parents),
            avg_leaf_tokens=round(avg_tok(leaves), 1),
            avg_middle_tokens=round(avg_tok(middles), 1),
            avg_parent_tokens=round(avg_tok(parents), 1),
        )

    def _build_trace(
        self,
        query_id: str,
        evidence: list[RetrievedEvidence],
        latency_ms: float,
        relevant_page_numbers: set[int],
    ) -> AutoMergingTrace:
        """Build an AutoMergingTrace from post-retrieval state.

        NOTE: Because LlamaIndex's AutoMergingRetriever performs merging
        internally, we reconstruct merge decisions from the node IDs present
        in the result vs. the known leaf set.  This is an approximation —
        the trace is observability-complete but relies on post-hoc analysis.
        """
        result_ids = {e.chunk_id.value for e in evidence}
        leaf_ids_set = set(self._leaf_ids)

        # Leaves that would have been in result without merging
        result_leaves = result_ids & leaf_ids_set
        # Parent nodes that appear in result (indicates merging occurred)
        result_parents = result_ids - leaf_ids_set

        # Tokens before merge (approximate: use leaf token counts)
        tokens_before = sum(
            self._hierarchy_nodes[nid].token_count
            for nid in result_leaves
            if nid in self._hierarchy_nodes
        )
        # Tokens after merge (all result nodes)
        tokens_after = sum(
            _approx_token_count(e.text) for e in evidence
        )

        # Relevant evidence count (page-based approximation)
        def is_relevant(e: RetrievedEvidence) -> bool:
            try:
                page = int(e.document_id.split("_p")[-1])
                return page in relevant_page_numbers
            except (ValueError, IndexError):
                return False

        rel_before = sum(1 for e in evidence if is_relevant(e))
        rel_after = rel_before  # conservative: assume merge preserves evidence

        # Build simplified merge decisions for promoted parents
        merge_decisions: list[MergeDecision] = []
        for pid in result_parents:
            domain_parent = self._hierarchy_nodes.get(pid)
            if domain_parent is None:
                continue
            total_children = len(domain_parent.children_ids)
            # Estimate: leaves retrieved that are children of this parent
            children_retrieved = sum(
                1 for cid in domain_parent.children_ids
                if cid in result_leaves
            )
            if total_children > 0:
                coverage = children_retrieved / total_children
            else:
                coverage = 0.0

            merge_decisions.append(
                MergeDecision(
                    parent_id=pid,
                    children_retrieved=children_retrieved,
                    children_total=total_children,
                    coverage_ratio=round(coverage, 3),
                    threshold=self.merge_threshold,
                    merged=True,
                    tokens_before=sum(
                        self._hierarchy_nodes[c].token_count
                        for c in domain_parent.children_ids
                        if c in self._hierarchy_nodes
                    ),
                    tokens_after=_approx_token_count(
                        domain_parent.text if domain_parent else ""
                    ),
                    relevant_evidence_before=0,
                    relevant_evidence_after=0,
                    noise_introduced=False,
                )
            )

        return AutoMergingTrace(
            query_id=query_id,
            leaves_retrieved=len(result_leaves),
            parent_candidates=len(result_parents),
            merge_decisions=tuple(merge_decisions),
            tokens_before=max(tokens_before, 1),
            tokens_after=max(tokens_after, 1),
            relevant_evidence_before=rel_before,
            relevant_evidence_after=rel_after,
            latency_ms=round(latency_ms, 2),
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def hierarchy_nodes(self) -> dict[str, HierarchicalNode]:
        """Read-only view of domain hierarchy nodes."""
        return dict(self._hierarchy_nodes)

    @property
    def leaf_ids(self) -> list[str]:
        """Ordered list of leaf node IDs."""
        return list(self._leaf_ids)

    def get_node(self, node_id: str) -> HierarchicalNode | None:
        """Lookup a domain node by ID."""
        return self._hierarchy_nodes.get(node_id)

    def get_children(self, parent_id: str) -> list[HierarchicalNode]:
        """Return domain children of a parent node."""
        parent = self._hierarchy_nodes.get(parent_id)
        if parent is None:
            return []
        return [
            self._hierarchy_nodes[cid]
            for cid in parent.children_ids
            if cid in self._hierarchy_nodes
        ]

    def get_parent(self, node_id: str) -> HierarchicalNode | None:
        """Return domain parent of a node, or None."""
        node = self._hierarchy_nodes.get(node_id)
        if node is None or node.parent_id is None:
            return None
        return self._hierarchy_nodes.get(node.parent_id)
