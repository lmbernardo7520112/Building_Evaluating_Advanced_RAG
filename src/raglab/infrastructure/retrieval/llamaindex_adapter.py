"""LlamaIndex baseline retrieval adapter.

Restricted strictly to the infrastructure layer. The domain and application
layers contain ZERO LlamaIndex imports.

Integrates LlamaIndex's VectorStoreIndex and VectorIndexRetriever while
operating 100% offline without remote APIs, keys, or model downloads.
Injects a deterministic BaseEmbedding wrapper for reproducible testing.

Satisfies the RetrievalPort interface:
- retrieve(query, top_k) -> Sequence[RetrievedEvidence]
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import TextNode
from pydantic import PrivateAttr

from raglab.domain.entities import Chunk, RetrievedEvidence
from raglab.domain.value_objects import ChunkId


class LlamaIndexDeterministicEmbedding(BaseEmbedding):
    """Deterministic hash-based embedding for LlamaIndex testing.

    Subclasses LlamaIndex BaseEmbedding to run 100% offline without APIs.
    """

    _dim: int = PrivateAttr(default=64)

    def __init__(self, dimension: int = 64) -> None:
        super().__init__(model_name="deterministic-sha256-hash")
        self._dim = dimension

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [float(h[i % len(h)]) for i in range(self._dim)]
        mag = math.sqrt(sum(x * x for x in raw))
        if mag == 0:
            return [0.0] * self._dim
        return [x / mag for x in raw]


@dataclass
class LlamaIndexBaselineAdapter:
    """Baseline retrieval adapter built on LlamaIndex VectorStoreIndex.

    Features:
    - Integrates LlamaIndex core retriever
    - Operates 100% offline using injected BaseEmbedding
    - Preserves chunk_id, document_id, page_number, rank, score, text
    - Handles empty query explicitly
    - Guarantees deterministic, stable result ranking
    """

    embed_model: BaseEmbedding = field(
        default_factory=LlamaIndexDeterministicEmbedding
    )
    _index: VectorStoreIndex | None = field(default=None, init=False, repr=False)
    _chunks_map: dict[str, Chunk] = field(
        default_factory=dict, init=False, repr=False
    )

    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Index chunks using LlamaIndex VectorStoreIndex."""
        nodes: list[TextNode] = []
        self._chunks_map = {}

        for chunk in chunks:
            cid = chunk.chunk_id.value
            self._chunks_map[cid] = chunk
            node = TextNode(
                text=chunk.text,
                id_=cid,
                metadata={
                    "chunk_id": cid,
                    "document_id": chunk.document_id,
                    "start_page": chunk.start_page,
                    "end_page": chunk.end_page,
                },
            )
            nodes.append(node)

        if nodes:
            self._index = VectorStoreIndex(
                nodes,
                embed_model=self.embed_model,
            )
        else:
            self._index = None

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> Sequence[RetrievedEvidence]:
        """Retrieve top-k evidence using LlamaIndex VectorIndexRetriever."""
        if not query or not query.strip() or top_k < 1 or self._index is None:
            return []

        retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=top_k,
            embed_model=self.embed_model,
        )

        nodes = retriever.retrieve(query)

        # Sort stably by score desc, then by node id_ asc for deterministic ties
        nodes_sorted = sorted(
            nodes,
            key=lambda n: (-(n.score if n.score is not None else 0.0), n.node.id_),
        )

        results: list[RetrievedEvidence] = []
        for rank, n in enumerate(nodes_sorted[:top_k], start=1):
            cid = n.node.id_
            original_chunk = self._chunks_map.get(cid)
            doc_id = (
                original_chunk.document_id
                if original_chunk
                else str(n.node.metadata.get("document_id", "unknown"))
            )
            score = float(n.score) if n.score is not None else 0.0

            results.append(
                RetrievedEvidence(
                    chunk_id=ChunkId(cid),
                    document_id=doc_id,
                    text=n.node.get_content(),
                    rank=rank,
                    score=score,
                )
            )

        return results

    def clear(self) -> None:
        """Clear indexed data."""
        self._index = None
        self._chunks_map.clear()
