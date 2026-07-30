"""Port: corpus ingestion and document management."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from raglab.domain.entities import Chunk, Corpus, Document
from raglab.domain.value_objects import DocumentPage, IntegrityDigest


class CorpusReaderPort(Protocol):
    """Read and parse source documents into pages."""

    def read_document(self, path: str) -> Sequence[DocumentPage]:
        """Parse a document file into pages with provenance."""
        ...

    def compute_fingerprint(self, path: str) -> IntegrityDigest:
        """Compute integrity digest for a document file."""
        ...


class CorpusStorePort(Protocol):
    """Persist and retrieve corpus metadata."""

    def save_corpus(self, corpus: Corpus) -> None: ...
    def save_document(self, document: Document) -> None: ...
    def save_chunks(self, chunks: Sequence[Chunk]) -> None: ...
    def load_corpus(self, corpus_id: str) -> Corpus | None: ...
    def load_chunks(self, corpus_id: str) -> Sequence[Chunk]: ...
