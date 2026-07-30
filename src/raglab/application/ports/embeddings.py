"""Port: embedding generation."""

from __future__ import annotations

from typing import Protocol, Sequence


class EmbeddingPort(Protocol):
    """Generate embeddings for text chunks."""

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Embed a batch of texts into vector representations."""
        ...

    @property
    def model_id(self) -> str:
        """Return the model identifier for reproducibility."""
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...
