"""Port: checkpoint persistence."""

from __future__ import annotations

from typing import Protocol

from raglab.domain.entities import Checkpoint
from raglab.domain.value_objects import RunId


class CheckpointPort(Protocol):
    """Persist and restore experiment checkpoints.

    Checkpoints are tied to specific corpus and config fingerprints.
    Loading a checkpoint with mismatched fingerprints must be detected
    and rejected by the domain layer.
    """

    def save(self, checkpoint: Checkpoint) -> None:
        """Persist a checkpoint."""
        ...

    def load(self, run_id: RunId) -> Checkpoint | None:
        """Load a checkpoint if it exists, None otherwise."""
        ...

    def exists(self, run_id: RunId) -> bool:
        """Check if a checkpoint exists for this run."""
        ...
