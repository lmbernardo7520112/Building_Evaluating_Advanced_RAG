"""Application DTOs — data transfer objects for cross-layer communication."""

from __future__ import annotations

from dataclasses import dataclass

from raglab.domain.enums import DatasetSplit, PipelineStrategy


@dataclass(frozen=True, slots=True)
class RunRequest:
    """Request to start or resume an experiment run."""

    strategy: PipelineStrategy
    corpus_path: str
    split: DatasetSplit
    resume_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Summary of a completed run for reporting."""

    run_id: str
    strategy: str
    total_queries: int
    completed_queries: int
    abstained_queries: int
    metrics_computed: int
