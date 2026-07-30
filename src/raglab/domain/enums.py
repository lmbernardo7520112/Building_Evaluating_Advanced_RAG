"""Domain enumerations for the RAG experiment pipeline."""

from __future__ import annotations

from enum import Enum, unique


@unique
class PipelineStrategy(Enum):
    """Retrieval pipeline strategy identifiers.

    Each strategy corresponds to a distinct chunking and retrieval
    approach tested in the controlled experiment.
    """

    BASELINE = "baseline"
    SENTENCE_WINDOW = "sentence_window"
    AUTO_MERGING = "auto_merging"


@unique
class DatasetSplit(Enum):
    """Dataset partition identifiers.

    Enforces explicit separation between development, test,
    and holdout sets to prevent contamination.
    """

    DEVELOPMENT = "development"
    TEST = "test"
    QUERY_HOLDOUT = "query_holdout"
    CORPUS_HOLDOUT = "corpus_holdout"

    @property
    def is_holdout(self) -> bool:
        return self in (DatasetSplit.QUERY_HOLDOUT, DatasetSplit.CORPUS_HOLDOUT)


@unique
class QuestionState(Enum):
    """Execution state for a question within a benchmark run.

    Mirrors the transactional model from v6.1 (cell 54).
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    RETRYABLE = "retryable"
    TERMINAL = "terminal"
    BLOCKED_BY_QUOTA = "blocked_by_quota"


@unique
class MetricName(Enum):
    """Recognized metric identifiers."""

    RECALL_AT_K = "recall_at_k"
    MRR = "mrr"
    NDCG_AT_K = "ndcg_at_k"
    CONTEXT_RELEVANCE = "context_relevance"
    GROUNDEDNESS = "groundedness"
    ANSWER_RELEVANCE = "answer_relevance"
    FACTUAL_CORRECTNESS = "factual_correctness"
    COMPLETENESS = "completeness"
    CITATION_PRECISION = "citation_precision"
    CITATION_RECALL = "citation_recall"
