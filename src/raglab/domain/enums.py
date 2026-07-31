"""Domain enumerations for the RAG experiment pipeline."""

from __future__ import annotations

from enum import Enum, unique


@unique
class PipelineStrategy(Enum):
    """Retrieval pipeline strategy identifiers.

    Each strategy corresponds to a distinct chunking and retrieval
    approach tested in the controlled experiment.

    Causal matrix (Slice 3):
      F0 = BASELINE               — fixed chunks, no window, no reranker
      S0 = SENTENCE_ANCHOR        — sentence indexed, returned as-is (no expansion)
      W0 = SENTENCE_WINDOW        — sentence anchor + window expansion, no reranker
      W1 = SENTENCE_WINDOW_RERANK — W0 + reranker (bi-encoder rescoring)
      H0 = HIERARCHICAL_LEAF      — hierarchy built, search leaves only, no merge
      H1 = AUTO_MERGING           — same hierarchy + auto-merging, no reranker
      H2 = AUTO_MERGING_RERANK    — H1 + reranker (bi-encoder rescoring)
    """

    BASELINE = "baseline"
    SENTENCE_ANCHOR = "sentence_anchor"
    SENTENCE_WINDOW = "sentence_window"
    SENTENCE_WINDOW_RERANK = "sentence_window_rerank"
    HIERARCHICAL_LEAF = "hierarchical_leaf"
    AUTO_MERGING = "auto_merging"
    AUTO_MERGING_RERANK = "auto_merging_rerank"


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


@unique
class RerankerClass(Enum):
    """Actual nature of the reranker — must be documented explicitly.

    Do NOT present a bi-encoder or heuristic as a cross-encoder.
    """

    CROSS_ENCODER = "cross_encoder"
    BI_ENCODER_RESCORING = "bi_encoder_rescoring"
    LATE_INTERACTION = "late_interaction"
    LEXICAL = "lexical"
    HYBRID = "hybrid"
    HEURISTIC = "heuristic"


@unique
class QrelAuditState(Enum):
    """State of a ground-truth annotation after audit.

    Mirrors the audit protocol of Slice 3, Section 4.
    """

    CONFIRMED = "CONFIRMED"
    CORRECTED_WITH_EVIDENCE = "CORRECTED_WITH_EVIDENCE"
    AMBIGUOUS = "AMBIGUOUS"
    UNANSWERABLE_IN_SUBCORPUS = "UNANSWERABLE_IN_SUBCORPUS"


@unique
class ExperimentClassification(Enum):
    """Scientific interpretation of an experimental result.

    Use inconclusive when IC includes zero or test recall is zero.
    Never use exploratory_signal when all gains are in development only.
    """

    ENGINEERING_VALID = "engineering_valid"
    EXPLORATORY_SIGNAL = "exploratory_signal"
    INCONCLUSIVE = "inconclusive"
    OBSERVED_REGRESSION = "observed_regression"
    EXPERIMENT_INVALID = "experiment_invalid"
