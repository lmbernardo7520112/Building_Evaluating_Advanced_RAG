"""Slice 5A.3 Agentic Evaluation Package."""

from raglab.agentic.evaluation.canonical_corpus import load_canonical_corpus_snapshot
from raglab.agentic.evaluation.parity_contracts import (
    ArmConfiguration,
    ArmRun,
    CanonicalCorpusSnapshot,
    CanonicalRetrievedItem,
    JudgmentStatus,
    MappingStatus,
    ParityComparison,
    ParityGateResult,
    ParityOutcomeCategory,
)
from raglab.agentic.evaluation.parity_gate import run_parity_gate_evaluation
from raglab.agentic.evaluation.reproducibility import (
    compare_arm_runs,
    compute_scientific_payload_hash,
)

__all__ = [
    "ArmConfiguration",
    "ArmRun",
    "CanonicalCorpusSnapshot",
    "CanonicalRetrievedItem",
    "JudgmentStatus",
    "MappingStatus",
    "ParityComparison",
    "ParityGateResult",
    "ParityOutcomeCategory",
    "compare_arm_runs",
    "compute_scientific_payload_hash",
    "load_canonical_corpus_snapshot",
    "run_parity_gate_evaluation",
]
