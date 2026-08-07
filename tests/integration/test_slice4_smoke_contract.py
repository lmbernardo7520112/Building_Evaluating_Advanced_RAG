"""Tests for Slice 4 smoke test evidence and evaluation contract.

Covers the 24 mandatory scenarios from the fix specification:
1.  Manifest absent
2.  Manifest invalid
3.  Model incompatible
4.  Hash UNRESOLVED
5.  Fingerprint valid
6.  Evidence serialized
7.  Relevant page found
8.  Relevant page absent
9.  Score absent != zero
10. evaluation=null rejected
11. Metric FAILED rejected
12. Metric NOT_EXECUTED rejected
13. Metric computed out of [0,1]
14. Correct abstention
15. Incorrect abstention
16. Groundedness NA on ABSTAIN
17. Smoke positive valid
18. Smoke positive with ABSTAIN fails
19. Smoke abstention correct
20. Holdout access fails
21. Citation without evidence fails
22. Checkpoint incompatible fails
23. No SMOKE_*_OK after failure
24. No secrets in result

No network, no Gemini, no HuggingFace.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))

runner: Any = importlib.import_module("run_slice4_benchmark")


# ─── Helpers ──────────────────────────────────────────────────────

def _make_manifest(tmp_path: Path, **overrides: Any) -> Path:
    """Create a valid provision manifest file."""
    data: dict[str, Any] = {
        "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "fastembed_version": "0.8.0",
        "onnxruntime_version": "1.28.0",
        "dimension": 384,
        "pooling": "mean",
        "normalization": True,
        "cache_tree_sha256": "a" * 64,
        "canary_dim_ok": True,
        "canary_finite_ok": True,
    }
    data.update(overrides)
    path = tmp_path / "provision_manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _make_smoke_result(
    *,
    strategy: str = "W0_sentence_window",
    qid: str = "q_dev_01",
    abstained: bool = False,
    is_abstention_question: bool = False,
    metrics: list[dict[str, Any]] | None = None,
    fingerprint_sha: str = "b" * 64,
    manifest_fingerprint: str = "b" * 64,
    evaluation: dict[str, Any] | None = None,
    retrieval_evidence: dict[str, Any] | None = None,
    citation_pages: list[int] | None = None,
) -> dict[str, Any]:
    """Build a minimal smoke result dict for validation tests."""
    if metrics is None:
        if abstained:
            metrics = [
                {"name": "abstention_correctness", "status": "COMPUTED",
                 "score": 1.0, "reason": "CORRECT_ABSTENTION",
                 "evaluator_model": "deterministic", "attempts": 1},
                {"name": "context_relevance", "status": "NOT_APPLICABLE",
                 "score": None, "reason": "NO_EVIDENCE_RETRIEVED",
                 "evaluator_model": "", "attempts": 0},
                {"name": "groundedness", "status": "NOT_APPLICABLE",
                 "score": None, "reason": "ABSTAINED_WITHOUT_CLAIMS",
                 "evaluator_model": "", "attempts": 0},
                {"name": "answer_relevance", "status": "NOT_APPLICABLE",
                 "score": None, "reason": "ABSTAINED_WITHOUT_CLAIMS",
                 "evaluator_model": "", "attempts": 0},
            ]
        else:
            metrics = [
                {"name": "abstention_correctness", "status": "NOT_APPLICABLE",
                 "score": None, "reason": "SUBSTANTIVE_ANSWER_NOT_ABSTENTION_QUESTION",
                 "evaluator_model": "", "attempts": 0},
                {"name": "context_relevance", "status": "COMPUTED",
                 "score": 0.8, "reason": "", "evaluator_model": "gemini-3.1-flash-lite",
                 "attempts": 1},
                {"name": "groundedness", "status": "COMPUTED",
                 "score": 0.9, "reason": "", "evaluator_model": "gemini-3.1-flash-lite",
                 "attempts": 1},
                {"name": "answer_relevance", "status": "COMPUTED",
                 "score": 0.85, "reason": "", "evaluator_model": "gemini-3.1-flash-lite",
                 "attempts": 1},
            ]

    if evaluation is None:
        evaluation = {
            "schema_version": runner._EVAL_SCHEMA_VERSION,
            "metrics": metrics,
        }

    if retrieval_evidence is None:
        retrieval_evidence = {
            "candidate_count": 3,
            "candidates": [
                {"chunk_id": "doc_p92_c0", "page_number": 92,
                 "retrieval_rank": 1, "retrieval_score": 0.9,
                 "rerank_rank": None, "rerank_score": None,
                 "parent_node_id": None,
                 "text_sha256": "x" * 64, "text_preview": "some text"},
            ],
            "relevant_pages_expected": [92],
            "relevant_pages_found": [92],
            "relevant_pages_missing": [],
            "retrieval_hit": True,
        }

    if citation_pages is None:
        citation_pages = [92] if not abstained else []

    ans_text = "ABSTAIN" if abstained else "Test answer"
    ans_sha = hashlib.sha256(ans_text.encode("utf-8")).hexdigest()

    return {
        "experiment_id": "smoke_test",
        "schema": "slice4_v3",
        "embedding_fingerprints": {
            strategy: {
                "embedding_model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "fastembed_version": "0.8.0",
                "onnxruntime_version": "1.28.0",
                "pooling": "mean",
                "dimension": 384,
                "normalization": True,
                "cache_tree_sha256": fingerprint_sha,
            },
        },
        "manifest_fingerprint": manifest_fingerprint,
        "results": {
            strategy: [{
                "qid": qid,
                "split": "development",
                "strategy": strategy,
                "relevant_pages": [92] if not is_abstention_question else [],
                "abstained": abstained,
                "is_abstention_question": is_abstention_question,
                "answer": {
                    "text": ans_text,
                    "text_sha256": ans_sha,
                    "text_length_chars": len(ans_text),
                    "truncated": False,
                    "preview": ans_text,
                    "abstained": abstained,
                },
                "citation_mapping_status": "NOT_APPLICABLE" if abstained else "AVAILABLE",
                "citation_map": [] if abstained else [{"marker": "[1]", "page_number": 92, "chunk_id": "doc_p92_c0", "text_sha256": "x" * 64}],
                "evaluation": evaluation,
                "retrieval_evidence": retrieval_evidence,
                "citation_pages": citation_pages,
            }],
        },
    }


# ─── 1. Manifest absent ──────────────────────────────────────────

class TestManifestValidation:
    def test_manifest_absent_raises(self, tmp_path):

        with pytest.raises(ValueError, match="EMBEDDING_ATTESTATION_FAILED"):
            runner.load_provision_manifest(tmp_path / "nonexistent.json")

    def test_manifest_invalid_json_raises(self, tmp_path):

        bad = tmp_path / "bad.json"
        bad.write_text("{invalid json!", encoding="utf-8")
        with pytest.raises(ValueError, match="EMBEDDING_ATTESTATION_FAILED"):
            runner.load_provision_manifest(bad)

    def test_model_incompatible_raises(self, tmp_path):

        path = _make_manifest(
            tmp_path, model_id="wrong-model/something-else"
        )
        with pytest.raises(ValueError, match="EMBEDDING_ATTESTATION_FAILED.*Model mismatch"):
            runner.load_provision_manifest(path)

    def test_hash_unresolved_raises(self, tmp_path):

        path = _make_manifest(tmp_path, cache_tree_sha256="UNRESOLVED")
        with pytest.raises(ValueError, match="EMBEDDING_ATTESTATION_FAILED"):
            runner.load_provision_manifest(path)

    def test_valid_manifest_loads(self, tmp_path):

        path = _make_manifest(tmp_path)
        data = runner.load_provision_manifest(path)
        assert data["cache_tree_sha256"] == "a" * 64
        assert data["dimension"] == 384


# ─── 6. Evidence serialized ───────────────────────────────────────

class TestRetrievalEvidence:
    def test_evidence_serialized_correctly(self):
        from raglab.domain.entities import RetrievedEvidence
        from raglab.domain.value_objects import ChunkId

        ev = RetrievedEvidence(
            chunk_id=ChunkId("doc_p92_c0"),
            document_id="test_doc",
            text="Prova por exaustão é uma técnica de demonstração.",
            rank=1,
            score=0.95,
        )

        result = runner.serialize_retrieval_evidence([ev], [92])
        assert result["candidate_count"] == 1
        assert result["retrieval_hit"] is True
        assert result["relevant_pages_found"] == [92]
        assert result["candidates"][0]["chunk_id"] == "doc_p92_c0"
        assert result["candidates"][0]["retrieval_score"] == 0.95
        assert len(result["candidates"][0]["text_sha256"]) == 64

    def test_relevant_page_found(self):
        from raglab.domain.entities import RetrievedEvidence
        from raglab.domain.value_objects import ChunkId

        ev = RetrievedEvidence(
            chunk_id=ChunkId("doc_p92_c0"), document_id="doc",
            text="test", rank=1, score=0.8,
        )
        result = runner.serialize_retrieval_evidence([ev], [92])
        assert 92 in result["relevant_pages_found"]
        assert result["retrieval_hit"] is True

    def test_relevant_page_absent(self):
        from raglab.domain.entities import RetrievedEvidence
        from raglab.domain.value_objects import ChunkId

        ev = RetrievedEvidence(
            chunk_id=ChunkId("doc_p99_c0"), document_id="doc",
            text="test", rank=1, score=0.8,
        )
        result = runner.serialize_retrieval_evidence([ev], [92])
        assert result["retrieval_hit"] is False
        assert 92 in result["relevant_pages_missing"]

    def test_score_absent_different_from_zero(self):
        """None retrieval_score must be distinguishable from 0."""
        from raglab.domain.entities import RetrievedEvidence
        from raglab.domain.value_objects import ChunkId

        ev_zero = RetrievedEvidence(
            chunk_id=ChunkId("doc_p1_c0"), document_id="doc",
            text="t", rank=1, score=0.0,
        )
        result = runner.serialize_retrieval_evidence([ev_zero], [])
        assert result["candidates"][0]["retrieval_score"] == 0.0
        assert result["candidates"][0]["retrieval_score"] is not None


# ─── 10-13. Typed Metrics ─────────────────────────────────────────

class TestTypedMetrics:
    def test_evaluation_null_rejected_by_smoke_validator(self):

        data = _make_smoke_result(evaluation=None)
        # Set evaluation to None in the result
        data["results"]["W0_sentence_window"][0]["evaluation"] = None

        logger = logging.getLogger("test")
        verdict = runner.validate_smoke_result(
            data, "W0_sentence_window", "q_dev_01", False, logger
        )
        assert verdict == "SMOKE_FAILED"

    def test_metric_failed_rejected(self):

        metrics: list[dict[str, Any]] = [
            {"name": "context_relevance", "status": "FAILED",
             "score": None, "reason": "error", "evaluator_model": "",
             "attempts": 1},
            {"name": "abstention_correctness", "status": "NOT_APPLICABLE",
             "score": None, "reason": "", "evaluator_model": "", "attempts": 0},
            {"name": "groundedness", "status": "COMPUTED",
             "score": 0.5, "reason": "", "evaluator_model": "", "attempts": 1},
            {"name": "answer_relevance", "status": "COMPUTED",
             "score": 0.5, "reason": "", "evaluator_model": "", "attempts": 1},
        ]
        data = _make_smoke_result(metrics=metrics)
        logger = logging.getLogger("test")
        verdict = runner.validate_smoke_result(
            data, "W0_sentence_window", "q_dev_01", False, logger
        )
        assert verdict == "SMOKE_FAILED"

    def test_metric_not_executed_rejected(self):

        metrics: list[dict[str, Any]] = [
            {"name": "context_relevance", "status": "NOT_EXECUTED",
             "score": None, "reason": "", "evaluator_model": "",
             "attempts": 0},
            {"name": "abstention_correctness", "status": "NOT_APPLICABLE",
             "score": None, "reason": "", "evaluator_model": "", "attempts": 0},
            {"name": "groundedness", "status": "COMPUTED",
             "score": 0.5, "reason": "", "evaluator_model": "", "attempts": 1},
            {"name": "answer_relevance", "status": "COMPUTED",
             "score": 0.5, "reason": "", "evaluator_model": "", "attempts": 1},
        ]
        data = _make_smoke_result(metrics=metrics)
        logger = logging.getLogger("test")
        verdict = runner.validate_smoke_result(
            data, "W0_sentence_window", "q_dev_01", False, logger
        )
        assert verdict == "SMOKE_FAILED"

    def test_metric_out_of_range_raises(self):

        with pytest.raises(ValueError, match="out of"):
            runner.make_metric_entry(
                "test", "COMPUTED", score=1.5,
            )

    def test_metric_computed_requires_score(self):

        with pytest.raises(ValueError, match="COMPUTED requires a score"):
            runner.make_metric_entry("test", "COMPUTED", score=None)

    def test_metric_non_computed_rejects_score(self):

        with pytest.raises(ValueError, match="must have score=None"):
            runner.make_metric_entry("test", "NOT_APPLICABLE", score=0.5)


# ─── 14-16. Abstention ───────────────────────────────────────────

class TestAbstentionMetrics:
    def test_correct_abstention_score_1(self):

        result = runner.compute_abstention_correctness(
            is_abstention_question=True, abstained=True,
        )
        assert result["status"] == "COMPUTED"
        assert result["score"] == 1.0
        assert result["reason"] == "CORRECT_ABSTENTION"

    def test_incorrect_abstention_on_answerable(self):

        result = runner.compute_abstention_correctness(
            is_abstention_question=False, abstained=True,
        )
        assert result["status"] == "COMPUTED"
        assert result["score"] == 0.0
        assert "INCORRECT_ABSTENTION" in result["reason"]

    def test_failed_to_abstain_on_unanswerable(self):

        result = runner.compute_abstention_correctness(
            is_abstention_question=True, abstained=False,
        )
        assert result["status"] == "COMPUTED"
        assert result["score"] == 0.0

    def test_groundedness_na_on_abstain_in_smoke(self):

        data = _make_smoke_result(
            strategy="F0_baseline",
            qid="q_test_04",
            abstained=True,
            is_abstention_question=True,
            retrieval_evidence={
                "candidate_count": 0, "candidates": [],
                "relevant_pages_expected": [],
                "relevant_pages_found": [],
                "relevant_pages_missing": [],
                "retrieval_hit": False,
            },
        )
        logger = logging.getLogger("test")
        verdict = runner.validate_smoke_result(
            data, "F0_baseline", "q_test_04", True, logger
        )
        assert verdict == "SMOKE_ABSTENTION_OK"


# ─── 17-18. Smoke Positive ────────────────────────────────────────

class TestSmokePositive:
    def test_smoke_positive_valid(self):

        data = _make_smoke_result()
        logger = logging.getLogger("test")
        verdict = runner.validate_smoke_result(
            data, "W0_sentence_window", "q_dev_01", False, logger
        )
        assert verdict == "SMOKE_POSITIVE_OK"

    def test_smoke_positive_with_abstain_fails(self):

        data = _make_smoke_result(abstained=True)
        logger = logging.getLogger("test")
        verdict = runner.validate_smoke_result(
            data, "W0_sentence_window", "q_dev_01", False, logger
        )
        assert verdict == "SMOKE_FAILED"


# ─── 19. Smoke abstention ────────────────────────────────────────

class TestSmokeAbstention:
    def test_smoke_abstention_correct(self):

        data = _make_smoke_result(
            strategy="F0_baseline",
            qid="q_test_04",
            abstained=True,
            is_abstention_question=True,
            retrieval_evidence={
                "candidate_count": 0, "candidates": [],
                "relevant_pages_expected": [],
                "relevant_pages_found": [],
                "relevant_pages_missing": [],
                "retrieval_hit": False,
            },
        )
        logger = logging.getLogger("test")
        verdict = runner.validate_smoke_result(
            data, "F0_baseline", "q_test_04", True, logger
        )
        assert verdict == "SMOKE_ABSTENTION_OK"


# ─── 20. Holdout ──────────────────────────────────────────────────

class TestHoldoutGuard:
    def test_holdout_access_fails(self):

        data = _make_smoke_result(qid="q_holdout_01")
        logger = logging.getLogger("test")
        verdict = runner.validate_smoke_result(
            data, "W0_sentence_window", "q_holdout_01", False, logger
        )
        assert verdict == "SMOKE_FAILED"


# ─── 21. Citation without evidence ───────────────────────────────

class TestCitationValidation:
    def test_citation_without_evidence_fails(self):

        data = _make_smoke_result(
            citation_pages=[99],  # page 99 not in evidence
            retrieval_evidence={
                "candidate_count": 1,
                "candidates": [
                    {"chunk_id": "doc_p92_c0", "page_number": 92,
                     "retrieval_rank": 1, "retrieval_score": 0.9,
                     "rerank_rank": None, "rerank_score": None,
                     "parent_node_id": None,
                     "text_sha256": "x" * 64, "text_preview": "text"},
                ],
                "relevant_pages_expected": [92],
                "relevant_pages_found": [92],
                "relevant_pages_missing": [],
                "retrieval_hit": True,
            },
        )
        logger = logging.getLogger("test")
        verdict = runner.validate_smoke_result(
            data, "W0_sentence_window", "q_dev_01", False, logger
        )
        assert verdict == "SMOKE_FAILED"


# ─── 22. Checkpoint incompatible ─────────────────────────────────

class TestCheckpointSchema:
    def test_wrong_eval_schema_fails(self):

        data = _make_smoke_result()
        data["results"]["W0_sentence_window"][0]["evaluation"]["schema_version"] = "slice4_v1"
        logger = logging.getLogger("test")
        verdict = runner.validate_smoke_result(
            data, "W0_sentence_window", "q_dev_01", False, logger
        )
        assert verdict == "SMOKE_FAILED"


# ─── 23. No SMOKE_*_OK after failure ─────────────────────────────

class TestNoOKAfterFailure:
    def test_no_smoke_ok_emitted_on_failure(self, capsys):

        data = _make_smoke_result()
        data["results"]["W0_sentence_window"][0]["evaluation"] = None
        logger = logging.getLogger("test")
        verdict = runner.validate_smoke_result(
            data, "W0_sentence_window", "q_dev_01", False, logger
        )
        assert verdict == "SMOKE_FAILED"
        captured = capsys.readouterr()
        assert "SMOKE_POSITIVE_OK" not in captured.out
        assert "SMOKE_ABSTENTION_OK" not in captured.out


# ─── 24. No secrets ──────────────────────────────────────────────

class TestNoSecrets:
    def test_secret_in_result_fails(self):

        data = _make_smoke_result()
        # Inject a fake secret into the answer text
        data["results"]["W0_sentence_window"][0]["answer"]["text"] = (
            "The key is GEMINI_API_KEY=abc123"
        )
        logger = logging.getLogger("test")
        verdict = runner.validate_smoke_result(
            data, "W0_sentence_window", "q_dev_01", False, logger
        )
        assert verdict == "SMOKE_FAILED"


# ─── Fingerprint with manifest ───────────────────────────────────

class TestFingerprintWithManifest:
    def test_fingerprint_uses_manifest_sha(self, tmp_path):
        """get_embedding_fingerprint must use manifest's cache_tree_sha256."""
        from raglab.infrastructure.retrieval.baseline_adapter import (
            DeterministicEmbedding,
        )

        class _FakeAdapter:
            model_id = "fake"
            dimension = 64
            pooling = "mean"
            normalization = True

            def embed(self, text: str) -> list[float]:
                return DeterministicEmbedding(dimension=64).embed(text)

        manifest = {
            "cache_tree_sha256": "c" * 64,
            "model_id": "fake",
        }
        fp = runner.get_embedding_fingerprint(_FakeAdapter(), manifest)
        assert fp["cache_tree_sha256"] == "c" * 64

    def test_fingerprint_without_manifest_still_works(self):
        """Without manifest, adapter attribute is used if present."""

        class _FakeAdapterWithSha:
            model_id = "fake"
            dimension = 64
            pooling = "mean"
            normalization = True
            cache_tree_sha256 = "d" * 64

        fp = runner.get_embedding_fingerprint(_FakeAdapterWithSha())
        assert fp["cache_tree_sha256"] == "d" * 64
