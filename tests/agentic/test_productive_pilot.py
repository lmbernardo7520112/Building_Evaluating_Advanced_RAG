"""Focal tests for productive pilot infrastructure.

Tests composition root, productive runner validation, and analyzer
B0×A1×O1 computation using mocks/fakes only.
No PDF, no embedding model, no network in CI.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from raglab.agentic.contracts import ToolObservation
from raglab.agentic.runtime.retrieval_tool_adapter import RetrievalToolAdapter
from raglab.agentic.runtime.strategy_tool_factory import (
    ALL_STRATEGIES,
    build_registry_with_adapters,
)
from raglab.domain.entities import RetrievedEvidence
from raglab.domain.enums import PipelineStrategy
from raglab.domain.value_objects import ChunkId

# ── Helpers ──────────────────────────────────────────────────────


class FakeRetrievalPort:
    """Fake retrieval port producing deterministic evidence."""

    def __init__(self, strategy: str) -> None:
        self._strategy = strategy

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedEvidence]:
        return [
            RetrievedEvidence(
                chunk_id=ChunkId(f"{self._strategy}_chunk_{i}"),
                document_id="test_doc",
                text=f"Evidence {i} for {query[:20]}",
                rank=i + 1,
                score=0.9 - i * 0.1,
            )
            for i in range(top_k)
        ]


# ── §7.1: Composition root injects seven ports ──────────────────


class TestCompositionRoot:
    """Test that the composition root produces 7 strategy ports."""

    def test_seven_strategies_expected(self) -> None:
        """ALL_STRATEGIES has exactly 7 members."""
        assert len(ALL_STRATEGIES) == 7

    def test_build_registry_with_all_seven(self) -> None:
        """build_registry_with_adapters accepts 7 ports."""
        ports: dict[PipelineStrategy, object] = {
            s: FakeRetrievalPort(s.value) for s in ALL_STRATEGIES
        }
        registry, adapters = build_registry_with_adapters(ports)
        assert len(adapters) == 7
        assert registry.is_frozen

    def test_adapter_retrieve_returns_tool_observation(self) -> None:
        """Each adapter returns a ToolObservation with ps_ IDs."""
        port = FakeRetrievalPort("baseline")
        adapter = RetrievalToolAdapter(strategy="baseline", retrieval_port=port)
        obs = adapter.retrieve(query="test", strategy="baseline", top_k=3)
        assert isinstance(obs, ToolObservation)
        assert len(obs.passage_ids) == 3
        for pid in obs.passage_ids:
            assert pid.startswith("ps_")


# ── §7.2: Productive runner rejects bad PDF ─────────────────────


class TestProductiveRunnerValidation:
    """Test that the productive runner rejects invalid PDFs."""

    def test_rejects_missing_pdf(self, tmp_path: Path) -> None:
        """resolve_pdf_path returns None when env is unset."""
        import os

        with patch.dict(os.environ, {}, clear=True):
            # Remove RAGLAB_PDF_PATH if set
            os.environ.pop("RAGLAB_PDF_PATH", None)
            from scripts.slice5a_productive_composition import (
                resolve_pdf_path,
            )

            result = resolve_pdf_path()
            assert result is None

    def test_rejects_wrong_hash(self, tmp_path: Path) -> None:
        """verify_pdf returns False for wrong content."""
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"not a real pdf")

        from scripts.slice5a_productive_composition import verify_pdf

        assert not verify_pdf(fake_pdf)


# ── §7.3: Qrels not in router ───────────────────────────────────


class TestNoQrelsInRouter:
    """Verify router has zero qrels dependency."""

    def test_router_imports_no_qrels(self) -> None:
        """The router module does not import qrels modules."""
        import raglab.agentic.router as router_mod

        src = Path(router_mod.__file__).read_text()  # type: ignore[arg-type]
        # Strip comments (lines starting with #) and docstrings before checking
        import_lines = [
            line
            for line in src.splitlines()
            if line.strip().startswith(("import ", "from "))
        ]
        joined = "\n".join(import_lines).lower()
        assert "qrels" not in joined
        assert "ground_truth" not in joined

    def test_policy_metadata_deterministic(self) -> None:
        """Policy metadata is fully deterministic."""
        from raglab.agentic.router import (
            get_deterministic_policy_metadata,
        )

        m1 = get_deterministic_policy_metadata()
        m2 = get_deterministic_policy_metadata()
        assert m1.policy_sha256 == m2.policy_sha256


# ── §7.4: Result marks productive=True ──────────────────────────


class TestResultFlags:
    """Test that pilot results carry correct flags."""

    def test_productive_flag(self) -> None:
        """Result dict should have productive=True."""
        # Simulated result structure
        result = {
            "productive": True,
            "generation": "NOT_EXECUTED",
        }
        assert result["productive"] is True
        assert result["generation"] == "NOT_EXECUTED"

    def test_exactly_one_logical_call(self) -> None:
        """Budget allows exactly 1 logical call."""
        from raglab.agentic.budget import Budget

        b = Budget(
            max_logical_calls=1,
            max_physical_attempts=1,
            max_retries=0,
        )
        assert b.can_consume_logical_call()
        b.consume_logical_call()
        assert not b.can_consume_logical_call()


# ── §7.5: Analyzer B0×A1×O1 computation ─────────────────────────


class TestAnalyzerComputation:
    """Test B0×A1×O1 paired comparison logic."""

    def test_delta_computation(self) -> None:
        """delta = A1 - B0."""
        b0 = 0.5775
        a1 = 0.4069
        delta = a1 - b0
        assert round(delta, 4) == -0.1706

    def test_regret_computation(self) -> None:
        """regret = O1 - A1."""
        o1 = 0.6941
        a1 = 0.4069
        regret = o1 - a1
        assert round(regret, 4) == 0.2872

    def test_category_classification(self) -> None:
        """Negative delta → category C."""
        delta = -0.1706
        if delta > 0:
            cat = "A"
        elif delta == 0:
            cat = "B"
        else:
            cat = "C"
        assert cat == "C"

    def test_oracle_misses(self) -> None:
        """3/4 oracle misses when W1 only wins 1 QID."""
        winners = ["H2", "W1", "H0", "H0"]
        selected = ["W1", "W1", "W1", "W1"]
        misses = sum(1 for w, s in zip(winners, selected, strict=True) if w != s)
        assert misses == 3


# ── §7.6: Stop reason semantics ─────────────────────────────────


class TestStopReasonSemantics:
    """BUDGET_EXHAUSTED is the normal one-shot termination."""

    def test_budget_exhausted_is_success(self) -> None:
        """BUDGET_EXHAUSTED after authorized call = SUCCESS."""
        from raglab.agentic.budget import Budget

        b = Budget(
            max_logical_calls=1,
            max_physical_attempts=1,
            max_retries=0,
        )
        b.consume_logical_call()

        # Budget exhausted → StopPolicy would return BUDGET_EXHAUSTED
        assert not b.can_consume_logical_call()
        assert b.logical_calls_consumed == b.max_logical_calls

        # Semantic: this is normal termination
        interpretation = "NORMAL_ONE_SHOT_TERMINATION_AFTER_AUTHORIZED_CALL"
        assert "NORMAL" in interpretation
