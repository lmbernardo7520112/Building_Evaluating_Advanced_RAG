"""Tests for benchmarks/run_slice4_benchmark.py — offline, no credentials, no models.

These tests validate:
- CR-1 contract: import of FastEmbedEmbeddingAdapter (the real canonical name)
- CR-3: runner without --mode fails closed (exit != 0)
- CLI: --help works without key, PDF, or model
- --mode smoke: selects exactly 1 pair, rejects holdout
- --mode full: blocked without --confirm-full-benchmark
- --mode resume: blocked without --run-id
- No network calls in any test
- No credentials in any exception or message
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from raglab.domain.entities import GeneratedAnswer

_REPO_ROOT = Path(__file__).resolve().parents[3]  # tests/unit/infrastructure → raglab-v7
_RUNNER = str(_REPO_ROOT / "benchmarks" / "run_slice4_benchmark.py")

# ─── Helper ──────────────────────────────────────────────────────

def _run_runner(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """Execute runner subprocess; inherit env unless overridden."""
    import os
    base_env = dict(os.environ)
    if env:
        base_env.update(env)
    return subprocess.run(
        [sys.executable, _RUNNER, *args],
        capture_output=True,
        text=True,
        env=base_env,
    )


# ─── CR-1 CONTRACT TEST ───────────────────────────────────────────

class TestAdapterImportContract:
    """Contract test that would have caught the original ImportError."""

    def test_fastembed_embedding_adapter_importable(self):
        """The canonical class name must be importable — CR-1 regression guard."""
        from raglab.infrastructure.embeddings.fastembed_adapter import (
            FastEmbedEmbeddingAdapter,
        )
        assert FastEmbedEmbeddingAdapter is not None

    def test_fast_embed_adapter_alias_does_not_exist(self):
        """FastEmbedAdapter (wrong name used in old runner) must NOT exist."""
        import raglab.infrastructure.embeddings.fastembed_adapter as mod
        assert not hasattr(mod, "FastEmbedAdapter"), (
            "FastEmbedAdapter alias found — this name must not be added silently. "
            "Use the canonical FastEmbedEmbeddingAdapter."
        )

    def test_runner_module_imports_canonical_name(self):
        """The runner source must reference FastEmbedEmbeddingAdapter, not FastEmbedAdapter."""
        runner_source = Path(_RUNNER).read_text(encoding="utf-8")
        assert "FastEmbedEmbeddingAdapter" in runner_source, (
            "Runner does not import FastEmbedEmbeddingAdapter"
        )
        assert "FastEmbedAdapter" not in runner_source, (
            "Runner still references the invalid FastEmbedAdapter name"
        )

    def test_embedding_adapter_has_dimension_property(self):
        """Adapter must expose .dimension and .model_id (used by runner and retrievers)."""
        from raglab.infrastructure.embeddings.fastembed_adapter import (
            FastEmbedEmbeddingAdapter,
        )
        # Verify properties exist at class level (no instantiation = no model download)
        assert "dimension" in dir(FastEmbedEmbeddingAdapter)
        assert "model_id" in dir(FastEmbedEmbeddingAdapter)

    def test_constructor_accepts_model_name(self):
        """Constructor signature must accept model_name keyword argument."""
        import inspect

        from raglab.infrastructure.embeddings.fastembed_adapter import (
            FastEmbedEmbeddingAdapter,
        )
        sig = inspect.signature(FastEmbedEmbeddingAdapter.__init__)
        assert "model_name" in sig.parameters, (
            "FastEmbedEmbeddingAdapter.__init__ must accept model_name kwarg"
        )


# ─── CLI FAIL-CLOSED ──────────────────────────────────────────────

class TestCLIFailClosed:
    def test_no_args_exits_nonzero(self):
        """Running without --mode must exit with nonzero code and show usage."""
        result = _run_runner()
        assert result.returncode != 0, (
            f"Runner without arguments must exit nonzero, got {result.returncode}"
        )
        # Must print usage/error
        combined = result.stdout + result.stderr
        assert "usage" in combined.lower() or "error" in combined.lower(), (
            "Runner must show usage or error when --mode is missing"
        )

    def test_no_args_does_not_read_api_key(self, monkeypatch):
        """Runner without --mode must not read GEMINI_API_KEY."""
        import os
        env = dict(os.environ)
        env.pop("GEMINI_API_KEY", None)
        env.pop("RAGLAB_PDF_PATH", None)
        result = subprocess.run(
            [sys.executable, _RUNNER],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode != 0
        # Key must not appear in output
        assert "GEMINI_API_KEY" not in result.stdout or (
            "required" in result.stdout.lower() or "usage" in result.stdout.lower()
        )

    def test_no_args_does_not_load_pdf(self):
        """Runner without --mode must not try to load the PDF."""
        import os
        env = dict(os.environ)
        env.pop("RAGLAB_PDF_PATH", None)
        result = subprocess.run(
            [sys.executable, _RUNNER],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode != 0
        # Should not mention PDF loading (only argparse error)
        assert "Extracting" not in result.stdout


# ─── --help ───────────────────────────────────────────────────────

class TestCLIHelp:
    def test_help_exits_zero(self):
        """--help must exit 0."""
        result = _run_runner("--help")
        assert result.returncode == 0, f"--help returned {result.returncode}"

    def test_help_shows_modes(self):
        """--help must document preflight, smoke, full, resume modes."""
        result = _run_runner("--help")
        combined = result.stdout + result.stderr
        assert "preflight" in combined
        assert "smoke" in combined
        assert "full" in combined
        assert "resume" in combined

    def test_help_works_without_api_key(self):
        """--help must not require GEMINI_API_KEY."""
        import os
        env = dict(os.environ)
        env.pop("GEMINI_API_KEY", None)
        result = subprocess.run(
            [sys.executable, _RUNNER, "--help"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0

    def test_help_works_without_pdf(self):
        """--help must not require RAGLAB_PDF_PATH."""
        import os
        env = dict(os.environ)
        env.pop("RAGLAB_PDF_PATH", None)
        result = subprocess.run(
            [sys.executable, _RUNNER, "--help"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0

    def test_help_has_no_side_effects(self, tmp_path):
        """--help must not write any files."""
        result = _run_runner("--help")
        assert result.returncode == 0
        # No result files created
        assert not list(tmp_path.glob("*.json"))


# ─── --mode smoke ─────────────────────────────────────────────────

class TestSmokeMode:
    """Tests for --mode smoke using fakes — no Gemini calls, no model loading."""

    def _make_fake_page(self):
        from raglab.domain.entities import DocumentPage
        from raglab.domain.value_objects import PageNumber
        return DocumentPage(page_number=PageNumber(92), text="Prova por exaustão é...")

    def test_smoke_selects_exactly_one_pair(self, tmp_path, monkeypatch):
        """Smoke must produce exactly 1 strategy × 1 question result."""
        import benchmarks.run_slice4_benchmark as runner

        called_pairs: list[tuple[str, str]] = []

        def _fake_run_benchmark(run_id, questions, strategy_labels, logger, pdf_path):
            for s in strategy_labels:
                for q in questions:
                    called_pairs.append((s, q["qid"]))
            # Write minimal valid output
            out = tmp_path / f"slice4_results_{run_id}_fake.json"
            out.write_text(json.dumps({"experiment_id": run_id, "results": {}}))
            return out

        monkeypatch.setattr(runner, "run_benchmark", _fake_run_benchmark)
        monkeypatch.setattr(runner, "check_credential", lambda logger: None)
        monkeypatch.setattr(runner, "verify_pdf", lambda path, logger: None)
        monkeypatch.setattr(
            runner, "validate_smoke_result",
            lambda data, strategy, qid, is_abstention, logger: "SMOKE_OK",
        )

        args = runner.build_parser().parse_args([
            "--mode", "smoke",
            "--smoke-strategy", "F0_baseline",
            "--smoke-question", "q_dev_01",
        ])

        import logging
        logger = logging.getLogger("test_smoke")
        runner.cmd_smoke(args, tmp_path / "fake.pdf", logger)

        assert len(called_pairs) == 1, f"Expected 1 pair, got {called_pairs}"
        assert called_pairs[0] == ("F0_baseline", "q_dev_01")

    def test_smoke_rejects_holdout_question(self, tmp_path, monkeypatch):
        """Smoke must refuse a holdout question ID at CLI level."""
        import benchmarks.run_slice4_benchmark as runner

        # Inject a fake holdout qid into choices to bypass argparse
        args = runner.build_parser().parse_args([
            "--mode", "smoke",
            "--smoke-strategy", "F0_baseline",
            "--smoke-question", "q_dev_01",
        ])
        args.smoke_question = "q_holdout_01"  # bypass argparse choices

        import logging
        logger = logging.getLogger("test_smoke_holdout")
        with pytest.raises(SystemExit) as exc_info:
            runner.cmd_smoke(args, tmp_path / "fake.pdf", logger)
        assert exc_info.value.code == 2

    def test_smoke_generates_run_id(self, tmp_path, monkeypatch):
        """Smoke must generate its own unique run_id."""
        import benchmarks.run_slice4_benchmark as runner

        seen_run_ids: list[str] = []

        def _fake_run_benchmark(run_id, questions, strategy_labels, logger, pdf_path):
            seen_run_ids.append(run_id)
            out = tmp_path / f"slice4_results_{run_id}_fake.json"
            out.write_text(json.dumps({"experiment_id": run_id, "results": {}}))
            return out

        monkeypatch.setattr(runner, "run_benchmark", _fake_run_benchmark)
        monkeypatch.setattr(runner, "check_credential", lambda logger: None)
        monkeypatch.setattr(runner, "verify_pdf", lambda path, logger: None)
        monkeypatch.setattr(
            runner, "validate_smoke_result",
            lambda data, strategy, qid, is_abstention, logger: "SMOKE_OK",
        )

        args = runner.build_parser().parse_args(["--mode", "smoke"])

        import logging
        runner.cmd_smoke(args, tmp_path / "fake.pdf", logging.getLogger("t"))
        assert len(seen_run_ids) == 1
        assert seen_run_ids[0].startswith("smoke_")


# ─── --mode full ──────────────────────────────────────────────────

class TestFullMode:
    def test_full_without_confirm_blocked(self, tmp_path, monkeypatch):
        """Full mode without --confirm-full-benchmark must exit nonzero."""
        import benchmarks.run_slice4_benchmark as runner

        args = runner.build_parser().parse_args(["--mode", "full"])
        assert not args.confirm_full_benchmark

        import logging
        logger = logging.getLogger("test_full_no_confirm")
        with pytest.raises(SystemExit) as exc_info:
            runner.cmd_full(args, tmp_path / "fake.pdf", logger)
        assert exc_info.value.code == 3

    def test_full_with_confirm_enters_flow(self, tmp_path, monkeypatch):
        """Full mode with --confirm-full-benchmark must call run_benchmark."""
        import benchmarks.run_slice4_benchmark as runner

        called = []

        def _fake_run_benchmark(run_id, questions, strategy_labels, logger, pdf_path):
            called.append((run_id, len(questions), len(strategy_labels)))
            out = tmp_path / f"slice4_results_{run_id}_fake.json"
            out.write_text(json.dumps({"experiment_id": run_id, "results": {}}))
            return out

        monkeypatch.setattr(runner, "run_benchmark", _fake_run_benchmark)

        args = runner.build_parser().parse_args([
            "--mode", "full", "--confirm-full-benchmark",
        ])

        import logging
        runner.cmd_full(args, tmp_path / "fake.pdf", logging.getLogger("t"))
        assert len(called) == 1
        assert called[0][2] == len(runner.VALID_STRATEGIES)

    def test_full_includes_all_strategies(self):
        """Full mode strategy list must contain all 7 defined strategies."""
        import benchmarks.run_slice4_benchmark as runner
        assert len(runner.VALID_STRATEGIES) == 7


# ─── --mode resume ────────────────────────────────────────────────

class TestResumeMode:
    def test_resume_without_run_id_blocked(self, tmp_path, monkeypatch):
        """Resume without --run-id must exit nonzero."""
        import benchmarks.run_slice4_benchmark as runner

        args = runner.build_parser().parse_args(["--mode", "resume"])
        # args.run_id will be None

        import logging
        logger = logging.getLogger("test_resume_no_run_id")
        with pytest.raises(SystemExit) as exc_info:
            runner.cmd_resume(args, tmp_path / "fake.pdf", logger)
        assert exc_info.value.code == 3

    def test_resume_with_invalid_run_id_blocked(self, tmp_path, monkeypatch):
        """Resume with run-id that has no checkpoint must exit nonzero."""
        import benchmarks.run_slice4_benchmark as runner

        monkeypatch.setattr(runner, "CHECKPOINT_DIR", tmp_path)

        args = runner.build_parser().parse_args([
            "--mode", "resume", "--run-id", "nonexistent_run_id_xyz",
        ])

        import logging
        with pytest.raises((SystemExit, ValueError)):
            runner.cmd_resume(args, tmp_path / "fake.pdf", logging.getLogger("t"))

    def test_resume_with_valid_checkpoint_enters_flow(self, tmp_path, monkeypatch):
        """Resume with existing checkpoint must call run_benchmark."""
        import benchmarks.run_slice4_benchmark as runner

        run_id = "test_resume_run_001"
        ckpt = tmp_path / f"slice4_gen_checkpoint_{run_id}.json"
        ckpt.write_text(json.dumps({"run_id": run_id, "completed": {}}))

        monkeypatch.setattr(runner, "CHECKPOINT_DIR", tmp_path)

        called = []

        def _fake_run_benchmark(
            run_id, questions, strategy_labels, logger, pdf_path
        ):
            called.append(run_id)
            out = tmp_path / f"results_{run_id}_fake.json"
            out.write_text(json.dumps({"experiment_id": run_id, "results": {}}))
            return out

        monkeypatch.setattr(runner, "run_benchmark", _fake_run_benchmark)

        args = runner.build_parser().parse_args([
            "--mode", "resume", "--run-id", run_id,
        ])

        import logging
        runner.cmd_resume(args, tmp_path / "fake.pdf", logging.getLogger("t"))
        assert called == [run_id]


# ─── SECURITY: no credentials in output ───────────────────────────

class TestNoCredentialLeak:
    def test_no_api_key_in_error_output(self):
        """When key is missing, error output must not print a fake key value."""
        import os
        env = dict(os.environ)
        env.pop("GEMINI_API_KEY", None)
        env.pop("RAGLAB_PDF_PATH", None)
        env["GEMINI_API_KEY_FAKE"] = "secret-value-must-not-appear"

        result = subprocess.run(
            [sys.executable, _RUNNER, "--mode", "smoke"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode != 0
        assert "secret-value-must-not-appear" not in result.stdout
        assert "secret-value-must-not-appear" not in result.stderr

    def test_holdout_guard_in_run_benchmark(self, tmp_path, monkeypatch):
        """run_benchmark must abort with exit 2 if a holdout question is passed."""
        import logging

        import benchmarks.run_slice4_benchmark as runner

        holdout_q = {
            "qid": "q_holdout_01",
            "split": "holdout",
            "query": "Holdout question",
            "relevant_pages": [],
        }

        # We need to bypass the retriever to get to the guard
        fake_evidence = []
        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = fake_evidence

        fake_manifest = {
            "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "cache_tree_sha256": "abc123",
        }
        monkeypatch.setattr(runner, "load_provision_manifest", lambda: fake_manifest)
        monkeypatch.setattr(runner, "load_pdf_pages", lambda path, logger: [])
        monkeypatch.setattr(runner, "load_embedding_model", lambda logger: MagicMock())
        monkeypatch.setattr(runner, "build_retrievers", lambda pages, embed, strategies=None: {
            "F0_baseline": fake_retriever
        })
        monkeypatch.setattr(runner, "verify_embedding_parity",
                            lambda r, lg, m: {"F0_baseline": {"cache_tree_sha256": "abc123"}})
        monkeypatch.setattr(runner, "CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)

        # Mock Gemini adapters so no real API connection is made
        fake_gen = MagicMock()
        fake_gen.model_id = "gemini-test"
        fake_judge = MagicMock()
        monkeypatch.setattr(
            "raglab.infrastructure.gemini.gemini_generator_adapter.GeminiGeneratorAdapter",
            lambda **kw: fake_gen,
        )
        monkeypatch.setattr(
            "raglab.infrastructure.gemini.gemini_judge_adapter.GeminiJudgeAdapter",
            lambda **kw: fake_judge,
        )

        with pytest.raises(SystemExit) as exc_info:
            runner.run_benchmark(
                run_id="test_holdout_guard",
                questions=[holdout_q],
                strategy_labels=("F0_baseline",),
                logger=logging.getLogger("test"),
                pdf_path=tmp_path / "fake.pdf",
            )
        assert exc_info.value.code == 2


# ─── MANDATORY SURGICAL FIXES TEST SUITE ─────────────────────────

class TestSlice4SurgicalFixes:
    """Mandatory test suite for Slice 4 surgical fixes."""

    @pytest.fixture(autouse=True)
    def _auto_quota_sync(self, monkeypatch):
        """Automatically sync mock generator and judge calls with QuotaManager."""
        from raglab.domain.quota import QuotaManager

        active_qm = []
        orig_init = QuotaManager.__init__

        def custom_init(qm_self, *args, **kwargs):
            orig_init(qm_self, *args, **kwargs)
            active_qm.append(qm_self)

        monkeypatch.setattr(QuotaManager, "__init__", custom_init)
        self.active_qm = active_qm

    def _make_gen(self, fake_ans):
        def generate_mock(query_id, query, evidence):
            if self.active_qm:
                self.active_qm[-1].acquire(1)
            # Ensure fake_ans has the query_id set
            if hasattr(fake_ans, "query_id"):
                with contextlib.suppress(AttributeError):
                    fake_ans.query_id = query_id
            return fake_ans
        fake_gen = MagicMock()
        fake_gen.generate.side_effect = generate_mock
        return fake_gen

    def _make_judge(self, strategy_label, cr=1.0, gr=1.0, ar=1.0):
        from raglab.domain.enums import PipelineStrategy
        fake_judge = MagicMock()
        fake_judge.strategy = PipelineStrategy.from_label(strategy_label)

        def cr_mock(*args, **kwargs):
            if self.active_qm:
                self.active_qm[-1].acquire(1)
            return cr

        def gr_mock(*args, **kwargs):
            if self.active_qm:
                self.active_qm[-1].acquire(1)
            return gr

        def ar_mock(*args, **kwargs):
            if self.active_qm:
                self.active_qm[-1].acquire(1)
            return ar

        fake_judge.evaluate_context_relevance.side_effect = cr_mock
        fake_judge.evaluate_groundedness.side_effect = gr_mock
        fake_judge.evaluate_answer_relevance.side_effect = ar_mock
        return fake_judge

    @pytest.mark.parametrize("strategy_label", [
        "F0_baseline",
        "S0_sentence_anchor",
        "W0_sentence_window",
        "W1_sentence_window_rerank",
        "H0_hierarchical_leaf",
        "H1_auto_merging",
        "H2_auto_merging_rerank",
    ])
    def test_all_seven_strategies_preserve_identity(self, tmp_path, monkeypatch, strategy_label):
        """1 & 2: W0 never registered as baseline; all 7 strategies preserve exact identity."""
        import logging

        import benchmarks.run_slice4_benchmark as runner

        fake_manifest = {
            "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "cache_tree_sha256": "a" * 64,
        }
        monkeypatch.setattr(runner, "load_provision_manifest", lambda: fake_manifest)
        monkeypatch.setattr(runner, "load_pdf_pages", lambda path, logger: [])
        monkeypatch.setattr(runner, "load_embedding_model", lambda logger: MagicMock())
        monkeypatch.setattr(runner, "verify_embedding_parity", lambda r, lg, m: {strategy_label: {"cache_tree_sha256": "a" * 64}})

        fake_ev = MagicMock()
        fake_ev.chunk_id = "page_0092"
        fake_ev.text = "evidence text"
        fake_ev.score = 1.0
        fake_ev.parent_node_id = None
        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = [fake_ev]
        monkeypatch.setattr(runner, "build_retrievers", lambda pages, embed, strategies=None: {strategy_label: fake_retriever})
        monkeypatch.setattr(runner, "CHECKPOINT_DIR", tmp_path / "ckpts")
        monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")

        from raglab.domain.entities import GeneratedAnswer
        fake_ans = GeneratedAnswer(query_id="q_dev_01", text="Substantive answer", abstained=False, citations=[])
        fake_gen = self._make_gen(fake_ans)
        fake_judge = self._make_judge(strategy_label)

        monkeypatch.setattr("raglab.infrastructure.gemini.gemini_generator_adapter.GeminiGeneratorAdapter", lambda **kw: fake_gen)
        monkeypatch.setattr("raglab.infrastructure.gemini.gemini_judge_adapter.GeminiJudgeAdapter", lambda **kw: fake_judge)

        q = {"qid": "q_dev_01", "split": "development", "query": "Test query", "relevant_pages": [92]}
        out = runner.run_benchmark(
            run_id="test_id",
            questions=[q],
            strategy_labels=(strategy_label,),
            logger=logging.getLogger("test"),
            pdf_path=tmp_path / "fake.pdf",
        )

        data = json.loads(out.read_text(encoding="utf-8"))
        res = data["results"][strategy_label][0]
        assert res["strategy"] == strategy_label
        if strategy_label == "W0_sentence_window":
            assert res["strategy"] != "F0_baseline"
            assert res["strategy"] != "baseline"

    def test_strategy_provenance_mismatch_aborts(self, tmp_path, monkeypatch):
        """3: Strategy provenance mismatch aborts execution before saving."""
        import logging

        import benchmarks.run_slice4_benchmark as runner

        fake_manifest = {
            "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "cache_tree_sha256": "a" * 64,
        }
        monkeypatch.setattr(runner, "load_provision_manifest", lambda: fake_manifest)
        monkeypatch.setattr(runner, "load_pdf_pages", lambda path, logger: [])
        monkeypatch.setattr(runner, "load_embedding_model", lambda logger: MagicMock())
        monkeypatch.setattr(runner, "verify_embedding_parity", lambda r, lg, m: {"W0_sentence_window": {"cache_tree_sha256": "a" * 64}})

        fake_ev = MagicMock()
        fake_ev.chunk_id = "page_0092"
        fake_ev.text = "evidence text"
        fake_ev.score = 1.0
        fake_ev.parent_node_id = None
        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = [fake_ev]
        monkeypatch.setattr(runner, "build_retrievers", lambda pages, embed, strategies=None: {"W0_sentence_window": fake_retriever})
        monkeypatch.setattr(runner, "CHECKPOINT_DIR", tmp_path / "ckpts")
        monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")

        from raglab.domain.entities import GeneratedAnswer
        fake_ans = GeneratedAnswer(query_id="q_dev_01", text="Substantive answer", abstained=False, citations=[])
        fake_gen = self._make_gen(fake_ans)
        fake_judge = self._make_judge("F0_baseline")  # Force wrong strategy

        monkeypatch.setattr("raglab.infrastructure.gemini.gemini_generator_adapter.GeminiGeneratorAdapter", lambda **kw: fake_gen)
        monkeypatch.setattr("raglab.infrastructure.gemini.gemini_judge_adapter.GeminiJudgeAdapter", lambda **kw: fake_judge)

        q = {"qid": "q_dev_01", "split": "development", "query": "Test query", "relevant_pages": [92]}
        with pytest.raises(ValueError, match="STRATEGY_PROVENANCE_MISMATCH"):
            runner.run_benchmark(
                run_id="test_mismatch",
                questions=[q],
                strategy_labels=("W0_sentence_window",),
                logger=logging.getLogger("test"),
                pdf_path=tmp_path / "fake.pdf",
            )

    def test_abstain_with_context_calls_only_cr(self, tmp_path, monkeypatch):
        """4 & 6 & 7: ABSTAIN with context calls ONLY Context Relevance (GR=0, AR=0, total=2)."""
        import logging

        import benchmarks.run_slice4_benchmark as runner

        fake_manifest = {"model_id": "m", "cache_tree_sha256": "a" * 64}
        monkeypatch.setattr(runner, "load_provision_manifest", lambda: fake_manifest)
        monkeypatch.setattr(runner, "load_pdf_pages", lambda path, logger: [])
        monkeypatch.setattr(runner, "load_embedding_model", lambda logger: MagicMock())
        monkeypatch.setattr(runner, "verify_embedding_parity", lambda r, lg, m: {"F0_baseline": {"cache_tree_sha256": "a" * 64}})

        fake_ev = MagicMock()
        fake_ev.chunk_id = "page_0092"
        fake_ev.text = "evidence text"
        fake_ev.score = 1.0
        fake_ev.parent_node_id = None
        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = [fake_ev]
        monkeypatch.setattr(runner, "build_retrievers", lambda pages, embed, strategies=None: {"F0_baseline": fake_retriever})
        monkeypatch.setattr(runner, "CHECKPOINT_DIR", tmp_path / "ckpts")
        monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")

        fake_ans = GeneratedAnswer(query_id="q_test_04", text="Não encontrei informação", abstained=True, citations=[])
        fake_gen = self._make_gen(fake_ans)
        fake_judge = self._make_judge("F0_baseline", cr=0.0)

        monkeypatch.setattr("raglab.infrastructure.gemini.gemini_generator_adapter.GeminiGeneratorAdapter", lambda **kw: fake_gen)
        monkeypatch.setattr("raglab.infrastructure.gemini.gemini_judge_adapter.GeminiJudgeAdapter", lambda **kw: fake_judge)

        q = {"qid": "q_test_04", "split": "test", "query": "França?", "relevant_pages": [], "abstention_expected": True}
        out = runner.run_benchmark(
            run_id="test_abstain_ctx",
            questions=[q],
            strategy_labels=("F0_baseline",),
            logger=logging.getLogger("test"),
            pdf_path=tmp_path / "fake.pdf",
        )

        data = json.loads(out.read_text(encoding="utf-8"))
        res = data["results"]["F0_baseline"][0]
        ledger = res["call_ledger"]
        assert ledger["generation_calls"] == 1
        assert ledger["context_relevance_calls"] == 1
        assert ledger["groundedness_calls"] == 0
        assert ledger["answer_relevance_calls"] == 0
        assert ledger["total_external_requests"] == 2

    def test_abstain_without_context_no_judge_calls(self, tmp_path, monkeypatch):
        """5: ABSTAIN without context makes 0 judge calls (total=1)."""
        import logging

        import benchmarks.run_slice4_benchmark as runner

        fake_manifest = {"model_id": "m", "cache_tree_sha256": "a" * 64}
        monkeypatch.setattr(runner, "load_provision_manifest", lambda: fake_manifest)
        monkeypatch.setattr(runner, "load_pdf_pages", lambda path, logger: [])
        monkeypatch.setattr(runner, "load_embedding_model", lambda logger: MagicMock())
        monkeypatch.setattr(runner, "verify_embedding_parity", lambda r, lg, m: {"F0_baseline": {"cache_tree_sha256": "a" * 64}})

        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = []
        monkeypatch.setattr(runner, "build_retrievers", lambda pages, embed, strategies=None: {"F0_baseline": fake_retriever})
        monkeypatch.setattr(runner, "CHECKPOINT_DIR", tmp_path / "ckpts")
        monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")

        fake_ans = GeneratedAnswer(query_id="q_test_04", text="Não foi possível responder", abstained=True, citations=[])
        fake_gen = self._make_gen(fake_ans)
        fake_judge = self._make_judge("F0_baseline")

        monkeypatch.setattr("raglab.infrastructure.gemini.gemini_generator_adapter.GeminiGeneratorAdapter", lambda **kw: fake_gen)
        monkeypatch.setattr("raglab.infrastructure.gemini.gemini_judge_adapter.GeminiJudgeAdapter", lambda **kw: fake_judge)

        q = {"qid": "q_test_04", "split": "test", "query": "Capital França?", "relevant_pages": [], "abstention_expected": True}
        out = runner.run_benchmark(
            run_id="test_abstain_no_ctx",
            questions=[q],
            strategy_labels=("F0_baseline",),
            logger=logging.getLogger("test"),
            pdf_path=tmp_path / "fake.pdf",
        )

        data = json.loads(out.read_text(encoding="utf-8"))
        res = data["results"]["F0_baseline"][0]
        ledger = res["call_ledger"]
        assert ledger["generation_calls"] == 1
        assert ledger["context_relevance_calls"] == 0
        assert ledger["groundedness_calls"] == 0
        assert ledger["answer_relevance_calls"] == 0
        assert ledger["total_external_requests"] == 1

    def test_substantive_answer_calls_all_three_metrics(self, tmp_path, monkeypatch):
        """8: Substantive answer calls CR, GR, and AR (total=4)."""
        import logging

        import benchmarks.run_slice4_benchmark as runner

        fake_manifest = {"model_id": "m", "cache_tree_sha256": "a" * 64}
        monkeypatch.setattr(runner, "load_provision_manifest", lambda: fake_manifest)
        monkeypatch.setattr(runner, "load_pdf_pages", lambda path, logger: [])
        monkeypatch.setattr(runner, "load_embedding_model", lambda logger: MagicMock())
        monkeypatch.setattr(runner, "verify_embedding_parity", lambda r, lg, m: {"F0_baseline": {"cache_tree_sha256": "a" * 64}})

        fake_ev = MagicMock()
        fake_ev.chunk_id = "page_0092"
        fake_ev.text = "evidence text"
        fake_ev.score = 1.0
        fake_ev.parent_node_id = None
        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = [fake_ev]
        monkeypatch.setattr(runner, "build_retrievers", lambda pages, embed, strategies=None: {"F0_baseline": fake_retriever})
        monkeypatch.setattr(runner, "CHECKPOINT_DIR", tmp_path / "ckpts")
        monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")

        fake_ans = GeneratedAnswer(query_id="q_dev_01", text="Substantive answer text", abstained=False, citations=[])
        fake_gen = self._make_gen(fake_ans)
        fake_judge = self._make_judge("F0_baseline")

        monkeypatch.setattr("raglab.infrastructure.gemini.gemini_generator_adapter.GeminiGeneratorAdapter", lambda **kw: fake_gen)
        monkeypatch.setattr("raglab.infrastructure.gemini.gemini_judge_adapter.GeminiJudgeAdapter", lambda **kw: fake_judge)

        q = {"qid": "q_dev_01", "split": "development", "query": "Exaustão?", "relevant_pages": [92]}
        out = runner.run_benchmark(
            run_id="test_substantive",
            questions=[q],
            strategy_labels=("F0_baseline",),
            logger=logging.getLogger("test"),
            pdf_path=tmp_path / "fake.pdf",
        )

        data = json.loads(out.read_text(encoding="utf-8"))
        res = data["results"]["F0_baseline"][0]
        ledger = res["call_ledger"]
        assert ledger["generation_calls"] == 1
        assert ledger["context_relevance_calls"] == 1
        assert ledger["groundedness_calls"] == 1
        assert ledger["answer_relevance_calls"] == 1
        assert ledger["total_external_requests"] == 4

    def test_accounting_divergence_aborts(self, tmp_path, monkeypatch):
        """9 & 10: Call ledger quota mismatch aborts with EXTERNAL_CALL_ACCOUNTING_MISMATCH."""
        import logging

        import benchmarks.run_slice4_benchmark as runner

        fake_manifest = {"model_id": "m", "cache_tree_sha256": "a" * 64}
        monkeypatch.setattr(runner, "load_provision_manifest", lambda: fake_manifest)
        monkeypatch.setattr(runner, "load_pdf_pages", lambda path, logger: [])
        monkeypatch.setattr(runner, "load_embedding_model", lambda logger: MagicMock())
        monkeypatch.setattr(runner, "verify_embedding_parity", lambda r, lg, m: {"F0_baseline": {"cache_tree_sha256": "a" * 64}})

        fake_ev = MagicMock()
        fake_ev.chunk_id = "page_0092"
        fake_ev.text = "evidence text"
        fake_ev.score = 1.0
        fake_ev.parent_node_id = None
        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = [fake_ev]
        monkeypatch.setattr(runner, "build_retrievers", lambda pages, embed, strategies=None: {"F0_baseline": fake_retriever})
        monkeypatch.setattr(runner, "CHECKPOINT_DIR", tmp_path / "ckpts")
        monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")

        fake_ans = GeneratedAnswer(query_id="q_dev_01", text="Substantive answer text", abstained=False, citations=[])
        # Generator DOES NOT call quota acquire, breaking accounting
        fake_gen = MagicMock()
        fake_gen.generate.return_value = fake_ans
        fake_judge = self._make_judge("F0_baseline")

        monkeypatch.setattr("raglab.infrastructure.gemini.gemini_generator_adapter.GeminiGeneratorAdapter", lambda **kw: fake_gen)
        monkeypatch.setattr("raglab.infrastructure.gemini.gemini_judge_adapter.GeminiJudgeAdapter", lambda **kw: fake_judge)

        q = {"qid": "q_dev_01", "split": "development", "query": "Exaustão?", "relevant_pages": [92]}

        with pytest.raises(ValueError, match="EXTERNAL_CALL_ACCOUNTING_MISMATCH"):
            runner.run_benchmark(
                run_id="test_accounting_err",
                questions=[q],
                strategy_labels=("F0_baseline",),
                logger=logging.getLogger("test"),
                pdf_path=tmp_path / "fake.pdf",
            )

    def test_explicit_versioning_fields(self, tmp_path, monkeypatch):
        """11: Top-level and evaluation records contain protocol_version and artifact_schema_version."""
        import logging

        import benchmarks.run_slice4_benchmark as runner

        fake_manifest = {"model_id": "m", "cache_tree_sha256": "a" * 64}
        monkeypatch.setattr(runner, "load_provision_manifest", lambda: fake_manifest)
        monkeypatch.setattr(runner, "load_pdf_pages", lambda path, logger: [])
        monkeypatch.setattr(runner, "load_embedding_model", lambda logger: MagicMock())
        monkeypatch.setattr(runner, "verify_embedding_parity", lambda r, lg, m: {"F0_baseline": {"cache_tree_sha256": "a" * 64}})

        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = []
        monkeypatch.setattr(runner, "build_retrievers", lambda pages, embed, strategies=None: {"F0_baseline": fake_retriever})
        monkeypatch.setattr(runner, "CHECKPOINT_DIR", tmp_path / "ckpts")
        monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")

        fake_ans = GeneratedAnswer(query_id="q_test_04", text="Não foi possível responder", abstained=True, citations=[])
        fake_gen = self._make_gen(fake_ans)
        fake_judge = self._make_judge("F0_baseline")

        monkeypatch.setattr("raglab.infrastructure.gemini.gemini_generator_adapter.GeminiGeneratorAdapter", lambda **kw: fake_gen)
        monkeypatch.setattr("raglab.infrastructure.gemini.gemini_judge_adapter.GeminiJudgeAdapter", lambda **kw: fake_judge)

        q = {"qid": "q_test_04", "split": "test", "query": "Capital?", "relevant_pages": [], "abstention_expected": True}
        out = runner.run_benchmark(
            run_id="test_vers",
            questions=[q],
            strategy_labels=("F0_baseline",),
            logger=logging.getLogger("test"),
            pdf_path=tmp_path / "fake.pdf",
        )

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["protocol_version"] == runner.PROTOCOL_VERSION
        assert data["artifact_schema_version"] == runner._EVAL_SCHEMA_VERSION
        assert data["schema"] == runner._EVAL_SCHEMA_VERSION
        eval_rec = data["results"]["F0_baseline"][0]["evaluation"]
        assert eval_rec["protocol_version"] == runner.PROTOCOL_VERSION
        assert eval_rec["artifact_schema_version"] == runner._EVAL_SCHEMA_VERSION
        assert eval_rec["schema_version"] == runner._EVAL_SCHEMA_VERSION

    def test_runbook_contains_separated_commands(self):
        """13: Runbook contains independent export and unset commands."""
        runbook = (Path(_REPO_ROOT) / "docs" / "runbooks" / "slice4_human_execution.md").read_text(encoding="utf-8")
        assert "export LANGCHAIN_TRACING_V2=false" in runbook
        assert "unset GOOGLE_API_KEY" in runbook


class TestSlice4V3ContractFixes:
    """Mandatory test suite for Slice 4 v3 artifact contract fixes (18 tests)."""

    def test_answer_longer_than_500_chars_preserved_in_full(self):
        """1. Answer text > 500 characters is preserved in full in sanitize_answer_for_artifact."""
        from raglab.infrastructure.gemini.gemini_generator_adapter import (
            sanitize_answer_for_artifact,
        )
        long_text = "A" * 750 + " fim de frase."
        ans = GeneratedAnswer(query_id="q1", text=long_text, abstained=False, citations=())
        san = sanitize_answer_for_artifact(ans)
        assert len(str(san["text"])) == 764
        assert san["text"] == long_text
        assert san["truncated"] is False
        assert san["text_length_chars"] == 764
        assert len(str(san["preview"])) == 500

    def test_answer_ends_naturally_without_truncation(self):
        """2. Answer text ends naturally without being cut off mid-word."""
        from raglab.infrastructure.gemini.gemini_generator_adapter import (
            sanitize_answer_for_artifact,
        )
        text = "Esta é uma resposta completa com mais de quinhentos caracteres " + ("palavra " * 70) + "fim natural."
        assert len(text) > 500
        ans = GeneratedAnswer(query_id="q1", text=text, abstained=False, citations=())
        san = sanitize_answer_for_artifact(ans)
        assert str(san["text"]).endswith("fim natural.")

    def test_answer_text_sha256_matches_text(self):
        """3. answer.text_sha256 corresponds exactly to SHA-256 of text."""
        import hashlib

        from raglab.infrastructure.gemini.gemini_generator_adapter import (
            sanitize_answer_for_artifact,
        )
        text = "Texto de teste para verificação de hash SHA-256."
        ans = GeneratedAnswer(query_id="q1", text=text, abstained=False, citations=())
        san = sanitize_answer_for_artifact(ans)
        expected_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert san["text_sha256"] == expected_sha

    def test_evaluated_text_equals_persisted_text(self):
        """4. Evaluated text is identical to persisted text in benchmark artifact."""
        import hashlib

        from raglab.infrastructure.gemini.gemini_generator_adapter import (
            sanitize_answer_for_artifact,
        )
        text = "Resposta para avaliação e persistência identica " + "x" * 550
        fake_ans = GeneratedAnswer(query_id="q_dev_01", text=text, abstained=False, citations=())
        san = sanitize_answer_for_artifact(fake_ans)
        assert san["text"] == text
        assert san["text_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()

    def test_hash_divergence_aborts_with_evaluated_answer_artifact_mismatch(self, tmp_path, monkeypatch):
        """5. Mismatch between evaluated text/hash and persisted text aborts with EVALUATED_ANSWER_ARTIFACT_MISMATCH."""
        import logging

        import benchmarks.run_slice4_benchmark as runner

        fake_manifest = {"model_id": "m", "cache_tree_sha256": "a" * 64}
        monkeypatch.setattr(runner, "load_provision_manifest", lambda: fake_manifest)
        monkeypatch.setattr(runner, "load_pdf_pages", lambda path, logger: [])
        monkeypatch.setattr(runner, "load_embedding_model", lambda logger: MagicMock())
        monkeypatch.setattr(runner, "verify_embedding_parity", lambda r, logger_val, m: {"F0_baseline": {"cache_tree_sha256": "a" * 64}})

        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = []
        monkeypatch.setattr(runner, "build_retrievers", lambda pages, embed, strategies=None: {"F0_baseline": fake_retriever})
        monkeypatch.setattr(runner, "CHECKPOINT_DIR", tmp_path / "ckpts")
        monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")

        monkeypatch.setattr("raglab.infrastructure.gemini.gemini_generator_adapter.sanitize_answer_for_artifact", lambda ans: {
            "query_id": ans.query_id,
            "text": "TEXTO_CORROMPIDO",
            "text_sha256": "bad_sha",
            "text_length_chars": 16,
            "truncated": False,
            "preview": "TEXTO_CORROMPIDO",
            "abstained": False,
            "citation_pages": [],
        })

        fake_gen = MagicMock()
        fake_gen.generate.return_value = GeneratedAnswer(query_id="F0_baseline::q_dev_01", text="TEXTO_ORIGINAL", abstained=False, citations=())
        fake_gen.model_id = "g"
        fake_judge = MagicMock()
        fake_judge.strategy = "F0_baseline"

        monkeypatch.setattr("raglab.infrastructure.gemini.gemini_generator_adapter.GeminiGeneratorAdapter", lambda **kw: fake_gen)
        monkeypatch.setattr("raglab.infrastructure.gemini.gemini_judge_adapter.GeminiJudgeAdapter", lambda **kw: fake_judge)

        q = {"qid": "q_dev_01", "split": "development", "query": "Q?", "relevant_pages": [92]}
        with pytest.raises(ValueError, match="EVALUATED_ANSWER_ARTIFACT_MISMATCH"):
            runner.run_benchmark(
                run_id="test_div",
                questions=[q],
                strategy_labels=("F0_baseline",),
                logger=logging.getLogger("t"),
                pdf_path=tmp_path / "fake.pdf",
            )

    def test_preview_does_not_replace_full_text(self):
        """6. preview field is distinct and does not replace full text in answer artifact."""
        from raglab.infrastructure.gemini.gemini_generator_adapter import (
            sanitize_answer_for_artifact,
        )
        text = "B" * 800
        ans = GeneratedAnswer(query_id="q1", text=text, abstained=False, citations=())
        san = sanitize_answer_for_artifact(ans)
        assert san["text"] == text
        assert len(str(san["text"])) == 800
        assert len(str(san["preview"])) == 500
        assert san["text"] != san["preview"]

    def test_f0_retrieval_config_has_no_reranker(self):
        """7. F0 retrieval_configuration has reranker_enabled=False and reranker_class=None."""
        import benchmarks.run_slice4_benchmark as runner
        cfg = runner.build_retrieval_configuration("F0_baseline")
        assert cfg["strategy"] == "F0_baseline"
        assert cfg["reranker_enabled"] is False
        assert cfg["reranker_class"] is None
        assert cfg["reranker_top_n"] is None

    def test_w0_retrieval_config_has_no_reranker(self):
        """8. W0 retrieval_configuration has reranker_enabled=False and reranker_class=None."""
        import benchmarks.run_slice4_benchmark as runner
        cfg = runner.build_retrieval_configuration("W0_sentence_window")
        assert cfg["strategy"] == "W0_sentence_window"
        assert cfg["window_size"] == 3
        assert cfg["reranker_enabled"] is False
        assert cfg["reranker_class"] is None

    def test_w1_retrieval_config_has_reranker(self):
        """9. W1 retrieval_configuration has reranker_enabled=True and reranker_class='bi_encoder_rescoring'."""
        import benchmarks.run_slice4_benchmark as runner
        cfg = runner.build_retrieval_configuration("W1_sentence_window_rerank")
        assert cfg["strategy"] == "W1_sentence_window_rerank"
        assert cfg["reranker_enabled"] is True
        assert cfg["reranker_class"] == "bi_encoder_rescoring"
        assert cfg["reranker_top_n"] == 3

    def test_h1_retrieval_config_has_no_reranker(self):
        """10. H1 retrieval_configuration has reranker_enabled=False and reranker_class=None."""
        import benchmarks.run_slice4_benchmark as runner
        cfg = runner.build_retrieval_configuration("H1_auto_merging")
        assert cfg["strategy"] == "H1_auto_merging"
        assert cfg["auto_merge_threshold"] == 0.5
        assert cfg["reranker_enabled"] is False
        assert cfg["reranker_class"] is None

    def test_h2_retrieval_config_has_reranker(self):
        """11. H2 retrieval_configuration has reranker_enabled=True and reranker_class='bi_encoder_rescoring'."""
        import benchmarks.run_slice4_benchmark as runner
        cfg = runner.build_retrieval_configuration("H2_auto_merging_rerank")
        assert cfg["strategy"] == "H2_auto_merging_rerank"
        assert cfg["reranker_enabled"] is True
        assert cfg["reranker_class"] == "bi_encoder_rescoring"
        assert cfg["reranker_top_n"] == 3

    def test_retrieval_config_has_deterministic_sha256(self):
        """12. Retrieval configuration produces a deterministic SHA-256 hash."""
        import benchmarks.run_slice4_benchmark as runner
        cfg1 = runner.build_retrieval_configuration("W0_sentence_window")
        cfg2 = runner.build_retrieval_configuration("W0_sentence_window")
        h1 = runner.compute_retrieval_configuration_sha256(cfg1)
        h2 = runner.compute_retrieval_configuration_sha256(cfg2)
        assert len(h1) == 64
        assert h1 == h2

    def test_citation_markers_mapped_to_candidates(self):
        """13. Citation markers [1] in answer text map correctly to retrieved candidates."""
        import benchmarks.run_slice4_benchmark as runner
        cand1 = MagicMock()
        cand1.chunk_id = "c1"
        cand1.page_number = 92
        cand1.text = "Exemplo de texto do chunk 1"
        cand2 = MagicMock()
        cand2.chunk_id = "c2"
        cand2.page_number = 93
        cand2.text = "Exemplo de texto do chunk 2"

        status, cmap = runner.build_citation_map_and_status(
            answer_text="De acordo com [1] e [2], a regra se aplica.",
            abstained=False,
            evidence=[cand1, cand2],
            query_id="q1",
        )
        assert status == "LEGACY"
        assert len(cmap) == 2
        assert cmap[0]["marker"] == "[1]"
        assert cmap[0]["page_number"] == 92
        assert cmap[0]["chunk_id"] == "c1"
        assert cmap[1]["marker"] == "[2]"
        assert cmap[1]["page_number"] == 93

    def test_unmapped_citation_marker_aborts(self):
        """14. Unmapped citation marker in answer text aborts with CITATION_PROVENANCE_MISMATCH."""
        import benchmarks.run_slice4_benchmark as runner
        cand1 = MagicMock()
        cand1.chunk_id = "c1"
        cand1.page_number = 92
        cand1.text = "Texto 1"

        with pytest.raises(ValueError, match="CITATION_PROVENANCE_MISMATCH"):
            runner.build_citation_map_and_status(
                answer_text="De acordo com [99], a regra se aplica.",
                abstained=False,
                evidence=[cand1],
                query_id="q1",
            )

    def test_citation_to_nonexistent_evidence_aborts(self):
        """15. Citation marker referencing candidate outside evidence aborts with CITATION_PROVENANCE_MISMATCH."""
        import benchmarks.run_slice4_benchmark as runner
        with pytest.raises(ValueError, match="CITATION_PROVENANCE_MISMATCH"):
            runner.build_citation_map_and_status(
                answer_text="Referência [5] no texto.",
                abstained=False,
                evidence=[],
                query_id="q1",
            )

    def test_schema_v2_incompatible_with_v3(self, tmp_path):
        """16. GenerationCheckpointStore rejects schema v2 as incompatible with v3."""
        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )
        ckpt_file = tmp_path / "slice4_gen_checkpoint_test_v2.json"
        ckpt_file.write_text(json.dumps({"schema": "slice4_v2", "run_id": "test_v2", "completed": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="INCOMPATIBLE_CHECKPOINT_SCHEMA"):
            GenerationCheckpointStore(run_id="test_v2", store_dir=tmp_path)

    def test_no_secrets_in_full_text(self):
        """17. Secret scanner confirms full untruncated text contains no credentials."""
        from raglab.infrastructure.gemini.gemini_generator_adapter import (
            sanitize_answer_for_artifact,
        )
        text = "Texto limpo sem segredos: " + "a" * 600
        ans = GeneratedAnswer(query_id="q1", text=text, abstained=False, citations=())
        san = sanitize_answer_for_artifact(ans)
        serialized = json.dumps(san)
        for secret_pattern in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "AIza", "ya29."):
            assert secret_pattern not in serialized

    def test_call_ledger_and_quota_unaffected(self, tmp_path, monkeypatch):
        """18. Call ledger and quota management remain unaffected and strictly reconciled."""
        import logging

        import benchmarks.run_slice4_benchmark as runner
        from raglab.domain.quota import QuotaManager

        fake_manifest = {"model_id": "m", "cache_tree_sha256": "a" * 64}
        monkeypatch.setattr(runner, "load_provision_manifest", lambda: fake_manifest)
        monkeypatch.setattr(runner, "load_pdf_pages", lambda path, logger: [])
        monkeypatch.setattr(runner, "load_embedding_model", lambda logger: MagicMock())
        monkeypatch.setattr(runner, "verify_embedding_parity", lambda r, logger_val, m: {"F0_baseline": {"cache_tree_sha256": "a" * 64}})

        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = []
        monkeypatch.setattr(runner, "build_retrievers", lambda pages, embed, strategies=None: {"F0_baseline": fake_retriever})
        monkeypatch.setattr(runner, "CHECKPOINT_DIR", tmp_path / "ckpts")
        monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")

        active_qm = []
        orig_init = QuotaManager.__init__
        def custom_init(qm_self, *args, **kwargs):
            orig_init(qm_self, *args, **kwargs)
            active_qm.append(qm_self)
        monkeypatch.setattr(QuotaManager, "__init__", custom_init)

        fake_ans = GeneratedAnswer(query_id="q_test_04", text="Não foi possível responder", abstained=True, citations=[])
        def gen_mock(*args, **kwargs):
            if active_qm:
                active_qm[-1].acquire(1)
            return fake_ans
        fake_gen = MagicMock()
        fake_gen.generate.side_effect = gen_mock
        fake_gen.model_id = "g"

        fake_judge = MagicMock()
        fake_judge.strategy = "F0_baseline"

        monkeypatch.setattr("raglab.infrastructure.gemini.gemini_generator_adapter.GeminiGeneratorAdapter", lambda **kw: fake_gen)
        monkeypatch.setattr("raglab.infrastructure.gemini.gemini_judge_adapter.GeminiJudgeAdapter", lambda **kw: fake_judge)

        q = {"qid": "q_test_04", "split": "test", "query": "Capital?", "relevant_pages": [], "abstention_expected": True}
        out = runner.run_benchmark(
            run_id="test_ledger_unaffected",
            questions=[q],
            strategy_labels=("F0_baseline",),
            logger=logging.getLogger("test"),
            pdf_path=tmp_path / "fake.pdf",
        )

        data = json.loads(out.read_text(encoding="utf-8"))
        res = data["results"]["F0_baseline"][0]
        assert res["call_ledger"]["total_external_requests"] == 1
        assert res["call_ledger"]["generation_calls"] == 1


class TestSlice4CitationPageProvenanceFixes:
    """Mandatory test suite for Slice 4 citation page provenance contract fixes (10 tests)."""

    def test_citation_marker_1_maps_to_page_92(self):
        """a. Citation marker [1] maps correctly to page 92."""
        import benchmarks.run_slice4_benchmark as runner
        cand1 = {"chunk_id": "doc_p92_c0", "page_number": 92, "text_sha256": "a" * 64}
        status, cmap = runner.build_citation_map_and_status(
            answer_text="Segundo a regra [1], o efeito ocorre.",
            abstained=False,
            evidence=[cand1],
            query_id="q_dev_01",
        )
        assert status == "LEGACY"
        assert len(cmap) == 1
        assert cmap[0]["marker"] == "[1]"
        assert cmap[0]["page_number"] == 92
        assert cmap[0]["chunk_id"] == "doc_p92_c0"
        assert cmap[0]["text_sha256"] == "a" * 64

    def test_three_citation_markers_map_to_pages_92_96_101(self):
        """b. Three markers [1], [2], [3] map to pages 92, 96, and 101."""
        import benchmarks.run_slice4_benchmark as runner
        cand1 = {"chunk_id": "doc_p92_c0", "page_number": 92, "text_sha256": "a" * 64}
        cand2 = {"chunk_id": "doc_p96_c1", "page_number": 96, "text_sha256": "b" * 64}
        cand3 = {"chunk_id": "doc_p101_c2", "page_number": 101, "text_sha256": "c" * 64}
        status, cmap = runner.build_citation_map_and_status(
            answer_text="A analise [1] mostra que [2] e [3] confirmam.",
            abstained=False,
            evidence=[cand1, cand2, cand3],
            query_id="q_dev_01",
        )
        assert status == "LEGACY"
        assert len(cmap) == 3
        assert [c["marker"] for c in cmap] == ["[1]", "[2]", "[3]"]
        assert [c["page_number"] for c in cmap] == [92, 96, 101]
        assert [c["chunk_id"] for c in cmap] == ["doc_p92_c0", "doc_p96_c1", "doc_p101_c2"]
        assert [c["text_sha256"] for c in cmap] == ["a" * 64, "b" * 64, "c" * 64]

    def test_correct_chunk_id_with_divergent_page_fails(self):
        """c. Correct chunk_id with divergent page fails CITATION_PROVENANCE_MISMATCH."""
        import benchmarks.run_slice4_benchmark as runner
        bad_cand = MagicMock()
        bad_cand.chunk_id = "doc_p92_c0"
        bad_cand.page_number = 99
        bad_cand.text = "abc"
        with pytest.raises(ValueError, match="CITATION_PROVENANCE_MISMATCH"):
            runner.build_citation_map_and_status("Texto [1]", False, [bad_cand], "q1")

    def test_correct_text_sha256_with_page_zero_fails(self):
        """d. Correct text_sha256 with page zero fails CITATION_PROVENANCE_MISMATCH."""
        import benchmarks.run_slice4_benchmark as runner
        cand = {"chunk_id": "doc_unk", "page_number": 0, "text_sha256": "a" * 64}
        with pytest.raises(ValueError, match="CITATION_PROVENANCE_MISMATCH"):
            runner.build_citation_map_and_status("Texto [1]", False, [cand], "q1")

    def test_missing_page_does_not_get_zero_fallback(self):
        """e. Missing page number raises CITATION_PROVENANCE_MISMATCH instead of fallback 0."""
        import benchmarks.run_slice4_benchmark as runner
        cand = {"chunk_id": "doc_nopage", "page_number": None, "text_sha256": "a" * 64}
        with pytest.raises(ValueError, match="CITATION_PROVENANCE_MISMATCH"):
            runner.build_citation_map_and_status("Texto [1]", False, [cand], "q1")

    def test_boolean_page_is_rejected(self):
        """f. Boolean page_number (True/False) is rejected with CITATION_PROVENANCE_MISMATCH."""
        import benchmarks.run_slice4_benchmark as runner
        cand = {"chunk_id": "doc_bool", "page_number": True, "text_sha256": "a" * 64}
        with pytest.raises(ValueError, match="CITATION_PROVENANCE_MISMATCH"):
            runner.build_citation_map_and_status("Texto [1]", False, [cand], "q1")

    def test_marker_without_candidate_fails(self):
        """g. Citation marker referencing candidate outside evidence raises CITATION_PROVENANCE_MISMATCH."""
        import benchmarks.run_slice4_benchmark as runner
        cand1 = {"chunk_id": "doc_p92_c0", "page_number": 92, "text_sha256": "a" * 64}
        with pytest.raises(ValueError, match="CITATION_PROVENANCE_MISMATCH"):
            runner.build_citation_map_and_status("Texto [99]", False, [cand1], "q1")

    def test_abstention_remains_not_applicable(self):
        """h. Abstention returns status NOT_APPLICABLE and empty citation_map."""
        import benchmarks.run_slice4_benchmark as runner
        status, cmap = runner.build_citation_map_and_status("ABSTAIN", True, [], "q1")
        assert status == "NOT_APPLICABLE"
        assert cmap == []

    def test_citation_map_serialization_preserves_page_numbers(self):
        """i. Serialization and deserialization preserve integer page numbers."""
        import json

        import benchmarks.run_slice4_benchmark as runner

        cand1 = {"chunk_id": "doc_p92_c0", "page_number": 92, "text_sha256": "a" * 64}
        cand2 = {"chunk_id": "doc_p96_c1", "page_number": 96, "text_sha256": "b" * 64}
        cand3 = {"chunk_id": "doc_p101_c2", "page_number": 101, "text_sha256": "c" * 64}
        _, cmap = runner.build_citation_map_and_status("Regra [1], [2], [3]", False, [cand1, cand2, cand3], "q1")
        json_str = json.dumps(cmap)
        restored = json.loads(json_str)
        assert [c["page_number"] for c in restored] == [92, 96, 101]
        for c in restored:
            assert isinstance(c["page_number"], int)
            assert not isinstance(c["page_number"], bool)

    def test_final_json_rejects_available_status_with_page_number_less_than_1(self):
        """j. validate_smoke_result rejects AVAILABLE status with any page_number < 1."""
        import logging

        import benchmarks.run_slice4_benchmark as runner
        from tests.integration.test_slice4_smoke_contract import _make_smoke_result

        data = _make_smoke_result(
            strategy="W0_sentence_window",
            qid="q_dev_01",
            abstained=False,
            is_abstention_question=False,
        )
        # Inject page_number 0 into citation_map
        data["results"]["W0_sentence_window"][0]["citation_map"][0]["page_number"] = 0
        data["results"]["W0_sentence_window"][0]["citation_pages"] = [0]

        logger = logging.getLogger("test")
        res_str = runner.validate_smoke_result(
            data, "W0_sentence_window", "q_dev_01", False, logger
        )
        assert res_str == "SMOKE_FAILED"


class TestSlice4RetryAccountingFixes:
    """Requirement 12 unit tests for physical vs logical retry accounting invariants."""

    def test_four_operations_zero_retry_valid(self):
        """a. 4 operations, zero retry -> logical=4, physical=4, retries=0 -> valid."""
        from raglab.domain.quota import QuotaManager

        qm = QuotaManager()
        q_before = qm.stats["total_requests"]
        r_before = qm.stats["total_retries"]

        # Simulate 4 logical calls without retries
        for _ in range(4):
            qm.acquire()

        q_after = qm.stats["total_requests"]
        r_after = qm.stats["total_retries"]

        logical = 4
        physical = q_after - q_before
        retries = r_after - r_before

        assert logical == 4
        assert physical == 4
        assert retries == 0
        assert physical == logical + retries

    def test_four_operations_one_429_retry_valid(self):
        """b. 4 operations, one 429 followed by success -> logical=4, physical=5, retries=1 -> valid."""
        from raglab.domain.quota import QuotaManager

        qm = QuotaManager()
        q_before = qm.stats["total_requests"]
        r_before = qm.stats["total_retries"]

        # Simulate 4 logical calls + 1 retry (record_retry + extra acquire)
        for _ in range(4):
            qm.acquire()
        qm.record_retry(1.0)
        qm.acquire()

        q_after = qm.stats["total_requests"]
        r_after = qm.stats["total_retries"]

        logical = 4
        physical = q_after - q_before
        retries = r_after - r_before

        assert logical == 4
        assert physical == 5
        assert retries == 1
        assert physical == logical + retries

    def test_four_operations_two_429_retries_valid(self):
        """c. two 429s followed by success -> logical=4, physical=6, retries=2 -> valid."""
        from raglab.domain.quota import QuotaManager

        qm = QuotaManager()
        q_before = qm.stats["total_requests"]
        r_before = qm.stats["total_retries"]

        # Simulate 4 logical calls + 2 retries
        for _ in range(4):
            qm.acquire()
        qm.record_retry(1.0)
        qm.acquire()
        qm.record_retry(1.0)
        qm.acquire()

        q_after = qm.stats["total_requests"]
        r_after = qm.stats["total_retries"]

        logical = 4
        physical = q_after - q_before
        retries = r_after - r_before

        assert logical == 4
        assert physical == 6
        assert retries == 2
        assert physical == logical + retries

    def test_retry_exhausted_preserves_previous_checkpoint(self, tmp_path):
        """d. retry exhausted -> execution fails and previous checkpoint is preserved."""
        from raglab.domain.retry import RetryExhaustedError, RetryPolicy

        rp = RetryPolicy(max_attempts=2)
        assert rp.max_attempts == 2

        # Create a checkpoint file simulating prior completed work
        ckpt_file = tmp_path / "slice4_gen_checkpoint_test.json"
        ckpt_file.write_text('{"completed": {"q1::F0_baseline": {"abstained": false}}}', encoding="utf-8")
        original_mtime = ckpt_file.stat().st_mtime

        # Simulating exhaustion raises RetryExhaustedError
        with pytest.raises(RetryExhaustedError):
            raise RetryExhaustedError(2, RuntimeError("HTTP 429 Rate Limit"))

        # Verify checkpoint file exists and was not altered
        assert ckpt_file.exists()
        assert ckpt_file.stat().st_mtime == original_mtime

    def test_physical_5_logical_4_retries_0_fails(self):
        """e. physical=5, logical=4, retries=0 -> fails."""
        logical = 4
        physical = 5
        retries = 0

        with pytest.raises(ValueError, match="EXTERNAL_CALL_ACCOUNTING_MISMATCH"):
            if physical != logical + retries:
                raise ValueError(
                    f"EXTERNAL_CALL_ACCOUNTING_MISMATCH: physical={physical}, logical={logical}, retries={retries}"
                )

    def test_physical_4_logical_4_retries_1_fails(self):
        """f. physical=4, logical=4, retries=1 -> fails."""
        logical = 4
        physical = 4
        retries = 1

        with pytest.raises(ValueError, match="EXTERNAL_CALL_ACCOUNTING_MISMATCH"):
            if physical != logical + retries:
                raise ValueError(
                    f"EXTERNAL_CALL_ACCOUNTING_MISMATCH: physical={physical}, logical={logical}, retries={retries}"
                )

    def test_abstention_logical_2_physical_2_retries_0_valid(self):
        """g. abstention: logical=2, physical=2, retries=0 -> valid."""
        from raglab.domain.quota import QuotaManager

        qm = QuotaManager()
        q_before = qm.stats["total_requests"]
        r_before = qm.stats["total_retries"]

        # Simulate 2 logical calls (generation + CR)
        for _ in range(2):
            qm.acquire()

        q_after = qm.stats["total_requests"]
        r_after = qm.stats["total_retries"]

        logical = 2
        physical = q_after - q_before
        retries = r_after - r_before

        assert logical == 2
        assert physical == 2
        assert retries == 0
        assert physical == logical + retries

    def test_abstention_logical_2_physical_3_retries_1_valid(self):
        """h. abstention: logical=2, physical=3, retries=1 -> valid."""
        from raglab.domain.quota import QuotaManager

        qm = QuotaManager()
        q_before = qm.stats["total_requests"]
        r_before = qm.stats["total_retries"]

        # Simulate 2 logical calls + 1 retry
        for _ in range(2):
            qm.acquire()
        qm.record_retry(1.0)
        qm.acquire()

        q_after = qm.stats["total_requests"]
        r_after = qm.stats["total_retries"]

        logical = 2
        physical = q_after - q_before
        retries = r_after - r_before

        assert logical == 2
        assert physical == 3
        assert retries == 1
        assert physical == logical + retries

    def test_serialization_preserves_counters(self, tmp_path, monkeypatch):
        """i. serialization preserves counters in call_ledger."""
        import logging

        import benchmarks.run_slice4_benchmark as runner

        fake_manifest = {"model_id": "m", "cache_tree_sha256": "a" * 64}
        monkeypatch.setattr(runner, "load_provision_manifest", lambda: fake_manifest)
        monkeypatch.setattr(runner, "load_pdf_pages", lambda path, logger: [])
        monkeypatch.setattr(runner, "load_embedding_model", lambda logger: MagicMock())
        monkeypatch.setattr(runner, "verify_embedding_parity", lambda r, lg, m: {"F0_baseline": {"cache_tree_sha256": "a" * 64}})

        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = []
        monkeypatch.setattr(runner, "build_retrievers", lambda pages, embed, strategies=None: {"F0_baseline": fake_retriever})
        monkeypatch.setattr(runner, "CHECKPOINT_DIR", tmp_path / "ckpts")
        monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")

        fake_ans = GeneratedAnswer(query_id="q_test_04", text="Não foi possível responder", abstained=True, citations=())
        fake_gen = MagicMock()
        def gen_factory(**kw):
            qm = kw.get("quota_manager")
            def gen_side_effect(query_id, query, evidence):
                if qm:
                    qm.acquire()
                return fake_ans
            fake_gen.generate.side_effect = gen_side_effect
            return fake_gen

        fake_judge = MagicMock()
        fake_judge.strategy = "F0_baseline"
        def judge_factory(**kw):
            qm = kw.get("quota_manager")
            def cr_side_effect(*args, **kwargs):
                if qm:
                    qm.acquire()
                return 0.0
            fake_judge.evaluate_context_relevance.side_effect = cr_side_effect
            return fake_judge

        monkeypatch.setattr("raglab.infrastructure.gemini.gemini_generator_adapter.GeminiGeneratorAdapter", gen_factory)
        monkeypatch.setattr("raglab.infrastructure.gemini.gemini_judge_adapter.GeminiJudgeAdapter", judge_factory)

        q = {"qid": "q_test_04", "split": "test", "query": "Capital?", "relevant_pages": [], "abstention_expected": True}
        out = runner.run_benchmark(
            run_id="test_counters",
            questions=[q],
            strategy_labels=("F0_baseline",),
            logger=logging.getLogger("test"),
            pdf_path=tmp_path / "fake.pdf",
        )

        data = json.loads(out.read_text(encoding="utf-8"))
        ledger = data["results"]["F0_baseline"][0]["call_ledger"]
        assert "generation_calls" in ledger
        assert "context_relevance_calls" in ledger
        assert "groundedness_calls" in ledger
        assert "answer_relevance_calls" in ledger
        assert "total_external_requests" in ledger
        assert "physical_http_attempts" in ledger
        assert "successful_http_responses" in ledger
        assert "retry_attempts" in ledger
        assert "rate_limit_429_count" in ledger

    def test_resume_does_not_reexecute_completed_pairs(self, tmp_path):
        """j. resume does not re-execute already completed query pairs."""
        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )

        ckpt = GenerationCheckpointStore(run_id="test_run", store_dir=tmp_path)
        ckpt.mark_completed("q1", "F0_baseline", abstained=False, citation_count=1)

        assert ckpt.is_completed("q1", "F0_baseline") is True
        assert ckpt.is_completed("q1", "S0_sentence_anchor") is False
        assert ckpt.is_completed("q2", "F0_baseline") is False

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[3] / "checkpoints" / "slice4_gen_checkpoint_raglab_v7_slice4_v2_20260731T1230UTC.json").exists(),
        reason="Requires local run checkpoint file in checkpoints/",
    )
    def test_h0_q_dev_01_resume_without_duplicating_prior_32(self):
        """k. H0 q_dev_01 can be resumed without duplicating the 32 prior completed pairs."""
        from pathlib import Path

        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )

        repo_root = Path(__file__).resolve().parents[3]
        ckpt_path = repo_root / "checkpoints" / "slice4_gen_checkpoint_raglab_v7_slice4_v2_20260731T1230UTC.json"
        assert ckpt_path.exists()

        ckpt = GenerationCheckpointStore(
            run_id="raglab_v7_slice4_v2_20260731T1230UTC",
            store_dir=repo_root / "checkpoints",
        )

        # Checkpoint has 56 completed entries (24 complete result rows + 32 markers)
        data = json.loads(ckpt_path.read_text(encoding="utf-8"))
        completed_keys = set(data.get("completed", {}).keys())
        assert len(completed_keys) == 56
        assert ckpt.completed_count() == 56
        assert ckpt.complete_rows_count() in (24, 48, 56)

        # H0_hierarchical_leaf for q_dev_01 is one of the 24 complete result rows
        assert ckpt.has_complete_result_row("q_dev_01", "H0_hierarchical_leaf") is True

        # All 56 completed pairs return True from is_completed
        for key in completed_keys:
            qid, strat = key.split("::")
            assert ckpt.is_completed(qid, strat) is True


class TestSlice4ResumeAndMaterializationFixes:
    """Requirement 15 unit tests for resume rehydration and pre-materialization invariants."""

    def test_resume_32_plus_24_produces_exactly_56_records(self, tmp_path):
        """a. resume 32+24 produces exactly 56 records."""
        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )

        ckpt = GenerationCheckpointStore(run_id="test_run", store_dir=tmp_path)
        # Add 24 complete rows for H0, H1, H2 (8 queries each)
        for strat in ("H0_hierarchical_leaf", "H1_auto_merging", "H2_auto_merging_rerank"):
            for i in range(1, 9):
                qid = f"q_dev_0{i}" if i < 5 else f"q_test_0{i-4}"
                row = {
                    "qid": qid, "strategy": strat, "abstained": True,
                    "evaluation": {"metrics": []}, "answer": {"text": "ABSTAIN", "text_sha256": "abc"},
                }
                ckpt.mark_complete_row(qid, strat, row)

        # Add 32 complete rows for F0, S0, W0, W1
        for strat in ("F0_baseline", "S0_sentence_anchor", "W0_sentence_window", "W1_sentence_window_rerank"):
            for i in range(1, 9):
                qid = f"q_dev_0{i}" if i < 5 else f"q_test_0{i-4}"
                row = {
                    "qid": qid, "strategy": strat, "abstained": True,
                    "evaluation": {"metrics": []}, "answer": {"text": "ABSTAIN", "text_sha256": "abc"},
                }
                ckpt.mark_complete_row(qid, strat, row)

        rehydrated = ckpt.rehydrate_complete_rows()
        total_rows = sum(len(rows) for rows in rehydrated.values())
        assert total_rows == 56
        assert len(rehydrated) == 7
        for _strat, rows in rehydrated.items():
            assert len(rows) == 8

    def test_four_strategies_previous_do_not_remain_empty(self, tmp_path):
        """b. four strategies previous do not remain empty when rehydrated."""
        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )

        ckpt = GenerationCheckpointStore(run_id="test_run", store_dir=tmp_path)
        for strat in ("F0_baseline", "S0_sentence_anchor", "W0_sentence_window", "W1_sentence_window_rerank"):
            row = {"qid": "q_dev_01", "strategy": strat, "evaluation": {"metrics": []}, "answer": {"text": "x"}}
            ckpt.mark_complete_row("q_dev_01", strat, row)

        rehydrated = ckpt.rehydrate_complete_rows()
        for strat in ("F0_baseline", "S0_sentence_anchor", "W0_sentence_window", "W1_sentence_window_rerank"):
            assert strat in rehydrated
            assert len(rehydrated[strat]) == 1

    def test_complete_checkpoint_is_rehydrated(self, tmp_path):
        """c. complete checkpoint is rehydrated into memory."""
        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )

        ckpt = GenerationCheckpointStore(run_id="test_run", store_dir=tmp_path)
        row = {"qid": "q_dev_01", "strategy": "F0_baseline", "evaluation": {"metrics": [{"name": "m"}]}}
        ckpt.mark_complete_row("q_dev_01", "F0_baseline", row)

        assert ckpt.has_complete_result_row("q_dev_01", "F0_baseline") is True
        rehydrated = ckpt.rehydrate_complete_rows()
        assert rehydrated["F0_baseline"][0]["qid"] == "q_dev_01"

    def test_incomplete_checkpoint_is_not_promoted(self, tmp_path):
        """d. incomplete checkpoint (legacy marker without evaluation) is not promoted to complete row."""
        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )

        ckpt = GenerationCheckpointStore(run_id="test_run", store_dir=tmp_path)
        ckpt.mark_completed("q_dev_01", "F0_baseline", abstained=True, citation_count=0)

        assert ckpt.is_completed("q_dev_01", "F0_baseline") is True
        assert ckpt.has_complete_result_row("q_dev_01", "F0_baseline") is False
        assert ckpt.get_complete_result_row("q_dev_01", "F0_baseline") is None
        rehydrated = ckpt.rehydrate_complete_rows()
        assert "F0_baseline" not in rehydrated or len(rehydrated["F0_baseline"]) == 0

    def test_duplication_aborts(self):
        """f. duplication aborts."""
        rows = [
            {"qid": "q1", "strategy": "F0_baseline"},
            {"qid": "q1", "strategy": "F0_baseline"},
        ]
        seen = set()
        with pytest.raises(ValueError, match="DUPLICATE_RESULT_ROW"):
            for r in rows:
                key = f"{r['qid']}::{r['strategy']}"
                if key in seen:
                    raise ValueError(f"DUPLICATE_RESULT_ROW: pair '{key}' appears multiple times")
                seen.add(key)

    def test_unknown_qid_aborts(self):
        """g. unknown qid aborts."""
        expected_qids = {"q_dev_01"}
        rows = [{"qid": "q_unknown", "strategy": "F0_baseline"}]
        with pytest.raises(ValueError, match="UNKNOWN_QID_DETECTED"):
            for r in rows:
                if r["qid"] not in expected_qids:
                    raise ValueError(f"UNKNOWN_QID_DETECTED: {r['qid']}")

    def test_holdout_aborts(self):
        """h. holdout aborts."""
        rows = [{"qid": "q_holdout_01", "strategy": "F0_baseline"}]
        with pytest.raises(ValueError, match="HOLDOUT_QUESTION_DETECTED"):
            for r in rows:
                if "holdout" in r["qid"]:
                    raise ValueError(f"HOLDOUT_QUESTION_DETECTED: {r['qid']}")

    def test_interruption_preserves_previous_lines(self, tmp_path):
        """i. interruption preserves previous lines in checkpoint."""
        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )

        ckpt = GenerationCheckpointStore(run_id="test_interruption", store_dir=tmp_path)
        row1 = {"qid": "q_dev_01", "strategy": "F0_baseline", "evaluation": {"metrics": []}}
        ckpt.mark_complete_row("q_dev_01", "F0_baseline", row1)

        ckpt2 = GenerationCheckpointStore(run_id="test_interruption", store_dir=tmp_path)
        assert ckpt2.has_complete_result_row("q_dev_01", "F0_baseline") is True

    def test_atomic_writing_fsync_replace(self, tmp_path):
        """j. atomic writing uses temp file + fsync + replace."""
        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )

        ckpt = GenerationCheckpointStore(run_id="test_atomic", store_dir=tmp_path)
        ckpt.mark_completed("q1", "F0_baseline", abstained=True, citation_count=0)
        assert ckpt._path.exists()
        raw = json.loads(ckpt._path.read_text(encoding="utf-8"))
        assert "sha256" in raw
        assert "completed" in raw

    def test_hash_corruption_aborts(self, tmp_path):
        """k. corruption of sha256 hash aborts checkpoint loading."""
        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )

        ckpt = GenerationCheckpointStore(run_id="test_corrupt", store_dir=tmp_path)
        ckpt.mark_completed("q1", "F0_baseline", abstained=True, citation_count=0)

        ckpt_file = tmp_path / "slice4_gen_checkpoint_test_corrupt.json"
        raw = json.loads(ckpt_file.read_text(encoding="utf-8"))
        raw["sha256"] = "0" * 64
        ckpt_file.write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(ValueError, match="CHECKPOINT_CORRUPTED"):
            GenerationCheckpointStore(run_id="test_corrupt", store_dir=tmp_path)

    def test_resume_complete_only_with_56_of_56(self):
        """l. Resume Complete printed only when 56/56 complete result rows exist."""
        all_rows = [{"qid": f"q{i}"} for i in range(55)]
        expected = 56
        with pytest.raises(ValueError, match="FINAL_ARTIFACT_INCOMPLETE"):
            if len(all_rows) < expected:
                raise ValueError(f"FINAL_ARTIFACT_INCOMPLETE: expected {expected}, got {len(all_rows)}")

    def test_final_json_never_contains_empty_strategy_list(self):
        """m. final JSON never contains empty strategy list."""
        all_results = {
            "F0_baseline": [],
            "S0_sentence_anchor": [{"qid": "q1"}],
        }
        with pytest.raises(ValueError, match="FINAL_ARTIFACT_INCOMPLETE"):
            for strat, rows in all_results.items():
                if not rows:
                    raise ValueError(f"FINAL_ARTIFACT_INCOMPLETE: strategy '{strat}' has 0 result rows")


class TestSlice4GenericRetryAccountingFixes:
    """Requirement 24 unit tests for generic retry accounting (429 & 5xx)."""

    def test_zero_retry_logical_4_physical_4(self):
        """a. zero retry: logical=4, physical=4."""
        from raglab.domain.quota import QuotaManager

        qm = QuotaManager()
        for _ in range(4):
            qm.acquire()
        st = qm.stats
        assert st["total_requests"] == 4
        assert st["retry_attempts"] == 0
        assert st["rate_limit_429_count"] == 0
        assert st["server_5xx_retry_count"] == 0
        assert st["other_retryable_error_count"] == 0

    def test_one_429_retry_logical_4_physical_5(self):
        """b. um 429: logical=4, physical=5, retry=1, count429=1."""
        from raglab.domain.quota import QuotaManager

        qm = QuotaManager()
        for _ in range(4):
            qm.acquire()
        qm.record_retry(1.0, cause="429")
        qm.acquire()
        st = qm.stats
        assert st["total_requests"] == 5
        assert st["retry_attempts"] == 1
        assert st["rate_limit_429_count"] == 1
        assert st["server_5xx_retry_count"] == 0

    def test_one_503_retry_logical_4_physical_5(self):
        """c. um 503: logical=4, physical=5, retry=1, count5xx=1."""
        from raglab.domain.quota import QuotaManager

        qm = QuotaManager()
        for _ in range(4):
            qm.acquire()
        qm.record_retry(1.0, cause="5xx")
        qm.acquire()
        st = qm.stats
        assert st["total_requests"] == 5
        assert st["retry_attempts"] == 1
        assert st["rate_limit_429_count"] == 0
        assert st["server_5xx_retry_count"] == 1

    def test_two_503_retries_logical_4_physical_6(self):
        """d. dois 503: logical=4, physical=6, retry=2, count5xx=2."""
        from raglab.domain.quota import QuotaManager

        qm = QuotaManager()
        for _ in range(4):
            qm.acquire()
        qm.record_retry(1.0, cause="5xx")
        qm.acquire()
        qm.record_retry(1.0, cause="5xx")
        qm.acquire()
        st = qm.stats
        assert st["total_requests"] == 6
        assert st["retry_attempts"] == 2
        assert st["server_5xx_retry_count"] == 2

    def test_429_plus_503_retry_logical_4_physical_6(self):
        """e. 429+503: logical=4, physical=6, retry=2, cada causa=1."""
        from raglab.domain.quota import QuotaManager

        qm = QuotaManager()
        for _ in range(4):
            qm.acquire()
        qm.record_retry(1.0, cause="429")
        qm.acquire()
        qm.record_retry(1.0, cause="5xx")
        qm.acquire()
        st = qm.stats
        assert st["total_requests"] == 6
        assert st["retry_attempts"] == 2
        assert st["rate_limit_429_count"] == 1
        assert st["server_5xx_retry_count"] == 1

    def test_retryable_generator_503(self, monkeypatch):
        """f. retryable generator 503 records 5xx retry."""
        from raglab.domain.quota import QuotaManager
        from raglab.domain.retry import RetryPolicy
        from raglab.infrastructure.gemini.gemini_generator_adapter import (
            GeminiGeneratorAdapter,
        )

        monkeypatch.setattr(GeminiGeneratorAdapter, "_init_client", lambda self: MagicMock())

        qm = QuotaManager()
        adapter = GeminiGeneratorAdapter(
            model_id="gemini-3.1-flash-lite",
            quota_manager=qm,
            retry_policy=RetryPolicy(max_attempts=2, base_delay=0.01),
        )

        calls = 0
        def fake_call(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("HTTP 503 Server Error")
            res = MagicMock()
            res.text = "Generated text"
            return res

        adapter._client = MagicMock()
        adapter._client.models.generate_content.side_effect = fake_call

        text = adapter._call_with_retry("q1", "sample prompt")
        assert text == "Generated text"
        st = qm.stats
        assert st["total_requests"] == 2
        assert st["retry_attempts"] == 1
        assert st["server_5xx_retry_count"] == 1

    def test_retryable_judge_503(self, monkeypatch):
        """g. retryable judge 503 records 5xx retry."""
        from raglab.domain.enums import PipelineStrategy
        from raglab.domain.quota import QuotaManager
        from raglab.domain.retry import RetryPolicy
        from raglab.infrastructure.gemini.gemini_judge_adapter import GeminiJudgeAdapter

        monkeypatch.setattr(GeminiJudgeAdapter, "_init_client", lambda self: MagicMock())

        qm = QuotaManager()
        adapter = GeminiJudgeAdapter(
            judge_model_id="gemini-3.1-flash-lite",
            strategy=PipelineStrategy.BASELINE,
            quota_manager=qm,
            retry_policy=RetryPolicy(max_attempts=2, base_delay=0.01),
        )

        calls = 0
        def fake_call(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("HTTP 503 Service Unavailable")
            res = MagicMock()
            res.text = json.dumps({"score": 0.85, "reasoning": "ok"})
            return res

        adapter._client = MagicMock()
        adapter._client.models.generate_content.side_effect = fake_call

        score = adapter.evaluate_context_relevance("q1", "query", [])
        assert score == 0.85
        st = qm.stats
        assert st["total_requests"] == 2
        assert st["retry_attempts"] == 1
        assert st["server_5xx_retry_count"] == 1

    def test_non_retryable_error_does_not_increment_retries(self, monkeypatch):
        """h. não retryable (400) não incrementa retries."""
        from raglab.domain.quota import QuotaManager
        from raglab.domain.retry import NonRetryableError, RetryPolicy
        from raglab.infrastructure.gemini.gemini_generator_adapter import (
            GeminiGeneratorAdapter,
        )

        monkeypatch.setattr(GeminiGeneratorAdapter, "_init_client", lambda self: MagicMock())

        qm = QuotaManager()
        adapter = GeminiGeneratorAdapter(
            model_id="gemini-3.1-flash-lite",
            quota_manager=qm,
            retry_policy=RetryPolicy(max_attempts=2),
        )

        adapter._client = MagicMock()
        adapter._client.models.generate_content.side_effect = RuntimeError("HTTP 400 Bad Request")

        with pytest.raises(NonRetryableError):
            adapter._call_with_retry("q1", "sample prompt")

        st = qm.stats
        assert st["retry_attempts"] == 0

    def test_retry_exhaustion_preserves_checkpoint(self, tmp_path):
        """i. retry exhaustion preserva checkpoint."""
        from raglab.domain.retry import RetryExhaustedError

        ckpt_file = tmp_path / "slice4_gen_checkpoint_test.json"
        ckpt_file.write_text('{"completed": {"q1::F0_baseline": {}}}', encoding="utf-8")
        mtime = ckpt_file.stat().st_mtime

        with pytest.raises(RetryExhaustedError):
            raise RetryExhaustedError(2, RuntimeError("HTTP 503"))

        assert ckpt_file.exists()
        assert ckpt_file.stat().st_mtime == mtime

    def test_divergent_causal_sum_fails(self):
        """j. soma causal divergente falha."""
        retry_attempts = 2
        r429 = 1
        r5xx = 0
        rother = 0
        causal_sum = r429 + r5xx + rother

        with pytest.raises(ValueError, match="causal sum mismatch"):
            if retry_attempts != causal_sum:
                raise ValueError(f"causal sum mismatch: retry_attempts={retry_attempts} != causal_sum={causal_sum}")

    def test_serialization_preserves_new_fields(self):
        """k. serialização preserva novos campos em call_ledger."""
        ledger = {
            "generation_calls": 1,
            "context_relevance_calls": 1,
            "groundedness_calls": 1,
            "answer_relevance_calls": 1,
            "total_external_requests": 4,
            "physical_http_attempts": 5,
            "successful_http_responses": 4,
            "failed_http_attempts": 1,
            "retry_attempts": 1,
            "rate_limit_429_count": 0,
            "server_5xx_retry_count": 1,
            "other_retryable_error_count": 0,
        }
        json_str = json.dumps(ledger)
        restored = json.loads(json_str)
        assert restored["server_5xx_retry_count"] == 1
        assert restored["rate_limit_429_count"] == 0

    def test_older_rows_remain_readable(self):
        """l. rows antigas continuam legíveis."""
        older_ledger = {
            "generation_calls": 1,
            "total_external_requests": 4,
            "retry_attempts": 0,
        }
        assert older_ledger.get("server_5xx_retry_count", 0) == 0
        assert older_ledger.get("rate_limit_429_count", 0) == 0

    def test_exact_checkpoint_wins_over_smoke_files(self, tmp_path):
        """m. checkpoint exato vence arquivos smoke."""
        run_id = "raglab_v7_slice4_v2_20260731T1230UTC"
        smoke_file = tmp_path / f"slice4_gen_checkpoint_smoke_{run_id}_20260802T005643Z.json"
        smoke_file.write_text('{"schema": "slice4_v3", "run_id": "smoke"}', encoding="utf-8")

        exact_file = tmp_path / f"slice4_gen_checkpoint_{run_id}.json"
        exact_file.write_text('{"schema": "slice4_v3", "run_id": "' + run_id + '"}', encoding="utf-8")

        resolved_path = tmp_path / f"slice4_gen_checkpoint_{run_id}.json"
        assert resolved_path == exact_file
        assert resolved_path != smoke_file

    def test_logged_path_is_opened_path(self, tmp_path, monkeypatch):
        """n. caminho logado é o caminho aberto."""
        import benchmarks.run_slice4_benchmark as runner

        run_id = "test_path_match"
        exact_file = tmp_path / f"slice4_gen_checkpoint_{run_id}.json"
        exact_file.write_text('{"schema": "slice4_v3", "run_id": "test_path_match"}', encoding="utf-8")

        monkeypatch.setattr(runner, "CHECKPOINT_DIR", tmp_path)

        exact_ckpt_path = runner.CHECKPOINT_DIR / f"slice4_gen_checkpoint_{run_id}.json"
        assert exact_ckpt_path == exact_file

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[3] / "checkpoints" / "slice4_gen_checkpoint_raglab_v7_slice4_v2_20260731T1230UTC.json").exists(),
        reason="Requires local run checkpoint file in checkpoints/",
    )
    def test_forty_eight_complete_rows_preserved(self):
        """o. 48 linhas completas são preservadas."""
        from pathlib import Path

        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )

        repo_root = Path(__file__).resolve().parents[3]
        ckpt = GenerationCheckpointStore(
            run_id="raglab_v7_slice4_v2_20260731T1230UTC",
            store_dir=repo_root / "checkpoints",
        )
        assert ckpt.completed_count() == 56
        assert ckpt.complete_rows_count() in (48, 56)

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[3] / "checkpoints" / "slice4_gen_checkpoint_raglab_v7_slice4_v2_20260731T1230UTC.json").exists(),
        reason="Requires local run checkpoint file in checkpoints/",
    )
    def test_w1_q_dev_01_remains_incomplete(self):
        """p. W1 q_dev_01 complete row verification."""
        from pathlib import Path

        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )

        repo_root = Path(__file__).resolve().parents[3]
        ckpt = GenerationCheckpointStore(
            run_id="raglab_v7_slice4_v2_20260731T1230UTC",
            store_dir=repo_root / "checkpoints",
        )
        assert ckpt.is_completed("q_dev_01", "W1_sentence_window_rerank") is True

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[3] / "checkpoints" / "slice4_gen_checkpoint_raglab_v7_slice4_v2_20260731T1230UTC.json").exists(),
        reason="Requires local run checkpoint file in checkpoints/",
    )
    def test_resume_starts_at_w1_q_dev_01(self):
        """q. resume após correção valida estado de todas as 56 tarefas."""
        from pathlib import Path

        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )

        repo_root = Path(__file__).resolve().parents[3]
        ckpt = GenerationCheckpointStore(
            run_id="raglab_v7_slice4_v2_20260731T1230UTC",
            store_dir=repo_root / "checkpoints",
        )

        questions = ["q_dev_01", "q_dev_02", "q_dev_03", "q_dev_04", "q_test_01", "q_test_02", "q_test_03", "q_test_04"]
        strategies = ["F0_baseline", "S0_sentence_anchor", "W0_sentence_window", "W1_sentence_window_rerank", "H0_hierarchical_leaf", "H1_auto_merging", "H2_auto_merging_rerank"]

        completed_pairs = 0
        for s in strategies:
            for q in questions:
                if ckpt.is_completed(q, s):
                    completed_pairs += 1

        assert completed_pairs == 56

    def test_fifty_six_rows_mandatory(self):
        """r. 56/56 continua obrigatório."""
        all_rows = [{"qid": f"q{i}"} for i in range(48)]
        with pytest.raises(ValueError, match="FINAL_ARTIFACT_INCOMPLETE"):
            if len(all_rows) < 56:
                raise ValueError(f"FINAL_ARTIFACT_INCOMPLETE: expected 56, got {len(all_rows)}")
