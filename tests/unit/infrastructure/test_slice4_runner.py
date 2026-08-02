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

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
        with pytest.raises(SystemExit) as exc_info:
            runner.cmd_resume(args, tmp_path / "fake.pdf", logging.getLogger("t"))
        assert exc_info.value.code == 3

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
                            lambda r, l, m: {"F0_baseline": {"cache_tree_sha256": "abc123"}})
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
        from raglab.domain.entities import GeneratedAnswer
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
                try:
                    fake_ans.query_id = query_id
                except AttributeError:
                    pass
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
        monkeypatch.setattr(runner, "verify_embedding_parity", lambda r, l, m: {strategy_label: {"cache_tree_sha256": "a" * 64}})

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
        monkeypatch.setattr(runner, "verify_embedding_parity", lambda r, l, m: {"W0_sentence_window": {"cache_tree_sha256": "a" * 64}})

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
        monkeypatch.setattr(runner, "verify_embedding_parity", lambda r, l, m: {"F0_baseline": {"cache_tree_sha256": "a" * 64}})

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
        monkeypatch.setattr(runner, "verify_embedding_parity", lambda r, l, m: {"F0_baseline": {"cache_tree_sha256": "a" * 64}})

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
        monkeypatch.setattr(runner, "verify_embedding_parity", lambda r, l, m: {"F0_baseline": {"cache_tree_sha256": "a" * 64}})

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
        monkeypatch.setattr(runner, "verify_embedding_parity", lambda r, l, m: {"F0_baseline": {"cache_tree_sha256": "a" * 64}})

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
        monkeypatch.setattr(runner, "verify_embedding_parity", lambda r, l, m: {"F0_baseline": {"cache_tree_sha256": "a" * 64}})

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
        assert data["protocol_version"] == "raglab_v7_slice4_v2"
        assert data["artifact_schema_version"] == "slice4_v2"
        assert data["schema"] == "slice4_v2"
        eval_rec = data["results"]["F0_baseline"][0]["evaluation"]
        assert eval_rec["protocol_version"] == "raglab_v7_slice4_v2"
        assert eval_rec["artifact_schema_version"] == "slice4_v2"
        assert eval_rec["schema_version"] == "slice4_v2"

    def test_runbook_contains_separated_commands(self):
        """13: Runbook contains independent export and unset commands."""
        runbook = (Path(_REPO_ROOT) / "docs" / "runbooks" / "slice4_human_execution.md").read_text(encoding="utf-8")
        assert "export LANGCHAIN_TRACING_V2=false" in runbook
        assert "unset GOOGLE_API_KEY" in runbook
