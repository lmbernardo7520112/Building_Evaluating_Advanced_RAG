"""Integration tests for Slice 4 retriever builder registry.

These tests validate exactly the bugs observed in production:
- ImportError: cannot import name 'AutoMergingAdapter' (wrong class name)
- Eager import of all modules regardless of selected strategy
- Wrong constructor signatures throughout build_retrievers
- Missing retrieve() method on reranker composites

Tests use ONLY fake pages and deterministic embeddings — no network,
no Gemini, no FastEmbed model, no credentials.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# tests/integration/ → raglab-v7/
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))


# ─── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture()
def fake_embedding():
    """A deterministic embedding adapter satisfying the interface subset."""
    from raglab.infrastructure.retrieval.baseline_adapter import DeterministicEmbedding

    class FakeEmbeddingAdapter:
        def __init__(self, dim: int = 64) -> None:
            self._det = DeterministicEmbedding(dimension=dim)
            self.dimension = dim
            self.model_id = "fake-deterministic"

        def _embed(self, text: str) -> list[float]:
            return self._det.embed(text)

        def _get_query_embedding(self, text: str) -> list[float]:
            return self._det.embed(text)

        def _get_text_embedding(self, text: str) -> list[float]:
            return self._det.embed(text)

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [self._det.embed(t) for t in texts]

    return FakeEmbeddingAdapter()


@pytest.fixture()
def fake_pages():
    """Minimal pages for indexing."""
    from raglab.domain.value_objects import DocumentPage

    return [
        DocumentPage(
            document_id="test_doc",
            page_number=1,
            text="Prova por exaustão é uma técnica de demonstração que verifica todos os casos. " * 10,
        ),
        DocumentPage(
            document_id="test_doc",
            page_number=2,
            text="O princípio da indução matemática estabelece a base e o passo indutivo. " * 10,
        ),
    ]


# ─── 1. F0 ISOLADO ────────────────────────────────────────────────

class TestF0Isolated:
    """F0_baseline must build WITHOUT importing auto_merging or reranker."""

    def test_f0_builds_and_retrieves(self, fake_pages, fake_embedding):
        import benchmarks.run_slice4_benchmark as runner

        retrievers = runner.build_retrievers(
            pages=fake_pages, embed_model=fake_embedding,
            strategies=("F0_baseline",),
        )
        assert "F0_baseline" in retrievers
        assert len(retrievers) == 1

        r = retrievers["F0_baseline"]
        results = r.retrieve("prova por exaustão", top_k=2)
        assert isinstance(results, (list, tuple))

    def test_f0_does_not_import_auto_merging(self, fake_pages, fake_embedding):
        """After building F0 only, auto_merging_adapter must NOT be in sys.modules."""
        # Clear any previous imports
        mods_to_clear = [
            k for k in sys.modules
            if "auto_merging_adapter" in k
        ]
        for m in mods_to_clear:
            del sys.modules[m]

        import benchmarks.run_slice4_benchmark as runner

        runner.build_retrievers(
            pages=fake_pages, embed_model=fake_embedding,
            strategies=("F0_baseline",),
        )

        auto_merging_loaded = any(
            "auto_merging_adapter" in k for k in sys.modules
        )
        assert not auto_merging_loaded, (
            "auto_merging_adapter was imported when only F0_baseline requested"
        )

    def test_f0_does_not_import_reranker(self, fake_pages, fake_embedding):
        """After building F0 only, reranker_adapter must NOT be in sys.modules."""
        mods_to_clear = [
            k for k in sys.modules if "reranker_adapter" in k
        ]
        for m in mods_to_clear:
            del sys.modules[m]

        import benchmarks.run_slice4_benchmark as runner

        runner.build_retrievers(
            pages=fake_pages, embed_model=fake_embedding,
            strategies=("F0_baseline",),
        )

        reranker_loaded = any(
            "reranker_adapter" in k for k in sys.modules
        )
        assert not reranker_loaded, (
            "reranker_adapter was imported when only F0_baseline requested"
        )


# ─── 2. W0 ISOLADO ────────────────────────────────────────────────

class TestW0Isolated:
    """W0 must NOT build W1 nor import auto-merging."""

    def test_w0_builds_alone(self, fake_pages, fake_embedding):
        import benchmarks.run_slice4_benchmark as runner

        retrievers = runner.build_retrievers(
            pages=fake_pages, embed_model=fake_embedding,
            strategies=("W0_sentence_window",),
        )
        assert "W0_sentence_window" in retrievers
        assert len(retrievers) == 1

    def test_w0_does_not_import_auto_merging(self, fake_pages, fake_embedding):
        mods_to_clear = [
            k for k in sys.modules if "auto_merging_adapter" in k
        ]
        for m in mods_to_clear:
            del sys.modules[m]

        import benchmarks.run_slice4_benchmark as runner

        runner.build_retrievers(
            pages=fake_pages, embed_model=fake_embedding,
            strategies=("W0_sentence_window",),
        )
        assert not any("auto_merging_adapter" in k for k in sys.modules)


# ─── 3. H1 CANONICAL CLASS ────────────────────────────────────────

class TestH1CanonicalClass:
    """H1 must import HierarchicalRetrievalAdapter (the real class)."""

    def test_h1_uses_canonical_class(self, fake_pages, fake_embedding):
        import benchmarks.run_slice4_benchmark as runner

        retrievers = runner.build_retrievers(
            pages=fake_pages, embed_model=fake_embedding,
            strategies=("H1_auto_merging",),
        )
        r = retrievers["H1_auto_merging"]
        # Must have retrieve method
        assert hasattr(r, "retrieve")
        results = r.retrieve("indução matemática", top_k=2)
        assert isinstance(results, (list, tuple))

    def test_h1_class_name_is_correct(self):
        from raglab.infrastructure.retrieval.auto_merging_adapter import (
            HierarchicalRetrievalAdapter,
        )
        assert HierarchicalRetrievalAdapter is not None
        # AutoMergingAdapter must NOT exist
        mod = importlib.import_module(
            "raglab.infrastructure.retrieval.auto_merging_adapter"
        )
        assert not hasattr(mod, "AutoMergingAdapter"), (
            "AutoMergingAdapter alias found — use HierarchicalRetrievalAdapter"
        )


# ─── 4. ALL SEVEN ─────────────────────────────────────────────────

class TestAllSevenStrategies:
    """Full registry: all 7 strategies build and return RetrievalPort."""

    def test_all_seven_build(self, fake_pages, fake_embedding):
        import benchmarks.run_slice4_benchmark as runner

        retrievers = runner.build_retrievers(
            pages=fake_pages, embed_model=fake_embedding,
        )
        assert len(retrievers) == 7
        assert set(retrievers.keys()) == set(runner.VALID_STRATEGIES)

    def test_all_seven_have_retrieve(self, fake_pages, fake_embedding):
        import benchmarks.run_slice4_benchmark as runner

        retrievers = runner.build_retrievers(
            pages=fake_pages, embed_model=fake_embedding,
        )
        for label, r in retrievers.items():
            assert hasattr(r, "retrieve"), f"{label} missing retrieve()"
            results = r.retrieve("teste", top_k=1)
            assert isinstance(results, (list, tuple)), (
                f"{label} retrieve returned {type(results)}"
            )

    def test_no_duplicate_labels(self):
        import benchmarks.run_slice4_benchmark as runner
        labels = list(runner.VALID_STRATEGIES)
        assert len(labels) == len(set(labels))


# ─── 5. IMPORT CONTRACT ───────────────────────────────────────────

class TestImportContract:
    """Every symbol referenced by the runner must exist in its module."""

    def test_canonical_adapter_names_exist(self):
        """Verify all canonical class names are importable."""
        from raglab.infrastructure.retrieval.auto_merging_adapter import (
            HierarchicalRetrievalAdapter,
        )
        from raglab.infrastructure.retrieval.baseline_adapter import (
            InMemoryBaselineAdapter,
        )
        from raglab.infrastructure.retrieval.reranker_adapter import (
            LocalRerankerAdapter,
        )
        from raglab.infrastructure.retrieval.sentence_anchor_adapter import (
            SentenceAnchorAdapter,
        )
        from raglab.infrastructure.retrieval.sentence_window_adapter import (
            SentenceWindowAdapter,
        )

        assert InMemoryBaselineAdapter is not None
        assert SentenceAnchorAdapter is not None
        assert SentenceWindowAdapter is not None
        assert HierarchicalRetrievalAdapter is not None
        assert LocalRerankerAdapter is not None

    def test_wrong_names_do_not_exist(self):
        """Names from the old broken runner must NOT exist as aliases."""
        mod_am = importlib.import_module(
            "raglab.infrastructure.retrieval.auto_merging_adapter"
        )
        assert not hasattr(mod_am, "AutoMergingAdapter")

        mod_bl = importlib.import_module(
            "raglab.infrastructure.retrieval.baseline_adapter"
        )
        assert not hasattr(mod_bl, "BaselineRetrieverAdapter")

    def test_runner_source_no_wrong_names(self):
        """The runner file must not reference AutoMergingAdapter or BaselineRetrieverAdapter."""
        runner_src = (_REPO_ROOT / "benchmarks" / "run_slice4_benchmark.py").read_text(
            encoding="utf-8"
        )
        assert "AutoMergingAdapter" not in runner_src, (
            "Runner still references non-existent AutoMergingAdapter"
        )
        assert "BaselineRetrieverAdapter" not in runner_src, (
            "Runner still references non-existent BaselineRetrieverAdapter"
        )


# ─── 6. CONSTRUCTOR CONTRACT ──────────────────────────────────────

class TestConstructorContract:
    """Instantiate each adapter with its real constructor signature."""

    def test_baseline_constructor(self, fake_embedding):
        from raglab.infrastructure.retrieval.baseline_adapter import (
            InMemoryBaselineAdapter,
        )
        # Default: DeterministicEmbedding
        adapter = InMemoryBaselineAdapter()
        assert adapter.embedding is not None

    def test_sentence_anchor_constructor(self, fake_embedding):
        from raglab.infrastructure.retrieval.sentence_anchor_adapter import (
            SentenceAnchorAdapter,
        )
        adapter = SentenceAnchorAdapter(embedding_adapter=fake_embedding)
        assert adapter.embedding_adapter is fake_embedding

    def test_sentence_window_constructor(self, fake_embedding):
        from raglab.infrastructure.retrieval.sentence_window_adapter import (
            SentenceWindowAdapter,
        )
        adapter = SentenceWindowAdapter(
            embedding_adapter=fake_embedding, window_size=3,
        )
        assert adapter.window_size == 3

    def test_hierarchical_constructor(self):
        from raglab.infrastructure.retrieval.auto_merging_adapter import (
            HierarchicalRetrievalAdapter,
        )
        adapter = HierarchicalRetrievalAdapter(
            auto_merge=True,
            merge_threshold=0.5,
            top_k=3,
        )
        assert adapter.auto_merge is True
        assert adapter.merge_threshold == 0.5

    def test_reranker_constructor(self, fake_embedding):
        from raglab.infrastructure.retrieval.reranker_adapter import (
            LocalRerankerAdapter,
        )
        reranker = LocalRerankerAdapter(embedding_adapter=fake_embedding)
        assert reranker.embedding_adapter is fake_embedding

    def test_baseline_rejects_unknown_kwarg(self):
        from raglab.infrastructure.retrieval.baseline_adapter import (
            InMemoryBaselineAdapter,
        )
        with pytest.raises(TypeError):
            InMemoryBaselineAdapter(pages=[], embed_model=None)  # type: ignore[call-arg]


# ─── 7. SMOKE F0 FAKE (full path: cmd_smoke → run_benchmark → build_retrievers) ─

class TestSmokeF0Fake:
    """F0 smoke with fake generator/judge — proves the real code path works."""

    def test_smoke_f0_end_to_end_fake(self, fake_pages, fake_embedding, tmp_path, monkeypatch):
        import logging

        import benchmarks.run_slice4_benchmark as runner

        # Patch out PDF loading, credential check, and Gemini
        monkeypatch.setattr(runner, "load_pdf_pages", lambda path, logger: fake_pages)
        monkeypatch.setattr(runner, "load_embedding_model", lambda logger, **kw: fake_embedding)
        monkeypatch.setattr(runner, "check_credential", lambda logger: None)
        monkeypatch.setattr(runner, "verify_pdf", lambda path, logger: None)
        monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(runner, "CHECKPOINT_DIR", tmp_path)

        # Mock Gemini adapter classes
        fake_answer = MagicMock()
        fake_answer.abstained = False
        fake_answer.citations = []

        fake_eval = MagicMock()

        fake_generator = MagicMock()
        fake_generator.generate.return_value = fake_answer
        fake_generator.model_id = "fake-gemini"

        fake_judge = MagicMock()
        fake_judge.evaluate.return_value = fake_eval

        # Patch the imports inside run_benchmark
        fake_gen_mod = MagicMock()
        fake_gen_mod.GeminiGeneratorAdapter.return_value = fake_generator
        fake_gen_mod.sanitize_answer_for_artifact.return_value = {"text": "fake"}

        fake_judge_mod = MagicMock()
        fake_judge_mod.GeminiJudgeAdapter.return_value = fake_judge
        fake_judge_mod.sanitize_evaluation_for_artifact.return_value = {"score": 0.5}

        monkeypatch.setitem(
            sys.modules,
            "raglab.infrastructure.gemini.gemini_generator_adapter",
            fake_gen_mod,
        )
        monkeypatch.setitem(
            sys.modules,
            "raglab.infrastructure.gemini.gemini_judge_adapter",
            fake_judge_mod,
        )

        logger = logging.getLogger("test_smoke_f0_fake")
        output_path = runner.run_benchmark(
            run_id="smoke_test_f0_fake",
            questions=[runner.ACTIVE_QUESTIONS[0]],
            strategy_labels=("F0_baseline",),
            logger=logger,
            pdf_path=tmp_path / "fake.pdf",
        )

        import json
        data = json.loads(output_path.read_text())
        assert "results" in data

    def test_smoke_f0_does_not_load_advanced_modules(
        self, fake_pages, fake_embedding, tmp_path, monkeypatch
    ):
        """After F0 smoke, auto_merging_adapter must NOT be loaded."""
        import logging

        import benchmarks.run_slice4_benchmark as runner

        monkeypatch.setattr(runner, "load_pdf_pages", lambda path, logger: fake_pages)
        monkeypatch.setattr(runner, "load_embedding_model", lambda logger, **kw: fake_embedding)
        monkeypatch.setattr(runner, "check_credential", lambda logger: None)
        monkeypatch.setattr(runner, "verify_pdf", lambda path, logger: None)
        monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(runner, "CHECKPOINT_DIR", tmp_path)

        fake_answer = MagicMock()
        fake_answer.abstained = True
        fake_answer.citations = []

        fake_generator = MagicMock()
        fake_generator.generate.return_value = fake_answer
        fake_generator.model_id = "fake"

        fake_gen_mod = MagicMock()
        fake_gen_mod.GeminiGeneratorAdapter.return_value = fake_generator
        fake_gen_mod.sanitize_answer_for_artifact.return_value = {}

        fake_judge_mod = MagicMock()
        fake_judge_mod.GeminiJudgeAdapter.return_value = MagicMock()
        fake_judge_mod.sanitize_evaluation_for_artifact.return_value = {}

        monkeypatch.setitem(sys.modules, "raglab.infrastructure.gemini.gemini_generator_adapter", fake_gen_mod)
        monkeypatch.setitem(sys.modules, "raglab.infrastructure.gemini.gemini_judge_adapter", fake_judge_mod)

        # Clear auto-merging modules
        for k in list(sys.modules):
            if "auto_merging_adapter" in k:
                del sys.modules[k]

        runner.run_benchmark(
            run_id="smoke_isolation_test",
            questions=[runner.ACTIVE_QUESTIONS[0]],
            strategy_labels=("F0_baseline",),
            logger=logging.getLogger("t"),
            pdf_path=tmp_path / "fake.pdf",
        )

        assert not any("auto_merging_adapter" in k for k in sys.modules)


# ─── 8. FULL FAKE ─────────────────────────────────────────────────

class TestFullFake:
    """All 7 strategies built before generation — incompatibility aborts early."""

    def test_all_seven_built_before_generate(
        self, fake_pages, fake_embedding, tmp_path, monkeypatch
    ):
        import logging

        import benchmarks.run_slice4_benchmark as runner

        monkeypatch.setattr(runner, "load_pdf_pages", lambda path, logger: fake_pages)
        monkeypatch.setattr(runner, "load_embedding_model", lambda logger, **kw: fake_embedding)
        monkeypatch.setattr(runner, "check_credential", lambda logger: None)
        monkeypatch.setattr(runner, "verify_pdf", lambda path, logger: None)
        monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(runner, "CHECKPOINT_DIR", tmp_path)

        build_called = []
        orig_build = runner.build_retrievers

        def _tracking_build(pages, embed_model, strategies=None):
            build_called.append(strategies)
            return orig_build(pages, embed_model, strategies=strategies)

        monkeypatch.setattr(runner, "build_retrievers", _tracking_build)

        fake_answer = MagicMock()
        fake_answer.abstained = True
        fake_answer.citations = []

        fake_generator = MagicMock()
        fake_generator.generate.return_value = fake_answer
        fake_generator.model_id = "fake"

        fake_gen_mod = MagicMock()
        fake_gen_mod.GeminiGeneratorAdapter.return_value = fake_generator
        fake_gen_mod.sanitize_answer_for_artifact.return_value = {}

        fake_judge_mod = MagicMock()
        fake_judge_mod.GeminiJudgeAdapter.return_value = MagicMock()
        fake_judge_mod.sanitize_evaluation_for_artifact.return_value = {}

        monkeypatch.setitem(sys.modules, "raglab.infrastructure.gemini.gemini_generator_adapter", fake_gen_mod)
        monkeypatch.setitem(sys.modules, "raglab.infrastructure.gemini.gemini_judge_adapter", fake_judge_mod)

        runner.run_benchmark(
            run_id="full_fake_test",
            questions=[runner.ACTIVE_QUESTIONS[0]],
            strategy_labels=runner.VALID_STRATEGIES,
            logger=logging.getLogger("t"),
            pdf_path=tmp_path / "fake.pdf",
        )

        # build_retrievers was called with all 7 strategies
        assert len(build_called) == 1
        assert build_called[0] == runner.VALID_STRATEGIES


# ─── 9. REGRESSION: AutoMergingAdapter must not exist ─────────────

class TestAutoMergingAdapterRegression:
    """Ensure the non-existent AutoMergingAdapter name cannot resurface."""

    def test_auto_merging_adapter_not_importable(self):
        mod = importlib.import_module(
            "raglab.infrastructure.retrieval.auto_merging_adapter"
        )
        assert not hasattr(mod, "AutoMergingAdapter"), (
            "REGRESSION: AutoMergingAdapter alias found — "
            "the canonical name is HierarchicalRetrievalAdapter"
        )

    def test_baseline_retriever_adapter_not_importable(self):
        mod = importlib.import_module(
            "raglab.infrastructure.retrieval.baseline_adapter"
        )
        assert not hasattr(mod, "BaselineRetrieverAdapter"), (
            "REGRESSION: BaselineRetrieverAdapter alias found — "
            "the canonical name is InMemoryBaselineAdapter"
        )


# ─── 10. UNKNOWN/DUPLICATE STRATEGY ──────────────────────────────

class TestStrategyValidation:
    def test_unknown_strategy_raises(self, fake_pages, fake_embedding):
        import benchmarks.run_slice4_benchmark as runner

        with pytest.raises(ValueError, match="Unknown strategies"):
            runner.build_retrievers(
                pages=fake_pages, embed_model=fake_embedding,
                strategies=("NONEXISTENT_strategy",),
            )

    def test_duplicate_strategy_raises(self, fake_pages, fake_embedding):
        import benchmarks.run_slice4_benchmark as runner

        with pytest.raises(ValueError, match="Duplicate strategies"):
            runner.build_retrievers(
                pages=fake_pages, embed_model=fake_embedding,
                strategies=("F0_baseline", "F0_baseline"),
            )


# ─── 11. PROVISIONER CLI ─────────────────────────────────────────

class TestProvisionerCLI:
    """Validate --help and bare invocation semantics."""

    _PROVISIONER = str(_REPO_ROOT / "scripts" / "provision_embedding_model.py")
    _ENV = {**__import__("os").environ}  # inherit env for venv

    def test_help_exits_zero(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, self._PROVISIONER, "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_help_shows_execute_flag(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, self._PROVISIONER, "--help"],
            capture_output=True, text=True,
        )
        assert "--execute" in result.stdout

    def test_bare_invocation_exits_two(self):
        """Without --execute, must exit 2 with no side effects."""
        import subprocess
        result = subprocess.run(
            [sys.executable, self._PROVISIONER],
            capture_output=True, text=True,
        )
        assert result.returncode == 2

    def test_bare_invocation_shows_instruction(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, self._PROVISIONER],
            capture_output=True, text=True,
        )
        assert "--execute" in (result.stdout + result.stderr)


# ─── 12. LLAMAINDEX EMBEDDING BRIDGE ─────────────────────────────

class TestLlamaIndexEmbeddingBridge:
    def test_bridge_text_embedding_delegation(self, fake_embedding):
        from raglab.infrastructure.retrieval.llamaindex_adapter import (
            LlamaIndexEmbeddingBridge,
        )

        bridge = LlamaIndexEmbeddingBridge(fake_embedding)
        vec = bridge._get_text_embedding("teste de texto")
        assert isinstance(vec, list)
        assert len(vec) == fake_embedding.dimension

    def test_bridge_query_embedding_delegation(self, fake_embedding):
        from raglab.infrastructure.retrieval.llamaindex_adapter import (
            LlamaIndexEmbeddingBridge,
        )

        bridge = LlamaIndexEmbeddingBridge(fake_embedding)
        vec = bridge._get_query_embedding("teste de busca")
        assert isinstance(vec, list)
        assert len(vec) == fake_embedding.dimension

    def test_bridge_preserves_dimension(self, fake_embedding):
        from raglab.infrastructure.retrieval.llamaindex_adapter import (
            LlamaIndexEmbeddingBridge,
        )

        bridge = LlamaIndexEmbeddingBridge(fake_embedding)
        assert bridge.dimension == fake_embedding.dimension

    def test_bridge_floats_are_finite(self, fake_embedding):
        import math

        from raglab.infrastructure.retrieval.llamaindex_adapter import (
            LlamaIndexEmbeddingBridge,
        )

        bridge = LlamaIndexEmbeddingBridge(fake_embedding)
        vec = bridge._get_text_embedding("exemplo para verificar finitude")
        assert all(math.isfinite(x) for x in vec)

    @pytest.mark.asyncio
    async def test_bridge_async_methods(self, fake_embedding):
        from raglab.infrastructure.retrieval.llamaindex_adapter import (
            LlamaIndexEmbeddingBridge,
        )

        bridge = LlamaIndexEmbeddingBridge(fake_embedding)
        vec_q = await bridge._aget_query_embedding("query async")
        vec_t = await bridge._aget_text_embedding("text async")
        assert len(vec_q) == fake_embedding.dimension
        assert len(vec_t) == fake_embedding.dimension


# ─── 13. EMBEDDING PARITY & INJECTED EMBEDDINGS ───────────────────

class TestEmbeddingParity:
    def test_h0_uses_injected_embedding(self, fake_pages, fake_embedding):
        import benchmarks.run_slice4_benchmark as runner

        retrievers = runner.build_retrievers(
            pages=fake_pages,
            embed_model=fake_embedding,
            strategies=("H0_hierarchical_leaf",),
        )
        root = runner.extract_underlying_embedding_adapter(
            retrievers["H0_hierarchical_leaf"]
        )
        assert root is fake_embedding

    def test_h1_uses_injected_embedding(self, fake_pages, fake_embedding):
        import benchmarks.run_slice4_benchmark as runner

        retrievers = runner.build_retrievers(
            pages=fake_pages,
            embed_model=fake_embedding,
            strategies=("H1_auto_merging",),
        )
        root = runner.extract_underlying_embedding_adapter(
            retrievers["H1_auto_merging"]
        )
        assert root is fake_embedding

    def test_h2_uses_injected_embedding_in_retrieval_and_rerank(
        self, fake_pages, fake_embedding
    ):
        import benchmarks.run_slice4_benchmark as runner

        retrievers = runner.build_retrievers(
            pages=fake_pages,
            embed_model=fake_embedding,
            strategies=("H2_auto_merging_rerank",),
        )
        h2 = retrievers["H2_auto_merging_rerank"]
        root_base = runner.extract_underlying_embedding_adapter(h2._base)  # noqa: SLF001
        root_reranker = runner.extract_underlying_embedding_adapter(
            h2._reranker  # noqa: SLF001
        )
        assert root_base is fake_embedding
        assert root_reranker is fake_embedding

    def test_no_llamaindex_deterministic_embedding_in_runner_source(self):
        runner_src = (_REPO_ROOT / "benchmarks" / "run_slice4_benchmark.py").read_text(
            encoding="utf-8"
        )
        assert "LlamaIndexDeterministicEmbedding" not in runner_src, (
            "Runner source must not reference LlamaIndexDeterministicEmbedding in scientific runtime"
        )

    def test_all_seven_fingerprints_equal(self, fake_pages, fake_embedding):
        import benchmarks.run_slice4_benchmark as runner

        retrievers = runner.build_retrievers(
            pages=fake_pages, embed_model=fake_embedding
        )
        fps = runner.verify_embedding_parity(retrievers)
        assert len(fps) == 7
        ref_fp = fps["F0_baseline"]
        for label, fp in fps.items():
            assert fp == ref_fp, f"Strategy {label} fingerprint {fp} != {ref_fp}"

    def test_divergent_fingerprint_aborts_before_generator(
        self, fake_pages, fake_embedding, tmp_path, monkeypatch
    ):
        import logging

        import benchmarks.run_slice4_benchmark as runner

        retrievers = runner.build_retrievers(
            pages=fake_pages, embed_model=fake_embedding
        )
        h1_root = runner.extract_underlying_embedding_adapter(
            retrievers["H1_auto_merging"]
        )
        monkeypatch.setattr(h1_root, "model_id", "divergent-model-xyz", raising=False)

        monkeypatch.setattr(runner, "load_pdf_pages", lambda path, logger: fake_pages)
        monkeypatch.setattr(
            runner, "load_embedding_model", lambda logger, **kw: fake_embedding
        )
        monkeypatch.setattr(
            runner,
            "build_retrievers",
            lambda pages, embed, strategies=None: retrievers,
        )

        gen_mock = MagicMock()
        monkeypatch.setitem(
            sys.modules,
            "raglab.infrastructure.gemini.gemini_generator_adapter",
            gen_mock,
        )

        with pytest.raises(ValueError, match="EMBEDDING_PARITY_FAILED"):
            runner.run_benchmark(
                run_id="test_parity_failure",
                questions=[runner.ACTIVE_QUESTIONS[0]],
                strategy_labels=runner.VALID_STRATEGIES,
                logger=logging.getLogger("t"),
                pdf_path=tmp_path / "fake.pdf",
            )
        assert not gen_mock.GeminiGeneratorAdapter.called

    def test_preflight_retrievers_emits_parity_ok(self, capsys):
        import logging

        import benchmarks.run_slice4_benchmark as runner

        logger = logging.getLogger("test_preflight_parity")
        args = runner.build_parser().parse_args(["--mode", "preflight-retrievers"])
        runner.cmd_preflight_retrievers(args, logger)
        captured = capsys.readouterr()
        assert (
            "EMBEDDING_PARITY_OK" in captured.out
            or "EMBEDDING_PARITY_OK" in captured.err
        )


# ─── 14. OFFLINE ATTESTATION CLI ─────────────────────────────────

class TestOfflineAttestationCLI:
    def test_attest_existing_offline_fake_cache(self, tmp_path):
        """Test --attest-existing on a fake cache directory without network or Gemini."""
        import os
        import subprocess

        fake_cache = tmp_path / "fake_model_cache"
        fake_cache.mkdir(parents=True, exist_ok=True)
        (fake_cache / "model.onnx").write_bytes(
            b"dummy onnx weights content for testing"
        )

        fake_manifest = tmp_path / "test_provision_manifest.json"

        # Mock FastEmbedEmbeddingAdapter._embed so fastembed/onnxruntime aren't called on dummy bytes
        from raglab.infrastructure.embeddings.fastembed_adapter import (
            FastEmbedEmbeddingAdapter,
        )

        def _fake_embed_impl(self, text: str) -> list[float]:
            return [0.1] * 384

        env = {
            **os.environ,
            "RAGLAB_MODEL_CACHE": str(fake_cache),
            "RAGLAB_MANIFEST_PATH": str(fake_manifest),
        }
        env.pop("GEMINI_API_KEY", None)
        env.pop("GOOGLE_API_KEY", None)

        provisioner = str(_REPO_ROOT / "scripts" / "provision_embedding_model.py")
        with patch.object(FastEmbedEmbeddingAdapter, "_embed", _fake_embed_impl):
            result = subprocess.run(
                [sys.executable, provisioner, "--attest-existing"],
                capture_output=True,
                text=True,
                env=env,
            )
            assert result.returncode == 0
            assert "ATTEST_OK" in result.stdout
            assert fake_manifest.exists()
            data = json.loads(fake_manifest.read_text(encoding="utf-8"))
            assert data["model_revision_status"] == "ATTESTED_OFFLINE"
            assert "cache_tree_sha256" in data
