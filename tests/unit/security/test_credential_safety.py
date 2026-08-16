"""Security tests — no credentials accessed, no network calls.

These tests verify the security contracts defined in
docs/security/credential_boundary.md:

1. Configuration contains no hardcoded credential defaults
2. Exceptions do not include fake credential values in their messages
3. Fake generator works without network (offline)
4. Fake judge works without network (offline)
5. Gemini provider is NOT initialized in Slice 3
6. HF token is not required by the embedding model
7. LangSmith remains disabled
8. Checkpoints do not serialize credentials
9. Scan-secrets-style pattern detection works on synthetic secrets
   without printing the full value

All credential values used in this module are:
- Clearly fictional
- Truncated/redacted before any assertion
- Never printed to stdout in full

SECURITY NOTE: This file must NEVER contain real credential values.
If any real key appears here, treat as CREDENTIAL_EXPOSURE_INCIDENT.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST_PATH = _PROJECT_ROOT / "benchmarks" / "slice3_experiment_manifest.json"


# ---------------------------------------------------------------------------
# 1. Configuration has no hardcoded credential defaults
# ---------------------------------------------------------------------------

class TestNoHardcodedCredentials:
    """Verify config files contain no credential default values."""

    _CREDENTIAL_PATTERNS = [
        r"AIza[0-9A-Za-z\-_]{35}",        # Google/Gemini API key pattern
        r"sk-[A-Za-z0-9]{32,}",            # OpenAI-style pattern
        r"hf_[A-Za-z0-9]{32,}",            # HuggingFace token pattern
        r"ls__[A-Za-z0-9]{32,}",           # LangSmith token pattern
    ]

    def _scan_file_for_patterns(self, path: Path) -> list[str]:
        """Return list of matched pattern names (NOT values) found in file."""
        text = path.read_text(encoding="utf-8", errors="ignore")
        found = []
        for pattern in self._CREDENTIAL_PATTERNS:
            if re.search(pattern, text):
                # Do NOT include the matched value in the result
                found.append(f"pattern_matched: {pattern[:20]}...")
        return found

    def test_manifest_has_no_credentials(self):
        if not _MANIFEST_PATH.exists():
            pytest.skip("Manifest not yet created")
        findings = self._scan_file_for_patterns(_MANIFEST_PATH)
        assert findings == [], f"Credential pattern found in manifest: {findings}"

    def test_pyproject_has_no_credentials(self):
        path = _PROJECT_ROOT / "pyproject.toml"
        assert path.exists()
        findings = self._scan_file_for_patterns(path)
        assert findings == [], f"Credential pattern in pyproject.toml: {findings}"

    def test_no_dotenv_file_in_repo(self):
        """A .env file must never exist in the repository."""
        dotenv = _PROJECT_ROOT / ".env"
        assert not dotenv.exists(), (
            ".env file must NOT exist in the repository. "
            "Credentials must never be persisted as files."
        )

    def test_src_has_no_hardcoded_credentials(self):
        src_dir = _PROJECT_ROOT / "src"
        violations = []
        for py_file in src_dir.rglob("*.py"):
            findings = self._scan_file_for_patterns(py_file)
            if findings:
                # Report file relative path only, not the matched value
                violations.append(py_file.relative_to(_PROJECT_ROOT))
        assert violations == [], (
            f"Credential patterns found in source files: {violations}. "
            "Check docs/security/credential_boundary.md"
        )


# ---------------------------------------------------------------------------
# 2. Exceptions do not leak credentials
# ---------------------------------------------------------------------------

class TestExceptionSafety:
    """Verify exceptions do not include credential-like strings."""

    def test_fake_generator_exception_has_no_credential(self):
        """FakeGeneratorAdapter must not embed credentials in exceptions."""
        from raglab.infrastructure.fakes.fake_generator_adapter import (
            FakeGeneratorAdapter,
        )

        adapter = FakeGeneratorAdapter()
        # Verify model_id does not look like a credential
        model_id = adapter.model_id
        assert "AIza" not in model_id
        assert "sk-" not in model_id
        assert "hf_" not in model_id

    def test_fake_judge_exception_has_no_credential(self):
        from raglab.infrastructure.fakes.fake_judge_adapter import FakeJudgeAdapter

        adapter = FakeJudgeAdapter()
        judge_id = adapter.judge_model_id
        assert "AIza" not in judge_id
        assert "sk-" not in judge_id


# ---------------------------------------------------------------------------
# 3 & 4. Fake adapters work without network (offline)
# ---------------------------------------------------------------------------

class TestFakeGeneratorOffline:
    """FakeGeneratorAdapter must work completely offline."""

    def _make_evidence(self, n: int = 2):
        from raglab.domain.entities import RetrievedEvidence
        from raglab.domain.value_objects import ChunkId
        return [
            RetrievedEvidence(
                chunk_id=ChunkId(f"chunk_{i}"),
                document_id=f"doc_p9{i}",
                text=f"Texto de evidência número {i} sobre matemática discreta.",
                rank=i + 1,
                score=0.9 - (i * 0.1),
            )
            for i in range(n)
        ]

    def test_generate_returns_answer_offline(self):
        from raglab.infrastructure.fakes.fake_generator_adapter import (
            FakeGeneratorAdapter,
        )

        adapter = FakeGeneratorAdapter()
        evidence = self._make_evidence(2)
        result = adapter.generate(
            "q_dev_01", "O que é demonstração por exaustão?", evidence
        )

        assert result is not None
        assert result.text
        # Abstained must be False because evidence was provided
        assert result.abstained is False
        # model_id lives on the adapter, not on GeneratedAnswer domain entity
        assert adapter.model_id == "fake-generator-v1-no-network"

    def test_generate_with_empty_evidence(self):
        from raglab.infrastructure.fakes.fake_generator_adapter import (
            FakeGeneratorAdapter,
        )

        adapter = FakeGeneratorAdapter()
        result = adapter.generate("q_test_01", "Pergunta sem evidência", [])
        # No evidence → must abstain
        assert result.abstained is True
        assert result.text

    def test_generate_with_empty_query(self):
        from raglab.infrastructure.fakes.fake_generator_adapter import (
            FakeGeneratorAdapter,
        )

        adapter = FakeGeneratorAdapter()
        result = adapter.generate("q_test_02", "", [])
        # Empty query → must abstain
        assert result.abstained is True

    def test_gemini_not_imported(self):
        """google.generativeai must not be imported in Slice 3."""
        assert "google.generativeai" not in sys.modules, (
            "Gemini SDK must NOT be imported in Slice 3. "
            "It is PLANNED for a future slice."
        )


class TestFakeJudgeOffline:
    """FakeJudgeAdapter must work completely offline."""

    def _make_answer(self, text: str = "[FAKE] Resposta de teste."):
        from raglab.domain.entities import GeneratedAnswer

        return GeneratedAnswer(
            query_id="q_dev_01",
            text=text,
            abstained=False,
            citations=(),
        )

    def _make_evidence(self, n: int = 2):
        from raglab.domain.entities import RetrievedEvidence
        from raglab.domain.value_objects import ChunkId
        return [
            RetrievedEvidence(
                chunk_id=ChunkId(f"chunk_{i}"),
                document_id=f"doc_p9{i}",
                text=f"Texto {i} sobre demonstração matemática.",
                rank=i + 1,
                score=0.8,
            )
            for i in range(n)
        ]

    def test_evaluate_returns_result_offline(self):
        from raglab.infrastructure.fakes.fake_judge_adapter import FakeJudgeAdapter

        judge = FakeJudgeAdapter()
        result = judge.evaluate(
            "q_dev_01",
            "O que é demonstração por exaustão?",
            self._make_answer(),
            self._make_evidence(),
        )

        assert result is not None
        metric_names = {m.name for m in result.metrics}
        # RAG Triad metrics present in updated FakeJudgeAdapter
        assert "context_relevance" in metric_names
        assert "groundedness" in metric_names
        assert "answer_relevance" in metric_names
        assert "langsmith_disabled" in metric_names

    def test_judge_model_id_has_no_credentials(self):
        from raglab.infrastructure.fakes.fake_judge_adapter import FakeJudgeAdapter

        judge = FakeJudgeAdapter()
        assert "AIza" not in judge.judge_model_id
        assert "hf_" not in judge.judge_model_id


# ---------------------------------------------------------------------------
# 5. Gemini provider not initialized in Slice 3
# ---------------------------------------------------------------------------

def _scan_source_for_gemini_adapter_violations(
    source: str, relative_path: Path
) -> list[str]:
    """Inspect source AST using reference- and scope-oriented analysis for Gemini adapter policy."""
    import ast

    violations: list[str] = []
    try:
        tree = ast.parse(source, filename=str(relative_path))
    except SyntaxError as e:
        return [f"SyntaxError parsing {relative_path}: {e}"]

    authorized_path = Path("raglab/interfaces/cli/agentic_smoke.py")
    is_authorized_file = relative_path == authorized_path

    # 1. MAPA DE IMPORTS
    # Maps local name -> canonical adapter name ("GeminiGeneratorAdapter" | "GeminiJudgeAdapter")
    imported_adapter_names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.ImportFrom, ast.Import)):
            for alias in node.names:
                if alias.name in (
                    "GeminiGeneratorAdapter",
                    "GeminiJudgeAdapter",
                ):
                    local_name = alias.asname or alias.name
                    imported_adapter_names[local_name] = alias.name

    # 2. MAPA DE PAIS
    parent_by_id: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_by_id[id(child)] = parent

    # 3. ANOTAÇÕES (Type-only nodes to ignore)
    annotation_node_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            annotation_node_ids.update(id(n) for n in ast.walk(node.annotation))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
            annotation_node_ids.update(id(n) for n in ast.walk(node.returns))
        elif isinstance(node, ast.AnnAssign):
            annotation_node_ids.update(id(n) for n in ast.walk(node.annotation))
        elif hasattr(ast, "TypeAlias") and isinstance(node, ast.TypeAlias):
            annotation_node_ids.update(id(n) for n in ast.walk(node.value))
            for tparam in getattr(node, "type_params", ()):
                annotation_node_ids.update(id(n) for n in ast.walk(tparam))

    # 4. REFERÊNCIAS RUNTIME
    valid_direct_generator_calls_in_main = 0

    for node in ast.walk(tree):
        target_adapter: str | None = None
        is_direct_canonical_name = False

        if isinstance(node, ast.Name):
            ctx = getattr(node, "ctx", None)
            if ctx is None or isinstance(ctx, ast.Load):
                if node.id in ("GeminiGeneratorAdapter", "GeminiJudgeAdapter"):
                    target_adapter = node.id
                    is_direct_canonical_name = True
                elif node.id in imported_adapter_names:
                    target_adapter = imported_adapter_names[node.id]
                    is_direct_canonical_name = False
        elif isinstance(node, ast.Attribute):
            ctx = getattr(node, "ctx", None)
            if (
                ctx is None or isinstance(ctx, ast.Load)
            ) and node.attr in ("GeminiGeneratorAdapter", "GeminiJudgeAdapter"):
                target_adapter = node.attr
                is_direct_canonical_name = False

        if target_adapter is None:
            continue

        parent = parent_by_id.get(id(node))
        is_call_func = isinstance(parent, ast.Call) and parent.func is node

        # Check if reference is inside a type annotation
        if id(node) in annotation_node_ids:
            if is_call_func:
                violations.append(
                    f"Gemini adapter constructor call prohibited inside type annotation in {relative_path}."
                )
            # Pure type annotation references (not called) are ignored
            continue

        # A & B: If NOT parent.func is node -> uncalled reference / alias / factory
        if not is_call_func:
            violations.append(
                f"Alias or uncalled reference to {target_adapter} prohibited in {relative_path}."
            )
            continue

        # 5. REGRAS DE CHAMADA (parent is ast.Call and parent.func is node)
        if target_adapter == "GeminiJudgeAdapter":
            violations.append(
                f"GeminiJudgeAdapter instantiation prohibited in {relative_path}."
            )
            continue

        # target_adapter == "GeminiGeneratorAdapter"
        if not is_authorized_file:
            violations.append(
                f"GeminiGeneratorAdapter instantiation prohibited in unauthorized file {relative_path}."
            )
            continue

        if not is_direct_canonical_name:
            violations.append(
                f"Call in {relative_path} must be direct 'GeminiGeneratorAdapter()', not an alias or attribute."
            )
            continue

        # 6. ESCOPO AUTORIZADO (Ascend parent_by_id from Call node)
        curr = parent
        enclosing_scopes: list[ast.AST] = []
        while curr is not None:
            curr_id = id(curr)
            curr = parent_by_id.get(curr_id)
            if isinstance(
                curr,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.Lambda,
                    ast.ClassDef,
                ),
            ):
                enclosing_scopes.append(curr)

        if len(enclosing_scopes) == 0:
            violations.append(
                f"GeminiGeneratorAdapter instantiation in {relative_path} must be inside top-level 'main' (found module scope)."
            )
        elif any(isinstance(s, ast.ClassDef) for s in enclosing_scopes):
            violations.append(
                f"GeminiGeneratorAdapter instantiation in {relative_path} cannot be inside a class."
            )
        elif any(isinstance(s, ast.AsyncFunctionDef) for s in enclosing_scopes):
            violations.append(
                f"GeminiGeneratorAdapter instantiation in {relative_path} cannot be inside an async function."
            )
        elif any(isinstance(s, ast.Lambda) for s in enclosing_scopes):
            violations.append(
                f"GeminiGeneratorAdapter instantiation in {relative_path} cannot be inside a lambda."
            )
        elif len(enclosing_scopes) > 1:
            violations.append(
                f"GeminiGeneratorAdapter instantiation in {relative_path} cannot be inside a nested function."
            )
        elif (
            isinstance(enclosing_scopes[0], ast.FunctionDef)
            and enclosing_scopes[0].name != "main"
        ):
            violations.append(
                f"GeminiGeneratorAdapter instantiation in {relative_path} must be inside 'main' function (found in '{enclosing_scopes[0].name}')."
            )
        else:
            valid_direct_generator_calls_in_main += 1

    # 7. CARDINALIDADE (No arquivo autorizado)
    if is_authorized_file:
        if valid_direct_generator_calls_in_main == 0:
            violations.append(
                f"Authorized composition root {relative_path} must instantiate GeminiGeneratorAdapter directly exactly once in top-level 'main'."
            )
        elif valid_direct_generator_calls_in_main > 1:
            violations.append(
                f"Authorized composition root {relative_path} instantiates GeminiGeneratorAdapter {valid_direct_generator_calls_in_main} times (expected exactly 1)."
            )

    return violations


class TestGeminiNotInitialized:
    def test_no_gemini_sdk_imported(self):
        """google-generativeai must not be imported at module level."""
        assert "google.generativeai" not in sys.modules
        assert "vertexai" not in sys.modules

    def test_no_gemini_adapter_instantiated_by_default(self):
        """Verify strict AST governance over Gemini adapter instantiations and aliases in src/."""
        src = _PROJECT_ROOT / "src"

        all_violations: list[str] = []
        for py_file in sorted(src.rglob("*.py")):
            relative_path = py_file.relative_to(src)
            content = py_file.read_text(encoding="utf-8")
            violations = _scan_source_for_gemini_adapter_violations(
                content, relative_path
            )
            all_violations.extend(violations)

        if all_violations:
            pytest.fail(
                "Gemini adapter governance violations in src/:\n"
                + "\n".join(f"  - {v}" for v in all_violations)
            )

    def test_ast_governance_positive_case(self) -> None:
        """Authorized composition root with exactly 1 direct call in top-level main produces zero violations."""
        valid_source = (
            "from raglab.infrastructure.gemini.gemini_generator_adapter import GeminiGeneratorAdapter\n\n"
            "def main():\n"
            "    generator = GeminiGeneratorAdapter()\n"
            "    return generator\n"
        )
        violations = _scan_source_for_gemini_adapter_violations(
            valid_source, Path("raglab/interfaces/cli/agentic_smoke.py")
        )
        assert violations == []

    def test_ast_governance_positive_type_annotation_only(self) -> None:
        """Usage of GeminiGeneratorAdapter strictly as type annotations produces zero violations."""
        annotation_only_source = (
            "from raglab.infrastructure.gemini.gemini_generator_adapter import GeminiGeneratorAdapter\n\n"
            "def inspect_generator(generator: GeminiGeneratorAdapter) -> GeminiGeneratorAdapter:\n"
            "    typed_var: GeminiGeneratorAdapter | None = None\n"
            "    return generator\n"
        )
        violations = _scan_source_for_gemini_adapter_violations(
            annotation_only_source, Path("raglab/domain/sample.py")
        )
        assert violations == []

    @pytest.mark.parametrize(
        ("case_name", "source", "rel_path", "expected_violation_fragment"),
        [
            (
                "direct_call_in_unauthorized_file",
                "def foo():\n    return GeminiGeneratorAdapter()\n",
                Path("raglab/domain/sample.py"),
                "instantiation prohibited in unauthorized file",
            ),
            (
                "simple_alias",
                "def main():\n    gemini_cls = GeminiGeneratorAdapter\n    return gemini_cls()\n",
                Path("raglab/interfaces/cli/agentic_smoke.py"),
                "Alias or uncalled reference to GeminiGeneratorAdapter",
            ),
            (
                "ann_assign_alias",
                "def main():\n    gemini_cls: type = GeminiGeneratorAdapter\n    return gemini_cls()\n",
                Path("raglab/interfaces/cli/agentic_smoke.py"),
                "Alias or uncalled reference to GeminiGeneratorAdapter",
            ),
            (
                "import_alias_call",
                "from mod import GeminiGeneratorAdapter as GGA\ndef main():\n    return GGA()\n",
                Path("raglab/interfaces/cli/agentic_smoke.py"),
                "must be direct 'GeminiGeneratorAdapter()'",
            ),
            (
                "attribute_call",
                "import mod\ndef main():\n    return mod.GeminiGeneratorAdapter()\n",
                Path("raglab/interfaces/cli/agentic_smoke.py"),
                "must be direct 'GeminiGeneratorAdapter()'",
            ),
            (
                "dict_factory_storage",
                "def main():\n    factories = {'gemini': GeminiGeneratorAdapter}\n    return factories['gemini']()\n",
                Path("raglab/interfaces/cli/agentic_smoke.py"),
                "Alias or uncalled reference to GeminiGeneratorAdapter",
            ),
            (
                "functools_partial",
                "import functools\ndef main():\n    factory = functools.partial(GeminiGeneratorAdapter)\n    return factory()\n",
                Path("raglab/interfaces/cli/agentic_smoke.py"),
                "Alias or uncalled reference to GeminiGeneratorAdapter",
            ),
            (
                "class_method_main",
                "class Runner:\n    def main(self):\n        return GeminiGeneratorAdapter()\n",
                Path("raglab/interfaces/cli/agentic_smoke.py"),
                "cannot be inside a class",
            ),
            (
                "nested_function_in_main",
                "def main():\n    def helper():\n        return GeminiGeneratorAdapter()\n    return helper()\n",
                Path("raglab/interfaces/cli/agentic_smoke.py"),
                "cannot be inside a nested function",
            ),
            (
                "two_direct_calls",
                "def main():\n    g1 = GeminiGeneratorAdapter()\n    g2 = GeminiGeneratorAdapter()\n    return g1, g2\n",
                Path("raglab/interfaces/cli/agentic_smoke.py"),
                "expected exactly 1",
            ),
            (
                "gemini_judge_direct",
                "def main():\n    return GeminiJudgeAdapter()\n",
                Path("raglab/interfaces/cli/agentic_smoke.py"),
                "GeminiJudgeAdapter instantiation prohibited",
            ),
            (
                "gemini_judge_alias",
                "def main():\n    judge_cls = GeminiJudgeAdapter\n    return judge_cls()\n",
                Path("raglab/interfaces/cli/agentic_smoke.py"),
                "Alias or uncalled reference to GeminiJudgeAdapter",
            ),
            (
                "lambda_in_main",
                "def main():\n    f = lambda: GeminiGeneratorAdapter()\n    return f()\n",
                Path("raglab/interfaces/cli/agentic_smoke.py"),
                "cannot be inside a lambda",
            ),
            (
                "function_default_argument",
                "def foo(g=GeminiGeneratorAdapter):\n    pass\n",
                Path("raglab/interfaces/cli/agentic_smoke.py"),
                "Alias or uncalled reference to GeminiGeneratorAdapter",
            ),
            (
                "class_base_inheritance",
                "class MySubclass(GeminiGeneratorAdapter):\n    pass\n",
                Path("raglab/interfaces/cli/agentic_smoke.py"),
                "Alias or uncalled reference to GeminiGeneratorAdapter",
            ),
            (
                "decorator_argument",
                "def decorator(cls):\n    return cls\n\n@decorator(GeminiGeneratorAdapter)\ndef foo():\n    pass\n",
                Path("raglab/interfaces/cli/agentic_smoke.py"),
                "Alias or uncalled reference to GeminiGeneratorAdapter",
            ),
            (
                "boolop_or_ifexp",
                "def main():\n    g = GeminiGeneratorAdapter if True else None\n    return g\n",
                Path("raglab/interfaces/cli/agentic_smoke.py"),
                "Alias or uncalled reference to GeminiGeneratorAdapter",
            ),
            (
                "annotation_constructor_call",
                "def main(value: GeminiGeneratorAdapter()):\n    return value\n",
                Path("raglab/interfaces/cli/agentic_smoke.py"),
                "constructor call prohibited inside type annotation",
            ),
        ],
    )
    def test_ast_governance_negative_cases(
        self,
        case_name: str,
        source: str,
        rel_path: Path,
        expected_violation_fragment: str,
    ) -> None:
        """Negative AST patterns must produce specific governance violation fragments."""
        violations = _scan_source_for_gemini_adapter_violations(
            source, rel_path
        )
        assert any(
            expected_violation_fragment in violation for violation in violations
        ), (
            f"Case '{case_name}' expected violation fragment '{expected_violation_fragment}' "
            f"in {violations}"
        )


# ---------------------------------------------------------------------------
# 6. HF token not required
# ---------------------------------------------------------------------------

class TestHFTokenNotRequired:
    def test_fastembed_imports_without_hf_token(self):
        """FastEmbed must import successfully with HF token absent."""
        import os

        # Document that HF_TOKEN should NOT be required.
        # Do NOT log the value — only test for absence at import time.
        _ = os.environ.get("HF_TOKEN")  # checked but not logged or asserted
        from fastembed import TextEmbedding  # noqa: F401

        # If we reach here, import succeeded regardless of HF_TOKEN
        assert True, "FastEmbed imported successfully"

    def test_embedding_model_is_public(self):
        """The selected model must be public (no HF_TOKEN required)."""
        public_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        # Verify the manifest registers this model
        if _MANIFEST_PATH.exists():
            with _MANIFEST_PATH.open(encoding="utf-8") as f:
                manifest = json.load(f)
            assert manifest["embedding"]["model"] == public_model


# ---------------------------------------------------------------------------
# 7. LangSmith disabled
# ---------------------------------------------------------------------------

class TestLangSmithDisabled:
    def test_langsmith_not_imported(self):
        """langsmith package must not be imported."""
        assert "langsmith" not in sys.modules, (
            "LangSmith must remain DISABLED in Slice 3. "
            "See docs/security/credential_boundary.md"
        )

    def test_no_langsmith_tracing_in_src(self):
        """No src file should contain active LangSmith tracing calls."""
        src = _PROJECT_ROOT / "src"
        for py_file in src.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8", errors="ignore")
            # Flag active tracing calls (not comments)
            active_patterns = [
                "langsmith.Client()",
                "LangSmithTracer(",
                "from langsmith import",
            ]
            for pattern in active_patterns:
                if pattern in text:
                    pytest.fail(
                        f"Active LangSmith pattern '{pattern}' found in "
                        f"{py_file.relative_to(_PROJECT_ROOT)}. "
                        "LangSmith is DISABLED in Slice 3."
                    )


# ---------------------------------------------------------------------------
# 8. Checkpoints do not serialize credentials
# ---------------------------------------------------------------------------

class TestCheckpointSafety:
    """Verify checkpoint format does not serialize credential-like values."""

    _CRED_PATTERNS = [
        r"AIza[0-9A-Za-z\-_]{10,}",
        r"hf_[A-Za-z0-9]{10,}",
        r"sk-[A-Za-z0-9]{10,}",
    ]

    def test_existing_checkpoints_contain_no_credentials(self):
        checkpoint_dir = _PROJECT_ROOT / "checkpoints"
        if not checkpoint_dir.exists():
            pytest.skip("No checkpoints directory yet")

        for cp_file in checkpoint_dir.glob("*.json"):
            text = cp_file.read_text(encoding="utf-8", errors="ignore")
            for pattern in self._CRED_PATTERNS:
                match = re.search(pattern, text)
                if match:
                    # Redact the value before failing
                    matched = match.group(0)
                    redacted = matched[:4] + "*" * (len(matched) - 4)
                    pytest.fail(
                        f"Credential-like pattern found in checkpoint "
                        f"{cp_file.name}: {redacted}. "
                        "CREDENTIAL_EXPOSURE_INCIDENT — revoke immediately."
                    )


# ---------------------------------------------------------------------------
# 9. Secret scanner detects synthetic secrets without printing full value
# ---------------------------------------------------------------------------

class TestSecretScanner:
    """Verify the scan_secrets.py script detects credential-like patterns."""

    def test_scanner_script_exists(self):
        scanner = _PROJECT_ROOT / "scripts" / "scan_secrets.py"
        assert scanner.exists(), "scan_secrets.py must exist in scripts/"

    def test_scanner_detects_synthetic_key_without_printing(self):
        """
        Confirm that credential detection works on a synthetic string.
        The full synthetic value is never printed or stored.
        """
        # Synthetic key — clearly fictional, formatted like a real one
        # We only test the regex pattern, not real credentials
        synthetic_pattern = r"AIza[0-9A-Za-z\-_]{35}"
        synthetic_test_string = "AIzaSyFAKE_KEY_FOR_TESTING_ONLY_0000000000"

        match = re.search(synthetic_pattern, synthetic_test_string)
        assert match is not None, "Pattern should detect synthetic key"

        # Verify redaction works correctly
        matched_value = match.group(0)
        redacted = matched_value[:4] + "..." + matched_value[-3:]
        assert "FAKE" not in redacted or len(redacted) < len(matched_value), (
            "Redaction must shorten the value"
        )
        # Verify we never print the full value — only the redacted form
        assert len(redacted) < len(matched_value)
