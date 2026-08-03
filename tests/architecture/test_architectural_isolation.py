"""AST Architectural Isolation Tests.

Verifies that Ground Truth v2 and evaluation contracts NEVER leak into:
1. domain/
2. application/ (including RAG inference pipeline)
3. infrastructure/ (retriever, generator, embeddings)
"""

from __future__ import annotations

import ast
from pathlib import Path


def _get_imports_from_file(file_path: Path) -> list[str]:
    """Parse python file with AST and extract all imported module names."""
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    except Exception:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


class TestArchitecturalIsolation:
    """Enforce strict directional isolation: runtime inference NEVER imports evaluation."""

    def test_domain_never_imports_evaluation(self):
        root = Path("src/raglab/domain")
        violating_files = []

        for py_file in root.rglob("*.py"):
            imports = _get_imports_from_file(py_file)
            for imp in imports:
                if "raglab.evaluation" in imp or imp.startswith("evaluation"):
                    violating_files.append(f"{py_file}: imports {imp}")

        assert not violating_files, f"Domain isolation violation found: {violating_files}"

    def test_application_never_imports_evaluation(self):
        root = Path("src/raglab/application")
        violating_files = []

        for py_file in root.rglob("*.py"):
            imports = _get_imports_from_file(py_file)
            for imp in imports:
                if "raglab.evaluation" in imp or imp.startswith("evaluation"):
                    violating_files.append(f"{py_file}: imports {imp}")

        assert not violating_files, f"Application isolation violation found: {violating_files}"

    def test_infrastructure_never_imports_evaluation(self):
        root = Path("src/raglab/infrastructure")
        violating_files = []

        for py_file in root.rglob("*.py"):
            imports = _get_imports_from_file(py_file)
            for imp in imports:
                if "raglab.evaluation" in imp or imp.startswith("evaluation"):
                    violating_files.append(f"{py_file}: imports {imp}")

        assert not violating_files, f"Infrastructure isolation violation found: {violating_files}"
