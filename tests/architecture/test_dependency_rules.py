"""Architectural dependency tests — enforce Clean Architecture rules.

These tests verify that the domain layer has NO imports from
infrastructure, frameworks, or external providers.
"""

import ast
import os
import sys
import unittest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "src", "raglab")

# Forbidden imports in the domain layer
FORBIDDEN_IN_DOMAIN = {
    "llamaindex", "llama_index",
    "trulens", "trulens_eval",
    "google.generativeai", "google.cloud", "google.colab",
    "openai",
    "langchain",
    "chromadb", "qdrant",
    "raglab.infrastructure",
    "raglab.application.ports",  # domain must not depend on application
}

# The domain directory
DOMAIN_DIR = os.path.join(REPO_ROOT, "domain")


def _collect_imports(filepath: str) -> set[str]:
    """Parse a Python file and collect all import module names."""
    with open(filepath, encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            return set()

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


class TestDomainDependencyRules(unittest.TestCase):
    """Verify domain layer purity — no infrastructure or framework imports."""

    def test_no_forbidden_imports_in_domain(self) -> None:
        violations: list[str] = []

        if not os.path.isdir(DOMAIN_DIR):
            self.fail(f"Domain directory not found: {DOMAIN_DIR}")

        for root, _dirs, files in os.walk(DOMAIN_DIR):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                filepath = os.path.join(root, fname)
                imports = _collect_imports(filepath)

                for imp in imports:
                    for forbidden in FORBIDDEN_IN_DOMAIN:
                        if imp == forbidden or imp.startswith(forbidden + "."):
                            violations.append(
                                f"{os.path.relpath(filepath, REPO_ROOT)}: "
                                f"imports '{imp}' (forbidden: {forbidden})"
                            )

        if violations:
            self.fail(
                "Domain layer has forbidden imports:\n"
                + "\n".join(f"  - {v}" for v in violations)
            )

    def test_domain_only_imports_domain_or_stdlib(self) -> None:
        """Domain may import from raglab.domain.* or stdlib only."""
        violations: list[str] = []

        if not os.path.isdir(DOMAIN_DIR):
            self.fail(f"Domain directory not found: {DOMAIN_DIR}")

        allowed_prefixes = {"raglab.domain"}
        # Standard library modules are allowed implicitly
        has_stdlib = hasattr(sys, "stdlib_module_names")
        stdlib_modules = set(sys.stdlib_module_names) if has_stdlib else set()

        for root, _dirs, files in os.walk(DOMAIN_DIR):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                filepath = os.path.join(root, fname)
                imports = _collect_imports(filepath)

                for imp in imports:
                    # Allow __future__
                    if imp == "__future__":
                        continue
                    # Allow raglab.domain.*
                    if any(imp.startswith(p) for p in allowed_prefixes):
                        continue
                    # Allow standard library
                    top_level = imp.split(".")[0]
                    if top_level in stdlib_modules:
                        continue
                    # Allow common stdlib not in sys.stdlib_module_names on all versions
                    if top_level in {
                        "math", "re", "hashlib", "json", "os", "sys",
                        "logging", "dataclasses", "typing", "enum",
                        "collections", "abc", "functools", "pathlib",
                        "datetime", "uuid", "copy",
                    }:
                        continue

                    violations.append(
                        f"{os.path.relpath(filepath, REPO_ROOT)}: "
                        f"imports '{imp}' (not domain or stdlib)"
                    )

        if violations:
            self.fail(
                "Domain imports non-domain, non-stdlib modules:\n"
                + "\n".join(f"  - {v}" for v in violations)
            )

    def test_ports_do_not_import_infrastructure(self) -> None:
        """Ports (application layer) must not import infrastructure."""
        ports_dir = os.path.join(REPO_ROOT, "application", "ports")
        violations: list[str] = []

        if not os.path.isdir(ports_dir):
            self.fail(f"Ports directory not found: {ports_dir}")

        for root, _dirs, files in os.walk(ports_dir):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                filepath = os.path.join(root, fname)
                imports = _collect_imports(filepath)
                for imp in imports:
                    if "infrastructure" in imp:
                        violations.append(
                            f"{os.path.relpath(filepath, REPO_ROOT)}: "
                            f"imports '{imp}'"
                        )

        if violations:
            self.fail(
                "Ports import infrastructure:\n"
                + "\n".join(f"  - {v}" for v in violations)
            )


if __name__ == "__main__":
    unittest.main()
