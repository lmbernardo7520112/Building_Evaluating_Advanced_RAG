"""Focal Unit Test Suite — ExperimentalPathPolicy (GREEN-2A).

Tests fail-closed path policy validation for durable artifact roots, rejecting
temporary directories (/tmp), repository root, benchmarks/, checkpoints/,
relative paths, path traversal, symlink escapes, and unallowed roots.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from raglab.agentic.experiments import ExperimentalPathPolicy, PathPolicyError


class TestExperimentalPathPolicyFocal(unittest.TestCase):
    """Focal test suite covering ExperimentalPathPolicy rules."""

    def setUp(self) -> None:
        self.fake_repo = Path("/srv/mock_repo").resolve()
        self.durable_external = Path("/srv/durable_storage").resolve()

    # 1. Raiz durável permitida é aceita
    def test_01_durable_allowed_root_accepted(self) -> None:
        policy = ExperimentalPathPolicy(
            repository_root=self.fake_repo,
            allowed_roots=[self.durable_external],
        )
        resolved = policy.validate_target_directory(self.durable_external)
        self.assertEqual(resolved, self.durable_external)

    # 2. /tmp é rejeitado
    def test_02_tmp_root_rejected(self) -> None:
        policy = ExperimentalPathPolicy(repository_root=self.fake_repo)
        with self.assertRaises(PathPolicyError):
            policy.validate_target_directory("/tmp")  # noqa: S108

    # 3. Descendente de /tmp é rejeitado
    def test_03_tmp_descendant_rejected(self) -> None:
        policy = ExperimentalPathPolicy(repository_root=self.fake_repo)
        with self.assertRaises(PathPolicyError):
            policy.validate_target_directory("/tmp/run_dir_123")  # noqa: S108

    # 4. Raiz do repositório é rejeitada
    def test_04_repo_root_rejected(self) -> None:
        policy = ExperimentalPathPolicy(repository_root=self.fake_repo)
        with self.assertRaises(PathPolicyError):
            policy.validate_target_directory(self.fake_repo)

    # 5. benchmarks/ é rejeitado
    def test_05_benchmarks_rejected(self) -> None:
        policy = ExperimentalPathPolicy(repository_root=self.fake_repo)
        bench_dir = self.fake_repo / "benchmarks" / "slice4"
        with self.assertRaises(PathPolicyError):
            policy.validate_target_directory(bench_dir)

    # 6. checkpoints/ é rejeitado
    def test_06_checkpoints_rejected(self) -> None:
        policy = ExperimentalPathPolicy(repository_root=self.fake_repo)
        check_dir = self.fake_repo / "checkpoints" / "run1"
        with self.assertRaises(PathPolicyError):
            policy.validate_target_directory(check_dir)

    # 7. Caminho relativo é rejeitado
    def test_07_relative_path_rejected(self) -> None:
        policy = ExperimentalPathPolicy(repository_root=self.fake_repo)
        with self.assertRaises(PathPolicyError):
            policy.validate_target_directory("relative/durable/path")

    # 8. Traversal (..) é rejeitado
    def test_08_path_traversal_rejected(self) -> None:
        policy = ExperimentalPathPolicy(repository_root=self.fake_repo)
        traversal_path = self.durable_external / ".." / "durable_storage"
        with self.assertRaises(PathPolicyError):
            policy.validate_target_directory(traversal_path)

    # 9. Symlink escape para /tmp é rejeitado
    def test_09_symlink_escape_rejected(self) -> None:
        policy = ExperimentalPathPolicy(repository_root=self.fake_repo)
        with tempfile.TemporaryDirectory() as tmp_dir:
            symlink_path = Path(tmp_dir) / "link_to_tmp"
            try:
                os.symlink("/tmp", symlink_path)  # noqa: S108
            except OSError:
                return
            with self.assertRaises(PathPolicyError):
                policy.validate_target_directory(symlink_path)

    # 10. Raiz fora da allowlist é rejeitada
    def test_10_unallowed_root_rejected(self) -> None:
        allowed = Path("/srv/allowed_dir").resolve()
        unallowed = Path("/srv/unallowed_dir").resolve()

        policy = ExperimentalPathPolicy(
            repository_root=self.fake_repo,
            allowed_roots=[allowed],
        )
        with self.assertRaises(PathPolicyError):
            policy.validate_target_directory(unallowed)

    # 11. /tmp na allowlist é rejeitado no construtor
    def test_11_tmp_in_allowed_roots_rejected(self) -> None:
        with self.assertRaises(PathPolicyError):
            ExperimentalPathPolicy(
                repository_root=self.fake_repo,
                allowed_roots=["/tmp"],  # noqa: S108
            )

    # 12. slice_id ou run_id inseguros são rejeitados
    def test_12_unsafe_identifiers_rejected(self) -> None:
        policy = ExperimentalPathPolicy(
            repository_root=self.fake_repo,
            allowed_roots=[self.durable_external],
        )
        with self.assertRaises(PathPolicyError):
            policy.derive_run_directory(self.durable_external, "../slice5", "run1")

        with self.assertRaises(PathPolicyError):
            policy.derive_run_directory(self.durable_external, "slice5", "run1/sub")

    # 13. Derivação determinística funciona para entradas válidas
    def test_13_deterministic_derivation_accepted(self) -> None:
        policy = ExperimentalPathPolicy(
            repository_root=self.fake_repo,
            allowed_roots=[self.durable_external],
        )
        derived = policy.derive_run_directory(
            self.durable_external, "slice_5b", "run_001"
        )
        expected = (self.durable_external / "slice_5b" / "run_001").resolve()
        self.assertEqual(derived, expected)

    # 14. Ausência de efeitos colaterais (validação não cria diretórios no disco)
    def test_14_no_side_effects_on_validation(self) -> None:
        policy = ExperimentalPathPolicy(
            repository_root=self.fake_repo,
            allowed_roots=[self.durable_external],
        )
        target = self.durable_external / "non_existent_folder"
        self.assertFalse(target.exists())

        resolved = policy.validate_target_directory(target)
        self.assertEqual(resolved, target.resolve())
        self.assertFalse(target.exists())

    # 15. must_be_empty rejeita diretório não-vazio existente
    def test_15_non_empty_directory_rejected(self) -> None:
        existing_non_empty = Path(__file__).parent.resolve()
        policy = ExperimentalPathPolicy(
            repository_root=self.fake_repo,
            allowed_roots=[existing_non_empty],
        )
        with self.assertRaises(PathPolicyError):
            policy.validate_target_directory(existing_non_empty, must_be_empty=True)


if __name__ == "__main__":
    unittest.main()
