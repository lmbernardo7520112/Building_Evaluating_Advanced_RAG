"""Tests for the raglab CLI.

Covers: smoke test, doctor, version, deterministic execution, offline.
"""

from __future__ import annotations

import subprocess
import sys
import unittest


class TestCLIVersion(unittest.TestCase):
    """Test --version flag."""

    def test_version_output(self) -> None:
        """raglab --version prints version string."""
        result = subprocess.run(
            [sys.executable, "-m", "raglab.interfaces.cli.main", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("7.0.0a1", result.stdout)


class TestCLISmoke(unittest.TestCase):
    """Test raglab smoke command."""

    def test_smoke_exits_zero(self) -> None:
        """Smoke test passes and exits 0."""
        result = subprocess.run(
            [sys.executable, "-m", "raglab.interfaces.cli.main", "smoke"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASSED", result.stdout)

    def test_smoke_produces_json(self) -> None:
        """Smoke test output contains valid JSON summary."""
        result = subprocess.run(
            [sys.executable, "-m", "raglab.interfaces.cli.main", "smoke"],
            capture_output=True, text=True, timeout=30,
        )
        import json
        # Find the JSON block in output
        lines = result.stdout.split("\n")
        json_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                json_start = i
                break
        self.assertIsNotNone(json_start)
        json_text = "\n".join(lines[json_start:])
        data = json.loads(json_text)
        self.assertEqual(data["overall"], "PASSED")
        self.assertIn("metrics", data)

    def test_smoke_computes_recall(self) -> None:
        """Smoke test computes Recall@k."""
        result = subprocess.run(
            [sys.executable, "-m", "raglab.interfaces.cli.main", "smoke"],
            capture_output=True, text=True, timeout=30,
        )
        import json
        lines = result.stdout.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                data = json.loads("\n".join(lines[i:]))
                break
        self.assertIn("recall_at_k", data["metrics"])

    def test_smoke_computes_mrr(self) -> None:
        """Smoke test computes MRR."""
        result = subprocess.run(
            [sys.executable, "-m", "raglab.interfaces.cli.main", "smoke"],
            capture_output=True, text=True, timeout=30,
        )
        import json
        lines = result.stdout.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                data = json.loads("\n".join(lines[i:]))
                break
        self.assertIsNotNone(data["metrics"]["mrr"])
        self.assertGreater(data["metrics"]["mrr"], 0.0)

    def test_smoke_deterministic(self) -> None:
        """Two runs produce same MRR."""
        results = []
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, "-m", "raglab.interfaces.cli.main", "smoke"],
                capture_output=True, text=True, timeout=30,
            )
            import json
            lines = result.stdout.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("{"):
                    data = json.loads("\n".join(lines[i:]))
                    results.append(data["metrics"]["mrr"])
                    break
        self.assertEqual(results[0], results[1])


class TestCLIDoctor(unittest.TestCase):
    """Test raglab doctor command."""

    def test_doctor_exits_zero(self) -> None:
        """Doctor runs without error."""
        result = subprocess.run(
            [sys.executable, "-m", "raglab.interfaces.cli.main", "doctor"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0)

    def test_doctor_reports_python(self) -> None:
        """Doctor reports Python version."""
        result = subprocess.run(
            [sys.executable, "-m", "raglab.interfaces.cli.main", "doctor"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertIn("python_version", result.stdout)


if __name__ == "__main__":
    unittest.main()
