"""Tests for configuration validation."""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from raglab.domain.errors import ConfigurationError
from raglab.infrastructure.config.settings import (
    CheckpointConfig,
    EmbeddingConfig,
    ExperimentConfig,
    ModelConfig,
    QuotaConfig,
    RerankingConfig,
    RetrievalConfig,
)


def _make_model(provider: str = "gemini", model_id: str = "gemini-1.5-flash") -> ModelConfig:
    return ModelConfig(provider=provider, model_id=model_id)


def _make_embedding() -> EmbeddingConfig:
    return EmbeddingConfig(provider="gemini", model_id="text-embedding-004")


class TestModelConfig(unittest.TestCase):
    def test_valid(self) -> None:
        mc = _make_model()
        self.assertEqual(mc.provider, "gemini")

    def test_empty_provider_raises(self) -> None:
        with self.assertRaises(ConfigurationError):
            ModelConfig(provider="", model_id="test")

    def test_empty_model_id_raises(self) -> None:
        with self.assertRaises(ConfigurationError):
            ModelConfig(provider="gemini", model_id="")

    def test_negative_temperature_raises(self) -> None:
        with self.assertRaises(ConfigurationError):
            ModelConfig(provider="gemini", model_id="test", temperature=-0.1)

    def test_zero_max_tokens_raises(self) -> None:
        with self.assertRaises(ConfigurationError):
            ModelConfig(provider="gemini", model_id="test", max_tokens=0)

    def test_zero_timeout_raises(self) -> None:
        with self.assertRaises(ConfigurationError):
            ModelConfig(provider="gemini", model_id="test", timeout_seconds=0.0)


class TestEmbeddingConfig(unittest.TestCase):
    def test_valid(self) -> None:
        ec = _make_embedding()
        self.assertEqual(ec.dimension, 768)

    def test_zero_dimension_raises(self) -> None:
        with self.assertRaises(ConfigurationError):
            EmbeddingConfig(provider="gemini", model_id="test", dimension=0)


class TestRetrievalConfig(unittest.TestCase):
    def test_valid_defaults(self) -> None:
        rc = RetrievalConfig()
        self.assertEqual(rc.top_k, 5)

    def test_zero_top_k_raises(self) -> None:
        with self.assertRaises(ConfigurationError):
            RetrievalConfig(top_k=0)

    def test_inf_threshold_raises(self) -> None:
        with self.assertRaises(ConfigurationError):
            RetrievalConfig(similarity_threshold=float("inf"))


class TestRerankingConfig(unittest.TestCase):
    def test_enabled_without_model_raises(self) -> None:
        with self.assertRaises(ConfigurationError):
            RerankingConfig(enabled=True, model_id="")


class TestQuotaConfig(unittest.TestCase):
    def test_valid_defaults(self) -> None:
        qc = QuotaConfig()
        self.assertEqual(qc.cooldown_seconds, 90.0)

    def test_zero_rpm_raises(self) -> None:
        with self.assertRaises(ConfigurationError):
            QuotaConfig(requests_per_minute=0)


class TestExperimentConfig(unittest.TestCase):
    def test_valid(self) -> None:
        cfg = ExperimentConfig(
            generator=_make_model(),
            judge=_make_model(model_id="gemini-1.5-pro"),
            embedding=_make_embedding(),
        )
        self.assertEqual(cfg.corpus_mode, "smoke")

    def test_invalid_corpus_mode_raises(self) -> None:
        with self.assertRaises(ConfigurationError):
            ExperimentConfig(
                generator=_make_model(),
                judge=_make_model(),
                embedding=_make_embedding(),
                corpus_mode="invalid",
            )

    def test_fingerprint_deterministic(self) -> None:
        cfg1 = ExperimentConfig(
            generator=_make_model(),
            judge=_make_model(model_id="gemini-1.5-pro"),
            embedding=_make_embedding(),
        )
        cfg2 = ExperimentConfig(
            generator=_make_model(),
            judge=_make_model(model_id="gemini-1.5-pro"),
            embedding=_make_embedding(),
        )
        self.assertEqual(cfg1.fingerprint(), cfg2.fingerprint())
        self.assertEqual(len(cfg1.fingerprint()), 64)

    def test_different_configs_different_fingerprints(self) -> None:
        cfg1 = ExperimentConfig(
            generator=_make_model(),
            judge=_make_model(model_id="gemini-1.5-pro"),
            embedding=_make_embedding(),
            seed=42,
        )
        cfg2 = ExperimentConfig(
            generator=_make_model(),
            judge=_make_model(model_id="gemini-1.5-pro"),
            embedding=_make_embedding(),
            seed=123,
        )
        self.assertNotEqual(cfg1.fingerprint(), cfg2.fingerprint())


if __name__ == "__main__":
    unittest.main()
