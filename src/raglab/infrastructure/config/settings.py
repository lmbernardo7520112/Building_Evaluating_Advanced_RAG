"""Typed, validated configuration — no secrets in defaults or errors.

Generator and judge are logically separate. Defaults are pedagogical,
not claimed as optimal. No coupling to Gemini/TruLens concrete classes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from raglab.domain.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for an LLM model (generator or judge)."""

    provider: str
    model_id: str
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not self.provider or not self.provider.strip():
            raise ConfigurationError("provider", "must be non-empty")
        if not self.model_id or not self.model_id.strip():
            raise ConfigurationError("model_id", "must be non-empty")
        if self.temperature < 0.0:
            raise ConfigurationError("temperature", "must be non-negative")
        if self.max_tokens < 1:
            raise ConfigurationError("max_tokens", "must be positive")
        if self.timeout_seconds <= 0.0:
            raise ConfigurationError("timeout_seconds", "must be positive")


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Configuration for the embedding model."""

    provider: str
    model_id: str
    dimension: int = 768

    def __post_init__(self) -> None:
        if not self.provider or not self.provider.strip():
            raise ConfigurationError("embedding.provider", "must be non-empty")
        if not self.model_id or not self.model_id.strip():
            raise ConfigurationError("embedding.model_id", "must be non-empty")
        if self.dimension < 1:
            raise ConfigurationError("embedding.dimension", "must be positive")


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Configuration for retrieval parameters."""

    top_k: int = 5
    similarity_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ConfigurationError("top_k", "must be positive")
        import math

        if not math.isfinite(self.similarity_threshold):
            raise ConfigurationError("similarity_threshold", "must be finite")


@dataclass(frozen=True, slots=True)
class RerankingConfig:
    """Configuration for reranking. Used in Slices 2+."""

    enabled: bool = False
    model_id: str = ""
    top_n: int = 3

    def __post_init__(self) -> None:
        if self.enabled and not self.model_id:
            raise ConfigurationError(
                "reranking.model_id", "required when reranking is enabled"
            )
        if self.top_n < 1:
            raise ConfigurationError("reranking.top_n", "must be positive")


@dataclass(frozen=True, slots=True)
class QuotaConfig:
    """Quota and rate limiting configuration."""

    requests_per_minute: int = 15
    cooldown_seconds: float = 90.0
    max_retries: int = 3
    circuit_breaker_threshold: int = 5

    def __post_init__(self) -> None:
        if self.requests_per_minute < 1:
            raise ConfigurationError("requests_per_minute", "must be positive")
        if self.cooldown_seconds < 0:
            raise ConfigurationError("cooldown_seconds", "must be non-negative")
        if self.max_retries < 0:
            raise ConfigurationError("max_retries", "must be non-negative")
        if self.circuit_breaker_threshold < 1:
            raise ConfigurationError(
                "circuit_breaker_threshold", "must be positive"
            )


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Configuration for checkpointing."""

    enabled: bool = True
    directory: str = "checkpoints"
    schema_version: str = "3.0"

    def __post_init__(self) -> None:
        if self.enabled and not self.directory:
            raise ConfigurationError(
                "checkpoint.directory", "required when checkpoints are enabled"
            )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Top-level experiment configuration.

    Generator and judge are kept logically separate to prevent
    contamination between answer generation and evaluation.

    No secrets in default values. Error messages never reveal
    actual config values that could contain credentials.
    """

    generator: ModelConfig
    judge: ModelConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = field(default_factory=RerankingConfig)
    quota: QuotaConfig = field(default_factory=QuotaConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    seed: int = 42
    corpus_mode: str = "smoke"

    def __post_init__(self) -> None:
        valid_modes = {"smoke", "controlled", "research", "stress"}
        if self.corpus_mode not in valid_modes:
            raise ConfigurationError(
                "corpus_mode", f"must be one of {valid_modes}"
            )
        if self.generator.model_id == self.judge.model_id:
            # Warning level, not error — but document the risk
            pass  # Separation is recommended, not enforced at config level

    def fingerprint(self) -> str:
        """Compute a deterministic fingerprint of this configuration.

        Used to tie checkpoints to specific configurations.
        Does NOT include secrets — only structural parameters.
        """
        config_dict = {
            "generator_provider": self.generator.provider,
            "generator_model": self.generator.model_id,
            "generator_temp": self.generator.temperature,
            "generator_max_tokens": self.generator.max_tokens,
            "judge_provider": self.judge.provider,
            "judge_model": self.judge.model_id,
            "judge_temp": self.judge.temperature,
            "embedding_provider": self.embedding.provider,
            "embedding_model": self.embedding.model_id,
            "embedding_dim": self.embedding.dimension,
            "retrieval_top_k": self.retrieval.top_k,
            "reranking_enabled": self.reranking.enabled,
            "seed": self.seed,
            "corpus_mode": self.corpus_mode,
        }
        serialized = json.dumps(config_dict, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
