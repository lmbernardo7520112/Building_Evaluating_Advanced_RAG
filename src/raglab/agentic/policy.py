"""Routing policy metadata and freezing — stdlib-only."""

from __future__ import annotations

from dataclasses import dataclass

from raglab.agentic.contracts import PolicyMetadata, _canonical_json, _sha256


@dataclass(frozen=True, slots=True)
class FrozenPolicy:
    """A frozen, hashable routing policy configuration.

    The hash is computed from the policy rules and version, ensuring
    deterministic reproducibility. Policies cannot be modified after creation.
    """

    metadata: PolicyMetadata
    rules: dict[str, str]  # query_class -> strategy mapping
    fallback_strategy: str

    def __post_init__(self) -> None:
        if not self.rules:
            raise ValueError("Policy must have at least one rule")
        if not self.fallback_strategy:
            raise ValueError("fallback_strategy must be non-empty")

    @staticmethod
    def compute_policy_hash(
        policy_id: str,
        policy_version: str,
        rules: dict[str, str],
        fallback_strategy: str,
    ) -> str:
        """Compute deterministic SHA-256 for a policy configuration."""
        payload = {
            "policy_id": policy_id,
            "policy_version": policy_version,
            "rules": rules,
            "fallback_strategy": fallback_strategy,
        }
        return _sha256(_canonical_json(payload))
