"""Immutable Receipt Core & Chain Integrity — RAGLab V7 Experimental Readiness.

Defines the RunState machine, immutable RunReceipt model, state transitions,
and append-only hash chain validation with fail-closed cryptographic checks.
"""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from raglab.agentic.experiments.integrity import (
    compute_canonical_json_sha256,
    is_valid_sha256,
    validate_sha256_digest,
)


class ReceiptStoreError(Exception):
    """Raised when receipt creation, state transition, or chain validation fails."""


class RunState(Enum):
    """Valid experiment run states."""

    PREPARED = "PREPARED"
    RUN_STARTED = "RUN_STARTED"
    RUN_COMPLETED = "RUN_COMPLETED"
    AUDIT_COMPLETED = "AUDIT_COMPLETED"
    FAILED = "FAILED"

    def __str__(self) -> str:
        return self.value


ALLOWED_TRANSITIONS: dict[RunState, set[RunState]] = {
    RunState.PREPARED: {RunState.RUN_STARTED},
    RunState.RUN_STARTED: {RunState.RUN_COMPLETED, RunState.FAILED},
    RunState.RUN_COMPLETED: {RunState.AUDIT_COMPLETED},
    RunState.AUDIT_COMPLETED: set(),  # terminal
    RunState.FAILED: set(),  # terminal
}

SECRET_FIELDS_FORBIDDEN: tuple[str, ...] = (
    "api_key",
    "secret",
    "token",
    "password",
    "credential",
    "bearer",
)

REQUIRED_RECEIPT_FIELDS: tuple[str, ...] = (
    "schema_version",
    "run_id",
    "slice_id",
    "state",
    "artifact_root",
    "run_directory",
    "implementation_commit",
    "protocol_commit",
    "protocol_sha256",
    "input_hashes",
    "runner_version",
    "created_at_utc",
    "started_at_utc",
    "finished_at_utc",
    "exit_code",
    "artifact_inventory",
    "artifact_hashes",
    "previous_receipt_sha256",
    "receipt_sha256",
)


def validate_state_transition(
    from_state: RunState | str, to_state: RunState | str
) -> bool:
    """Validate if transitioning from from_state to to_state is allowed."""
    try:
        from_enum = (
            RunState(from_state) if isinstance(from_state, str) else from_state
        )
        to_enum = RunState(to_state) if isinstance(to_state, str) else to_state
    except ValueError as exc:
        raise ReceiptStoreError(f"Invalid state enum value: {exc}") from exc

    allowed = ALLOWED_TRANSITIONS.get(from_enum, set())
    if to_enum not in allowed:
        raise ReceiptStoreError(
            f"Forbidden state transition from {from_enum.value} to {to_enum.value}"
        )
    return True


@dataclass(frozen=True, slots=True)
class RunReceipt:
    """Immutable record of an experiment run state transition."""

    schema_version: int
    run_id: str
    slice_id: str
    state: RunState | str
    artifact_root: str
    run_directory: str
    implementation_commit: str
    protocol_commit: str
    protocol_sha256: str
    input_hashes: Mapping[str, str]
    runner_version: str
    created_at_utc: str
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    exit_code: int | None = None
    artifact_inventory: tuple[str, ...] = field(default_factory=tuple)
    artifact_hashes: Mapping[str, str] = field(default_factory=dict)
    previous_receipt_sha256: str | None = None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        """Validate receipt attributes and convert collections to read-only proxies."""
        if self.schema_version < 1:
            raise ValueError(f"Invalid schema_version: {self.schema_version}")
        if not self.run_id:
            raise ValueError("run_id cannot be empty")
        if not self.slice_id:
            raise ValueError("slice_id cannot be empty")

        if isinstance(self.state, str):
            try:
                RunState(self.state)
            except ValueError:
                raise ValueError(f"Invalid state: {self.state}") from None

        # Validate protocol_sha256
        validate_sha256_digest(self.protocol_sha256, "protocol_sha256")

        # Validate and encapsulate input_hashes
        raw_input_hashes = dict(self.input_hashes or {})
        for k, v in raw_input_hashes.items():
            validate_sha256_digest(v, f"input_hashes[{k}]")
        object.__setattr__(
            self, "input_hashes", MappingProxyType(raw_input_hashes)
        )

        # Encapsulate artifact_inventory
        raw_inventory = tuple(str(x) for x in (self.artifact_inventory or ()))
        object.__setattr__(self, "artifact_inventory", raw_inventory)

        # Validate and encapsulate artifact_hashes
        raw_artifact_hashes = dict(self.artifact_hashes or {})
        for k, v in raw_artifact_hashes.items():
            validate_sha256_digest(v, f"artifact_hashes[{k}]")
        object.__setattr__(
            self, "artifact_hashes", MappingProxyType(raw_artifact_hashes)
        )

        # Validate previous_receipt_sha256 if present
        validate_sha256_digest(
            self.previous_receipt_sha256,
            "previous_receipt_sha256",
            allow_none=True,
        )

        # Validate receipt_sha256 if not empty (draft allow empty)
        validate_sha256_digest(
            self.receipt_sha256, "receipt_sha256", allow_empty=True
        )

    def compute_hash(self) -> str:
        """Compute deterministic SHA-256 hash of this receipt."""
        d = self.to_dict()
        d.pop("receipt_sha256", None)
        return compute_canonical_json_sha256(d)

    def to_dict(self) -> dict[str, Any]:
        """Convert receipt to dictionary representation matching required fields."""
        state_val = (
            self.state.value if isinstance(self.state, RunState) else str(self.state)
        )
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "slice_id": self.slice_id,
            "state": state_val,
            "artifact_root": self.artifact_root,
            "run_directory": self.run_directory,
            "implementation_commit": self.implementation_commit,
            "protocol_commit": self.protocol_commit,
            "protocol_sha256": self.protocol_sha256,
            "input_hashes": dict(self.input_hashes),
            "runner_version": self.runner_version,
            "created_at_utc": self.created_at_utc,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "exit_code": self.exit_code,
            "artifact_inventory": list(self.artifact_inventory),
            "artifact_hashes": dict(self.artifact_hashes),
            "previous_receipt_sha256": self.previous_receipt_sha256,
            "receipt_sha256": self.receipt_sha256,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunReceipt:
        """Construct RunReceipt from a dictionary after validation."""
        for field_name in REQUIRED_RECEIPT_FIELDS:
            if field_name not in d:
                raise ValueError(
                    f"Missing required field in receipt dictionary: {field_name}"
                )

        receipt_sha = d.get("receipt_sha256")
        if not receipt_sha or not isinstance(receipt_sha, str) or receipt_sha == "":
            raise ValueError(
                "Materialized receipt from_dict() requires non-empty receipt_sha256"
            )

        cls.validate_metadata_security(d)

        raw_state = d["state"]
        state_inst: RunState | str
        if isinstance(raw_state, str) and raw_state in RunState.__members__:
            state_inst = RunState[raw_state]
        else:
            state_inst = str(raw_state)

        return cls(
            schema_version=int(d["schema_version"]),
            run_id=str(d["run_id"]),
            slice_id=str(d["slice_id"]),
            state=state_inst,
            artifact_root=str(d["artifact_root"]),
            run_directory=str(d["run_directory"]),
            implementation_commit=str(d["implementation_commit"]),
            protocol_commit=str(d["protocol_commit"]),
            protocol_sha256=str(d["protocol_sha256"]),
            input_hashes=dict(d["input_hashes"]),
            runner_version=str(d["runner_version"]),
            created_at_utc=str(d["created_at_utc"]),
            started_at_utc=d.get("started_at_utc"),
            finished_at_utc=d.get("finished_at_utc"),
            exit_code=d.get("exit_code"),
            artifact_inventory=tuple(d.get("artifact_inventory", ())),
            artifact_hashes=dict(d.get("artifact_hashes", {})),
            previous_receipt_sha256=d["previous_receipt_sha256"],
            receipt_sha256=str(receipt_sha),
        )

    @staticmethod
    def validate_metadata_security(metadata: dict[str, Any]) -> None:
        """Scan keys and values in metadata dictionary for forbidden secret patterns."""

        def _check_key_and_value(k: str, v: Any) -> None:
            k_lower = str(k).lower()
            for sec in SECRET_FIELDS_FORBIDDEN:
                if sec in k_lower:
                    raise ValueError(
                        f"Forbidden secret field name '{k}' in receipt metadata"
                    )
            if isinstance(v, str) and ("sk-" in v or "AIzaSy" in v):
                raise ValueError(
                    f"Forbidden secret token string detected in key '{k}'"
                )
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    _check_key_and_value(sub_k, sub_v)

        for k, v in metadata.items():
            _check_key_and_value(k, v)


def verify_receipt_chain(receipts: list[RunReceipt]) -> bool:
    """Verify integrity of an append-only sequence of RunReceipts."""
    if not receipts:
        raise ReceiptStoreError("Receipt chain is empty")

    first = receipts[0]
    first_state = (
        RunState(first.state) if isinstance(first.state, str) else first.state
    )
    if first_state != RunState.PREPARED:
        raise ReceiptStoreError(
            f"First receipt state must be PREPARED, got {first_state.value}"
        )

    for i, curr in enumerate(receipts):
        if not curr.receipt_sha256 or not is_valid_sha256(curr.receipt_sha256):
            raise ReceiptStoreError(
                f"Receipt at index {i} has invalid receipt_sha256: "
                f"'{curr.receipt_sha256}'"
            )

        expected_hash = curr.compute_hash()
        if not hmac.compare_digest(curr.receipt_sha256, expected_hash):
            raise ReceiptStoreError(
                f"Receipt at index {i} hash mismatch: "
                f"recorded {curr.receipt_sha256}, expected {expected_hash}"
            )

        if i > 0:
            prev = receipts[i - 1]
            if (
                not curr.previous_receipt_sha256
                or not is_valid_sha256(curr.previous_receipt_sha256)
                or not hmac.compare_digest(
                    curr.previous_receipt_sha256, prev.receipt_sha256
                )
            ):
                raise ReceiptStoreError(
                    f"Receipt at index {i} previous_receipt_sha256 "
                    f"'{curr.previous_receipt_sha256}' does not match preceding "
                    f"receipt_sha256 '{prev.receipt_sha256}'"
                )

            validate_state_transition(prev.state, curr.state)

    return True
