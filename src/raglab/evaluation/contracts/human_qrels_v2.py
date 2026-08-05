# ruff: noqa: E501, E741
"""Domain Model & Fail-Closed Loader for Gate B Human-Validated Graded Qrels v2.

Enforces immutability, exact distribution validation (32, 18, 13, 6), SHA-256 cryptographic verification,
prohibition of machine-generated silver labels, sealed holdouts, and 10/10 negative controls for q_test_04.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Final

VALID_ACTIVE_QUESTIONS: Final[frozenset[str]] = frozenset({
    "q_dev_01",
    "q_dev_02",
    "q_dev_03",
    "q_dev_04",
    "q_test_01",
    "q_test_02",
    "q_test_03",
    "q_test_04",
})

ACTIVE_QUESTION_IDS: Final[frozenset[str]] = VALID_ACTIVE_QUESTIONS

VALID_EVIDENCE_ROLES: Final[frozenset[str]] = frozenset({
    "PRIMARY",
    "SUPPORTING",
    "CONTEXTUAL",
    "NEGATIVE_CONTROL",
})

VALID_PROVENANCES: Final[frozenset[str]] = frozenset({
    "HUMAN_EXACT_CONSENSUS",
    "HUMAN_ADJUDICATED",
})

EXPECTED_GRADE_DISTRIBUTION: Final[dict[int, int]] = {
    0: 32,
    1: 18,
    2: 13,
    3: 6,
}


@dataclasses.dataclass(frozen=True)
class HumanQrel:
    """Individual human-validated passage relevance qrel record."""

    question_id: str
    passage_id: str
    relevance_grade: int  # 0, 1, 2, 3
    evidence_role: str  # PRIMARY, SUPPORTING, CONTEXTUAL, NEGATIVE_CONTROL
    provenance: str  # HUMAN_EXACT_CONSENSUS, HUMAN_ADJUDICATED
    page_number: int | None = None
    question_text: str = ""
    passage_text: str = ""
    supporting_span_human: str = ""
    annotation_notes: str = ""
    schema_version: str = "2.0.0"

    def __post_init__(self) -> None:
        if "holdout" in self.question_id.lower():
            raise ValueError(f"Forbidden holdout question_id: {self.question_id}")
        if self.question_id not in VALID_ACTIVE_QUESTIONS:
            raise ValueError(f"Unauthorized question_id: {self.question_id}")
        if self.relevance_grade not in (0, 1, 2, 3):
            raise ValueError(f"Invalid relevance_grade: {self.relevance_grade}")
        if self.evidence_role not in VALID_EVIDENCE_ROLES:
            raise ValueError(f"Invalid evidence_role: {self.evidence_role}")
        if self.provenance not in VALID_PROVENANCES:
            raise ValueError(f"Invalid provenance: {self.provenance}")


@dataclasses.dataclass(frozen=True)
class HumanQrelsSet:
    """Immutable collection of human-validated graded qrels indexed by (question_id, passage_id)."""

    qrels: tuple[HumanQrel, ...]
    schema_version: str = "2.0.0"
    qrels_sha256: str = ""
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_pair", {(q.question_id, q.passage_id): q for q in self.qrels})
        by_qid: dict[str, list[HumanQrel]] = {}
        for q in self.qrels:
            by_qid.setdefault(q.question_id, []).append(q)
        object.__setattr__(self, "_by_qid", {qid: tuple(items) for qid, items in by_qid.items()})

    @property
    def total_pairs(self) -> int:
        return len(self.qrels)

    @property
    def consensus_count(self) -> int:
        return sum(1 for q in self.qrels if q.provenance == "HUMAN_EXACT_CONSENSUS")

    @property
    def adjudicated_count(self) -> int:
        return sum(1 for q in self.qrels if q.provenance == "HUMAN_ADJUDICATED")

    @property
    def grade_distribution(self) -> dict[int, int]:
        dist = {0: 0, 1: 0, 2: 0, 3: 0}
        for q in self.qrels:
            dist[q.relevance_grade] = dist.get(q.relevance_grade, 0) + 1
        return dist

    def get_qrel(self, question_id: str, passage_id: str) -> HumanQrel | None:
        if "holdout" in question_id.lower():
            raise ValueError(
                f"HOLDOUT_SEALED: Forbidden access to holdout question {question_id}"
            )
        by_pair: dict[tuple[str, str], HumanQrel] = self._by_pair  # type: ignore[attr-defined]
        return by_pair.get((question_id, passage_id))

    def get_qrels_for_question(self, question_id: str) -> tuple[HumanQrel, ...]:
        if "holdout" in question_id.lower():
            raise ValueError(
                f"HOLDOUT_SEALED: Forbidden access to holdout question {question_id}"
            )
        by_qid: dict[str, tuple[HumanQrel, ...]] = self._by_qid  # type: ignore[attr-defined]
        return by_qid.get(question_id, ())


    def get_relevant_passages(self, question_id: str, min_grade: int = 1) -> tuple[HumanQrel, ...]:
        return tuple(q for q in self.get_qrels_for_question(question_id) if q.relevance_grade >= min_grade)

    def get_irrelevant_passages(self, question_id: str) -> tuple[HumanQrel, ...]:
        return tuple(q for q in self.get_qrels_for_question(question_id) if q.relevance_grade == 0)

    def is_judged(self, question_id: str, passage_id: str) -> bool:
        return self.get_qrel(question_id, passage_id) is not None

    def is_abstention_question(self, question_id: str) -> bool:
        qrels = self.get_qrels_for_question(question_id)
        if not qrels:
            return False
        return all(q.relevance_grade == 0 and q.evidence_role == "NEGATIVE_CONTROL" for q in qrels)

    def get_graduated_gain(self, question_id: str, passage_id: str) -> float:
        qrel = self.get_qrel(question_id, passage_id)
        if qrel is None:
            return 0.0
        return float(qrel.relevance_grade)


def load_human_qrels_set(qrels_path: Path, manifest_path: Path) -> HumanQrelsSet:
    """Fail-closed loader for human-validated graded qrels."""
    if not qrels_path.exists():
        raise ValueError(f"HUMAN_QRELS_REQUIRED_OR_INVALID: Qrels file not found at {qrels_path}")
    if not manifest_path.exists():
        raise ValueError(f"HUMAN_QRELS_REQUIRED_OR_INVALID: Manifest file not found at {manifest_path}")

    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"HUMAN_QRELS_REQUIRED_OR_INVALID: Invalid manifest JSON: {exc}") from exc

    if manifest_data.get("authoritative_for_evaluation") is not True:
        raise ValueError("HUMAN_QRELS_REQUIRED_OR_INVALID: Manifest authoritative_for_evaluation must be True")
    if manifest_data.get("silver_used_as_ground_truth") is not False:
        raise ValueError("HUMAN_QRELS_REQUIRED_OR_INVALID: Manifest silver_used_as_ground_truth must be False")
    if manifest_data.get("holdout_sealed") is not True:
        raise ValueError("HUMAN_QRELS_REQUIRED_OR_INVALID: Manifest holdout_sealed must be True")

    qrels_bytes = qrels_path.read_bytes()
    actual_qrels_sha = hashlib.sha256(qrels_bytes).hexdigest()
    expected_qrels_sha = manifest_data.get("final_qrels_file_sha256") or manifest_data.get("actual_sha256")
    if expected_qrels_sha and actual_qrels_sha != expected_qrels_sha:
        raise ValueError(f"HUMAN_QRELS_REQUIRED_OR_INVALID: Qrels SHA-256 digest mismatch. Expected {expected_qrels_sha}, got {actual_qrels_sha}")

    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    records: list[HumanQrel] = []
    lines = [line_str for line_str in qrels_bytes.decode("utf-8").splitlines() if line_str.strip()]

    seen_pairs: set[tuple[str, str]] = set()
    for line_idx, line in enumerate(lines, 1):
        try:
            item = json.loads(line)
        except Exception as exc:
            raise ValueError(f"HUMAN_QRELS_REQUIRED_OR_INVALID: Invalid JSONL on line {line_idx}: {exc}") from exc

        qid = item.get("question_id")
        psid = item.get("passage_id")
        if not qid or not psid:
            raise ValueError(f"HUMAN_QRELS_REQUIRED_OR_INVALID: Missing question_id or passage_id on line {line_idx}")
        if (qid, psid) in seen_pairs:
            raise ValueError(f"HUMAN_QRELS_REQUIRED_OR_INVALID: Duplicate pair ({qid}, {psid}) found on line {line_idx}")
        seen_pairs.add((qid, psid))

        if "holdout" in qid.lower():
            raise ValueError(f"HUMAN_QRELS_REQUIRED_OR_INVALID: Forbidden holdout question found: {qid}")
        if qid not in VALID_ACTIVE_QUESTIONS:
            raise ValueError(f"HUMAN_QRELS_REQUIRED_OR_INVALID: Unauthorized question_id: {qid}")

        grade = item.get("relevance_grade")
        if grade not in (0, 1, 2, 3):
            raise ValueError(f"HUMAN_QRELS_REQUIRED_OR_INVALID: Invalid relevance_grade on line {line_idx}: {grade}")

        role = item.get("evidence_role")
        if role not in VALID_EVIDENCE_ROLES:
            raise ValueError(f"HUMAN_QRELS_REQUIRED_OR_INVALID: Invalid evidence_role on line {line_idx}: {role}")

        provenance = item.get("provenance")
        if provenance not in VALID_PROVENANCES:
            raise ValueError(f"HUMAN_QRELS_REQUIRED_OR_INVALID: Invalid provenance on line {line_idx}: {provenance}")

        records.append(
            HumanQrel(
                question_id=qid,
                passage_id=psid,
                relevance_grade=int(grade),
                evidence_role=str(role),
                provenance=str(provenance),
                page_number=item.get("page_number"),
                question_text=item.get("question_text", ""),
                passage_text=item.get("passage_text", ""),
                supporting_span_human=item.get("supporting_span_human", ""),
                annotation_notes=item.get("annotation_notes", ""),
                schema_version=item.get("schema_version", "2.0.0"),
            )
        )

    qrels_set = HumanQrelsSet(
        qrels=tuple(records),
        schema_version=manifest_data.get("schema_version", "2.0.0"),
        qrels_sha256=actual_qrels_sha,
        manifest_sha256=manifest_sha,
    )

    if qrels_set.total_pairs != manifest_data.get("total_pairs", 69):
        raise ValueError(f"HUMAN_QRELS_REQUIRED_OR_INVALID: Total pairs mismatch. Expected {manifest_data.get('total_pairs')}, got {qrels_set.total_pairs}")
    if qrels_set.total_pairs != 69:
        raise ValueError(f"HUMAN_QRELS_REQUIRED_OR_INVALID: Total pairs must be exactly 69, got {qrels_set.total_pairs}")
    if qrels_set.consensus_count != 41:
        raise ValueError(f"HUMAN_QRELS_REQUIRED_OR_INVALID: Consensus count must be exactly 41, got {qrels_set.consensus_count}")
    if qrels_set.adjudicated_count != 28:
        raise ValueError(f"HUMAN_QRELS_REQUIRED_OR_INVALID: Adjudicated count must be exactly 28, got {qrels_set.adjudicated_count}")

    if qrels_set.grade_distribution != EXPECTED_GRADE_DISTRIBUTION:
        raise ValueError(f"HUMAN_QRELS_REQUIRED_OR_INVALID: Grade distribution mismatch. Expected {EXPECTED_GRADE_DISTRIBUTION}, got {qrels_set.grade_distribution}")

    q4_items = qrels_set.get_qrels_for_question("q_test_04")
    if len(q4_items) != 10:
        raise ValueError(f"HUMAN_QRELS_REQUIRED_OR_INVALID: q_test_04 must contain exactly 10 passages, got {len(q4_items)}")
    for item in q4_items:
        if item.relevance_grade != 0 or item.evidence_role != "NEGATIVE_CONTROL":
            raise ValueError(f"HUMAN_QRELS_REQUIRED_OR_INVALID: q_test_04 item {item.passage_id} must have grade=0 and role=NEGATIVE_CONTROL")

    return qrels_set
