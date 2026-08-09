"""Tests for DeterministicPilot — end-to-end one-shot pilot with fakes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from raglab.agentic.runtime.deterministic_pilot import (
    DeterministicPilot,
    PilotQuestion,
    PilotRunConfig,
    build_pilot_manifest,
)
from raglab.agentic.runtime.strategy_tool_factory import (
    ALL_STRATEGIES,
    build_registry_with_adapters,
)
from raglab.domain.entities import RetrievedEvidence
from raglab.domain.value_objects import ChunkId


@dataclass
class FakePort:
    """Fake RetrievalPort returning deterministic evidence."""

    strategy_name: str

    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedEvidence]:
        return [
            RetrievedEvidence(
                chunk_id=ChunkId(value=f"{self.strategy_name}_{i}"),
                document_id="doc_01",
                text=f"evidence from {self.strategy_name} rank {i}",
                rank=i,
                score=1.0 / i,
                passage_id=f"ps_{self.strategy_name}_{i}",
            )
            for i in range(1, min(top_k + 1, 4))
        ]


def _build_pilot(
    run_id: str = "test_run",
) -> DeterministicPilot:
    """Build a pilot with all 7 fake ports."""
    ports = {s: FakePort(s.value) for s in ALL_STRATEGIES}
    registry, adapters = build_registry_with_adapters(ports)

    counter = 0

    def _fixed_clock() -> str:
        return "2026-01-01T00:00:00+00:00"

    def _fixed_id_gen() -> str:
        nonlocal counter
        counter += 1
        return f"inv_{counter:04d}"

    config = PilotRunConfig(run_id=run_id)
    return DeterministicPilot(
        registry=registry,
        adapters=adapters,
        config=config,
        clock=_fixed_clock,
        invocation_id_gen=_fixed_id_gen,
    )


SAMPLE_QUESTIONS = [
    PilotQuestion(qid="q_01", text="What is the main idea?", split="development"),
    PilotQuestion(qid="q_02", text="Compare approaches A and B.", split="development"),
    PilotQuestion(qid="q_03", text="How to build a pipeline?", split="test"),
]


class TestDeterministicPilot:
    def test_processes_all_questions(self) -> None:
        pilot = _build_pilot()
        result = pilot.run(SAMPLE_QUESTIONS)

        assert result.questions_processed == 3
        assert len(result.results) == 3
        assert len(result.routing_decisions) == 3
        assert len(result.trajectories) == 3

    def test_one_decision_per_qid(self) -> None:
        pilot = _build_pilot()
        result = pilot.run(SAMPLE_QUESTIONS)

        qids_routed = [rd["qid"] for rd in result.routing_decisions]
        assert qids_routed == ["q_01", "q_02", "q_03"]

    def test_generation_not_executed(self) -> None:
        pilot = _build_pilot()
        result = pilot.run(SAMPLE_QUESTIONS)

        for r in result.results:
            assert r["generation"] == "NOT_EXECUTED"

    def test_each_trajectory_has_one_step(self) -> None:
        pilot = _build_pilot()
        result = pilot.run(SAMPLE_QUESTIONS)

        for traj in result.trajectories:
            assert len(traj["steps"]) == 1

    def test_evidence_collected(self) -> None:
        pilot = _build_pilot()
        result = pilot.run(SAMPLE_QUESTIONS)

        for r in result.results:
            assert r["evidence_count"] == 3  # top_k=3 default

    def test_determinism_across_runs(self) -> None:
        pilot1 = _build_pilot(run_id="det_check")
        pilot2 = _build_pilot(run_id="det_check")

        r1 = pilot1.run(SAMPLE_QUESTIONS)
        r2 = pilot2.run(SAMPLE_QUESTIONS)

        # Routing decisions must be identical
        assert r1.routing_decisions == r2.routing_decisions

        # Results (excluding latency-dependent fields) must match
        for a, b in zip(r1.results, r2.results, strict=False):
            assert a["qid"] == b["qid"]
            assert a["strategy_selected"] == b["strategy_selected"]
            assert a["evidence_count"] == b["evidence_count"]
            assert a["stop_reason"] == b["stop_reason"]

    def test_qid_isolation(self) -> None:
        """Each QID must produce independent results."""
        pilot = _build_pilot()
        q1 = [PilotQuestion(qid="q_a", text="First question?", split="dev")]
        q2 = [
            PilotQuestion(qid="q_a", text="First question?", split="dev"),
            PilotQuestion(qid="q_b", text="Second question?", split="dev"),
        ]

        r1 = pilot.run(q1)
        pilot2 = _build_pilot()
        r2 = pilot2.run(q2)

        # q_a should have same routing regardless of q_b presence
        assert (
            r1.routing_decisions[0]["selected_strategy"]
            == r2.routing_decisions[0]["selected_strategy"]
        )

    def test_manifest_contains_required_fields(self) -> None:
        pilot = _build_pilot()
        result = pilot.run(SAMPLE_QUESTIONS)
        manifest = build_pilot_manifest(result)

        assert manifest["run_id"] == "test_run"
        assert manifest["credentials_used"] is False
        assert manifest["gemini_used"] is False
        assert manifest["generation_status"] == "NOT_EXECUTED"
        assert manifest["questions_processed"] == 3
