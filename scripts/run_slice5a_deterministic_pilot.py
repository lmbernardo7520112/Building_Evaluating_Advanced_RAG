"""CLI: Run deterministic pilot with stub/fake retrieval ports.

This script orchestrates the DeterministicPilot with synthetic retrieval
ports, producing trajectory and manifest artifacts. It does NOT:
- Load real PDF or indices
- Call Gemini or any external API
- Access qrels during routing/retrieval

If productive assets (PDF, indices, models) are unavailable, it reports:
  PRODUCTIVE_PILOT_NOT_EXECUTED

Usage:
  python scripts/run_slice5a_deterministic_pilot.py \
    --run-id pilot_v1 \
    --output-dir benchmarks/agentic/slice5/pilot_output \
    [--questions-json path/to/questions.json] \
    [--productive]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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

# ── Stub port (no real retrieval) ────────────────────────────────


@dataclass
class StubRetrievalPort:
    """Minimal port returning deterministic stub evidence."""

    strategy_name: str

    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedEvidence]:
        return [
            RetrievedEvidence(
                chunk_id=ChunkId(value=f"{self.strategy_name}_{i}"),
                document_id="doc_stub",
                text=f"stub evidence from {self.strategy_name} rank {i}",
                rank=i,
                score=1.0 / i,
                passage_id=f"ps_{self.strategy_name}_{i}",
            )
            for i in range(1, min(top_k + 1, 4))
        ]


# ── Default questions ────────────────────────────────────────────

DEFAULT_QUESTIONS = [
    PilotQuestion(
        qid="q_dev_01",
        text="What is the main concept of the paper?",
        split="development",
    ),
    PilotQuestion(
        qid="q_dev_02", text="Compare approaches A and B.", split="development"
    ),
    PilotQuestion(
        qid="q_dev_03", text="Describe the chunking strategy.", split="development"
    ),
    PilotQuestion(
        qid="q_dev_04", text="What is hierarchical retrieval?", split="development"
    ),
    PilotQuestion(
        qid="q_test_01", text="What are the evaluation metrics?", split="test"
    ),
    PilotQuestion(qid="q_test_02", text="How does sentence window work?", split="test"),
    PilotQuestion(qid="q_test_03", text="What is auto-merging?", split="test"),
    PilotQuestion(
        qid="q_test_04", text="Unanswerable question about Mars.", split="test"
    ),
]


def _load_questions(path: Path | None) -> list[PilotQuestion]:
    """Load questions from JSON or use defaults."""
    if path and path.exists():
        with open(path) as f:
            data = json.load(f)
        return [
            PilotQuestion(
                qid=q["qid"],
                text=q["text"],
                split=q.get("split", "unknown"),
            )
            for q in data["questions"]
        ]
    return DEFAULT_QUESTIONS


def _check_productive_assets() -> bool:
    """Check if productive retrieval assets are available."""
    required = [
        Path("data/Building_and_Evaluating_Advanced_RAG_Applications.pdf"),
    ]
    return all(p.exists() for p in required)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic pilot for Slice 5A.2",
    )
    parser.add_argument("--run-id", type=str, default="pilot_v1", help="Run identifier")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory"
    )
    parser.add_argument(
        "--questions-json", type=Path, default=None, help="Questions JSON file"
    )
    parser.add_argument(
        "--productive", action="store_true", help="Require productive assets"
    )
    parser.add_argument(
        "--dev-only", action="store_true", help="Run only DEV questions"
    )

    args = parser.parse_args()

    # Check productive mode
    if args.productive and not _check_productive_assets():
        print("PRODUCTIVE_PILOT_NOT_EXECUTED")
        print("Reason: Required productive assets not found.")
        print("Missing: data/Building_and_Evaluating_Advanced_RAG_Applications.pdf")
        sys.exit(0)

    questions = _load_questions(args.questions_json)
    if args.dev_only:
        questions = [q for q in questions if q.split == "development"]

    # Build ports (stub for now)
    ports = {s: StubRetrievalPort(s.value) for s in ALL_STRATEGIES}
    registry, adapters = build_registry_with_adapters(ports)

    # Fixed clock and ID generator for determinism
    counter = 0

    def _fixed_clock() -> str:
        return "2026-08-08T18:00:00+00:00"

    def _fixed_id_gen() -> str:
        nonlocal counter
        counter += 1
        return f"inv_{counter:04d}"

    config = PilotRunConfig(run_id=args.run_id)
    pilot = DeterministicPilot(
        registry=registry,
        adapters=adapters,
        config=config,
        clock=_fixed_clock,
        invocation_id_gen=_fixed_id_gen,
    )

    result = pilot.run(questions)
    manifest = build_pilot_manifest(result)

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_dict = {
        "run_id": result.run_id,
        "questions_processed": result.questions_processed,
        "results": result.results,
        "routing_decisions": result.routing_decisions,
        "errors": result.errors,
        "policy_metadata": result.policy_metadata,
    }

    (args.output_dir / "pilot_result.json").write_text(
        json.dumps(result_dict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "pilot_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "pilot_trajectories.json").write_text(
        json.dumps(result.trajectories, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Pilot completed: {result.questions_processed} questions processed")
    print(f"Errors: {len(result.errors)}")
    print(f"Output: {args.output_dir}")
    for r in result.results:
        qid = r["qid"]
        strat = r["strategy_selected"]
        ev = r["evidence_count"]
        gen = r["generation"]
        print(f"  {qid}: {strat} evidence={ev} gen={gen}")

    sys.exit(0)


if __name__ == "__main__":
    main()
