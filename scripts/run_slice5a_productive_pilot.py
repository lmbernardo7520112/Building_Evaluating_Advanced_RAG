#!/usr/bin/env python3
"""Run Slice 5A.2 deterministic pilot with productive retrievers.

Connects DeterministicPilot → StrategyToolFactory →
RetrievalToolAdapter → frozen retrieval pipelines from Slice 4.

Requirements:
- RAGLAB_PDF_PATH set to the authoritative corpus PDF
- Embedding cache at .model_cache
- No Gemini API key (generation = NOT_EXECUTED)
- No network access during execution

Usage:
  RAGLAB_PDF_PATH="path/to/book.pdf" \\
    .venv/bin/python scripts/run_slice5a_productive_pilot.py \\
    --run-id pilot_prod_v1 \\
    --output-dir benchmarks/agentic/slice5/pilot_prod_output \\
    --dev-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from slice5a_productive_composition import (  # noqa: E402
    build_productive_retrievers,
    check_all_assets,
    load_embedding_model,
    load_pages,
    resolve_pdf_path,
    sha256_file,
    verify_pdf,
)

from raglab.agentic.runtime.deterministic_pilot import (  # noqa: E402
    DeterministicPilot,
    PilotQuestion,
    PilotRunConfig,
    build_pilot_manifest,
)
from raglab.agentic.runtime.strategy_tool_factory import (  # noqa: E402
    build_registry_with_adapters,
)

logger = logging.getLogger("productive_pilot")

# Frozen DEV questions (same as benchmark)
DEV_QUESTIONS = [
    PilotQuestion(
        qid="q_dev_01",
        text="O que é um grafo bipartido?",
        split="development",
    ),
    PilotQuestion(
        qid="q_dev_02",
        text=("Compare os métodos de busca em largura e em profundidade."),
        split="development",
    ),
    PilotQuestion(
        qid="q_dev_03",
        text=("Descreva a estratégia de demonstração por indução."),
        split="development",
    ),
    PilotQuestion(
        qid="q_dev_04",
        text="O que é uma relação de equivalência?",
        split="development",
    ),
]


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Productive deterministic pilot",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="pilot_prod_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dev-only",
        action="store_true",
        default=True,
    )
    args = parser.parse_args()

    # ── Asset verification ──
    logger.info("Checking productive assets...")
    status = check_all_assets()
    if not status["all_ready"]:
        print("PRODUCTIVE_PILOT_NOT_EXECUTED")
        print("reason=ASSET_MISSING")
        print(json.dumps(status, indent=2, default=str))
        sys.exit(0)

    pdf_path = resolve_pdf_path()
    if pdf_path is None:
        print("PRODUCTIVE_PILOT_NOT_EXECUTED")
        print("reason=ASSET_MISSING (pdf_path is None)")
        sys.exit(1)
    if not verify_pdf(pdf_path):
        print("PRODUCTIVE_PILOT_NOT_EXECUTED")
        print("reason=PDF_HASH_MISMATCH")
        sys.exit(1)
    pdf_sha = sha256_file(pdf_path)
    logger.info("PDF verified: %s", pdf_sha[:16])

    # ── Load pages ──
    logger.info("Loading PDF pages...")
    t0 = time.monotonic()
    pages = load_pages(pdf_path)
    t_pages = time.monotonic() - t0
    logger.info("Loaded %d pages in %.1fs", len(pages), t_pages)

    # ── Load embedding model ──
    logger.info("Loading embedding model...")
    t0 = time.monotonic()
    embed_model = load_embedding_model()
    t_embed = time.monotonic() - t0
    logger.info("Embedding model loaded in %.1fs", t_embed)

    # ── Build retrievers ──
    logger.info("Building productive retrievers...")
    t0 = time.monotonic()
    ports = build_productive_retrievers(pages, embed_model)
    t_build = time.monotonic() - t0
    logger.info(
        "Built %d retrievers in %.1fs",
        len(ports),
        t_build,
    )

    # ── Assemble pilot ──
    registry, adapters = build_registry_with_adapters(ports)

    counter = 0

    def _clock() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())

    def _id_gen() -> str:
        nonlocal counter
        counter += 1
        return f"inv_{counter:04d}"

    config = PilotRunConfig(run_id=args.run_id)
    pilot = DeterministicPilot(
        registry=registry,
        adapters=adapters,
        config=config,
        clock=_clock,
        invocation_id_gen=_id_gen,
    )

    questions = DEV_QUESTIONS
    logger.info("Running pilot: %d DEV questions", len(questions))

    # ── Execute ──
    t0 = time.monotonic()
    result = pilot.run(questions)
    t_run = time.monotonic() - t0
    manifest = build_pilot_manifest(result)

    # ── Write outputs ──
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_dict = {
        "run_id": result.run_id,
        "questions_processed": result.questions_processed,
        "results": result.results,
        "routing_decisions": result.routing_decisions,
        "errors": result.errors,
        "policy_metadata": result.policy_metadata,
        "productive": True,
        "pdf_sha256": pdf_sha,
        "timing": {
            "pages_load_s": round(t_pages, 2),
            "embed_load_s": round(t_embed, 2),
            "build_s": round(t_build, 2),
            "run_s": round(t_run, 2),
        },
    }

    out = args.output_dir
    (out / "pilot_result.json").write_text(
        json.dumps(result_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "pilot_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "pilot_trajectories.json").write_text(
        json.dumps(
            result.trajectories,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ── Report ──
    print(f"\n{'=' * 60}")
    print("PRODUCTIVE DETERMINISTIC PILOT — DEV")
    print(f"{'=' * 60}")
    print(f"Questions: {result.questions_processed}")
    print(f"Errors: {len(result.errors)}")
    print(f"Run time: {t_run:.1f}s")
    print(f"Output: {out}")
    print()
    for r in result.results:
        qid = r["qid"]
        strat = r["strategy_selected"]
        ev = r["evidence_count"]
        gen = r["generation"]
        stop = r["stop_reason"]
        print(f"  {qid}: {strat} evidence={ev} gen={gen} stop={stop}")
    print(f"\n{'=' * 60}\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
