# RAGLab v7 — Agentic Track

## Overview

The Agentic Track adds a **governed routing and tool-calling layer** on top of
the existing RAGLab v7 pipeline. It does NOT replace the seven fixed retrieval
strategies — it selects among them based on query characteristics.

**Status:** Slice 5A — Foundation only (experimental).

## Key Clarifications

- The current RAG pipeline is **not agentic by execution** — strategies are
  pre-selected at pipeline configuration time.
- **L1** (this slice): One-shot deterministic routing — selects one strategy
  per query based on public text features.
- **L2** (Slice 5B, not implemented): Tool calling one-shot — invokes a
  retrieval tool governed by budget and authorization.
- **L3** (Slice 5C, not implemented): Multi-step reasoning loop with evidence
  accumulation and stop policy evaluation.
- **L4** (future): Single agent with tool retrieval — NOT multi-agent.

## Scientific Controls

- Slice 4/v5 results with human qrels remain **IMMUTABLE** and serve as the
  control group.
- `n=8` queries does NOT permit general superiority claims.
- The post-hoc oracle is a **ceiling only** — it uses evaluation outcomes
  that are unavailable at inference time.
- Chain-of-thought is NOT evidence — it is an internal reasoning trace.
- All APIs from the old notebooks were abandoned, not copied.
- **DOCUMENT_FILTERING_DISABLED_V1**: Document, page, or metadata filtering remains strictly disabled in v1 (`_ALLOWED_FILTER_KEYS = frozenset()`). No filter keys are authorized.

## Architecture

See [architecture.md](architecture.md) for the full module diagram.

## Quick Start

```bash
# Run tests
.venv/bin/python -m pytest tests/agentic/ -v

# Policy replay (CI mode — synthetic fixture)
.venv/bin/python scripts/analyze_agentic_policy_replay.py \
  --slice4-result benchmarks/agentic/slice5/fixtures/synthetic_slice4_result_v1.json \
  --config benchmarks/agentic/slice5/configs/policy_replay_v1.json \
  --output-dir /tmp/replay_output
```

## File Map

```
src/raglab/agentic/
├── __init__.py          # Package init with schema version
├── contracts.py         # Domain contracts (frozen dataclasses)
├── enums.py             # QueryClass, StopReason, DecisionCode, etc.
├── errors.py            # Structured errors (no silent fallbacks)
├── policy.py            # Frozen policy metadata and hashing
├── router.py            # Deterministic + LLM router contracts
├── tool_registry.py     # Read-only capability registry
├── tool_executor.py     # Governed execution with validation
├── evidence_state.py    # Evidence accumulator with deduplication
├── budget.py            # Resource limits and consumption
├── stop_policy.py       # Stop decision evaluation
├── trajectory_ledger.py # Append-only JSONL ledger
└── adapters/
    ├── __init__.py
    └── llamaindex.py    # Optional availability barrier

tests/agentic/
├── test_contracts.py    # Contract validation
├── test_router.py       # Router stability, leakage, policy hash
├── test_tool_registry.py# Registry, executor, budget, stop policy
├── test_ledger.py       # Ledger + vertical end-to-end flow
└── test_security.py     # Domain purity, injection, adapter isolation

benchmarks/agentic/slice5/
├── configs/             # Versionable configuration
├── protocols/           # Scientific protocol
└── fixtures/            # Synthetic CI fixture (not scientific data)
```
