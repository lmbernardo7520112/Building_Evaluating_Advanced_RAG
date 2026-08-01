# ADR-001: Embedding Pooling Policy — mean pooling (FastEmbed ≥ 0.6)

**Status:** Accepted  
**Date:** 2026-08-01  
**Context:** Slice 4 embedding provisioning fix

## Context

FastEmbed 0.8.0 (installed) uses **mean pooling** by default for
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

FastEmbed ≤ 0.5.1 used **CLS pooling** for the same model.

The warning observed:
> "FastEmbed informou que o modelo passou a usar mean pooling em vez de CLS
> pooling e que FastEmbed 0.5.1 preservaria o comportamento anterior."

## Decision

**Option A — Preserve FastEmbed 0.8.0 with mean pooling as canonical policy.**

Rationale:

1. `pyproject.toml` declares `fastembed>=0.3.0,<1.0.0` and `licenses.json` records
   version 0.8.0. The lockfile was generated with 0.8.0.

2. **No evidence exists** in the codebase that any previous Slice benchmark results
   were generated under CLS pooling. The Slice 3 experiment manifest records
   `"backend": "fastembed>=0.3.0,<1.0.0"` and `"dimension": 384` but does NOT
   record the pooling strategy or the exact FastEmbed version used at execution time.

3. The Slice 3 benchmark runner (`run_slice3_benchmark.py`) has pre-existing
   unstaged modifications and was never re-executed under a controlled pooling
   policy. Gate 2 report does not mention pooling.

4. Downgrading to FastEmbed 0.5.1 would require re-pinning all transitive
   dependencies, which is disproportionate and introduces regression risk.

5. **Since pooling strategy was never recorded, all previous retrieval results
   are NOT DIRECTLY COMPARABLE with results produced under this fix.** Any future
   comparison must note the pooling change.

## Consequences

- All new Slice 4 (and future) embeddings use **mean pooling** under FastEmbed 0.8.0.
- The embedding model manifest records `pooling: mean` explicitly.
- Previous Slice 2/3 results are classified as **pooling-ambiguous** — they cannot
  be directly compared without re-execution under the same policy.
- If re-execution of Slice 2/3 is desired, it must use the same FastEmbed 0.8.0
  + mean pooling to ensure comparability.
- No downgrade to FastEmbed 0.5.1 is performed.
