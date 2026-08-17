# Slice 5A.3 — Canonical Coverage and Retrieval Parity Gate

## Protocol Overview

This benchmark directory contains the pre-registration protocol, schemas, and governance specifications for **Slice 5A.3** of RAGLab v7.

### Core Objectives
1. Verify 100% canonical passage ID coverage across all 7 retrieval arms (F0, S0, S1, W0, W1, C0, C1).
2. Enforce zero synthetic fallback IDs or synthetic human judgments.
3. Validate deterministic retrieval repeatability across same-snapshot runs.
4. Calculate same-run operational oracle metrics without historical O1 reuse.
5. Fail closed (`NOT_EVALUABLE_JUDGED_COVERAGE` / `NOT_EVALUABLE_CANONICAL_COVERAGE`) if any retrieved item lacks explicit human judgment or canonical passage mapping.
