# Slice 5A.2 — Definitive Scientific Closure Report

**Repository**: `raglab-v7`
**Branch**: `feat/agentic-rag-slice5a-runtime-pilot`
**HEAD**: `686291e08be7a010112981899f8de920ec94c5b1`
**Date**: 2026-08-09

---

## 1. Definitive Scientific Classification

```text
SLICE5A2_OPERATIONAL_COMPLETE
SLICE5A2_SCIENTIFIC_NOT_EVALUABLE
CATEGORY_D_CONFIRMED
LABEL = PILOT_NOT_EVALUABLE_DUE_TO_CANONICAL_MAPPING_FAILURE
```

---

## 2. Key Findings & Gate Audit

1. **Operational Execution**: `SLICE5A2_OPERATIONAL_COMPLETE` (4/4 DEV QIDs executed, 1 logical call per QID).
2. **Scientific Evaluation**: `SLICE5A2_SCIENTIFIC_NOT_EVALUABLE` (Category D confirmed).
3. **Canonical Mapping Audit**:
   - 23 out of 24 evidence items mapped via `EXACT_SUBSTRING`.
   - **1 unmapped evidence item**: `q_dev_01`, Arm B0 (`H0_hierarchical_leaf`), Rank 2, Page 117.
   - Page 117 falls outside the human-annotated passage registry universe (pages 91–115).
   - **Zero Synthetic Fallback**: `ZERO_SYNTHETIC_PS_PREFIX_FALLBACK = true`.
4. **Descriptive Metrics Only**:
   - Raw mean nDCG@3: B0 = 0.1642, A1 = 0.1707, Raw Δ = +0.0064.
   - Raw Δ is **descriptive only** and does **NOT** demonstrate routing superiority.
   - `METRIC_VALIDITY = DESCRIPTIVE_ONLY`, `SUPERIORITY_CLAIMED = false`.
5. **Historical Oracle Calibration**:
   - Retrieval divergence detected across all 8 arm × QID pairs (`RETRIEVAL_DIVERGENCE_DETECTED`).
   - `O1_STATUS = HISTORICAL_NONCOMPARABLE_REFERENCE`.
   - `REGRET_STATUS = NOT_VALID_FOR_CURRENT_PILOT`.
   - `DIVERGENCE_CAUSE = UNDETERMINED`.
6. **TEST Split**: `TEST_SPLIT_STATUS = SEALED_UNEXECUTED`.
7. **No Superiority Claim**: `NO_SUPERIORITY_CLAIM = true`.

---

## 3. Criteria for Future Valid Evaluation

To establish a scientifically valid comparison in a future study:
1. Expand the human annotation pool and passage registry to include page 117.
2. Maintain zero-tolerance fail-closed canonical mapping.
3. Keep the TEST split sealed until all DEV mapping gates pass cleanly.
