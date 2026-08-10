# Slice 5A.3 / 5A.3.1 Scientific Closure — Canonical Coverage & Retrieval Parity Gate

## 1. Executive Summary

| Parameter | V1 Diagnostic Status | V2 Remediation Status |
| :--- | :--- | :--- |
| **Branch** | `feat/agentic-rag-slice5a3-parity-gate` | `feat/agentic-rag-slice5a3-parity-gate` |
| **Pre-registration Protocol** | `preregistration_v1.json` | `preregistration_v2.json` |
| **Qrels Contract** | `human_qrels.jsonl` (legacy) | `human_qrels_final.jsonl` (authoritative) |
| **Evaluation Split** | DEV Split (`q_dev_01` .. `q_dev_04`) | DEV Split (`q_dev_01` .. `q_dev_04`) |
| **TEST Split Status** | **100% SEALED and UNEXECUTED** | **100% SEALED and UNEXECUTED** |
| **Consolidated Control Arms** | `F0, S0, S1, W0, W1, C0, C1` (erroneous) | `F0, H0, H1, H2, S0, W0, W1` (authoritative 7 arms) |
| **Canonical Mapper** | Raw string match | `CanonicalPassageMapper` (exact, offset, SHA256, substring) |
| **Canonical Coverage** | 14.29% (72/84 unmapped) | **100.00% (0 unmapped)** |
| **Judged Coverage** | 0.00% (84/84 unjudged) | **49.40% (41/83 judged, 42 unjudged)** |
| **Synthetic Fallback Count** | 0 | 0 |
| **Repeatability Topk Identity** | N/A | **1.0 (100% deterministic)** |
| **Final Outcome Status** | `NOT_EVALUABLE_CANONICAL_COVERAGE` | `NOT_EVALUABLE_JUDGED_COVERAGE` |
| **Scientific Claim** | **NO_SUPERIORITY_CLAIM** | **NO_SUPERIORITY_CLAIM** |
| **Metrics Status** | `NOT_APPLICABLE` | `NOT_APPLICABLE` |

---

## 2. Empirical Verification Results (V2 Remediation Protocol)

```text
Total Retrieved Items (4 DEV QIDs x 7 arms x top_k=3): 83 items
Canonical Mapped Items:                               83 / 83 (100.00%)
Unmapped Retrieved Items:                             0 / 83 (0.00%)
Explicit Judged Items:                                41 / 83 (49.40%)
Unjudged Items Queued for Review:                     42 / 83 (50.60%)
Synthetic Fallback Items:                             0 (0.00%)
Repeatability Top-k Identity:                         1.0000 (100.00%)
Repeatability Rank Match:                             True
Same-run Arm Completeness:                            True
```

---

## 3. Root Cause Analysis & Governance Enforcement

1. **V1 Historical Negative Diagnostic**:
   - V1 used legacy `human_qrels.jsonl` instead of `human_qrels_final.jsonl`.
   - V1 compared dynamic sub-chunk IDs (e.g., `doc_p91_s0`, `node_...`) directly against `passage_registry.jsonl` without canonical mapping, resulting in 85.71% unmapped items.
   - V1 used silver checkpoint hash instead of Slice 4 Full checkpoint hash `371a78e5b3e53ce3d69b0a6c9fe9d243bad7c85967e5e8a3e65fdccfc0a21f7c`.

2. **V2 Canonical Provenance & Mapping Remediation**:
   - Separated technical chunk identity (`chunk_id`/`node_id`) from canonical passage identity (`canonical_passage_id`).
   - Applied `CanonicalPassageMapper` across all 7 consolidated control arms (`F0`, `H0`, `H1`, `H2`, `S0`, `W0`, `W1`).
   - Result: **100% Canonical Coverage** (0 unmapped items).

3. **Fail-Closed Governance Trigger (Judged Coverage)**:
   - Evaluated retrieved canonical passages against `human_qrels_final.jsonl` (hash `9c83aa9dc75924f5d9942cc2d6fb518368f2ab34f95306f080dbb111b4138d3e`).
   - 41 retrieved passages were explicitly rated by human annotators (grades 0, 1, 2).
   - 42 retrieved passages were unjudged in `human_qrels_final.jsonl` and queued cleanly into `human_review_queue.jsonl`.
   - In accordance with pre-registered fail-closed governance (`preregistration_v2.json`), because `judged_coverage < 1.0`, execution immediately halts with status `NOT_EVALUABLE_JUDGED_COVERAGE` without generating synthetic ratings or assuming implicit zeros.

---

## 4. Definitive Scientific Conclusion

```text
SLICE5A3_OPERATIONAL_COMPLETE
SLICE5A3_V2_REMEDIATION_COMPLETE
SLICE5A3_STATUS_NOT_EVALUABLE_JUDGED_COVERAGE
CATEGORY_D_CONFIRMED
NO_SUPERIORITY_CLAIM
```
