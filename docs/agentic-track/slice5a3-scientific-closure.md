# Slice 5A.3 Scientific Closure — Canonical Coverage & Retrieval Parity Gate

## 1. Executive Summary

| Parameter | Value / Status |
| :--- | :--- |
| **Branch** | `feat/agentic-rag-slice5a3-parity-gate` |
| **Pre-registration Protocol** | `benchmarks/agentic/slice5/slice5a3/protocols/preregistration_v1.json` |
| **Evaluation Split** | DEV Split (`q_dev_01`, `q_dev_02`, `q_dev_03`, `q_dev_04`) |
| **TEST Split Status** | **100% SEALED and UNEXECUTED** |
| **Final Outcome Status** | `NOT_EVALUABLE_CANONICAL_COVERAGE` |
| **Scientific Claim** | **NO_SUPERIORITY_CLAIM** |
| **Metrics Status** | `NOT_APPLICABLE` (Evaluations halted under fail-closed governance) |

---

## 2. Empirical Verification Results

```
Total Retrieved Items:        84 (4 DEV QIDs x 7 arms x top_k=3)
Canonical Mapped Items:        12 / 84 (14.29%)
Unmapped Retrieved Items:      72 / 84 (85.71%)
Explicit Judged Items:         0 / 84 (0.00%)
Unjudged Items Queued:         84 / 84 (100.00%)
Synthetic Fallback Items:      0 (0.00%)
```

---

## 3. Root Cause Analysis & Governance Enforcement

1. **Sub-chunk Canonical Mismatch**: Non-baseline arms (`S0`, `S1`, `W0`, `W1`, `C0`, `C1`) generate dynamic internal chunk identifiers (e.g., `doc_p91_s0`, `node_...`) during sentence/window/hierarchical indexing. These identifiers lack the required `ps_` prefix and fail exact string mapping against `passage_registry.jsonl` (25 canonical passages `ps_001` .. `ps_025`).
2. **Missing Human Qrel Coverage**: Because retrieved item identifiers do not match canonical passage IDs, 0% of retrieved items find matching human ratings in `human_qrels.jsonl`. Under pre-registered fail-closed governance, unjudged items are assigned `UNJUDGED` and queued into `human_review_queue.jsonl` without synthetic fallback ratings or implicit zeros.
3. **Fail-Closed Gate Trigger**: In accordance with `preregistration_v1.json`, because `canonical_coverage < 1.0` and `judged_coverage < 1.0`, execution immediately halts with status `NOT_EVALUABLE_CANONICAL_COVERAGE`.

---

## 4. Definitive Scientific Conclusion

```text
SLICE5A3_OPERATIONAL_COMPLETE
SLICE5A3_NOT_EVALUABLE_CANONICAL_COVERAGE
CATEGORY_D_CONFIRMED
NO_SUPERIORITY_CLAIM
```
