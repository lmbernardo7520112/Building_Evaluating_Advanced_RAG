# Slice 5A.2 — Deterministic Pilot Protocol

## Protocol ID: `slice5a_runtime_pilot_v1`
## Protocol SHA-256: `f035077a23401e0ca90838102d20d7f83ea042408b19d931cbfdff56da911266`
## Registered: BEFORE policy replay observation

## Hypothesis

A deterministic one-shot router selecting per-QID strategies from a frozen
policy can match or exceed the best fixed global strategy on at least one
primary retrieval metric across the answerable subset.

## Null Hypothesis

The deterministic router does not surpass the best fixed global strategy on
any primary retrieval metric across the answerable subset.

## Systems

| ID | Label | Operational | Notes |
|----|-------|-------------|-------|
| B0 | Best fixed global | Yes | Determined by replay |
| B2 | Deterministic router | Yes | Frozen policy v1 |
| A1 | Tool calling one-shot | Yes | Via OneShotRunner |
| O1 | Post-hoc oracle | No | Uses evaluation outcomes |

## Metrics

- **Primary retrieval**: nDCG@3, Recall@3, MRR@3
- **Router**: selection distribution, fallback rate, policy stability
- **Abstention**: correct abstention, false abstention
- **Operation**: logical calls per QID, tool failures
- **Generation**: NOT_EXECUTED

## Denominators

- Answerable: QIDs with ≥1 relevant passage (retrieval metrics)
- Unanswerable: QIDs marked `UNANSWERABLE_IN_SUBCORPUS` (abstention metrics)
- Never mixed.

## Anti-Leakage

- qrels: FORBIDDEN during routing/retrieval; permitted in post-execution analysis
- gold answers: FORBIDDEN everywhere
- holdout: FORBIDDEN
- historical results: FORBIDDEN during routing
- document filtering: DISABLED

## Headroom Results (observed post-registration)

| Metric | Oracle | Best Fixed | Δ |
|--------|--------|------------|---|
| nDCG@3 | 0.7010 | 0.5262 (H0) | +0.1749 |
| Recall@3 | 0.5857 | 0.5024 (W1) | +0.0833 |
| MRR@3 | 1.0000 | 1.0000 (H0) | +0.0000 |

## Limitations

- n=8 is insufficient for formal statistical significance
- Deterministic router only
- Single-document corpus
- Generation NOT_EXECUTED
