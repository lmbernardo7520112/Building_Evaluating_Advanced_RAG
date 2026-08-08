# Slice 5A.2 — Runtime Integration

## Status: IMPLEMENTED

## Objective

Connect the agentic domain (contracts, router, runner from Slice 5A.1) to the
existing seven fixed-strategy retrieval pipelines via read-only port/adapters,
enabling a deterministic one-shot pilot.

## Headroom Gate

Policy replay over the Slice 4 Full result (n=8) demonstrated:

- **Oracle nDCG@3 = 0.7010** vs. best fixed (H0) = 0.5262 → +0.1749 headroom
- **5 distinct strategy winners** across QIDs (F0, H0, H2, S0, W1)
- **147/147 COMPUTED** metrics, 0 NA
- **Verdict: `ROUTING_HEADROOM_PRESENT`**

## Architecture

```
Query → DeterministicRouter → ToolAuthorization → RetrievalToolAdapter
        (frozen policy)       (ToolExecutor)      (wraps RetrievalPort)
                                                        ↓
                                                  ToolObservation
                                                        ↓
                                               OneShotRunner result
```

### Modules

| Module | Purpose |
|--------|---------|
| `runtime/retrieval_tool_adapter.py` | Bridge: `RetrievalBackend` protocol → concrete `RetrievalPort` |
| `runtime/strategy_tool_factory.py` | Maps `PipelineStrategy` → `ToolSpecification` + adapter |
| `runtime/deterministic_pilot.py` | Composes `OneShotRunner` with real adapters for end-to-end pilot |

### Key Design Decisions

1. **Port/Adapter boundary**: The agentic layer never imports infrastructure
   modules directly. All retrieval goes through the `RetrievalPort` protocol.

2. **Existing strategies are not reconstructed**: W1, H1, H2 etc. are accessed
   via their existing factories/ports, not rebuilt inside the agentic layer.

3. **One decision, one call**: Each QID gets exactly one routing decision and
   one logical retrieval call. No iterative refinement.

4. **qrels are post-execution only**: During routing and retrieval, no qrels,
   gold answers, or historical results are accessible.

5. **Generation is NOT_EXECUTED**: No LLM generation in this task. The field
   is explicitly `NOT_EXECUTED`, never `0`.

## Limitations

- n=8 is insufficient for statistical significance
- Deterministic router only (no LLM router)
- Single-document corpus
- No generation evaluation
