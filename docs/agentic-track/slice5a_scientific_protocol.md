# Slice 5A — Scientific Protocol

## Research Question

> In which query classes, under which costs and with which risks, does a
> dynamic routing policy surpass, complement, or worsen fixed RAG strategies?

## Experimental Design

### Control Group
- Slice 4/v5 results with human qrels (8 queries, 7 strategies)
- **IMMUTABLE** — no modifications permitted

### Experimental Condition
- Deterministic routing policy (`deterministic_v1`)
- Selects one of seven strategies per query based on public text features only
- No external model execution (Slice 5A)

### Metrics
- nDCG@3 (primary)
- Recall@3
- MRR@3

### Analysis Methods
1. **Strategy summary**: Average metrics per strategy
2. **Fixed-best baseline**: Strategy with highest average primary metric
3. **Router performance**: Average metrics under routing policy
4. **Post-hoc oracle**: Per-query best strategy (CEILING ONLY)
5. **Regret**: Oracle − Router (lower is better)
6. **Win/Tie/Loss**: Per-query comparison vs fixed-best

## Constraints

| Constraint | Status |
|-----------|--------|
| No Gemini execution | ✅ Enforced |
| No new dependencies | ✅ Enforced |
| No Slice 4 mutation | ✅ Verified by SHA-256 |
| No network in tools | ✅ Enforced by ToolSpecification |
| No LlamaIndex in domain | ✅ Verified by AST scan |
| No chain-of-thought in ledger | ✅ Verified by field sanitizer |
| No qrels/gold/holdout leakage | ✅ Verified by regex patterns |

## Statistical Caveat

With `n=8` queries, statistical power is insufficient for general superiority
claims. Results should be interpreted as:
- Directional indicators, not definitive conclusions
- Hypothesis generators for future experiments with larger query sets
- Baseline measurements for Slice 5B/5C comparisons

## Oracle Disclaimer

The post-hoc oracle:
- Uses evaluation outcomes (nDCG, MRR, Recall) that are **unavailable at
  inference time**
- Is labeled `POST_HOC_ORACLE` and `NON_OPERATIONAL`
- Represents a theoretical ceiling, not a deployable policy
- Must NEVER be reported as achievable router performance

## Future Ground Truth Needs

For Slice 5B+ with multi-document retrieval:
- Multi-document hierarchical ground truth required
- Current passage-level qrels may be insufficient for composite questions
- Annotation guidelines need expansion for multi-hop relevance
