# Threat Model — Agentic Track

## Scope

This threat model covers the Slice 5A governed routing and tool-calling
foundation. It does NOT cover Gemini/LLM execution (not present in 5A).

## Threat Categories

### T1 — Evaluation Leakage

| Threat | Mitigation | Status |
|--------|-----------|--------|
| Router accesses qrels | Regex filter on router features | ✅ Tested |
| Router accesses gold answers | Regex filter + validation | ✅ Tested |
| Router accesses holdout | Regex filter + validation | ✅ Tested |
| Tool filters by relevant pages | Forbidden filter patterns | ✅ Tested |
| Router uses historical QID performance | No QID lookup in classification | ✅ By design |

### T2 — Information Flow Violations

| Threat | Mitigation | Status |
|--------|-----------|--------|
| Chain-of-thought persisted | Field sanitizer on ledger | ✅ Tested |
| Credentials in trajectory | Forbidden field check | ✅ Tested |
| Prompts leaked to ledger | No prompt field in contracts | ✅ By design |
| Full corpus in observation | IDs + hashes only, no text | ✅ By design |

### T3 — Tool Safety

| Threat | Mitigation | Status |
|--------|-----------|--------|
| Write tool registered | ToolSpecification rejects | ✅ Tested |
| Network tool registered | ToolSpecification rejects | ✅ Tested |
| Unknown tool executed | UnknownToolError | ✅ Tested |
| Invalid arguments accepted | ToolExecutor validation | ✅ Tested |
| Non-canonical IDs accepted | ToolObservation validation | ✅ Tested |

### T4 — Budget Attacks

| Threat | Mitigation | Status |
|--------|-----------|--------|
| Unbounded tool calls | Budget.max_logical_calls | ✅ Tested |
| Budget bypass | Executor checks before execution | ✅ Tested |
| Budget rollback | Consumption is monotonic | ✅ By design |

### T5 — Prompt Injection

| Threat | Mitigation | Status |
|--------|-----------|--------|
| Injection in retrieved text | Observations carry IDs/hashes, not text | ✅ By design |
| Injection alters budget | Budget cannot be modified by data | ✅ Tested |
| Injection overrides router | Router uses only public query text | ✅ By design |

### T6 — Reproducibility

| Threat | Mitigation | Status |
|--------|-----------|--------|
| Non-deterministic routing | SHA-256 policy hash | ✅ Tested |
| Non-deterministic hashing | Canonical JSON (sort_keys) | ✅ Tested |
| Ledger modification | Append-only + hash chain | ✅ Tested |
| Scientific artifact mutation | SHA-256 immutability matrix | ✅ Pre/post check |

## Residual Risks

1. **LLM Router (5B+)**: When the LLM router is activated, model
   non-determinism introduces routing variance. Mitigated by:
   - Temperature=0
   - Structured output schema
   - Validation against governance rules
   - Fallback to deterministic router on failure

2. **Side-channel timing**: Tool execution latency may leak information
   about document structure. Accepted risk for Slice 5A.

3. **Sample size**: `n=8` prevents reliable statistical inference.
   Mitigated by oracle ceiling labeling and regret analysis.
