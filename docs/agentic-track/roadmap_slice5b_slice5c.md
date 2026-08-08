# Roadmap — Slices 5B and 5C

## Current State: Slice 5A (Completed)

- Governed routing foundation
- Deterministic router with text-only classification
- Tool registry (read-only, frozen)
- Evidence accumulator with deduplication
- Budget tracking (separate from stop policy)
- Append-only trajectory ledger
- Policy replay CLI
- LLM router contract (no execution)

## Slice 5B — Tool Calling One-Shot (NOT IMPLEMENTED)

### Scope
- Execute a single governed tool call per query
- Integrate with actual LlamaIndex retrieval adapters via `adapters/llamaindex.py`
- Real evidence from the existing index
- LLM router execution (Gemini, with quota)

### Prerequisites
- Working LlamaIndex retrieval adapters in the current environment
- Gemini API key for router (not for generation in 5B)
- Extended synthetic fixture with expected tool observations

### Non-Goals
- Multi-step loop
- Multiple tool calls per query
- Answer generation

## Slice 5C — Bounded Reasoning Loop (NOT IMPLEMENTED)

### Scope
- Multi-step evidence accumulation (max N steps)
- Stop policy evaluation at each step
- Evidence delta tracking (repeated-no-new detection)
- Full trajectory with state-before/state-after per step
- Regret analysis comparing loop vs one-shot vs fixed

### Prerequisites
- Slice 5B completed and validated
- Extended ground truth for multi-hop questions
- Budget tuning based on 5B observations

### Non-Goals
- Open-ended reasoning
- Multi-agent orchestration
- Self-modifying policies

## Future: L4 — Single Agent with Tool Retrieval

- One agent, multiple tools (retrieval + lookup + summarization)
- Still governed by budget, authorization, and ledger
- NOT multi-agent
- Requires new threat model extension
