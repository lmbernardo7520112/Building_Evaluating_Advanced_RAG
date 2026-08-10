# Slice 5A.3 — Canonical Coverage and Retrieval Parity Gate Architecture

## 1. Overview and Rationale

The **Slice 5A.3 Parity Gate** establishes a governed, fail-closed experimental harness designed to evaluate retrieval parity, passage ID canonicality, and human qrel coverage across all 7 frozen retrieval strategies in RAGLab v7:
- **F0**: InMemory Baseline
- **S0**: SentenceAnchor
- **S1**: SentenceAnchor + LocalReranker
- **W0**: SentenceWindow
- **W1**: SentenceWindow + LocalReranker
- **C0**: AutoMerging (Hierarchical)
- **C1**: AutoMerging + LocalReranker

## 2. Core Governance Invariants

1. **100% Canonical Passage Coverage**: Every retrieved item must map to a valid `ps_*` ID from the single canonical corpus snapshot (`benchmarks/ground_truth/v2/passage_registry.jsonl`). Zero synthetic fallback IDs are admitted.
2. **Explicit Human Judgment Coverage**: Every retrieved item must possess an explicit human relevance rating in `human_qrels.jsonl`. If any retrieved item is missing a qrel, it is assigned `UNJUDGED` (never converted to 0.0) and triggers `NOT_EVALUABLE_JUDGED_COVERAGE`.
3. **Same-Run Materialization**: All 7 arms and the deterministic routing policy A1 are materialized in the same execution run. Operational oracle metrics are computed retrospectively across same-run materialized arms without historical O1 reuse.
4. **Deterministic Repeatability**: Dual execution runs under identical corpus and configuration snapshots must produce 100% identical top-k passage ID and rank alignments.
5. **Sealed TEST Split**: Evaluation is strictly restricted to DEV QIDs (`q_dev_01`, `q_dev_02`, `q_dev_03`, `q_dev_04`). TEST split questions remain 100% sealed and unexecuted.

## 3. CLI Interface

```bash
.venv/bin/python scripts/run_slice5a3_parity_gate.py \
  --protocol benchmarks/agentic/slice5/slice5a3/protocols/preregistration_v1.json \
  --qrels benchmarks/ground_truth/v2/hybrid/qrels/human_qrels.jsonl \
  --qrels-manifest benchmarks/ground_truth/v2/hybrid/qrels/human_qrels_manifest.json \
  --passage-registry benchmarks/ground_truth/v2/passage_registry.jsonl \
  --passage-registry-manifest benchmarks/ground_truth/v2/passage_registry_manifest.json \
  --questions-file benchmarks/questions/controlled_chapter2.json \
  --output-dir /tmp/raglab_slice5a3_dev_run1 \
  --offline
```
