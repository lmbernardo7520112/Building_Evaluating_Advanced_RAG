# Gate B2 Reconciliation — Authoritative Report

## State

| Item | Value |
|------|-------|
| Branch | `feat/hybrid-human-validated-eval` |
| Evidence v2 commit | `4b03f8e feat(eval): materialize full retrieval evidence for gate b2` |
| Integration commit | (pending — this commit) |
| Evidence v2 SHA-256 | `22e2aaf324999d8b...` (279 records) |
| Evidence schema | `retrieval_evidence_v2` |
| Pool schema | `3.0.0` |
| Protocol | `raglab_v7_slice4_v3` |

## Evidence v2 Materialization

The evidence v2 artifact was produced by `scripts/materialize_retrieval_evidence_v2.py`
running all 7 retrieval strategies offline against the Gersting PDF (SHA-256 `33e2e9f1...`,
pages 91-115) using cached embeddings. Zero network, zero API, zero credentials.

| Strategy | Records | Type |
|----------|---------|------|
| F0_baseline | 24 | 8 x TOP_K=3 |
| S0_sentence_anchor | 24 | Anchor sentences |
| W0_sentence_window | 24 | Window text (window_size=3) |
| W1_sentence_window_rerank | 80 | 3 final + 7 dropped per question |
| H0_hierarchical_leaf | 24 | Leaf nodes |
| H1_auto_merging | 24 | Auto-merged nodes |
| H2_auto_merging_rerank | 79 | 3 final + ~7 dropped per question |
| **Total** | **279** | |

## Experimental Unit

| Property | Value |
|----------|-------|
| canonical_registry_entry_count | 25 |
| canonical_evaluation_unit | PAGE_LEVEL |
| raw_retrieval_unit | sub-page chunks, sentences, windows |
| generation_context_unit | retriever output |

The 25 canonical evaluation units are the 25 pages (91-115) in the passage
registry, **not** 123 sub-page passages.

## Pool Reconstruction

| Metric | Value |
|--------|-------|
| Main canonical pool items | 53 |
| Outside pool audit items | 16 |
| Queue A total | 69 |
| Pool intersection Outside | empty (disjoint) |
| Multisystem families | 4 (hierarchical, sentence_anchor, sentence_window, standard_chunking) |
| Strategies present | 7 |

## Accounting Identity (279 Records)

```
279 = 168 final + 111 dropped + 0 invalid
```

| Category | Count |
|----------|-------|
| raw_candidates_final | 168 |
| raw_candidates_dropped_by_reranker | 111 |
| invalid_records | 0 |
| unique_raw_candidates | 168 |
| canonical_pool_items | 53 |
| outside_pool_audit_items | 16 |

## Queue A Union

```
queue_a_total = 69 = main_canonical_pool (53) + outside_pool_audit (16)
intersection_count = 0
```

### Old Discrepancy Explained

The old accounting showed main=53, outside=72, queue_a=107. This was because
the old builder used a different select_outside_pool_audit_sample function
with max(10, ceil(0.10 * count)) which sampled from all 25 pages per question.
The new builder samples from the complement (registry minus pool) which is
typically ~18 pages, yielding ~2 outside items per question.

## Human Queues

| Queue | Count | Status |
|-------|-------|--------|
| Annotator A | 69 | PENDING |
| Annotator B | 16 | PENDING |
| Adjudication | 16 | PENDING_HUMAN_ANNOTATIONS |
| Queue status | PROVISIONAL_WITHOUT_SILVER | |
| Blinding verified | yes | |
| Overlap rate | 23.19% | Within 15-25% target |

## Test Suite

| Metric | Value |
|--------|-------|
| Baseline (1280744) | 741 |
| Current HEAD | 772 |
| Net change | +31 |
| MISSING_BLOCKING | 0 |

See gate_b2_test_invariant_matrix.md for the full invariant replacement table.

## QA

| Check | Result |
|-------|--------|
| Evidence v2 validation | PASSED (0 errors, 0 warnings) |
| Pool builder | PASSED |
| Queue builder | PASSED |
| Determinism (2nd run SHA match) | DETERMINISTIC |
| Full test suite | 772 passed |
| Holdout sealed | yes |
| No ground truth leak | yes |
| Blinding verified | yes |

## Limitations

1. No silver triage executed. Queue status is PROVISIONAL_WITHOUT_SILVER.
2. No human annotations. All relevance grades are null.
3. No gold answers. Ground truth v2 gold answers are not yet authored.
4. No Gemini. No LLM was called during this integration.
5. Outside pool sample is small (16 items across 8 questions = ~2 per question)
   because the canonical pool already covers most pages per question.
6. Reranker damage assessment not yet complete. The dropped candidates
   are cataloged but not yet evaluated for relevance.

## Declarations

```
FULL_RETRIEVAL_EVIDENCE_MATERIALIZED
POOL_CONSUMES_EVIDENCE_V2
EVALUATION_UNIT_EXPLICIT
RERANKER_DAMAGE_CANDIDATES_PRESERVED
QUEUE_ACCOUNTING_CLOSED
TEST_INVARIANTS_PRESERVED
MOCK_SILVER_ISOLATED
HOLDOUT_SEALED
OFFLINE_QA_PASSED
GATE_B2_OPERATIONAL_READY
READY_FOR_CONTROLLED_SILVER_TRIAGE
```
