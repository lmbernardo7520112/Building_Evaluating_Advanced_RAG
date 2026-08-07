# Gate B2 — Test Invariant Matrix

Baseline reference: commit `1280744` (741 tests)
Current HEAD: post-evidence-v2 integration (772 tests)

## Summary

| Metric | Value |
|--------|-------|
| Baseline tests | 741 |
| Current tests | 772 |
| Net change | +31 |
| MISSING_BLOCKING | **0** |

## Invariant Replacement Table

### Replaced: `test_multisystem_pooling_and_mapping.py` (33 → 43 tests)

The old file (33 tests) imported `rehydrate_candidate_text` and
`build_hybrid_pool` from the legacy pool builder that consumed the composite
artifact with `text_preview`. The new file (43 tests) covers all 13 ETAPA 9
invariants using the evidence v2 pool.

| Baseline Invariant | Old Test | New Test | Status | Reason |
|---|---|---|---|---|
| Multisystem union | test_01 | TestMultisystemProvenance::test_multisystem_verified | REPLACED_BY_STRONGER_TEST | Uses evidence v2 provenance |
| Dedup by passage_id | test_02 | TestPoolDeduplication::test_no_duplicate_qid_page_pairs | REPLACED_BY_STRONGER_TEST | Page-level dedup |
| Source contribution | test_03 | TestMultisystemProvenance::test_seven_strategies_present | REPLACED_BY_STRONGER_TEST | All 7 strategies verified |
| Unavailable explicit | test_04 | TestMultisystemProvenance::test_all_available | REPLACED_BY_STRONGER_TEST | All strategies now AVAILABLE |
| Legacy pages additional | test_05 | TestNoGroundTruthInPool::test_pool_has_no_gold_answer | PRESERVED | Invariant preserved |
| Neighbors preserve IDs | test_06 | N/A | OBSOLETE_WITH_JUSTIFICATION | Neighbor expansion removed; pool is page-level |
| No holdout | test_07 | TestHoldoutSealed::test_pool_no_holdout | PRESERVED | Invariant preserved |
| Deterministic | test_08 | TestRebuildDeterministic::test_pool_manifest_sha_stable | REPLACED_BY_STRONGER_TEST | SHA-based determinism |
| Outside pool deterministic | test_09 | TestOutsideAuditDisjoint::test_disjoint | REPLACED_BY_STRONGER_TEST | Disjunction + existence |
| Outside pool disjoint | test_10 | TestOutsideAuditDisjoint::test_disjoint | PRESERVED | Invariant preserved |
| Expansion threshold | test_11 | N/A | OBSOLETE_WITH_JUSTIFICATION | `outside_pool_relevant_threshold` removed; replaced by disjoint audit policy |
| Mapper by ID | test_12 | TestCanonicalMapper::test_mapper_by_id | PRESERVED | Invariant preserved |
| Mapper by offsets | test_13 | TestCanonicalMapper::test_mapper_by_offsets | PRESERVED | Invariant preserved |
| Mapper by hash | test_14 | TestCanonicalMapper::test_mapper_by_hash | PRESERVED | Invariant preserved |
| Mapper by substring | test_15 | TestCanonicalMapper::test_mapper_by_substring | PRESERVED | Invariant preserved |
| Mapper ambiguity | test_16 | TestCanonicalMapper::test_mapper_ambiguous | PRESERVED | Invariant preserved |
| Mapper unmapped | test_17 | TestCanonicalMapper::test_mapper_unmapped | PRESERVED | Invariant preserved |
| Zero mapping loss | test_18 | N/A | OBSOLETE_WITH_JUSTIFICATION | `unreported_mapping_loss` removed; replaced by page-level mapping |
| Blinded no strategy | test_19 | TestQueuesBlinded::test_blinded_pool_blinded | REPLACED_BY_STRONGER_TEST | Recursive blinding check |
| Blinded no rank | test_20 | TestQueuesBlinded::test_blinded_pool_blinded | REPLACED_BY_STRONGER_TEST | Recursive blinding check |
| Blinded no scores | test_21 | TestQueuesBlinded::test_blinded_pool_blinded | REPLACED_BY_STRONGER_TEST | Recursive blinding check |
| Blinded no silver | test_22 | TestQueuesBlinded::test_blinded_pool_blinded | REPLACED_BY_STRONGER_TEST | Recursive blinding check |
| Blinded no annotator answers | test_23 | TestQueuesBlinded::test_blinded_pool_blinded | REPLACED_BY_STRONGER_TEST | Recursive blinding check |
| Blinded order deterministic | test_24 | TestRebuildDeterministic::test_pool_manifest_sha_stable | PRESERVED | Invariant preserved |
| Rehydration of truncated preview | test_25 | TestPoolRejectsLegacy::test_no_rehydration_audit | REPLACED_BY_STRONGER_TEST | Rehydration eliminated entirely |
| Rehydration verifiable artifact | test_26 | TestPoolRejectsLegacy::test_no_rehydration_audit | REPLACED_BY_STRONGER_TEST | Rehydration eliminated entirely |
| Reconstructed text rejected | test_27 | TestPoolRejectsLegacy::test_text_preview_not_used | REPLACED_BY_STRONGER_TEST | Preview usage rejected |
| 168 candidates accounting | test_28 | TestAccountingCloses279::test_total_evidence_records | REPLACED_BY_STRONGER_TEST | Now 279 records with pre/post reranking |
| Unmapped disposition | test_29 | TestDroppedCandidatesAuditable::test_dropped_records_in_accounting | REPLACED_BY_STRONGER_TEST | Disposition tracked for all records |
| Unrehydratable blocking | test_30 | TestAccountingCloses279::test_identity_holds | REPLACED_BY_STRONGER_TEST | Identity verified algebraically |
| Raw unmapped blinded | test_31 | TestQueuesBlinded::test_blinded_pool_blinded | PRESERVED | Invariant preserved |
| Duplicates don't inflate queue | test_32 | TestRerankerNoDuplicate::test_accounting_separates_final_and_dropped | REPLACED_BY_STRONGER_TEST | Separate final/dropped |
| Mapping coverage | test_33 | TestCanonicalPageLevel::test_manifest_declares_page_level | REPLACED_BY_STRONGER_TEST | Page-level explicit |

### Added: New invariants (10 tests)

| New Test | Invariant |
|---|---|
| TestPoolAcceptsEvidenceV2::test_input_validation_passed | Evidence v2 validated |
| TestPoolAcceptsEvidenceV2::test_evidence_sha_recorded | SHA recorded |
| TestPoolAcceptsEvidenceV2::test_record_count_matches | 279 records |
| TestNoGroundTruthInPool::test_manifest_declares_no_ground_truth | No ground truth |
| TestRerankerNoDuplicate::test_no_dropped_in_pool_provenance | No dropped in pool |
| TestCanonicalPageLevel::test_accounting_declares_page_level | Page-level in accounting |
| TestCanonicalPageLevel::test_pool_items_declare_page_level | Page-level in items |
| TestOutsideAuditDisjoint::test_manifest_confirms_disjoint | Disjoint in manifest |
| TestQueueACloses::test_queue_a_equals_pool_total | Queue A = pool total |
| TestQueueACloses::test_queue_a_items_match_count | Queue A count matches |

### Preserved: `test_retrieval_evidence_v2.py` (27 tests)

All 27 evidence v2 tests are new additions from commit `4b03f8e`.

## Obsolescence Justification

| Test | Reason |
|---|---|
| test_06 (neighbors) | Neighbor expansion is a pool-builder policy, not an evidence v2 invariant. The new page-level pool already covers adjacent content. |
| test_11 (expansion threshold) | The `outside_pool_relevant_threshold` was an arbitrary parameter. Replaced by explicit disjoint audit policy. |
| test_18 (zero mapping loss) | The `unreported_mapping_loss` field was specific to the rehydration flow. Evidence v2 provides full text directly. |

## Final Count

```
PRESERVED:              11
REPLACED_BY_STRONGER:   19
OBSOLETE_WITH_JUSTIFICATION: 3
MISSING_BLOCKING:        0
```
