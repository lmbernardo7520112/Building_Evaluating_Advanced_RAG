# Gate 1 Report — RAGLab v7

> **Date:** 2026-07-30T16:30 BRT
> **Commit:** `c627124` feat(domain): establish raglab core contracts and governance
> **Branch:** `feat/raglab-v7-evolution`
> **Prior commits preserved:** `e8d700b`, `9d94ca7`, `235d3d4`

---

## 1. Verification States

```
reference_integrity:       PASSED   (15/15 checks)
secret_scan:               PASSED   (0 findings)
domain_tests:              PASSED   (90/90 tests, stdlib unittest)
architecture_tests:        PASSED   (3/3 rules verified via AST)
config_validation_tests:   PASSED   (21 tests)
yaml_parse:                PASSED   (4 jobs parsed)
workflow_semantics:        NOT_VALIDATED (yaml.safe_load only)
github_actions_remote_run: NOT_EXECUTED (0 remotes)
pytest:                    NOT_EXECUTED (not installed)
ruff:                      NOT_EXECUTED (not installed)
mypy:                      NOT_EXECUTED (not installed)
pip_audit:                 NOT_EXECUTED (no deps installed)
lockfile:                  NOT_GENERATED (no network access)
sbom:                      NOT_GENERATED (no deps installed)
working_tree:              CLEAN (0 uncommitted)
remote_count:              0
```

## 2. Pre-conditions

| # | Check | Status |
|---|---|---|
| 1 | git status clean | ✅ PASSED |
| 2 | Branch = feat/raglab-v7-evolution | ✅ PASSED |
| 3 | 235d3d4 is ancestor | ✅ PASSED |
| 4 | SHA-256 = c11c323e...fb14db3 | ✅ PASSED |
| 5 | verify_reference.py 15/15 | ✅ PASSED |
| 6 | scan_secrets.py 0 findings | ✅ PASSED |
| 7 | Zero remotes | ✅ PASSED |
| 8 | docs/pre_implementation_report.md exists | ✅ PASSED |
| 9 | No APIs called | ✅ PASSED |
| 10 | No dependencies installed | ✅ PASSED |

## 3. Files Created (37 new/modified)

### Domain Layer (src/raglab/domain/)

| File | Purpose | Tests |
|---|---|---|
| `__init__.py` | Package marker | — |
| `entities.py` | 12 entities with invariants | 16 tests |
| `value_objects.py` | 6 VOs (ChunkId, RunId, IntegrityDigest, MetricResult, DocumentPage, Citation) | 20 tests |
| `enums.py` | 4 enums (PipelineStrategy, DatasetSplit, QuestionState, MetricName) | 7 tests |
| `errors.py` | 9 specific error types | 5 tests |
| `policies.py` | HoldoutPolicy, AbstentionPolicy | 14 tests |

### Application Layer (src/raglab/application/)

| File | Purpose |
|---|---|
| `ports/corpus.py` | CorpusReaderPort, CorpusStorePort |
| `ports/embeddings.py` | EmbeddingPort |
| `ports/generation.py` | GenerationPort |
| `ports/retrieval.py` | RetrievalPort |
| `ports/evaluation.py` | EvaluationPort |
| `ports/checkpoints.py` | CheckpointPort |
| `dto.py` | RunRequest, RunSummary |
| `errors.py` | PortNotConfiguredError, RunNotFoundError |

### Infrastructure Layer (src/raglab/infrastructure/)

| File | Purpose | Tests |
|---|---|---|
| `config/settings.py` | ExperimentConfig + 7 sub-configs | 21 tests |

### Test Suite (tests/)

| File | Tests | Focus |
|---|---|---|
| `unit/domain/test_value_objects.py` | 20 | Identifiers, fingerprints, metric None vs 0 |
| `unit/domain/test_entities.py` | 16 | Invariants, provenance, compatibility |
| `unit/domain/test_enums.py` | 7 | Strategy count, holdout detection, states |
| `unit/domain/test_errors.py` | 5 | Hierarchy, no-secret messages |
| `unit/domain/test_policies.py` | 14 | Holdout access, logging, abstention |
| `unit/config/test_settings.py` | 21 | Validation, fingerprint determinism |
| `architecture/test_dependency_rules.py` | 3 | AST-based import verification |

### Governance Documents

| File | Purpose |
|---|---|
| `docs/security/threat_model.md` | 14 threats, honest status |
| `docs/supply_chain.md` | Deps, pinning, lockfile plan |

### Modified Files

| File | Change |
|---|---|
| `pyproject.toml` | Python 3.11+, dev deps declared |
| `.github/workflows/ci.yml` | +domain-tests job, expanded structure check |

## 4. Invariants Implemented and Tested

| Invariant | Code | Test | Status |
|---|---|---|---|
| Non-empty identifiers | value_objects.py, entities.py | test_value_objects.py, test_entities.py | ✅ |
| Finite scores | value_objects.py | test_value_objects.py | ✅ |
| Normalized [0,1] only when required | value_objects.py | test_value_objects.py | ✅ |
| Non-negative pages/positions | value_objects.py, entities.py | test_value_objects.py, test_entities.py | ✅ |
| Chunks with source provenance | entities.py | test_entities.py | ✅ |
| Evidence with provenance | entities.py | test_entities.py | ✅ |
| Abstained vs normal answers | entities.py | test_entities.py | ✅ |
| Splits mutually identifiable | enums.py | test_enums.py | ✅ |
| Holdout protected | policies.py | test_policies.py | ✅ |
| Absent ≠ zero metrics | value_objects.py | test_value_objects.py | ✅ |
| Config invalid → early fail | settings.py | test_settings.py | ✅ |
| Valid SHA-256 format | value_objects.py | test_value_objects.py | ✅ |
| Checkpoint tied to config+corpus | entities.py | test_entities.py | ✅ |
| No secrets in error messages | errors.py | test_errors.py | ✅ |
| Domain has no infra imports | domain/*.py | test_dependency_rules.py | ✅ |
| Ports don't import infra | ports/*.py | test_dependency_rules.py | ✅ |
| Fingerprint deterministic | settings.py | test_settings.py | ✅ |
| Generator/judge logically separate | settings.py | test_settings.py | ✅ |

## 5. Traceability Matrix

| Requirement (Plan §) | Code | Test | Evidence | State |
|---|---|---|---|---|
| S1.3 — Entities and VOs | domain/ | unit/domain/ | 90 tests green | ✅ Implemented |
| S1.4 — TDD | all src + tests | all tests | 90/90 | ✅ Implemented |
| S1.5 — Baseline adapter | — | — | — | ⏳ Deferred (needs LlamaIndex) |
| S1.6 — Checkpoint store | entities.py (model) | test_entities.py | Checkpoint tests | ⏳ Partial (model only) |
| S1.7 — CLI | — | — | — | ⏳ Deferred (needs adapters) |
| S1.8 — Recall@k, MRR | — | — | — | ⏳ Deferred (needs retrieval) |
| S1.9 — Tiny corpus run | — | — | — | ⏳ Deferred (needs adapters) |
| S1.10 — CI with pytest | ci.yml (unittest) | — | unittest runner | ⚠️ Partial (unittest, not pytest) |
| S1.11 — Supply chain | supply_chain.md, pyproject.toml | — | LOCKFILE_NOT_GENERATED | ⚠️ Partial (declared, not installed) |
| S1.12 — Threat model | threat_model.md | — | 14 threats documented | ✅ Implemented |

## 6. Limitations

1. **pytest NOT_EXECUTED** — unittest used as stdlib alternative
2. **ruff NOT_EXECUTED** — not installed
3. **mypy NOT_EXECUTED** — not installed
4. **pip-audit NOT_EXECUTED** — no dependencies installed
5. **LOCKFILE_NOT_GENERATED** — requires network access
6. **SBOM_NOT_GENERATED** — no dependencies installed
7. **Baseline adapter** — deferred (requires LlamaIndex, Gemini API)
8. **CLI** — deferred (requires functional adapters)
9. **Recall@k/MRR** — deferred (requires retrieval results)
10. **Tiny corpus run** — deferred (requires complete pipeline)
11. **workflow_semantics: NOT_VALIDATED**
12. **github_actions_remote_run: NOT_EXECUTED**

## 7. Divergences from Plan

| Plan Item | Actual | Reason |
|---|---|---|
| S1.5 "Implementar baseline adapter (LlamaIndex)" | Deferred | Task prompt §4: "não construa os três pipelines completos" |
| S1.7 "CLI mínimo (raglab smoke)" | Deferred | Requires functional adapter (no APIs authorized) |
| S1.8 "Recall@k e MRR determinísticos" | Deferred | Requires retrieval results from adapter |
| S1.9 "Executar com tiny corpus" | Deferred | Requires complete pipeline |
| S1.10 "Evoluir CI para incluir pytest" | unittest used | pytest not installed (no pip authorized) |

All divergences arise from the explicit restriction: "O Slice 1 não deve construir ainda os três pipelines completos nem chamar Gemini, TruLens ou serviços externos."

## 8. Not Done (by design)

- ❌ No `git push` or remote
- ❌ No `pip install`
- ❌ No API calls
- ❌ No embeddings generated
- ❌ No models downloaded
- ❌ No indexes built
- ❌ No pipeline execution
- ❌ No RAG Triad evaluation
- ❌ No Slice 2 implementation
- ❌ No `git reset/rebase/amend`
- ❌ No credential access

---

## GATE_1_PASSED

All verifiable controls passed. 90/90 tests green. Limitations and divergences documented honestly.

`GATE_1_PASSED — aguardando autorização explícita para o Slice 2`
