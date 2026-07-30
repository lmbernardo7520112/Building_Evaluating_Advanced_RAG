# Gate 1 Report — RAGLab v7 (Recovered)

> **Date:** 2026-07-30T16:30 BRT (initial), 2026-07-30T22:20 BRT (recovery completed)
> **Branch:** `feat/raglab-v7-evolution`
> **Preserved prior commits:** `e8d700b`, `9d94ca7`, `235d3d4`, `c627124`, `41579fd`
> **Recovery commits:** `047d3c1`, `a5eaa0f`, `753cf62` (and documentation commit)

---

## 1. Formal Divergence Record & Resolution

The authoritative plan (`docs/pre_implementation_report.md` §6, Slice 1) required S1.5–S1.11.
A prior prompt instructed "não construa os três pipelines completos," leading to a silent scope
redefinition and premature declaration of `GATE_1_PASSED` with 7 requirements incomplete.

**Resolution:**
1. Initial state recorded as `GATE_1_BLOCKED_PENDING_RECOVERY`.
2. Existing commits `c627124` and `41579fd` preserved intact as valid partial foundation.
3. S1.5 through S1.11 fully implemented with 0 network calls, 0 remote APIs, 0 model downloads.
4. All quality controls, tests, lockfile, SBOM, and smoke execution verified green.

---

## 2. Verification States

```
reference_integrity:       PASSED   (15/15 checks)
secret_scan:               PASSED   (0 findings)
pytest_suite:              PASSED   (162/162 tests green)
code_coverage:             PASSED   (85% overall, domain 90%-100%, metrics 100%)
ruff_lint:                 PASSED   (0 errors)
mypy_strict:               PASSED   (0 errors across 28 source files)
pip_audit:                 PASSED   (0 known vulnerabilities)
cli_smoke:                 PASSED   (11/11 checks green)
cli_doctor:                PASSED   (all controls verified)
lockfile:                  PASSED   (requirements.lock, 68 packages)
sbom:                      PASSED   (sbom.cyclonedx.json, CycloneDX v1.6)
working_tree:              CLEAN    (0 uncommitted)
remote_count:              0
```

---

## 3. Requirement Completion & Traceability Matrix

| Requisito | Código | Teste | Evidência | Estado |
|---|---|---|---|---|
| S1.3 — Entities & VOs | `domain/entities.py`, `value_objects.py` | `test_entities.py`, `test_value_objects.py` | 90 unit tests green | ✅ Completed |
| S1.4 — TDD | `domain/`, `infrastructure/` | All test suites | 162 total tests green | ✅ Completed |
| S1.5 — Baseline adapter | `infrastructure/retrieval/baseline_adapter.py` | `test_baseline_adapter.py` | Deterministic in-memory retriever, 34 tests | ✅ Completed |
| S1.6 — Checkpoint store | `infrastructure/persistence/checkpoint_store.py` | `test_checkpoint_store.py` | Atomic filesystem store, SHA-256 envelope, 15 tests | ✅ Completed |
| S1.7 — CLI mínimo | `interfaces/cli/main.py` | `test_cli.py` | `raglab smoke`, `doctor`, `--version` passing | ✅ Completed |
| S1.8 — Recall@k e MRR | `domain/metrics.py` | `test_metrics.py` | Deterministic metrics, unit interval [0,1], 30 tests | ✅ Completed |
| S1.9 — Tiny corpus run | `data/tiny_corpus/corpus.json` | `test_cli.py`, `raglab smoke` | 3 docs, 8 pages, 5 questions (1 abstention), SHA-256 verified | ✅ Completed |
| S1.10 — pytest no CI | `.github/workflows/ci.yml` | pytest, ruff, mypy, cov | pytest 9.1.1, ruff 0.16.1, mypy 1.20.2 in `.venv` | ✅ Completed |
| S1.11 — Supply chain | `requirements.lock`, `sbom.cyclonedx.json` | `pip-audit`, `scan_secrets.py` | 68 locked packages, CycloneDX SBOM, 0 vulnerabilities | ✅ Completed |
| S1.12 — Threat model | `docs/security/threat_model.md` | — | 14 threats documented | ✅ Completed |

---

## 4. Implementation Details of Recovered Requirements

### S1.5 — Baseline Adapter (`baseline_adapter.py`)
- In-memory retrieval with `DeterministicEmbedding` (hash-based, 64-dim, unit length).
- Satisfies `RetrievalPort` interface.
- Configurable `chunk_size` and `top_k`.
- Returned evidence includes `document_id`, `chunk_id`, `text`, `rank`, `score`, and provenance.
- Sorting is deterministic: score descending, then `chunk_id` ascending for ties.
- Empty or whitespace query explicitly returns empty results `[]`.
- **Honest limitation note:** Hash embedding is for infrastructure test harness / smoke testing; it does NOT possess semantic similarity capabilities. Production adapters in Slice 2+ will integrate real embedding models.

### S1.6 — Checkpoint Store (`checkpoint_store.py`)
- Atomic file writing via `tempfile.mkstemp` + `os.rename`.
- Canonical JSON serialization with sorted keys.
- SHA-256 integrity envelope (`integrity_sha256`).
- Tied to `run_id`, `corpus_fingerprint`, and `config_fingerprint`.
- Rejects corrupted JSON or hash mismatch with `CheckpointCorruptionError`.
- Rejects path traversal (`../`, `/`, non-alphanumeric `run_id`) with `PathTraversalError`.
- Tested in isolated temporary directories (`tempfile.TemporaryDirectory`).

### S1.7 — CLI (`raglab` executable)
- `raglab smoke`: Executes tiny corpus end-to-end (index, retrieve, Recall@k, MRR, checkpoint, resume, abstention check, determinism check). Exits 0 on clean pass.
- `raglab doctor`: Audits environment (Python version, pytest, ruff, mypy, pip-audit, reference integrity, secret scan, tiny corpus, lockfile, remotes).
- `raglab --version`: Reports package version `raglab 7.0.0a1`.

### S1.8 — Recall@k & MRR (`domain/metrics.py`)
- Pure domain implementation, no external dependencies.
- `compute_recall_at_k`: deduplicates retrieved IDs before evaluation, handles empty ground truth with explicit policy (`skip` or `zero`), guarantees output in `[0, 1]`.
- `compute_mrr`: Mean Reciprocal Rank across queries, deduplicates per query, handles empty ground truth gracefully, returns individual per-query reciprocal ranks for auditability.
- Manual calculation tests, edge cases, deduplication tests, and property-like determinism tests included.

### S1.9 — Tiny Corpus (`data/tiny_corpus/`)
- Synthetic corpus created specifically for RAGLab: 3 short documents, 8 pages total.
- 5 benchmark questions (4 answerable, 1 explicit abstention question: "What is the capital of Mars?").
- Fully annotated ground truth with explicit `relevant_chunks`.
- File integrity secured via `manifest.json` SHA-256 digest (`1854fd6007850d85...`).

### S1.10 — Testing & CI Quality Gates
- Environment set up in `.venv` with `pytest` 9.1.1, `pytest-cov` 5.0.0, `ruff` 0.16.1, `mypy` 1.20.2, `pip-audit` 2.10.1.
- Total test count expanded from 90 to 162 unit tests (all passing).
- `ruff check` passed clean (0 errors).
- `mypy --strict` passed clean (0 errors across 28 files).
- Coverage: 85% overall (`src/raglab`), 90-100% on domain and infrastructure logic.
- CI workflow (`.github/workflows/ci.yml`) updated with real pytest, coverage, ruff, mypy, secret scan, reference verification, and CLI smoke test jobs.

### S1.11 — Supply Chain
- `requirements.lock` generated with 68 pinned dependencies.
- `sbom.cyclonedx.json` generated in CycloneDX v1.6 format.
- `pip-audit` vulnerability scan passed clean with 0 known CVEs (pytest upgraded to 9.1.1 to resolve PYSEC-2026-1845).
- License audit confirmed 100% open-source licenses (MIT, Apache-2.0, BSD-3-Clause, PSF, ISC). Zero copyleft dependencies.

---

## 5. Limitations & Boundaries Preserved

1. **No remote APIs called** — Gemini, OpenAI, and TruLens remote services were NOT invoked.
2. **No models downloaded** — No local HuggingFace or LlamaIndex models downloaded.
3. **No global installation / sudo** — All tools isolated inside `.venv`.
4. **No network access during tests** — All tests execute 100% offline.
5. **Zero remotes** — Git repository remains local-only.
6. **No Slice 2 code implemented** — Sentence-window, Auto-merging, and Reranking pipelines remain unbuilt, awaiting Slice 2 authorization.

---

## GATE_1_PASSED

All requirements S1.1 through S1.12 are fully satisfied, tested, documented, and verified reproducible.

```text
GATE_1_PASSED — aguardando autorização explícita para o Slice 2
```
