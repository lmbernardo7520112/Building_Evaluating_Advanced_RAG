# Threat Model — RAGLab v7

> **Version:** S1 (initial)
> **Created:** 2026-07-30
> **Review schedule:** S3 (post-pipeline integration), S6 (final consolidation)
> **Status:** DRAFT — mitigations are planned, NOT all implemented

---

## Assets

| Asset | Sensitivity | Custodian |
|---|---|---|
| v6.1 notebook (reference) | High — intellectual property | Git + verify_reference.py |
| Corpus PDF(s) | Medium — copyrighted content | Local filesystem |
| Ground truth annotations | High — experiment integrity | Git (after creation) |
| Holdout queries | Critical — confirmatory validity | HoldoutPolicy |
| API credentials (Gemini) | Critical — cost and access | Environment variables only |
| Checkpoints | Medium — reproducibility | Local filesystem |
| Generated answers | Low — reproducible | Checkpoint store |
| Evaluation results | Medium — experiment integrity | Checkpoint store |
| Model manifests | Medium — reproducibility | Git |
| Experiment configuration | Low | Git (no secrets) |

## Trust Boundaries

```
┌────────────────────────────────────────────────┐
│ Local Execution Environment                     │
│  ┌──────────────┐  ┌──────────────────────────┐│
│  │ Domain Layer  │  │ Infrastructure Layer     ││
│  │ (trusted)     │  │ (adapters to external)   ││
│  └──────────────┘  └──────────┬───────────────┘│
│                                │                │
└────────────────────────────────┼────────────────┘
                                 │ Trust boundary
                    ┌────────────▼────────────────┐
                    │ External Services            │
                    │ Gemini API, HuggingFace      │
                    │ PyPI, PDF parsers             │
                    └──────────────────────────────┘
```

## Actors

| Actor | Trust Level | Access |
|---|---|---|
| Developer (human) | Trusted | Full |
| AI agent (Antigravity) | Constrained | Per authorization.md |
| Gemini API | Semi-trusted | API calls only |
| PyPI packages | Untrusted until verified | install-time |
| Corpus PDFs | Untrusted content | parse-time |
| End users of results | Consumers | Read-only |

## Threats

### T01 — Prompt Injection via Corpus

| Field | Value |
|---|---|
| **Scenario** | Malicious text in PDF corpus causes LLM to ignore instructions, exfiltrate data, or produce biased answers |
| **Impact** | High — compromised answers, data exfiltration |
| **Probability** | Medium — requires adversarial corpus |
| **Mitigation (planned)** | Input sanitization, prompt hardening (QA_PROMPT with defenses from v6.1 F21), output validation |
| **Residual risk** | Medium — LLMs remain vulnerable to novel injection |
| **Slice** | S3 (security tests) |
| **Evidence** | test_prompt_injection.py (S3) |

### T02 — Malicious/Malformed PDF

| Field | Value |
|---|---|
| **Scenario** | Crafted PDF exploits parser vulnerability (buffer overflow, XXE, path traversal) |
| **Impact** | High — code execution, file access |
| **Probability** | Low — requires targeted attack |
| **Mitigation (planned)** | Sandboxed parsing, file size limits, extension whitelist |
| **Residual risk** | Low |
| **Slice** | S1 (config limits), S3 (tests) |
| **Evidence** | File size validation in config |

### T03 — Secret Exfiltration

| Field | Value |
|---|---|
| **Scenario** | API keys leaked in logs, error messages, checkpoints, or committed files |
| **Impact** | Critical — unauthorized API access, cost |
| **Probability** | Medium — common in development |
| **Mitigation (implemented S0)** | scan_secrets.py, .gitignore, error messages without values, env-only credentials |
| **Residual risk** | Low |
| **Slice** | S0 (scanner), S1 (error design) |
| **Evidence** | scan_secrets.py PASSED, test_errors.py (no secrets in messages) |

### T04 — Path Traversal

| Field | Value |
|---|---|
| **Scenario** | Malicious corpus path escapes intended directory |
| **Impact** | Medium — read/write arbitrary files |
| **Probability** | Low |
| **Mitigation (planned)** | Path canonicalization, directory confinement |
| **Residual risk** | Low |
| **Slice** | S1 (config validation), S3 (tests) |
| **Evidence** | Config validates paths (planned) |

### T05 — Dependency Confusion / Typosquatting

| Field | Value |
|---|---|
| **Scenario** | Malicious package substituted via typosquatting or dependency confusion |
| **Impact** | Critical — arbitrary code execution |
| **Probability** | Low — requires supply chain attack |
| **Mitigation (planned)** | Pinned versions, hash verification, lockfile, pip-audit |
| **Residual risk** | Low with mitigation |
| **Slice** | S1 (pinning strategy), S4 (vulnerability scan) |
| **Evidence** | pyproject.toml with pinned deps (S1), pip-audit (S4) |

### T06 — Corpus Poisoning

| Field | Value |
|---|---|
| **Scenario** | Adversary modifies corpus to bias retrieval or answers |
| **Impact** | High — experiment integrity compromised |
| **Probability** | Low — requires write access |
| **Mitigation (implemented)** | Corpus fingerprinting (IntegrityDigest), source_manifest.json |
| **Residual risk** | Low |
| **Slice** | S0 (manifests), S1 (domain model) |
| **Evidence** | IntegrityDigest tests, verify_reference.py |

### T07 — Holdout Manipulation

| Field | Value |
|---|---|
| **Scenario** | Developer accidentally or deliberately uses holdout data during development, invalidating confirmatory results |
| **Impact** | Critical — experiment validity destroyed |
| **Probability** | Medium — common mistake |
| **Mitigation (implemented S1)** | HoldoutPolicy with auditable logging, access denied by default |
| **Residual risk** | Medium — policy is programmatic, not cryptographic |
| **Slice** | S1 (policy + tests) |
| **Evidence** | test_policies.py — holdout denial, authorization, logging |

### T08 — Generator/Judge Contamination

| Field | Value |
|---|---|
| **Scenario** | Same model instance used for both generation and evaluation, creating circular validation |
| **Impact** | High — inflated metrics, unreliable evaluation |
| **Probability** | Medium — easy configuration mistake |
| **Mitigation (implemented S1)** | Separate ModelConfig for generator and judge in ExperimentConfig |
| **Residual risk** | Medium — same model_id is allowed (warning only) |
| **Slice** | S1 (config), S4 (TruLens adapter) |
| **Evidence** | ExperimentConfig separates generator/judge |

### T09 — Checkpoint Tampering

| Field | Value |
|---|---|
| **Scenario** | Checkpoint modified to skip evaluations or inject fake results |
| **Impact** | High — undetectable result manipulation |
| **Probability** | Low — requires filesystem access |
| **Mitigation (planned)** | Checkpoint tied to config+corpus fingerprints, integrity validation |
| **Residual risk** | Medium — no signature, only fingerprint match |
| **Slice** | S1 (model), S4 (implementation) |
| **Evidence** | Checkpoint.is_compatible() tests |

### T10 — Denial of Service via API Cost

| Field | Value |
|---|---|
| **Scenario** | Runaway evaluation loop exhausts API quota or budget |
| **Impact** | Medium — financial cost, blocked development |
| **Probability** | Medium — quota is limited |
| **Mitigation (planned)** | QuotaConfig limits, circuit breaker, budget tracking |
| **Residual risk** | Low with mitigation |
| **Slice** | S1 (config), S4 (circuit breaker) |
| **Evidence** | QuotaConfig validation tests |

### T11 — PII Exposure in Logs

| Field | Value |
|---|---|
| **Scenario** | Corpus text, user queries, or answers logged containing PII |
| **Impact** | Medium — privacy violation |
| **Probability** | Medium — academic corpus may contain names |
| **Mitigation (planned)** | Structured logging, redaction policy |
| **Residual risk** | Medium |
| **Slice** | S4 (telemetry adapter) |
| **Evidence** | Logging policy (planned) |

### T12 — Serialization / Deserialization Risks

| Field | Value |
|---|---|
| **Scenario** | Pickle or unsafe deserialization of checkpoints/indices |
| **Impact** | Critical — arbitrary code execution |
| **Probability** | Low — requires malicious checkpoint |
| **Mitigation (planned)** | JSON-only serialization, no pickle, no eval() |
| **Residual risk** | Low |
| **Slice** | S4 (checkpoint store) |
| **Evidence** | Implementation review (S4) |

### T13 — Excessively Large Files

| Field | Value |
|---|---|
| **Scenario** | PDF bomb or extremely large corpus exhausts memory/disk |
| **Impact** | Medium — denial of service |
| **Probability** | Low |
| **Mitigation (planned)** | File size limits in config, streaming parsers |
| **Residual risk** | Low |
| **Slice** | S1 (config), S2 (parsers) |
| **Evidence** | Config validation |

### T14 — Sensitive Logs in Error Messages

| Field | Value |
|---|---|
| **Scenario** | Exception handlers include API keys, tokens, or credential values in error messages |
| **Impact** | High — credential exposure |
| **Probability** | Medium — common coding mistake |
| **Mitigation (implemented S1)** | Domain errors never include actual values, ConfigurationError redacts values |
| **Residual risk** | Low |
| **Slice** | S1 (error design + tests) |
| **Evidence** | test_errors.py — TestErrorMessagesNoSecrets |

---

## Review Schedule

| Phase | Scope | Trigger |
|---|---|---|
| S1 | Initial threat model (this document) | Gate 1 |
| S3 | Review after pipeline integration and persistence | Gate 3 |
| S6 | Consolidation and final acceptance | Gate 6+ |
