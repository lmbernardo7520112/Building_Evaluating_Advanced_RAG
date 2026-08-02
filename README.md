# 🧠 RAGLab v7 — Production-Grade RAG Retrieval & Evaluation Laboratory

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-Retrieval-6C63FF?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Gemini-3.1_Flash_Lite-4285F4?style=for-the-badge&logo=google&logoColor=white)
![FastEmbed](https://img.shields.io/badge/FastEmbed-ONNX-FF6F00?style=for-the-badge&logo=onnx&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-Quality_Gates-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-Clean_%2B_DDD-2E8B57?style=for-the-badge)
![Security](https://img.shields.io/badge/Security-Credential_Isolation-B22222?style=for-the-badge)
![Research](https://img.shields.io/badge/Research-Reproducible-7B1FA2?style=for-the-badge)

> [!IMPORTANT]
> **Engineering-validated, scientifically exploratory.** RAGLab v7 completed a governed 56-pair benchmark across seven retrieval strategies. The experiment demonstrated pipeline integrity, reproducible evidence capture, safe checkpoint recovery, and honest negative-result reporting. It did **not** establish statistically significant superiority of any advanced retrieval method — and the framework is explicitly designed to prevent such unsupported claims.

---

## 📘 Executive Overview

**RAGLab v7** is a production-oriented experimental framework for building, comparing, and auditing Retrieval-Augmented Generation systems.

It evolves a notebook-based RAG course into an engineered research platform that can answer questions many demos avoid:

- Did the retriever actually recover the expected evidence?
- Did sentence-window expansion improve retrieval — or merely add context?
- Did reranking preserve relevant passages or silently discard them?
- Did auto-merging outperform retrieval over hierarchical leaves?
- Is the generated answer grounded, relevant, correctly abstained, and properly cited?
- Can an interrupted experiment resume without duplicating requests or corrupting results?
- Can API credentials remain outside the AI-assisted development environment?
- Do the data justify a superiority claim — or only an exploratory observation?

This is not a “chat with PDF” prototype. It is a **governed RAG experimentation and evaluation system** built around traceability, controlled comparison, failure recovery, and scientific restraint.

---

## 🎯 Why This Project Matters

Advanced RAG techniques are frequently presented as universally superior. In practice, their benefit depends on corpus structure, ground-truth quality, retrieval depth, segmentation, reranking behavior, evaluation design, and the question distribution.

RAGLab v7 treats every technique as a **testable hypothesis**, not a product claim.

The framework isolates and compares seven variants while preserving the same multilingual embedding model, source corpus, question set, generation model, and evaluation protocol. It records not only scores, but also the retrieved passages, page provenance, citations, abstentions, configuration fingerprints, API-call ledgers, retry causes, and checkpoint state.

> **Core engineering principle:** a sophisticated pipeline is not better because it is sophisticated; it is better only when controlled evidence demonstrates a relevant improvement.

---

## 🧩 Experimental Architecture

```text
┌──────────────────────────────────────────────┐
│       Specifications & Domain Contracts       │
│  Questions · Splits · Evidence · Invariants  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│        Corpus & Provenance Infrastructure     │
│ PDF Audit · Page Extraction · SHA-256 · IDs  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│          Controlled Retrieval Matrix          │
│ F0 · S0 · W0 · W1 · H0 · H1 · H2            │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│        Generation & Evaluation Protocol       │
│ Gemini · RAG Triad · Abstention · Citations  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│       Reliability & Governance Controls       │
│ Checkpoints · Resume · Retry Ledger · Gates  │
└──────────────────────────────────────────────┘
```

The architecture follows **Domain-Driven Design, Specification-Driven Development, Clean Architecture, Clean Code, TDD, and gated delivery**. Domain contracts remain independent of external providers, while infrastructure adapters implement PDF extraction, embeddings, retrieval, generation, evaluation, and persistence.

---

## 🔎 Retrieval Strategies Under Test

| ID | Strategy | Experimental Role |
|---|---|---|
| **F0** | Fixed-chunk baseline | Reference pipeline using flat chunks and semantic top-k retrieval. |
| **S0** | Sentence anchor | Indexes individual sentences to test fine-grained semantic matching without contextual expansion. |
| **W0** | Sentence-window | Retrieves sentence anchors and expands them to neighboring sentences before generation. |
| **W1** | Sentence-window + reranking | Tests whether second-stage bi-encoder rescoring improves or damages W0 candidates. |
| **H0** | Hierarchical leaves | Retrieves small leaf nodes from a multi-level document hierarchy. |
| **H1** | Auto-merging | Promotes sufficiently covered child nodes into larger parent contexts. |
| **H2** | Auto-merging + reranking | Measures the incremental effect of rescoring over hierarchical auto-merging. |

### Sentence-window — precision first, context second

Sentence-window separates two units that conventional chunking conflates:

1. **Retrieval unit:** a small sentence anchor optimized for semantic precision.
2. **Generation unit:** the anchor plus neighboring sentences, providing local explanatory context.

This design investigates whether retrieving small and answering with more context can reduce the precision–coherence trade-off.

### Auto-merging — retrieve leaves, promote context conditionally

Auto-merging organizes content into parent–child hierarchies. Retrieval begins at fine-grained leaves. When enough related children from the same parent are recovered, the system can promote them to a larger parent node.

The intended benefit is coherent context without searching only over large, noisy chunks. RAGLab does not assume that promotion is beneficial: it records whether the merged context actually improves evidence recovery.

### Reranking — a fallible statistical filter

Reranking is treated as an intervention with potential benefit **and potential damage**. Unlike alpha–beta pruning, a statistical reranker provides no formal guarantee that the correct passage survives in the final top-n.

RAGLab therefore evaluates reranker damage explicitly by comparing first-stage and post-reranking evidence rather than reporting only the final answer.

---

# 🔬 Methodological Rigor

## Controlled Variables

All seven strategies use the same:

- audited PDF sub-corpus;
- question and split definitions;
- multilingual embedding model;
- embedding dimension and normalization;
- Gemini generation model;
- abstention policy;
- evaluation protocol;
- evidence and citation schema.

The final artifact confirms fingerprint parity across all strategies:

```text
embedding_model:  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
dimension:        384
normalization:    true
pooling:          mean
FastEmbed:        0.8.0
ONNX Runtime:     1.28.0
fingerprint:      identical across 7/7 strategies
```

## Evaluation Dimensions

| Dimension | Question Answered | Important Limitation |
|---|---|---|
| **Retrieval hit** | Was an annotated relevant page present in the final context? | Depends on ground-truth completeness. |
| **Context relevance** | Does the retrieved context address the question? | LLM-judge score, not retrieval recall. |
| **Groundedness** | Is the generated answer supported by retrieved context? | Grounded does not necessarily mean externally true. |
| **Answer relevance** | Does the answer directly address the question? | Relevant does not necessarily mean correct. |
| **Abstention correctness** | Did the pipeline answer or abstain appropriately? | Requires representative positive and negative cases. |
| **Citation provenance** | Do cited pages map to captured source evidence? | Mapping alone does not prove claim-level entailment. |
| **Reranker damage** | Did reranking eliminate previously recovered evidence? | Requires first-stage candidates to be retained. |

This distinction prevents the common mistake:

```text
high RAG Triad score ≠ proven factual correctness
```

---

## 📊 Completed Benchmark

### Execution integrity

```text
Strategies:                7
Questions per strategy:    8
Completed pairs:           56 / 56
Answerable questions:      7
Abstention controls:       1
Logical external calls:    154
Physical HTTP attempts:    155
Successful responses:      154
Failed attempts:           1
Accounted retries:         1
Holdout status:            SEALED
Artifact schema:           slice4_v3
```

The request ledger satisfies both reliability invariants:

```text
physical_attempts = logical_requests + retries
physical_attempts = successful_responses + failed_attempts
```

### Retrieval results

| Strategy | Relevant-page hits | Hit rate | Incorrect abstentions |
|---|---:|---:|---:|
| **F0 — Fixed chunks** | 1/7 | 14.3% | 6/7 |
| **S0 — Sentence anchor** | 2/7 | 28.6% | 5/7 |
| **W0 — Sentence-window** | 2/7 | 28.6% | **2/7** |
| **W1 — Window + reranking** | 2/7 | 28.6% | 4/7 |
| **H0 — Hierarchical leaves** | 2/7 | 28.6% | 4/7 |
| **H1 — Auto-merging** | 2/7 | 28.6% | 4/7 |
| **H2 — Auto-merging + reranking** | 1/7 | 14.3% | 3/7 |

### Evidence-based interpretation

- **Sentence-window showed only an exploratory gain:** W0 added one net retrieval hit over F0.
- **Auto-merging showed no observed retrieval gain:** H1 and H0 produced identical hit counts.
- **Sentence-window reranking was net neutral:** W1 lost one W0 hit and gained a different one.
- **Hierarchical reranking caused observed damage:** H2 fell from 2/7 to 1/7 relative to H1.
- **No advanced strategy demonstrated statistically supported superiority.**
- **All seven strategies correctly abstained on the single out-of-corpus control question.**

> [!NOTE]
> Reporting this result is a project strength, not a failure. The benchmark successfully prevented architectural sophistication from being mistaken for empirical superiority and exposed where the next experimental cycle must improve.

---

## 🛡 Reliability Engineering

### Idempotent checkpoint recovery

Long-running evaluation can be interrupted by notebook shutdowns, API quotas, network failures, or local session loss. RAGLab persists progress after each valid unit of work and resumes by exact `RUN_ID`.

Key guarantees:

- exact checkpoint selection instead of ambiguous glob matching;
- already completed strategy–question pairs are not repeated;
- integrity-bound manifests connect checkpoints to configuration and corpus;
- incomplete rows are rejected rather than silently accepted;
- final materialization requires all mandatory pairs;
- recovered and composite runs are labeled honestly.

### Generic retry accounting

The retry ledger distinguishes:

- HTTP `429` rate limiting;
- retryable `5xx` server errors;
- other retryable transport failures;
- successful and failed physical attempts;
- logical requests versus physical attempts.

This prevents a recovered `503`, for example, from disappearing from operational statistics merely because the subsequent attempt succeeded.

### Fail-closed behavior

The benchmark rejects:

- execution without an explicit mode;
- full benchmark execution without confirmation;
- resume without an exact run identifier;
- holdout leakage;
- incompatible fingerprints;
- missing mandatory evidence;
- invalid or non-numeric metric values;
- inconsistent call ledgers;
- corrupted checkpoint envelopes.

---

## 🔐 Credential Security Boundary

Gemini credentials are never stored in notebooks, source files, checkpoints, result artifacts, prompts, or the AI-assisted IDE.

The human operator decrypts the credential in an isolated local terminal using `systemd-creds`, exports it only for the benchmark process, and removes it through an exit trap.

```bash
set +x

cleanup_credentials() {
  unset GEMINI_API_KEY
  unset GOOGLE_API_KEY
}

trap cleanup_credentials EXIT INT TERM HUP

export GEMINI_API_KEY="$(
  sudo systemd-creds decrypt \
    /home/lg-runner/.config/credstore.encrypted/GEMINI_API_KEY \
    -
)"
```

Additional controls include secret scanning, telemetry restrictions, offline embedding execution, sanitized result serialization, and explicit prohibition of credential output.

---

## 🏛 Engineering & Governance Model

| Practice | Application in RAGLab v7 |
|---|---|
| **DDD** | Domain entities, value objects, policies, invariants, and provider-independent ports. |
| **SDD** | Versioned requirements, experimental contracts, gates, and traceability matrices. |
| **Clean Architecture** | Domain and application layers isolated from Gemini, PDF, embedding, and retrieval adapters. |
| **TDD** | Unit, contract, architecture, integration, resume, retry, and artifact validation tests. |
| **Clean Code** | Typed interfaces, narrow responsibilities, explicit error taxonomy, deterministic serialization. |
| **CI/CD readiness** | Automated reference verification, secret scanning, tests, linting, typing, dependency audit, and SBOM generation. |
| **Supply-chain governance** | Version lock, hashes, dependency audit, license inventory, and CycloneDX SBOM. |
| **Scientific governance** | Sealed holdout, controlled variables, paired comparisons, limitations, and calibrated conclusions. |

---

## 🧰 Technology Stack

| Area | Technologies |
|---|---|
| Language | Python 3.11+ |
| RAG orchestration | LlamaIndex adapters and provider-independent application ports |
| Embeddings | FastEmbed, ONNX Runtime, multilingual MiniLM |
| Generation and judging | Gemini 3.1 Flash Lite through isolated adapters |
| PDF processing | pypdf with page-level provenance and integrity hashes |
| Data contracts | Typed domain entities, value objects, configuration fingerprints |
| Testing | pytest, contract tests, architecture tests, deterministic fixtures |
| Static quality | Ruff, mypy, reference verification, secret scanner |
| Supply chain | Locked dependencies, pip-audit, license inventory, CycloneDX SBOM |
| Persistence | JSON checkpoints and versioned experimental artifacts with SHA-256 envelopes |

---

## 🧩 Repository Structure

```text
raglab-v7/
├── src/raglab/
│   ├── domain/                  # Entities, value objects, policies, invariants
│   ├── application/
│   │   └── ports/               # Corpus, embeddings, retrieval, generation,
│   │                            # evaluation, and checkpoint contracts
│   └── infrastructure/
│       ├── config/              # Validated experiment configuration
│       ├── embeddings/          # FastEmbed multilingual adapter
│       ├── pdf_parsers/         # Audited page extraction and provenance
│       ├── retrieval/           # Fixed, sentence, and hierarchical strategies
│       ├── generation/          # Gemini generation boundary
│       ├── evaluation/          # RAG Triad and abstention evaluation
│       └── checkpoints/         # Idempotent filesystem persistence
├── benchmarks/
│   ├── questions/               # Versioned benchmark definitions and splits
│   ├── results/                 # Sanitized experimental artifacts
│   └── run_slice4_benchmark.py  # Governed benchmark runner
├── checkpoints/                 # Local resumable execution state (not committed)
├── tests/
│   ├── unit/                    # Domain and infrastructure behavior
│   ├── integration/             # Retrieval, runner, artifact, and resume contracts
│   └── architecture/            # Dependency-direction enforcement
├── scripts/                     # Provisioning, verification, security, and audits
├── docs/
│   ├── runbooks/                # Human-only credential-safe execution
│   └── security/                # Threat model and governance controls
├── requirements.lock            # Reproducible dependency resolution
├── licenses.json                # Dependency license decisions
└── sbom.product.cyclonedx.json  # Software Bill of Materials
```

---

## 🚀 Governed Execution Modes

The runner is deliberately explicit:

```bash
# Safe preflight without a full benchmark
python benchmarks/run_slice4_benchmark.py --mode preflight-retrievers

# One strategy × one question
python benchmarks/run_slice4_benchmark.py \
  --mode smoke \
  --smoke-strategy F0_baseline \
  --smoke-question q_dev_01

# Resume an interrupted governed run
python benchmarks/run_slice4_benchmark.py \
  --mode resume \
  --run-id <EXACT_RUN_ID>

# Full execution requires explicit confirmation
python benchmarks/run_slice4_benchmark.py \
  --mode full \
  --confirm-full-benchmark
```

Execution without an explicit mode fails closed.

---

## 📌 Current Status

```text
ARCHITECTURE:              Established — DDD + Clean Architecture
RETRIEVAL STRATEGIES:      7 operational variants
MULTILINGUAL EMBEDDING:    Fingerprint parity confirmed across 7/7
BENCHMARK COMPLETENESS:    56/56 strategy–question pairs
CHECKPOINT RECOVERY:       Operational and idempotent
RETRY ACCOUNTING:          Logical/physical/causal invariants satisfied
CREDENTIAL BOUNDARY:       Human terminal only; no IDE access
HOLDOUT:                   Sealed
ENGINEERING VALIDATION:    Passed
SCIENTIFIC STATUS:         Exploratory — superiority not demonstrated
PRODUCTION READINESS:      Not yet declared
```

---

## 🔮 Next Research & Engineering Directions

1. Expand the benchmark to achieve adequate statistical power.
2. Replace single-annotator page labels with dual annotation and adjudication.
3. Support multiple relevant passages and claim-level evidence labels.
4. Add Recall@k, MRR, nDCG, citation precision, citation recall, and selective-risk curves.
5. Introduce independently calibrated factual-correctness evaluation.
6. Compare bi-encoder rescoring with a genuine multilingual cross-encoder.
7. Measure first-stage recall and explicit relevant-passage dropped rate.
8. Execute a clean confirmatory run after freezing protocol and thresholds.
9. Evaluate robustness across additional corpora, domains, document layouts, and languages.
10. Calibrate LLM-as-a-judge metrics against blinded human assessment.

---

## 💼 Competencies Demonstrated

RAGLab v7 provides concrete evidence of capability in:

- production-oriented RAG architecture;
- semantic and hierarchical retrieval;
- sentence-window context engineering;
- reranking analysis and failure diagnosis;
- LLM evaluation and RAG Triad interpretation;
- experimental design and paired comparison;
- checkpointing and idempotent recovery;
- API quota and retry engineering;
- credential isolation and secure operations;
- DDD, SDD, Clean Architecture, TDD, and CI/CD governance;
- reproducibility, SBOM, dependency locking, and auditability;
- responsible communication of uncertainty and negative results.

> **The differentiator is not that every advanced RAG technique won. The differentiator is that the system was rigorous enough to discover when they did not.**

---

> 💬 *RAGLab v7 bridges production-grade AI engineering with applied scientific discipline: every retrieval decision is traceable, every interruption is recoverable, every external request is accounted for, and every conclusion must be earned by evidence.*  
> — **Leonardo Maximino Bernardo, 2026**
