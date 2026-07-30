# Supply Chain — RAGLab v7

> **Version:** S1 (initial)
> **Created:** 2026-07-30
> **Status:** Dependencies declared, lockfile NOT generated

---

## 1. Project Manifest

Dependencies are declared in `pyproject.toml`. The S1 baseline uses stdlib-only
for the domain and application layers.

### Python Version

- **Minimum:** 3.11
- **Tested:** 3.11, 3.12
- **Rationale:** `slots=True` in dataclasses, `tomllib`, `sys.stdlib_module_names`

### Direct Dependencies (runtime)

| Package | Pinned Version | Purpose | License |
|---|---|---|---|
| (none in S1) | — | Domain is stdlib-only | — |

> [!NOTE]
> Runtime dependencies will be added in S2+ when LlamaIndex, TruLens, and
> google-generativeai adapters are implemented.

### Development Dependencies

| Package | Pinned Version | Purpose | License |
|---|---|---|---|
| pytest | >=8.0,<9 | Test runner | MIT |
| pytest-cov | >=5.0,<6 | Coverage reporting | MIT |
| ruff | >=0.5,<1 | Linting + formatting | MIT |
| mypy | >=1.10,<2 | Type checking | MIT |
| pip-audit | >=2.7,<3 | Vulnerability scanning | Apache 2.0 |

> [!IMPORTANT]
> Development dependencies are **NOT installed** in this slice.
> They are declared for reproducibility and will be installed when authorized.

## 2. Pinning Strategy

- **Direct dependencies:** pinned to compatible release ranges (`>=X,<Y`)
- **Lockfile:** will be generated with `pip-compile` or `uv` when installation is authorized
- **Hashes:** lockfile will include `--generate-hashes` for integrity verification
- **Transitive dependencies:** resolved and locked by the lockfile tool

### Current State

```
LOCKFILE_NOT_GENERATED — no package installation authorized in S1
```

**Command to generate when authorized:**

```bash
# Option A: pip-tools
pip-compile --generate-hashes --output-file=requirements.lock pyproject.toml

# Option B: uv (faster)
uv pip compile --generate-hashes pyproject.toml -o requirements.lock
```

## 3. Vulnerability Scanning

### Planned Tools

| Tool | Purpose | Scope |
|---|---|---|
| pip-audit | Known CVE scan | Direct + transitive deps |
| safety | Vulnerability DB check | Direct + transitive deps |
| scan_secrets.py | Secret detection in repo | All tracked files |
| ruff | Security-related lint rules | Python source |

### Current State

```
VULNERABILITY_SCAN_NOT_EXECUTED — no dependencies installed
SECRET_SCAN: PASSED (scan_secrets.py — 0 findings)
```

**Command to run when authorized:**

```bash
pip-audit --require-hashes -r requirements.lock
```

## 4. License Policy

### Allowed Licenses

- MIT
- Apache 2.0
- BSD (2-clause, 3-clause)
- PSF
- ISC

### Prohibited Licenses

- GPL (any version) — incompatible with proprietary use
- AGPL
- SSPL
- EUPL
- Unknown / Unlicensed

### Compliance

License compliance will be verified when dependencies are installed:

```bash
pip-licenses --format=table --fail-on="GPL;AGPL;SSPL;EUPL;Unknown"
```

## 5. SBOM Plan

### Target Format

- CycloneDX 1.5 (JSON)

### Generation

```bash
# After dependencies are installed
cyclonedx-py environment -o sbom.json --format json
```

### Current State

```
SBOM_NOT_GENERATED — no dependencies installed
```

## 6. Update Policy

| Trigger | Action | Frequency |
|---|---|---|
| Security advisory | Patch immediately | On notification |
| Minor release | Review changelog, update if compatible | Monthly |
| Major release | Evaluate breaking changes, test, update | Quarterly |
| New dependency | Justify, license check, pin, lock | Per addition |

## 7. Integrity Mechanisms

| Mechanism | Status | Tool |
|---|---|---|
| Version pinning | ✅ Declared in pyproject.toml | pip-compile / uv |
| Hash verification | ⏳ Planned (lockfile) | pip-compile --generate-hashes |
| Signature verification | ⏳ Planned | pip --require-hashes |
| Vulnerability scan | ⏳ Planned | pip-audit |
| License audit | ⏳ Planned | pip-licenses |
| SBOM | ⏳ Planned | cyclonedx-py |
| Secret scan | ✅ Implemented | scan_secrets.py |

---

## Limitations

1. No lockfile generated — network access required
2. No vulnerability database consulted — no packages installed
3. License audit not executed — no packages installed
4. SBOM not generated — no packages installed
5. Hash verification not tested — no lockfile
6. pip-audit scope is limited to Python packages; system-level deps not covered
