# Supply Chain — RAGLab v7

> **Version:** S1 Gate 1 Recovered
> **Created:** 2026-07-30
> **Status:** Dependencies installed in `.venv`, lockfile generated, SBOM generated, scans clean

---

## 1. Project Manifest

Dependencies are declared in `pyproject.toml` and installed locally in `.venv`.

### Python Version

- **Minimum:** 3.11
- **Tested:** 3.12.3
- **Rationale:** `slots=True` in dataclasses, `tomllib`, `sys.stdlib_module_names`

### Direct Dependencies (runtime)

| Package | Version Constraint | Purpose | License |
|---|---|---|---|
| (none in S1) | — | Domain and application layers are stdlib-only | — |

> [!NOTE]
> Runtime dependencies will be added in S2+ when LlamaIndex, TruLens, and
> google-generativeai adapters are implemented.

### Development Dependencies

| Package | Version Constraint | Installed Version | Purpose | License |
|---|---|---|---|---|
| pytest | >=8.0,<10 | 9.1.1 | Test runner | MIT |
| pytest-cov | >=5.0,<6 | 5.0.0 | Coverage reporting | MIT |
| ruff | >=0.5,<1 | 0.16.1 | Linting + formatting | MIT |
| mypy | >=1.10,<2 | 1.20.2 | Type checking | MIT |
| pip-audit | >=2.7,<3 | 2.10.1 | Vulnerability scanning | Apache 2.0 |
| pip-tools | >=7.0,<8 | 7.6.0 | Lockfile generation | BSD-3-Clause |
| cyclonedx-bom | >=7.0,<8 | 7.3.1 | SBOM generation | Apache-2.0 |

## 2. Pinning Strategy & Lockfile

- **Direct dependencies:** pinned to compatible release ranges (`>=X,<Y`) in `pyproject.toml`
- **Lockfile:** generated as `requirements.lock` via `.venv/bin/pip freeze`
- **Reproducibility:** isolated `.venv` environment ensures exact version reproducibility

### Lockfile Status

```
requirements.lock — GENERATED (68 pinned packages)
```

## 3. Vulnerability Scanning

### Scan Results

| Tool | Status | Findings | Action Taken |
|---|---|---|---|
| pip-audit | PASSED | 0 known vulnerabilities | Upgraded pytest to 9.1.1 (fixed PYSEC-2026-1845) |
| scan_secrets.py | PASSED | 0 findings | Scanned tracked files; `.venv` excluded |
| ruff check | PASSED | 0 lint issues | All rules passed clean |
| mypy --strict | PASSED | 0 type errors | Strict type checking on `src/` |

## 4. SBOM (Software Bill of Materials)

- **Format:** CycloneDX v1.6 JSON
- **Location:** `sbom.cyclonedx.json`
- **Generator:** `cyclonedx-py` (v7.3.1)
- **Status:** GENERATED and verified

```bash
.venv/bin/cyclonedx-py environment .venv -o sbom.cyclonedx.json
```

## 5. License Inventory & Policy

### Allowed Licenses

- MIT
- Apache 2.0
- BSD (2-clause, 3-clause)
- PSF
- ISC

### License Audit Summary

All 68 installed packages use open-source licenses compatible with MIT project licensing (MIT, Apache-2.0, BSD-3-Clause, PSF, ISC). Zero GPL/AGPL copyleft dependencies present.

## 6. Update Policy

1. All package updates must be tested locally with `pytest`, `ruff`, `mypy`, and `pip-audit` prior to commit.
2. No automatic major version updates.
3. Vulnerability fixes (patch/minor) must be applied immediately when `pip-audit` flags a CVE.
