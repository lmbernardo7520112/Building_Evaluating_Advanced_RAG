# Relatório de Implementação — Gate B1: Infraestrutura de Anotação Humana (Ground Truth v2)

> **Status**: GATE_B1_OFFLINE_QA_PASSED  
> **Protocolo**: Ground Truth v2 (raglab_v7_slice4_v3)  
> **Data**: 2026-08-03  

---

## 1. Resumo Executivo

O **Gate B1** estabelece a infraestrutura offline, reproduzível e cega para a anotação humana independente do Ground Truth v2.
Toda a infraestrutura foi construída sem acesso a credenciais, sem chamadas a APIs/Gemini e mantendo o **Holdout estritamente lacrado**.

---

## 2. Artefatos Criados e Estrutura de Arquivos

```text
benchmarks/ground_truth/v2/
├── passage_registry.jsonl                   (Registro canônico de passagens)
├── passage_registry_manifest.json          (Manifesto de extração com SHA-256)
├── annotation_guidelines.md                 (Manual de anotação 0-3 e answerability)
├── agreement_report.json                    (Relatório de concordância inter-anotadores)
├── adjudication_template.jsonl              (Template de adjudicação sem decisões prévias)
└── annotation_packages/
    ├── annotator_a/
    │   ├── development.jsonl                (Pacotes cegos Anotador A - dev)
    │   └── test.jsonl                       (Pacotes cegos Anotador A - test)
    ├── annotator_b/
    │   ├── development.jsonl                (Pacotes cegos Anotador B - dev)
    │   └── test.jsonl                       (Pacotes cegos Anotador B - test)
    └── package_manifest.json                (Manifesto dos pacotes de anotação)

scripts/
├── build_passage_registry.py                (Builder determinístico do registro)
├── build_blinded_annotation_packages.py      (Builder dos pacotes cegos A/B e adjudicação)
├── validate_human_annotations.py            (Validador offline: template, completed, adjudicated)
└── compute_annotation_agreement.py          (Calculador de concordância Kappa/Weighted Kappa)

src/raglab/evaluation/contracts/
└── human_annotation_v2.py                   (Contratos tipados Pydantic/Dataclasses)

tests/unit/evaluation/
└── test_human_annotation_v2.py              (Suíte de 28 testes de invariantes)
```

---

## 3. Registro Canônico de Passagens e Segmentação

- **Corpus PDF**: `Fundamentos matemáticos para a ciência da computação (Gersting)`
- **PDF SHA-256**: `33e2e9f1e190158b3e99c19fced1acd050720247c7556780bad82b2f93bf1254`
- **Páginas físicas**: 91–115 (inclusive, 1-indexed)
- **Adapter de Extração**: `PyPdfExtractorAdapter`
- **Política de Segmentação**: `paragraph_split_with_min_50_chars`
- **Quantidade de Passagens**: 123 passagens canônicas
- **Fingerprint do Registry (`passage_registry.jsonl SHA-256`)**: `c2d1b31e5eaeb98e2a31094628cf2e4291c4ff73bb7ddb1f79929204163799e1`
- **Regra "Página não é passagem"**: Todo `passage_id` segue o formato determinístico `ps_<16 hex chars>` derivado do digest SHA-256 do documento, página, offsets e conteúdo. Nenhum ID artificial como `p92` ou `page_92` foi gerado.

---

## 4. Pacotes Cegos e Cegamento Aplicado

- **Perguntas Empacotadas**: 4 perguntas (Split `development`: 2 | Split `test`: 2)
- **Candidate Pool**: União de passagens das páginas relevantes e controles negativos amostrados deterministicamente do registro.
- **Fontes Indisponíveis Offline**: `CANDIDATE_SOURCE_NOT_AVAILABLE_OFFLINE` (registrado no manifesto).
- **Cegamento Estrito**:
  - Removidos: `strategy`, `original_rank`, `score`, `retriever_name`, `reranker_score`, respostas LLM (Gemini), RAG Triad e páginas legadas.
  - As passagens dos candidatos são apresentadas em ordem determinística e embaralhada via `seed(question_id)`.
  - Anotadores A e B recebem os mesmos candidatos, mas em arquivos isolados sem acesso às respostas um do outro.

---

## 5. QA e Suíte de Validação Completa

Todos os comandos de QA foram executados sem pipes que mascarem exit codes:

| Verificação | Comando | Resultado | Exit Code |
| :--- | :--- | :--- | :--- |
| **Suíte de Invariantes Gate B1** | `pytest tests/unit/evaluation/test_human_annotation_v2.py -vv` | **28/28 PASSED** | `0` |
| **Suíte Global Pytest** | `pytest tests/ -q` | **687/687 PASSED** | `0` |
| **Linter Ruff** | `ruff check src/ tests/ scripts/` | **All checks passed!** | `0` |
| **Type Checker Mypy** | `mypy src --ignore-missing-imports` | **Success (0 errors)** | `0` |
| **Referência de Notebook** | `python scripts/verify_reference.py` | **15/15 PASSED** | `0` |
| **Scanner de Segredos** | `python scripts/scan_secrets.py` | **0 findings (PASSED)** | `0` |
| **Git Diff Check** | `git diff --check` | **CLEAN** | `0` |
| **Validação de Templates** | `validate_human_annotations.py --mode template` | **PASSED** | `0` |
| **Reconstrução Reproduzível** | `build_passage_registry.py` (2x tmp) | **Identical SHA256** | `0` |

---

## 6. Estado de Isolamento do Holdout

- **Holdout Status**: `HOLDOUT_SEALED`
- Perguntas de holdout foram estritamente filtradas de `ACTIVE_QUESTIONS` na geração dos pacotes.
- Nenhum texto de pergunta de holdout foi exposto, empacotado ou anotado.

---

## 7. Declarações Autorizadas de Encerramento do Gate B1

```text
PASSAGE_REGISTRY_BUILT
BLINDED_PACKAGES_BUILT
ANNOTATION_GUIDELINES_READY
ANNOTATION_VALIDATOR_READY
AGREEMENT_TOOL_READY
ADJUDICATION_TEMPLATE_READY
HOLDOUT_SEALED
GATE_B1_OFFLINE_QA_PASSED

GEMINI_NOT_CALLED
CREDENTIALS_NOT_ACCESSED
NETWORK_NOT_USED
HUMAN_LABELS_NOT_CREATED
GOLD_ANSWERS_NOT_CREATED
HOLDOUT_SEALED
FULL_BENCHMARK_NOT_EXECUTED
READY_FOR_INDEPENDENT_HUMAN_ANNOTATION
```
