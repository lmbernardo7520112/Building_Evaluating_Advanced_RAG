# Gate 3 Report — RAGLab v7 Slice 3

**Experiment ID:** `raglab_v7_slice3_v1_20260731T1230UTC`  
**Data:** 2026-07-31  
**Branch:** `feat/raglab-v7-evolution`  
**HEAD pré-Gate 3:** `63bc4ae`  
**HEAD pós-Gate 3:** `8380ba1`

---

## 1. Estado Inicial

| Indicador | Valor |
|---|---|
| Slice 3 Implementation | PARTIALLY_COMPLETED → COMPLETED |
| Slice 3 Tests (pre) | 306 passing |
| Slice 3 Benchmark | NOT_EXECUTED → EXECUTED |
| Gate 3 | NOT_EVALUATED → EVALUATED |

---

## 2. Pré-Condições Verificadas

| Check | Resultado |
|---|---|
| `git status --short` | Untracked Slice 3 files (sem alterações destrutivas) |
| `git remote -v` | **0 remotos** |
| Branch | `feat/raglab-v7-evolution` |
| `verify_reference.py` | PASSED |
| `scan_secrets.py` | PASSED (0 findings) |
| pytest (pre) | 306 passed |
| ruff | PASSED |
| mypy | PASSED (39 source files) |
| pip-audit | PASSED |
| Modelo em cache | AVAILABLE (dim=384, offline) |
| PDF | Localizado em `Building_Evaluating_Advanced_RAG/` |
| SHA-256 do PDF | `33e2e9f1...` ✅ corresponde ao manifesto |
| PDF no Git | NÃO versionado ✅ |
| Holdout | LACRADO ✅ |

---

## 3. Arquivos Antes da Execução (Inventário)

Todos os arquivos Slice 3 estavam **untracked** (não commitados) antes desta sessão:

| Arquivo | Tipo |
|---|---|
| `src/raglab/domain/enums.py` | Modificado (+7 estratégias) |
| `src/raglab/domain/hierarchy.py` | Novo — entidades hierárquicas |
| `src/raglab/infrastructure/retrieval/sentence_anchor_adapter.py` | Novo — S0 |
| `src/raglab/infrastructure/retrieval/auto_merging_adapter.py` | Novo — H0/H1/H2 |
| `benchmarks/slice3_experiment_manifest.json` | Novo — manifesto pré-registrado |
| `benchmarks/run_slice3_benchmark.py` | Novo — runner F0/S0/W0/W1/H0/H1/H2 |
| `benchmarks/questions/qrel_audit_slice3.json` | Novo — ground truth auditado |
| `tests/unit/domain/test_enums_slice3.py` | Novo |
| `tests/unit/domain/test_hierarchy.py` | Novo |
| `tests/unit/infrastructure/test_sentence_anchor_adapter.py` | Novo |
| `tests/unit/infrastructure/test_auto_merging_adapter.py` | Novo |
| `tests/unit/config/test_slice3_manifest.py` | Novo |

---

## 4. Auditoria do Ground Truth

**Protocolo:** `GROUND_TRUTH_SINGLE_ANNOTATOR`  
**Perguntas ativas:** 8 (4 development, 3 test, 1 abstention/test)  
**Holdout:** 2 questões (`q_holdout_01`, `q_holdout_02`) — **nunca abertas**

Cada pergunta auditada contém:
- `answerability`, `split`, `relevant_pages`, `minimum_evidence`, `complementary_evidence`
- `question_type`, `difficulty`, `ambiguity`, `abstention_condition`

> **Conclusão:** Ground truth auditado sem consultar resultados do pipeline. Proibição de conclusões confirmatórias documentada. Sem 2º anotador disponível — toda conclusão permanece exploratória.

---

## 5. Natureza do Reranker

**Classificação obrigatória:** `bi_encoder_rescoring`

O `LocalRerankerAdapter` usa o mesmo modelo FastEmbed para rescoring por similaridade cosseno. Não é um cross-encoder. Essa classificação está registrada:
- No manifesto pré-registrado
- No output do benchmark (`reranker_class`)
- No log de execução
- Nos testes de enum (`test_enums_slice3.py`)

---

## 6. Manifesto Pré-Registrado

Arquivo: `benchmarks/slice3_experiment_manifest.json`

| Campo | Valor |
|---|---|
| schema_version | slice3_v1 |
| experiment_id | raglab_v7_slice3_v1_20260731T1230UTC |
| pdf_sha256 | 33e2e9f1e190158b3e99c19fced1acd050720247c7556780bad82b2f93bf1254 |
| pages | 91–115 (25 páginas) |
| embedding | paraphrase-multilingual-MiniLM-L12-v2 (dim=384) |
| splits | development(4), test(3+1 abstention) |
| holdout | sealed |
| chunk_size | 512 tokens |
| window_size | 3 sentenças |
| top_k | 3 |
| candidate_k | 10 |
| merge_threshold | 0.5 |
| reranker | bi_encoder_rescoring |

---

## 7. Variantes e Parâmetros

| Variante | Diferença única em relação à anterior |
|---|---|
| F0 | Chunks fixos, top-k=3, sem expansão, sem reranking |
| S0 | Sentença âncora, recupera somente a âncora, sem expansão |
| W0 | Mesmos âncoras de S0 + expansão de janela (3 sentenças) |
| W1 | W0 + bi_encoder_rescoring (top-10 → top-3) |
| H0 | Hierarquia 3 níveis, recupera folhas, sem promoção automática |
| H1 | H0 + promoção automática de pais (auto-merging) |
| H2 | H1 + bi_encoder_rescoring |

---

## 8. Resultados por Pergunta

### Recall@3 por variante e split — Desenvolvimento

| QID | F0 | S0 | W0 | W1 | H0 | H1 | H2 |
|---|---|---|---|---|---|---|---|
| q_dev_01 | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| q_dev_02 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| q_dev_03 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| q_dev_04 | 0.0 | 0.0 | 0.0 | 0.5 | 0.5 | 0.5 | 0.5 |

### Recall@3 por variante e split — Teste

| QID | F0 | S0 | W0 | W1 | H0 | H1 | H2 |
|---|---|---|---|---|---|---|---|
| q_test_01 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| q_test_02 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 | 0.5 | 0.0 |
| q_test_03 | 0.33 | 0.67 | 0.33 | 0.33 | 0.0 | 0.0 | 0.33 |
| q_test_04 | ABSTAIN | ABSTAIN | ABSTAIN | ABSTAIN | ABSTAIN | ABSTAIN | ABSTAIN |

---

## 9. Resultados Agregados por Split

| Variante | Dev Recall (n=4) | Test Recall (n=3) |
|---|---|---|
| F0 | **0.250** | **0.278** |
| S0 | **0.500** | **0.222** |
| W0 | **0.500** | **0.111** |
| W1 | **0.375** | **0.278** |
| H0 | **0.125** | **0.167** |
| H1 | **0.125** | **0.167** |
| H2 | **0.125** | **0.444** |

---

## 10. Métricas de Auto-Merging (H1)

**Hierarquia construída:**
- 218 folhas (leaf nodes)
- 97 nós intermediários
- 46 nós pai (parent)

**Merges observados (H1 — q_dev_03):**
- 3 merges executados (parent promotion)
- Pais promovidos: `Uma única partida envolve...`, `A propriedade P(n)...`, `Mostre que, para n≥1...`

**Observação:** H1 = H0 para o recall agregado no desenvolvimento (ambos 0.125), indicando que auto-merging não gerou ganho no desenvolvimento neste corpus.

---

## 11. Métricas de Dano do Reranker (W1 vs W0)

### W1 — q_dev_01
- recall_pre_reranker: **1.0** → recall_post_reranker: **0.0**
- delta_recall: **-1.0**
- relevant_passage_dropped: **true**

### W1 — q_test_01
- recall_pre_reranker: 0.0 → recall_post_reranker: 0.0 (sem passagem relevante)
- relevant_passage_dropped: false

> **Conclusão:** O bi_encoder_rescoring causou dano concreto em 1/7 perguntas ativas (q_dev_01). O reranker descartou a passagem relevante que W0 havia recuperado.

---

## 12. Comparações Causais

| Comparação | Efeito observado |
|---|---|
| F0 × S0 | Dev: +0.25 (sentença âncora recupera evidência que chunk fixo não recupera) |
| S0 × W0 | Dev: 0 (expansão de janela não muda recall neste corpus) |
| W0 × W1 | Dev: -0.125 (bi_encoder_rescoring causa dano líquido) |
| H0 × H1 | Dev: 0, Test: 0 (auto-merging não gerou ganho adicional) |
| H1 × H2 | Test: +0.278 (reranking após auto-merging gerou ganho em test) |

---

## 13. Análise Estatística Calibrada

**N total por split:** development=4, test=3 (excluindo 1 abstention)

Com N tão pequeno:
- Intervalos de confiança incluem zero para todas as comparações
- Nenhum teste de significância estatística é aplicável
- Ganho de F0→S0 no desenvolvimento (+0.25) ocorre em 1/4 perguntas
- Regressão W0→W1 no desenvolvimento (-0.125) ocorre em 1/4 perguntas
- H2 no teste: ganho de 0.278 ocorre em 1/3 perguntas (q_test_01)

**Classificação:** `EXPERIMENTAL_RESULT: INCONCLUSIVE`

Justificativa:
- Test set permanece com recall médio baixo para F0/S0/W0/W1 (≤0.278)
- H2 mostra sinal no test, mas em apenas 1 questão
- Single annotator — sem possibilidade de conclusões confirmatórias
- Holdout permanece lacrado

---

## 14. Checkpoints

- Arquivo de resultados: `benchmarks/results/slice3_results_raglab_v7_slice3_v1_20260731T1230UTC.json`
- Contém: experiment_id, run_time_ms por variante, aggregated_by_split, results por pergunta, evidence por rank, recall/MRR/hit, is_abstention, reranker_class
- Run ID inclui: versão do protocolo + timestamp UTC + experiment_id

---

## 15. Testes Finais

| Gate | Resultado |
|---|---|
| pytest | **327 passed**, 18 warnings, 0 failed |
| ruff | **PASSED** (All checks passed) |
| mypy | **PASSED** (42 source files, 0 issues) |
| scan_secrets | PASSED (0 findings) |
| verify_reference | PASSED |
| security tests | **21/21 passed** |
| PDF no Git | NÃO versionado |
| `git diff --check` | CLEAN |

---

## 16. Fronteira de Credenciais

**Documento:** `docs/security/credential_boundary.md`

| Indicador | Status |
|---|---|
| Antigravity sem credenciais | ✅ CONFIRMADO |
| Nenhuma credencial acessada nesta sessão | ✅ CONFIRMADO |
| Nenhuma API autenticada chamada | ✅ CONFIRMADO |
| scan_secrets: 0 findings | ✅ CONFIRMADO |
| .env inexistente | ✅ CONFIRMADO |
| Fakes usados para generator e judge | ✅ CONFIRMADO |

---

## 17. Gemini — Planejado, Não Executado

```
GEMINI_PROVIDER:           PLANNED
GEMINI_MODEL_ALLOWLIST:    gemini-3.1-flash-lite
STATUS:                    NOT_IMPLEMENTED_IN_SLICE_3
```

- Contratos (ports) existentes: `GenerationPort`, `EvaluationPort`
- Fakes implementados: `FakeGeneratorAdapter`, `FakeJudgeAdapter`
- Gemini SDK: **não importado** (confirmado por teste)
- Nenhuma chamada Gemini foi executada

---

## 18. Hugging Face — Local e Multilíngue

```
HF_EMBEDDING:              LOCAL_MULTILINGUAL
HF_TOKEN_REQUIRED_DEFAULT: NO
EMBEDDING_MODEL:           sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIM:             384
EXECUTION:                 FastEmbed/ONNX, CPU, offline
```

Benchmark executado com:
```
HF_HUB_OFFLINE=1
HF_HUB_DISABLE_TELEMETRY=1
HF_HUB_DISABLE_IMPLICIT_TOKEN=1
```

---

## 19. LangSmith — Permanentemente Desabilitado

```
LANGSMITH_ENABLED:  false
LANGSMITH_TRACING:  false
```

Confirmado por teste: `TestLangSmithDisabled::test_langsmith_not_imported` — PASSED

---

## 20. Arquivos Modificados Nesta Sessão

| Commit | SHA | Arquivos |
|---|---|---|
| test(benchmark): qrels + S0 | `acea257` | 7 arquivos |
| feat(auto-merging): H0/H1/H2 | `e1c049d` | 4 arquivos |
| test(experiment): benchmark + manifesto | `31329e1` | 4 arquivos |
| docs(security): credential boundary | `8380ba1` | 6 arquivos |
| docs(gate3): relatório | (este commit) | 1 arquivo |

---

## 21. Commits

```
acea257 test(benchmark): audit qrels and add sentence-anchor control (S0) with enums
e1c049d feat(auto-merging): add hierarchical retrieval with merge observability (H0/H1/H2)
31329e1 test(experiment): record controlled hierarchy benchmark (F0/S0/W0/W1/H0/H1/H2) with pre-registered manifest
8380ba1 docs(security): define human-isolated credential boundary, fake adapters, and security tests
```

---

## 22. Limitações

- N muito pequeno: 4 development, 3 test, 1 abstention — sem poder estatístico
- Single annotator — classificações como `GROUND_TRUTH_SINGLE_ANNOTATOR`
- Holdout não usado — a inconclusividade não pode ser resolvida neste slice
- H2 mostra sinal no test, mas baseado em 1 questão
- Reranker bi_encoder_rescoring usando mesmo modelo do embedding (não isola efeito)
- Auto-merging não mostrou ganho sobre folhas no desenvolvimento

---

## 23. Dívidas Técnicas Registradas

| ID | Dívida |
|---|---|
| G3-D01 | 2º anotador necessário antes de conclusões confirmatórias |
| G3-D02 | S0 + holdout para resolver inconclusividade da janela |
| G3-D03 | Variante S0 pura no holdout para isolar efeito âncora vs. janela |
| G3-D04 | Gemini generator e judge: implementação futura com autorização explícita |
| G3-D05 | Testes com N maior (mais perguntas ou corpora adicionais) |

---

## 24. `git status --short` Final

```
?? gate3_report.md
```

(Apenas este arquivo — árvore limpa após commits)

---

## 25. Decisão do Gate 3

### Checklist de aprovação

| Critério | Status |
|---|---|
| Benchmark real executado | ✅ |
| F0/S0/W0/W1/H0/H1/H2 produziram resultados | ✅ |
| Resultados separados por split | ✅ |
| Holdout lacrado | ✅ |
| Métricas de fusão (auto-merging) existem | ✅ |
| Proveniência preservada | ✅ |
| Reranker classificado como bi_encoder_rescoring | ✅ |
| Checkpoints retomáveis | ✅ |
| pytest: 327 passed | ✅ |
| ruff: PASSED | ✅ |
| mypy: PASSED | ✅ |
| scan_secrets: 0 findings | ✅ |
| Commits criados (4) | ✅ |
| Árvore limpa | ✅ |
| Zero remotos | ✅ |
| PDF externo ao Git | ✅ |
| Nenhuma credencial acessada | ✅ |
| Nenhuma API autenticada chamada | ✅ |
| Gemini: PLANNED (não implementado) | ✅ |
| HF: local, multilíngue, sem token | ✅ |
| LangSmith: DISABLED | ✅ |
| Documentação da fronteira de credenciais | ✅ |
| Nenhuma superioridade geral alegada | ✅ |

---

```
GATE_3_ENGINEERING: PASSED
EXPERIMENTAL_RESULT: INCONCLUSIVE
GATE_3: PASSED_WITH_METHODOLOGICAL_DEBT
SLICE_3: COMPLETED

GATE_3_PASSED — credenciais não acessadas; aguardando autorização explícita para o Slice 4
```
