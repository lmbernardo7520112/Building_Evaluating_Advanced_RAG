# Relatório de Reconciliação Técnico-Científica — Gate B2

- **Status do Gate**: GATE_B2_OPERATIONAL_READY
- **HEAD Inicial**: `1280744`
- **Branch**: `feat/hybrid-human-validated-eval`
- **Protocolo**: `raglab_v7_slice4_v3`
- **Schema de Artefato**: `2.0.0`

---

## 1. Auditoria de Proveniência do Candidate Pool

O pool multissistema foi auditado e reconstruído com base estrita em evidências brutas recuperadas de execuções de benchmark offline materializadas (`slice4_final_composite_recovered_run.json`), eliminando qualquer seleção direta a partir do registro ou das páginas do Ground Truth.

### Fontes Efetivamente Executadas e Verificadas:
1. **`F0_baseline`** (Standard Chunking)
2. **`W0_sentence_window`** (Sentence Window sem Reranker)
3. **`W1_sentence_window_rerank`** (Sentence Window com Reranker)
4. **`H0_hierarchical_leaf`** (Hierárquico Folha)
5. **`H1_auto_merging`** (Hierárquico Auto-Merging)
6. **`H2_auto_merging_rerank`** (Hierárquico Auto-Merging com Reranker)
7. **`S0_sentence_anchor`** (Sentence Anchor)
8. **`legacy_pages_pool`** (Fonte legada identificada e adicional)
9. **`neighbor_expansion`** (Expansão de vizinhos estruturais)

### Fontes Indisponíveis Offline:
- **`lexical_bm25`**: `CANDIDATE_SOURCE_NOT_AVAILABLE_OFFLINE`
- **`dense_canonical`**: `CANDIDATE_SOURCE_NOT_AVAILABLE_OFFLINE`

### Resumo da Proveniência:
- **Famílias Independentes Ativas**: 4 (`hierarchical`, `sentence_anchor`, `sentence_window`, `standard_chunking`)
- **Declaração Proveniência Verificada**: `multisystem_provenance_verified = True`

---

## 2. Auditoria de Mapeamento Canônico

- **Total de Candidatos Brutos**: 168
- **Mapeados Exatos/Substring**: 48 (`exact_mapped_count`)
- **Ambíguos**: 0 (`ambiguous_count`)
- **Não Mapeados**: 120 (`unmapped_count`)
- **Cobertura de Mapeamento (`mapping_coverage`)**: 0.2857 (28,57%)
- **Perda de Mapeamento Não Reportada (`unreported_mapping_loss`)**: **0** (todos os não mapeados são preservados e roteados para revisão humana)

---

## 3. Estado Real dos Arquivos Silver e Filas Humanas

### Isolamento de Mock Silver:
- **Modo de Execução**: `VALIDATION_ONLY` / `TEST_FIXTURE`
- **Flag Autoridade**: `authoritative = False`
- **Status de Calibração**: `SILVER_CALIBRATION_NOT_EXECUTED`
- **Status Qrels Silver**: `silver_qrels.jsonl` não consome nem é alimentado por registros de fixture.

### Filas de Anotação Humana (Modo Provisório sem Silver Real):
- **Status das Filas**: `PROVISIONAL_WITHOUT_SILVER`
- **Fila Anotador A**: 107 itens (Pool completo + Amostra fora do pool)
- **Fila Anotador B**: 84 itens (Casos de risco obrigatórios + Amostra aleatória)
- **Faixa de Sobreposição Alvo**: `[0.15, 0.25]` (15% a 25%)
- **Sobreposição Observada**: `0.785` (78,5%)
- **Excesso de Sobreposição**: `overlap_exceeded_due_to_mandatory_risk_cases = True` (Risco obrigatório preservado sem remoção de candidatos).

---

## 4. Testes e Validações de QA Final

| Teste / Validador | Status | Detalhes |
| :--- | :--- | :--- |
| **Pytest Suíte Completa** | PASSED | 705/705 testes aprovados (34.34s) |
| **Ruff Linter** | PASSED | All checks passed (0 errors) |
| **Mypy Type Checker** | PASSED | Success (0 errors in 59 files) |
| **Verify Reference** | PASSED | 15/15 checks passed |
| **Secret Scanner** | PASSED | 0 findings |
| **Git Diff Check** | PASSED | Clean |
| **Holdout Status** | SEALED | 100% ausente de pools e filas |

---

## 5. Declarações Autorizadas do Gate B2 Reconciliado

```text
MULTISYSTEM_POOL_PROVENANCE_VERIFIED
CANONICAL_MAPPING_OPERATIONALLY_AUDITED
OUTSIDE_POOL_AUDIT_SAMPLE_VERIFIED
MOCK_SILVER_ISOLATED
HUMAN_ROUTING_RISK_RULES_VERIFIED
HOLDOUT_SEALED
OFFLINE_QA_PASSED
GATE_B2_OPERATIONAL_READY
```
