# Documento de Fechamento Científico e Governança — Slice 5A.3 (V3 Final)

> **Status Final do Slice 5A.3 (V3)**: `SLICE5A3_OPERATIONAL_COMPLETE`  
> **Classificação Científica**: `SLICE5A3_STATUS_NOT_EVALUABLE_JUDGED_COVERAGE`  
> **Conclusão Metodológica**: `NO_SUPERIORITY_CLAIM` (Inadmissível inferir superioridade do pipeline agentivo devido à cobertura humana insuficiente nos 7 braços de recuperação).

---

## 1. Metadados Autoritativos do Pré-Registro V3

- **Identificador do Protocolo**: `slice5a3_parity_gate_v3`
- **SHA-256 do Pré-Registro**: `6fe435e2abc98eff05728370dfff7bb0d64744a22d0c94a77331a194c20d3290`
- **Commit da Implementação (Código Congelado)**: `856a91a1caefcf5fdd1e09c853f090b8ce1bf297`
- **Commit do Pré-Registro V3**: `53807b5597793d5fbab1b54a72dcfebcbeeb0efc`
- **Path do Pré-Registro**: [preregistration_v3.json](file:///home/leonardomaximinobernardo/Downloads/huggingface_agents_course_bootcamp_AI_triggo_2026/Building_Evaluating_Advanced_RAG/raglab-v7/benchmarks/agentic/slice5/slice5a3/protocols/preregistration_v3.json)
- **Modo de Execução**: 100% Offline (Sem APIs pagas, sem Gemini, sem credenciais externas).

---

## 2. Resumo Executivo e Conclusão Científica Controlada

A remediação governada V3 do **Slice 5A.3 (Parity Gate para RAG Agentivo)** foi concluída com sucesso operacional sob o protocolo estritamente pré-registrado. 

### Conclusão Científica Oficial e Imutável:
> *"O Slice 5A.3 não demonstrou superioridade. A execução controlada V3 confirmou que a cobertura de julgamentos humanos disponível em `human_qrels_final.jsonl` (49.4%) é insuficiente para avaliar inferencialmente o Parity Gate do pipeline agentivo frente aos 7 braços de recuperação pré-registrados."*

---

## 3. Tabela Consolidada de Cobertura e Reprodutibilidade (Execuções A & B)

| Métrica / Indicador | Execução A (Run A) | Execução B (Run B) | Status do Gate |
| :--- | :--- | :--- | :--- |
| **Status de Avaliabilidade** | `NOT_EVALUABLE_JUDGED_COVERAGE` | `NOT_EVALUABLE_JUDGED_COVERAGE` | **FAIL-CLOSED OK** |
| **QIDs do Split DEV** | 4 (`q_dev_01`..`q_dev_04`) | 4 (`q_dev_01`..`q_dev_04`) | **CONFORME** |
| **Braços de Recuperação** | 7 (`F0, H0, H1, H2, S0, W0, W1`) | 7 (`F0, H0, H1, H2, S0, W0, W1`) | **CONFORME** |
| **Top-K Solicitado** | 3 | 3 | **CONFORME** |
| **Slots Técnicos Esperados** | 84 | 84 | **CONFORME** |
| **Itens Técnicos Retornados** | 83 | 83 | **EXPLICAÇÃO FORMAL OK** |
| **Cobertura Canônica (`canonical_coverage`)** | **100.0%** (83/83) | **100.0%** (83/83) | **PASS (Target >= 1.0)** |
| **Itens Não Mapeados (`unmapped_count`)** | **0** | **0** | **PASS (Target == 0)** |
| **Projeções Ambíguas (`ambiguous_count`)** | **0** | **0** | **PASS (Target == 0)** |
| **Cobertura Julgada (`judged_coverage`)** | **49.4%** (41/83) | **49.4%** (41/83) | **NOT EVALUABLE (< 1.0)** |
| **Fila de Revisão Humana (`unjudged_count`)** | **42 itens** | **42 itens** | **FILA GERADA OK** |
| **Ratings Sintéticos Utilizados** | **0** | **0** | **PROIBIÇÃO CUMPRIDA** |
| **Repetibilidade Top-K Identity** | **1.0** | **1.0** | **PASS (Target == 1.0)** |
| **Paridade de Rank (`rank_match`)** | **True** | **True** | **PASS** |
| **Completude de Braços (`same_run_arm_completeness`)** | **True** | **True** | **PASS** |
| **Métricas nDCG/MRR** | `NOT_APPLICABLE` (`null`) | `NOT_APPLICABLE` (`null`) | **GOVERNANÇA CUMPRIDA** |

---

## 4. Explicação Empírica dos 83 vs 84 Slots Técnicos Retornados

A auditoria empírica do pipeline comprovou de forma determinística por que exatamente 83 itens foram retornados em vez de 84:
- Para `q_dev_01`, `q_dev_02` e `q_dev_03`, todos os 7 braços retornam top_k=3 completo ($3 \times 7 = 21$ itens por QID, totalizando 63 itens).
- Para `q_dev_04`, 6 braços (`F0, H0, H2, S0, W0, W1`) retornam top_k=3 completo (18 itens).
- O braço `H1` (`H1_auto_merging`) sob a consulta `q_dev_04` retorna **apenas 2 itens** (rank 1 e rank 2).
- **Razão Técnica**: O algoritmo de auto-merging hierárquico consolida nós folha filhos no seu nó pai quando o limiar de fusão é atingido. Para `q_dev_04`, restam apenas 2 nós pais candidatos únicos na estrutura da árvore, esgotando o conjunto de candidatos antes do rank 3.

---

## 5. Auditoria de Identidade e Isolamento de Contratos

1. **Separação Rígida**: O campo técnico `technical_chunk_id` (ex: `doc_p91_s0` ou UUIDs in-memory) e o identificador nó `technical_node_id` são isolados dos campos canônicos.
2. **Proibição de Chaining/Fallback**: A propriedade `passage_id` do item retornado devolve exclusivamente `anchor_passage_id or ""`. Em caso de item não mapeado, a propriedade devolve `""` (e JAMAIS o ID técnico).
3. **Offset Canônico e Projeção**: Todos os 83 itens mapeados apontam para passagens canônicas válidas no `passage_registry.jsonl` com offset exato e hash SHA-256 verificado.

---

## 6. Auditoria da Fila de Revisão Humana (`human_review_queue.jsonl`)

As execuções A e B geraram uma fila de revisão humana limpa e não contaminada contendo exatamente **42 itens unjudged**:
- Nenhum item possui `suggested_grade`, `gold` ou `silver` atribuído sinteticamente.
- Todos os 42 itens registram com precisão: `qid`, `passage_id` (canônico `ps_*`), `page_number`, `text`, `arms_recovering` e `ranks_recovering`.
- O artefato está disponível em [human_review_queue.jsonl](file:///home/leonardomaximinobernardo/Downloads/huggingface_agents_course_bootcamp_AI_triggo_2026/Building_Evaluating_Advanced_RAG/raglab-v7/benchmarks/agentic/slice5/slice5a3/results/human_review_queue.jsonl) para posterior anotação humana offline.

---

## 7. Verificação dos Gates de Encerramento (Phase 10)

- [x] **Branch**: `feat/agentic-rag-slice5a3-parity-gate`
- [x] **Commit da Implementação**: `856a91a1caefcf5fdd1e09c853f090b8ce1bf297`
- [x] **Commit do Pré-Registro V3**: `53807b5597793d5fbab1b54a72dcfebcbeeb0efc`
- [x] **Sem push, PR ou merge**: Mantido estritamente em ambiente local.
- [x] **Sem alteração do histórico Git**: Histórico preservado e sequencial.
- [x] **24 Arquivos Não Rastreados Preexistentes**: 100% preservados e intocados.
- [x] **Zero Alegações de Superioridade**: Conclusão nula e fail-closed respeitada.

---

## 8. Próximos Passos e Transição para Slice 5B

Com o encerramento probatório definitivo e irrevogável do **Slice 5A.3 (V3)**:
1. O repositório permanece no estado fail-closed `NOT_EVALUABLE_JUDGED_COVERAGE`.
2. O inventário de 42 itens em `human_review_queue.jsonl` servirá de entrada exclusiva para a campanha offline de anotação humana prévia a futuras avaliações.
3. Não haverá novas tentativas de reavaliação ou refatoração no Slice 5A. O trabalho do Slice 5A está 100% concluído.
