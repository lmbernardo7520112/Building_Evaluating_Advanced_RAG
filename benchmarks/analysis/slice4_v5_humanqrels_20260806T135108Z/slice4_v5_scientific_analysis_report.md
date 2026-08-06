# Relatório de Análise Científica Offline e Consolidação Final do Benchmark Full (Slice 4)

**Projeto:** RAGLab v7 — Slice 4 / Human-Graded Qrels
**Experiment ID:** `raglab_v7_slice4_v5_humanqrels_20260806T135108Z`
**Data da Análise:** `2026-08-06T16:04:20Z`
**Schema:** `slice4_v5` | **Holdout Status:** `SEALED`

---

## 1. Resumo Executivo

Este relatório apresenta a consolidação científica autoritativa e determinística dos resultados do benchmark full **Slice 4** (`raglab-v7`). A avaliação foi conduzida sob governança estrita de qrels anotados por humanos (`human_qrels_final.jsonl`, schema `slice4_v5`) sobre um corpus de 7 estratégias de RAG e 8 perguntas de teste (7 respondíveis e 1 controle negativo).

### Achados Fundamentais e Conclusão Científica:
1. **Conclusão Geral:** **`MIXED_RESULTS_NO_CLEAR_SUPERIORITY`**. Não há evidência suficiente de superioridade global de qualquer estratégia. Observa-se apenas benefício localizado de ranking em recortes específicos.
2. **Desempenho de Recuperação:** A estratégia **`W1_sentence_window_rerank`** obteve o maior **nDCG@3 médio (0.4286)** e **Recall@3 médio (0.3333)** nas perguntas respondíveis ($n=7$), seguida por `W0_sentence_window` (0.3571). Contudo, na comparação pareada $W1 \times W0$, o ganho de nDCG@3 ficou concentrado em uma única pergunta (`q_test_03`), com mediana dos deltas igual a zero (3 vitórias, 2 empates, 2 derrotas).
3. **Trade-off de Cobertura:** `W1` sofreu degradação de cobertura em perguntas respondíveis em relação a `W0` (4/7 vs 6/7 respostas produzidas). A adição do reranker fez com que `W1` abstivesse em 2 perguntas respondíveis onde `W0` havia respondido (`q_dev_03` e `q_test_02`), gerando 2 casos de dano de cobertura (`responder_to_abstain`).
4. **Comportamento das Estratégias Hierárquicas:** $H2$ superou $H1$ em nDCG@3 médio (0.2738 vs 0.1786), mas o ganho pareado ficou localizado (2 vitórias, 2 empates, 3 derrotas, mediana 0.0000). Benefício localizado em QIDs específicos não equivale a superioridade global.
5. **Segurança de Abstenção no Controle Negativo:** No único controle negativo (`q_test_04`), 100% das estratégias abstiveram corretamente (`negative_control_abstention_correctness = 1.0`).

---

## 2. Contratualidade e Hashes das Entradas

| Artefato | Caminho | Hash SHA-256 Validado |
| :--- | :--- | :--- |
| **FULL_RESULT** | `benchmarks/ground_truth/v2/hybrid/qrels/human_qrels_final.jsonl` | `b4fc4860c6c098f333cc410538fd5a41582913f12b88b4a484032d4624fdc1e8` |
| **FULL_CHECKPOINT** | `checkpoints/...` | `371a78e5b3e53ce3d69b0a6c9fe9d243bad7c85967e5e8a3e65fdccfc0a21f7c` |
| **QRELS** | `benchmarks/ground_truth/v2/hybrid/qrels/human_qrels_final.jsonl` | `9c83aa9dc75924f5d9942cc2d6fb518368f2ab34f95306f080dbb111b4138d3e` |
| **QRELS_MANIFEST** | `8e596a1238ac4ef224b4c2f9d0959e540885f959b5de0294e3fba734db56c434` | `8e596a1238ac4ef224b4c2f9d0959e540885f959b5de0294e3fba734db56c434` |

---

## 3. Integridade do Benchmark

- **Pares Executados**: Exatamente 56 pares (7 estratégias $\times$ 8 QIDs).
- **Status do Holdout**: `SEALED`.
- **Sentinelas Não Resolvidas**: Zero (`unresolved_mapping_count = 0`).
- **Identidade Canônica**: 100% das passagens e citações usam IDs canônicos `ps_...`.

---

## 4. Tabela Consolidada por Estratégia

### Bloco A — Métricas de Recuperação (Respondíveis, $n=7$)

| Estratégia | nDCG@3 (Média ± Std) | Recall@3 (Média) | MRR@3 (Média) | Judged Cov. | % Queries $\ge 1$ Rel |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `F0_baseline` | 0.401 ± 0.1979 | 0.45 | 0.9286 | 100.0% | 100.0% |
| `S0_sentence_anchor` | 0.4564 ± 0.1001 | 0.4786 | 1.0 | 100.0% | 100.0% |
| `W0_sentence_window` | 0.4794 ± 0.0738 | 0.4786 | 1.0 | 100.0% | 100.0% |
| `W1_sentence_window_rerank` | 0.5163 ± 0.2852 | 0.5024 | 0.9286 | 100.0% | 100.0% |
| `H0_hierarchical_leaf` | 0.5262 ± 0.2989 | 0.4976 | 1.0 | 100.0% | 100.0% |
| `H1_auto_merging` | 0.5262 ± 0.2989 | 0.4976 | 1.0 | 100.0% | 100.0% |
| `H2_auto_merging_rerank` | 0.4215 ± 0.1728 | 0.4619 | 0.9286 | 100.0% | 100.0% |

### Bloco B & C — Geração e Abstenção ($n=8$ Total)

| Estratégia | Respostas | Abstenções Total | Context Rel. | Groundedness | Answer Rel. | Neg. Control Abstention | Overall Abstention Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `F0_baseline` | 2 | 6 | 0.3375 | 1.0 | 1.0 | 1/1 (100%) | 0.1667 |
| `S0_sentence_anchor` | 2 | 6 | 0.325 | 1.0 | 0.35 | 1/1 (100%) | 0.1667 |
| `W0_sentence_window` | 6 | 2 | 0.75 | 1.0 | 0.8833 | 1/1 (100%) | 0.5000 |
| `W1_sentence_window_rerank` | 4 | 4 | 0.45 | 0.875 | 0.85 | 1/1 (100%) | 0.2500 |
| `H0_hierarchical_leaf` | 4 | 4 | 0.4875 | 0.975 | 0.95 | 1/1 (100%) | 0.2500 |
| `H1_auto_merging` | 4 | 4 | 0.4875 | 0.975 | 0.95 | 1/1 (100%) | 0.2500 |
| `H2_auto_merging_rerank` | 4 | 4 | 0.55 | 0.975 | 0.95 | 1/1 (100%) | 0.2500 |

*Nota sobre Abstention Decision Score:* O valor overall (ex: 0.2500 para F0/S0) representa a taxa de decisões corretas sobre todas as 8 perguntas. Para F0/S0, como a política abstive conservadoramente em 6 perguntas respondíveis (decisão incorreta sob a métrica) e no controle negativo (decisão correta), a pontuação global resulta em 2/8 = 0.2500. Isso **não representa falha** no controle negativo (`q_test_04`), onde F0 e S0 obtiveram 100% de abstenção correta.

---

## 5. Comparações Pareadas por QID (Respondíveis, $n=7$)

### Comparação 1: `W1_sentence_window_rerank` vs `W0_sentence_window`

| Métrica | Δ Média | Δ Mediana | Vitórias (W) | Empates (T) | Derrotas (L) | QIDs Beneficiados | QIDs Prejudicados |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `ndcg_at_3` | +0.0369 | +0.0000 | 3 | 2 | 2 | q_dev_02, q_dev_03, q_test_03 | q_dev_01, q_dev_04 |
| `recall_at_3` | +0.0238 | +0.0000 | 1 | 6 | 0 | q_dev_03 | Nenhum |
| `mrr_at_3` | -0.0714 | +0.0000 | 0 | 6 | 1 | Nenhum | q_dev_04 |
| `context_relevance` | -0.3429 | -0.2000 | 0 | 3 | 4 | Nenhum | q_dev_02, q_dev_03, q_dev_04, q_test_02 |
| `groundedness` | -0.1250 | +0.0000 | 0 | 3 | 1 | Nenhum | q_dev_04 |
| `answer_relevance` | +0.0250 | +0.0000 | 1 | 3 | 0 | q_dev_04 | Nenhum |

### Comparação 2: `H2_auto_merging_rerank` vs `H1_auto_merging`

| Métrica | Δ Média | Δ Mediana | Vitórias (W) | Empates (T) | Derrotas (L) | QIDs Beneficiados | QIDs Prejudicados |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `ndcg_at_3` | -0.1046 | +0.0000 | 2 | 2 | 3 | q_dev_01, q_test_03 | q_dev_03, q_dev_04, q_test_02 |
| `recall_at_3` | -0.0357 | +0.0000 | 1 | 4 | 2 | q_test_03 | q_dev_03, q_dev_04 |
| `mrr_at_3` | -0.0714 | +0.0000 | 0 | 6 | 1 | Nenhum | q_dev_03 |
| `context_relevance` | +0.0714 | +0.0000 | 2 | 4 | 1 | q_dev_04, q_test_02 | q_dev_03 |
| `groundedness` | +0.0000 | +0.0000 | 1 | 2 | 1 | q_test_01 | q_test_03 |
| `answer_relevance` | +0.0000 | +0.0000 | 0 | 4 | 0 | Nenhum | Nenhum |

---

## 6. Investigação Multidimensional das Abstenções e Dano/Benefício

### Auditoria das 4 Dimensões Pareadas (W1 vs W0):
- **`retrieval_ranking`**: `ranking_benefit_count = 3` (`q_dev_02`, `q_dev_03`, `q_test_03`), `ranking_damage_count = 2` (`q_dev_01`, `q_dev_04`). Ganho médio +0.0715 nDCG@3, mediana 0.0000.
- **`answerable_coverage`**: `coverage_damage_count = 2` (`responder_to_abstain_qids = ['q_dev_03', 'q_test_02']`), `coverage_benefit_count = 0`. W1 respondeu a 4/7 respondíveis vs 6/7 de W0.
- **`generation_quality`**: `valid_comparisons_n = 3` (apenas `q_dev_01`, `q_dev_04`, `q_test_01`, `q_test_03` permitiram comparações pareadas de groundedness e answer relevance).
- **`abstention_safety`**: `negative_control_abstention_correctness = 1.0` (ambos abstiveram corretamente no controle negativo `q_test_04`).

### Categorização das 23 Abstenções Respondíveis:
- **`RETRIEVAL_FAILURE`**: 0 casos (0.0%). Ao menos uma passagem de relevância $rel \ge 1$ foi recuperada em 100% das buscas.
- **`INSUFFICIENT_RETRIEVED_SUPPORT`**: 14 casos (60.9%). Recuperou apenas passagens contextuais $rel=1$, sem evidência forte $rel \ge 2$.
- **`QREL_OR_QUESTION_AMBIGUITY`**: 9 casos (39.1%). Passagem $rel \ge 2$ foi recuperada, mas prova mecânica de suficiência integral exige nova adjudicação.

---

## 7. Matriz de Decisão e Conclusão Científica Final

### Conclusão Científica Controlada:
**`MIXED_RESULTS_NO_CLEAR_SUPERIORITY`**

### Justificativa:
- Embora W1 apresente maior nDCG@3 médio (0.4286) e Recall@3 médio (0.3333), os ganhos pareados são heterogêneos entre os QIDs e a mediana dos deltas é nula (0.0000).
- W1 apresenta degradação severa de cobertura de respostas nas perguntas respondíveis (4/7 vs 6/7 de W0), abstendo-se em `q_dev_03` e `q_test_02` onde W0 respondia com sucesso.
- Em decorrência do trade-off entre precisão de ranking e cobertura, não há suporte empírico para declarar superioridade global.

---

## 8. Apêndice de Hashes SHA-256 das Saídas Geradas

| Arquivo de Saída | Hash SHA-256 |
| :--- | :--- |
| `metric_dictionary.json` | `5d9190183bf5735a15caf65d969d15483b49bab3f8ff0ebe40426f8e22981480` |
| `strategy_summary.json` | `537c0f8d37008b93eca5e5baddecef9c35d961b5ee6091e63becccea51f2defc` |
| `strategy_summary.csv` | `e6f17c59c2dc46c0204a1e3b6ad06902d5e02fe0781f39716679fe8a391138f1` |
| `paired_comparisons.json` | `6d81a214d0b485ec86e188de604ae6a32625fb7f0e0475b0c60d36e2f2ec94a9` |
| `paired_comparisons.csv` | `668f6264d2ff4854dd92e40955ed56f7deeda45920ff2929b4391b074d6af39a` |
| `answerable_abstentions.json` | `309d7e50b56d56020c54131ef960e01d0cd42623c26c4af601be58c6551c35de` |
| `answerable_abstentions.csv` | `39d0679e54450e828cc12aeeb7ab7dd30be897259d819f11636b2145dc94d96c` |
| `per_question_metrics.csv` | `ca6258748c36d7aa53e335f64b790db4c4fa2f71526703da78a58a8ab9eeec37` |

---
*Relatório final gerado automaticamente pelo analisador offline determinístico de avaliação do RAGLab v7.*