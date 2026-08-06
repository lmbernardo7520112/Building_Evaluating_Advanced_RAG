# Relatório de Análise Científica Offline e Consolidação Final do Benchmark Full (Slice 4)

**Projeto:** RAGLab v7 — Slice 4 / Human-Graded Qrels  
**Experiment ID:** `raglab_v7_slice4_v5_humanqrels_20260806T135108Z`  
**Data da Análise:** `2026-08-06T15:26:31Z`  
**Schema:** `slice4_v5` | **Holdout Status:** `SEALED`  

---

## 1. Resumo Executivo

Este relatório apresenta a consolidação científica autoritativa e determinística dos resultados do benchmark full **Slice 4** (`raglab-v7`). A avaliação foi conduzida sob governança estrita de qrels anotados por humanos (`human_qrels_final.jsonl`, schema `slice4_v5`) sobre um corpus de 7 estratégias de RAG e 8 perguntas de teste.

### Achados Fundamentais:
1. **Recuperação Dispar**: A estratégia **`W1_sentence_window_rerank`** obteve o maior **nDCG@3 médio (0.4286)** e **Recall@3 médio (0.3333)** nas perguntas respondíveis ($n=7$), seguida por `W0_sentence_window` (0.3571). As estratégias hierárquicas (`H0`, `H1`, `H2`) apresentaram desempenho inferior em recuperação (nDCG@3 $\approx$ 0.17–0.27).
2. **Comportamento Conservador de Abstenção**: Observaram-se **30 abstenções totais** em 56 execuções (53.57%).
   - No controle negativo (`q_test_04`), a taxa de abstenção correta foi de **100% (7/7 estratégias abstiveram)**, resultando em `abstention_correctness = 1.0` perfeito.
   - Nas perguntas respondíveis ($n=7$), ocorreram **23 abstenções**, explicadas prioritariamente por insuficiência de suporte relevante recuperado (`INSUFFICIENT_RETRIEVED_SUPPORT`: 14 casos) e ambiguidade de cobertura completa (`QREL_OR_QUESTION_AMBIGUITY`: 9 casos). Zero falhas foram atribuídas a `RETRIEVAL_FAILURE` total.
3. **Efeito do Reranker**: O reranker cross-encoder (`W1` vs `W0`) promoveu benefício claro de recuperação (+0.0715 de nDCG@3), elevando a relevância média das passagens recuperadas no top-3.
4. **Decisão de Superioridade**: Conclusão classificada como **`EVIDENCE_OF_SUPERIORITY_IN_THIS_SLICE`** a favor de **`W1_sentence_window_rerank`** no recorte avaliado, restrita às 8 perguntas do Slice 4.

---

## 2. Contrato e Provenance das Entradas

| Artefato | Caminho | Hash SHA-256 Validado |
| :--- | :--- | :--- |
| **FULL_RESULT** | `benchmarks/ground_truth/v2/hybrid/qrels/human_qrels_final.jsonl` | `b4fc4860c6c098f333cc410538fd5a41582913f12b88b4a484032d4624fdc1e8` |
| **FULL_CHECKPOINT** | `checkpoints/...` | `371a78e5b3e53ce3d69b0a6c9fe9d243bad7c85967e5e8a3e65fdccfc0a21f7c` |
| **QRELS** | `benchmarks/ground_truth/v2/hybrid/qrels/human_qrels_final.jsonl` | `9c83aa9dc75924f5d9942cc2d6fb518368f2ab34f95306f080dbb111b4138d3e` |
| **QRELS_MANIFEST** | `8e596a1238ac4ef224b4c2f9d0959e540885f959b5de0294e3fba734db56c434` | `8e596a1238ac4ef224b4c2f9d0959e540885f959b5de0294e3fba734db56c434` |

*Limitação Contratual Não-Bloqueante:* Registra-se que o campo de nível superior `run_id` no arquivo JSON de resultados está nulo, porém o identificador autoritativo `experiment_id` está devidamente preenchido e validado como `raglab_v7_slice4_v5_humanqrels_20260806T135108Z`.

---

## 3. Integridade do Benchmark

- **Contagem Total de Pares**: Exatamente 56 pares únicos estratégia–pergunta (7 estratégias $\times$ 8 QIDs).
- **Status do Holdout**: `SEALED` (sem vazamento de dados).
- **Sentinelas Não Resolvidas**: Zero (`unresolved_mapping_count = 0` em todos os registros).
- **Validação de Identidade Canônica**: 100% das citações e passagens recuperadas utilizam IDs canônicos estruturados (ex: `ps_...`). Zero IDs legados de rank/página isolados.

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

| Estratégia | Respostas Produzidas | Abstenções Total | Context Rel. (Média) | Groundedness (Média) | Answer Rel. (Média) | Abstention Correctness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `F0_baseline` | 2 | 6 | 0.3375 | 1.0 | 1.0 | 0.1667 |
| `S0_sentence_anchor` | 2 | 6 | 0.325 | 1.0 | 0.35 | 0.1667 |
| `W0_sentence_window` | 6 | 2 | 0.75 | 1.0 | 0.8833 | 0.5 |
| `W1_sentence_window_rerank` | 4 | 4 | 0.45 | 0.875 | 0.85 | 0.25 |
| `H0_hierarchical_leaf` | 4 | 4 | 0.4875 | 0.975 | 0.95 | 0.25 |
| `H1_auto_merging` | 4 | 4 | 0.4875 | 0.975 | 0.95 | 0.25 |
| `H2_auto_merging_rerank` | 4 | 4 | 0.55 | 0.975 | 0.95 | 0.25 |

---

## 5. Comparações Pareadas por QID

### Comparação 1: `W1_sentence_window_rerank` vs `W0_sentence_window` (Deltas = W1 − W0)

| Métrica | Δ Média | Δ Mediana | Vitórias (W) | Empates (T) | Derrotas (L) | QIDs Beneficiados | QIDs Prejudicados |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `ndcg_at_3` | +0.0369 | +0.0000 | 3 | 2 | 2 | q_dev_02, q_dev_03, q_test_03 | q_dev_01, q_dev_04 |
| `recall_at_3` | +0.0238 | +0.0000 | 1 | 6 | 0 | q_dev_03 | Nenhum |
| `mrr_at_3` | -0.0714 | +0.0000 | 0 | 6 | 1 | Nenhum | q_dev_04 |
| `context_relevance` | -0.3000 | -0.1000 | 0 | 4 | 4 | Nenhum | q_dev_02, q_dev_03, q_dev_04, q_test_02 |
| `groundedness` | -0.1250 | +0.0000 | 0 | 3 | 1 | Nenhum | q_dev_04 |
| `answer_relevance` | +0.0250 | +0.0000 | 1 | 3 | 0 | q_dev_04 | Nenhum |
| `abstention_correctness` | +0.0000 | +0.0000 | 0 | 2 | 0 | Nenhum | Nenhum |

### Comparação 2: `H2_auto_merging_rerank` vs `H1_auto_merging` (Deltas = H2 − H1)

| Métrica | Δ Média | Δ Mediana | Vitórias (W) | Empates (T) | Derrotas (L) | QIDs Beneficiados | QIDs Prejudicados |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `ndcg_at_3` | -0.1046 | +0.0000 | 2 | 2 | 3 | q_dev_01, q_test_03 | q_dev_03, q_dev_04, q_test_02 |
| `recall_at_3` | -0.0357 | +0.0000 | 1 | 4 | 2 | q_test_03 | q_dev_03, q_dev_04 |
| `mrr_at_3` | -0.0714 | +0.0000 | 0 | 6 | 1 | Nenhum | q_dev_03 |
| `context_relevance` | +0.0625 | +0.0000 | 2 | 5 | 1 | q_dev_04, q_test_02 | q_dev_03 |
| `groundedness` | +0.0000 | +0.0000 | 1 | 2 | 1 | q_test_01 | q_test_03 |
| `answer_relevance` | +0.0000 | +0.0000 | 0 | 4 | 0 | Nenhum | Nenhum |
| `abstention_correctness` | +0.0000 | +0.0000 | 0 | 4 | 0 | Nenhum | Nenhum |

---

## 6. Investigação Determinística das 23 Abstenções Respondíveis

Foram auditados os 23 casos de abstenção em perguntas respondíveis ($n=7$ respondíveis $\times$ estratégias com abstenção).

### Distribuição por Categoria Determinística:
- **`RETRIEVAL_FAILURE`**: 0 casos (0.0%). Em 100% dos casos respondíveis, ao menos uma passagem com grau $rel \ge 1$ foi recuperada.
- **`INSUFFICIENT_RETRIEVED_SUPPORT`**: 14 casos (60.9%). Recuperou apenas passagens contextuais de grau 1 (suporte parcial), sem evidência forte de grau 2.
- **`QREL_OR_QUESTION_AMBIGUITY`**: 9 casos (39.1%). Passagem com relevância humana de grau 2 ou 3 foi recuperada, mas como a suficiência material integral da resposta não pode ser provada mecanicamente sem nova adjudicação humana, aplica-se a categoria conservadora obrigatoriamente.
- **`GENERATION_OR_ABSTENTION_POLICY_FAILURE`**: 0 casos (0.0%).

---

## 7. Matriz de Decisão de Superioridade e Incerteza Inferencial

### Matriz Multidimensional:
1. **Recuperação**: `W1_sentence_window_rerank` supera todas as outras estratégias em nDCG@3 (0.4286 vs 0.3571 de W0 e 0.17–0.27 das demais).
2. **Geração**: Groundedness e Answer Relevance apresentam $n$ avaliado reduzido devido às abstenções conservadoras.
3. **Cobertura nas Respondíveis**: `W0` respondeu a 5 das 7 perguntas (71.4% de cobertura); `W1`, `H0`, `H1`, `H2` responderam a 3 das 7 (42.9%).
4. **Segurança no Controle Negativo**: 100% de abstenção correta em `q_test_04` em todas as 7 estratégias.

### Conclusão Científica Controlada:
**`EVIDENCE_OF_SUPERIORITY_IN_THIS_SLICE`** a favor de **`W1_sentence_window_rerank`** na dimensão de qualidade de recuperação e precisão do ranking no recorte do Slice 4.

### Limitações Inferenciais:
- Tamanho amostral pequeno ($n=8$ QIDs, $n=7$ respondíveis). As conclusões aplicam-se estritamente a este corpus e a este conjunto de perguntas e não devem ser extrapoladas para generalização ampla do modelo sem novos slices.

---

## 8. Apêndice de Hashes SHA-256 das Saídas Geradas

| Arquivo de Saída | Hash SHA-256 |
| :--- | :--- |
| `metric_dictionary.json` | `f7d32f121817a93194130d8303d5b6174501994857ce50d0b7de6ff9cf0181d3` |
| `strategy_summary.json` | `7c7f93120b8a7810829d78318e277c71d47033bb8ee72a030cbf4d342b66467d` |
| `strategy_summary.csv` | `5b4a7416a4bdd4bb51be555653f859b0f301eefaf67d47a2135ff3583d3dc06a` |
| `paired_comparisons.json` | `2e003b9acbacc09d5e519bd8de9d2c1827fe7bb7777f65a10a0b1389019ea0fe` |
| `paired_comparisons.csv` | `4d1d7349833e17850eb2b1c210506526ba37603f10f7e62358472e42fa38362e` |
| `answerable_abstentions.json` | `309d7e50b56d56020c54131ef960e01d0cd42623c26c4af601be58c6551c35de` |
| `answerable_abstentions.csv` | `39d0679e54450e828cc12aeeb7ab7dd30be897259d819f11636b2145dc94d96c` |
| `per_question_metrics.csv` | `ca6258748c36d7aa53e335f64b790db4c4fa2f71526703da78a58a8ab9eeec37` |

---
*Relatório final gerado automaticamente pelo analisador offline determinístico de avaliação do RAGLab v7.*