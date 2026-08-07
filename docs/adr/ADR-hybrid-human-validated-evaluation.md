# ADR: Hybrid Human-Validated Evaluation Protocol (Ground Truth v2)

- **Status**: APPROVED / IMPLEMENTED
- **Data**: 2026-08-04
- **Nível de Evidência Pretendido**: **E3 — COMPARAÇÃO CONTROLADA NO RECORTE**
- **Autores**: Equipe de Avaliação RAGLab v7

---

## 1. Contexto e Problema

O protocolo inicial do Ground Truth v2 previa a anotação humana exaustiva e dupla (`DOUBLE_EXHAUSTIVE_HUMAN_ANNOTATION`) sobre todas as passagens canônicas do corpus registradas em `passage_registry.jsonl` (123 passagens nas páginas físicas 91–115 da obra de Gersting).

### Análise Econômica e Operacional:
- **Custo Operacional Total**: $123 \text{ passagens} \times 8 \text{ perguntas} \times 2 \text{ anotadores} = 1.968 \text{ julgamentos individuais}$.
- **Rendimento Factual**: Devido à esparsidade inerente dos conjuntos de evidência em RAG, mais de 92% das passagens anotadas em um esquema exaustivo cego são estritamente irrelevantes (Grau 0), gerando alto custo humano sem incremento proporcional na precisão da avaliação de ranking.
- **Limitação Metodológica**: O recorte avaliado possui 8 perguntas ativas (dev/test), o que impede conclusões universais sobre a generalização de estratégias de RAG fora deste corpus e recorte.

---

## 2. Decisão Arquitetural

Substituir a política padrão `DOUBLE_EXHAUSTIVE_HUMAN_ANNOTATION` pela política **`HUMAN_VALIDATED_HYBRID_EVAL_SET`**.

A nova arquitetura introduz quatro pilares metodológicos:
1. **Pooling Multissistema Offline**: União deduplicada de candidatos recuperados por múltiplas famílias RAG (Baseline F0, Sentence-Window W0/W1, Hierárquico H0/H1/H2, busca lexical e densa) acrescida de vizinhos canônicos e controles negativos.
2. **Machine Silver (Triagem por Juiz LLM)**: Rotulagem automática preliminar identificada explicitamente como `MACHINE_SILVER`, utilizada exclusivamente para triagem e priorização de risco.
3. **Revisão Humana Orientada por Risco**: Roteamento direcionado das passagens de maior incerteza, desacordo ou relevância para os Anotadores A e B.
4. **Auditoria Aleatória Fora do Pool**: Amostragem determinística fora do pool para detectar falsos negativos silenciosos e auditar a cobertura do pooling.
5. **Fronteira Rígida de Autoridade**: Separação estrita entre qrels humanas (`human_qrels.jsonl`) e qrels automáticas (`silver_qrels.jsonl`).

---

## 3. Alegações Permitidas e Proibidas

### Alegação Máxima Permitida (Nível E3):
> *"No recorte e no conjunto de perguntas avaliados, a estratégia X apresentou melhor desempenho nas métricas Y, segundo qrels humanamente validadas."*

### Alegações Proibidas:
- ❌ *"Sentence-window é superior em geral."*
- ❌ *"Auto-merging é superior em RAG."*
- ❌ *"Técnicas avançadas são universalmente superiores."*
- ❌ *"MACHINE_SILVER é Ground Truth humano."*
- ❌ *"Passagem não julgada significa estritamente irrelevante (relevance_grade = 0)."*
- ❌ *"Dois agentes com o mesmo modelo LLM configuram anotadores independentes."*

---

## 4. Referências Bibliográficas e Metodológicas

- **TREC / NIST Pooling**: Harman, D. (1993). *Overview of the First Text REtrieval Conference (TREC-1)*. NIST Special Publication. (Uso de pooling de múltiplos sistemas de busca para construção eficiente de coleções de teste).
- **ARES / PPI (Prediction-Powered Inference)**: Saad-Falcon et al. (2023). *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems*; Angelopoulos et al. (2023). *Prediction-Powered Inference*. (Premissas de calibração e inferência assistida por modelos com garantia de validade).
- **Selective Evaluation & Human-in-the-Loop**: Spärck Jones, K., & van Rijsbergen, C. J. (1976). *Report on the need for documentation and statistics of test collections*. (Princípios de auditoria de não julgados e controle de esparsidade de relevância).

---

## 5. Status do Pacote Exaustivo Legado

O pacote de anotação exaustiva de 1.968 itens gerado no Gate B1 é preservado intacto e reclassificado como:
`EXHAUSTIVE_ANNOTATION_TEMPLATE_NOT_SELECTED_AS_DEFAULT`.
