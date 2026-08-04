# Relatório de Governança — Gate B2: Conjunto de Avaliação Híbrido Humanamente Validado

- **Status do Gate**: READY_FOR_CONTROLLED_SILVER_TRIAGE_AND_HUMAN_REVIEW
- **Branch**: `feat/hybrid-human-validated-eval`
- **Protocolo**: `raglab_v7_slice4_v3`
- **Schema de Artefato**: `2.0.0`
- **Nível de Evidência Pretendido**: **E3 — COMPARAÇÃO CONTROLADA NO RECORTE**

---

## 1. Decisão Metodológica

Substituição da política de anotação exaustiva dupla (`DOUBLE_EXHAUSTIVE_HUMAN_ANNOTATION`) pela política **`HUMAN_VALIDATED_HYBRID_EVAL_SET`**.

### Justificativa Econômica e Técnica:
- **Redução do Custo Operacional**: Elimina a necessidade de 1.968 anotações humanas redundantes em passagens irrelevantes.
- **Foco Orientado por Risco**: Direciona o esforço humano dos Anotadores A e B para os candidatos de maior incerteza, divergência e positivos da triagem automática.
- **Fronteira Rígida de Autoridade**: Separação completa entre rótulos automáticos (`MACHINE_SILVER`) e qrels autoritativas (`HUMAN_GOLD`, `HUMAN_VALIDATED`, `HUMAN_ADJUDICATED`).

---

## 2. Resumo da Infraestrutura Construída

1. **Candidate Pool Multissistema (`candidate_pool/`)**:
   - `pool.jsonl` (Visão de Auditoria Interna com proveniente de busca);
   - `blinded_pool.jsonl` (Visão Cega para Anotadores, sem qualquer metadado de estratégia/score/rank);
   - `mapping_audit.json` (Mapeamento canônico 100% auditado, `unreported_mapping_loss = 0`).
2. **Runner e Contratos Silver (`silver/`)**:
   - Prompts de triagem com barreira estrita contra injection (`passage_text` tratado como não confiável);
   - Infrastructure runner resiliente (`scripts/run_silver_annotation.py`) com modos `validate-only`, `smoke`, `full` e `resume`.
3. **Filas de Anotação Humana Orientadas por Risco (`human_queues/`)**:
   - `annotator_a.jsonl`: Avaliação primária (pool completo + amostragem fora do pool + casos ambíguos + abstenção);
   - `annotator_b.jsonl`: Avaliação secundária (positivos silver + desacordos + baixa confiança + sobreposição planejada de 20% com A);
   - `adjudication.jsonl`: Template de adjudicação para conflitos.
4. **Ferramenta de Calibração (`scripts/calibrate_silver_against_human.py`)**:
   - Pronta para calcular Matriz de Confusão, Kappa, FNR da classe relevante e Recall relevante contra alvos predefinidos (`TARGET_RELEVANT_RECALL = 0.95`).
   - Retorna status explícito `SILVER_CALIBRATION_NOT_EXECUTED` na ausência de anotações humanas.

---

## 3. Matriz de Integridade e Segurança

| Item de Verificação | Status Autorizado |
| :--- | :--- |
| **Corpus SHA-256 (Gersting p. 91-115)** | `33e2e9f1e190158b3e99c19fced1acd050720247c7556780bad82b2f93bf1254` |
| **Passage Registry SHA-256** | `c2d1b31e5eaeb98e2a31094628cf2e4291c4ff73bb7ddb1f79929204163799e1` |
| **Holdout Sealing** | **SEALED (100% ausente de pools e filas)** |
| **Chamadas API Gemini** | **0 (GEMINI_NOT_CALLED)** |
| **Acesso a Credenciais** | **0 (CREDENTIALS_NOT_ACCESSED)** |
| **Uso de Rede** | **0 (NETWORK_NOT_USED)** |
| **Anotações Humanas** | **HUMAN_ANNOTATION_NOT_EXECUTED** |
| **Respostas Gold** | **GOLD_ANSWERS_NOT_CREATED** |
| **Calibração Silver** | **SILVER_CALIBRATION_NOT_EXECUTED** |
| **Benchmark Completo** | **FULL_BENCHMARK_NOT_EXECUTED** |

---

## 4. Declarações Autorizadas do Gate B2

```text
HYBRID_EVAL_INFRASTRUCTURE_READY
EVIDENCE_LEVEL_TARGET_E3
MACHINE_SILVER_NOT_HUMAN_GOLD
SILVER_CALIBRATION_NOT_EXECUTED
HUMAN_ANNOTATION_NOT_EXECUTED
GOLD_ANSWERS_NOT_CREATED
GEMINI_NOT_CALLED
CREDENTIALS_NOT_ACCESSED
NETWORK_NOT_USED
HOLDOUT_SEALED
FULL_BENCHMARK_NOT_EXECUTED
READY_FOR_CONTROLLED_SILVER_TRIAGE_AND_HUMAN_REVIEW
```
