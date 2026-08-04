# Manual de Anotação Humana do Ground Truth v2 — RAGLab v7 (Gate B1)

> [!IMPORTANT]
> Este manual é o guia autoritativo para anotação cega do dataset de avaliação de RAG (Protocolo v2).
> As anotações devem ser estritamente baseadas nas passagens canônicas do corpus registradas no `passage_registry.jsonl`.

---

## 1. Escala de Relevância Graduada (Relevance Grade 0–3)

Cada passagem candidata apresentada para uma pergunta deve receber exatamente uma nota de relevância numérica de **0 a 3**:

- **`0` — Irrelevante**: A passagem não possui relação com a pergunta ou contém apenas palavras-chave coincidentes fora de contexto (controle negativo ou ruído).
- **`1` — Relacionado / Contextual (Insuficiente)**: A passagem aborda o tópico geral ou domínio da pergunta, mas **não contém informações suficientes** para responder nem parcialmente à questão.
- **`2` — Parcialmente Relevante / Suporte Útil**: A passagem responde a **uma parte da pergunta** ou fornece uma premissa/passo intermediário essencial, mas necessita de outra passagem complementar para formar uma resposta completa.
- **`3` — Diretamente Suficiente / Evidência Principal**: A passagem contém **evidência direta, explícita e suficiente** para responder integralmente à pergunta (ou à premissa central da questão).

---

## 2. Papéis Semânticos da Evidência (Evidence Role)

Para passagens com nota de relevância $\ge 1$, atribua um dos seguintes papéis:

- **`PRIMARY`**: Passagem principal contendo o trecho central da resposta.
- **`SUPPORTING`**: Passagem complementar (ex: definição de termo, premissa matemática, exemplo de apoio).
- **`CONTEXTUAL`**: Passagem de contextualização geral do capítulo ou seção.
- **`NEGATIVE_CONTROL`**: Passagem introduzida propositalmente para testar se o modelo/anotador evita falsos positivos.

---

## 3. Classificação de Respondibilidade (Answerability)

Para cada pergunta, o anotador deve classificar se a pergunta é respondível **exclusivamente com base no corpus fornecido**:

- **`ANSWERABLE`**: O corpus contém evidências suficientes para responder à pergunta de forma factual e completa.
- **`UNANSWERABLE_INFORMATION_ABSENT`**: A informação solicitada não consta no corpus (ex: fatos do mundo externo, atualidades).
- **`UNANSWERABLE_AMBIGUOUS_QUESTION`**: A pergunta é formulada de maneira ambígua ou contraditória.
- **`UNANSWERABLE_INSUFFICIENT_EVIDENCE`**: O corpus trata do tema, mas os trechos disponíveis na amostra são insuficientes para garantir uma resposta segura.

---

## 4. Regras Fundamentais de Anotação

1. **Julgamento Exclusivo pelo Corpus**: Avalie as passagens **somente** com base no texto exato apresentado. É **proibido** utilizar conhecimento prévio externo ou assumir fatos não escritos.
2. **Cegamento de Número de Página e Ranking**: Não infira relevância pelo número da página nem assuma que a primeira passagem da lista é melhor. As passagens foram apresentadas em ordem embaralhada.
3. **Isolamento entre Anotadores**: Anotadores A e B trabalham de forma independente e cega. Não consulte respostas de modelos LLM (Gemini) nem pareceres de outros anotadores.
4. **Perguntas Irrespondíveis**: Se uma pergunta for classificada como `UNANSWERABLE_*`, **não fabrique evidências artificiais**. Todos os `relevance_grade` devem ser 0 ou 1 (sem evidência primária suficiente).
5. **Suficiência Conjunta (`evidence_sets`)**: Se a resposta exigir 2 ou mais passagens conjuntamente (ex: premissa + conclusão em páginas diferentes), registre os `passage_id`s no array `evidence_sets` com `jointly_sufficient: true`.
6. **Respostas de Referência (`gold_answer`)**: Ao escrever a resposta de referência para perguntas `ANSWERABLE`, utilize apenas fatos presentes nas passagens anotadas com nota $\ge 2$ e cite explicitamente os `passage_id`s de suporte em `gold_supporting_passage_ids`.
7. **Dúvidas e Casos Limítrofes**: Qualquer incerteza deve ser registrada textualmente no campo `annotation_notes`.

---

## 5. Exemplos Didáticos de Referência

> [!NOTE]
> Os exemplos a seguir são puramente didáticos e marcados como **NÃO AUTORITATIVOS**.

### Exemplo Didático 1: Passagem Suficiente (Nota 3)
- **Pergunta**: *"O que é o princípio da indução matemática?"*
- **Passagem (`ps_didactic_01`)**: *"O Princípio da Indução Matemática estabelece que, para provar P(n) para todo n natural, deve-se provar a base P(1) e o passo indutivo P(k) => P(k+1)."*
- **Anotação**:
  - `relevance_grade`: `3`
  - `evidence_role`: `PRIMARY`
  - `answerability`: `ANSWERABLE`

### Exemplo Didático 2: Pergunta Irrespondível (Controle Negativo)
- **Pergunta**: *"Qual é a capital da França?"*
- **Passagem (`ps_didactic_02`)**: *"Na Seção 2.1 estudamos teoremas de indução e estruturas algébricas."*
- **Anotação**:
  - `relevance_grade`: `0`
  - `evidence_role`: `NEGATIVE_CONTROL`
  - `answerability`: `UNANSWERABLE_INFORMATION_ABSENT`
  - `gold_answer`: `null`
  - `gold_supporting_passage_ids`: `[]`

---

## 6. Fluxo de Trabalho e Finalização

1. Abra o arquivo JSONL designado (`annotator_a/development.jsonl` ou `annotator_b/test.jsonl`).
2. Preencha todos os campos nulos de avaliação para cada pergunta.
3. Altere o `annotation_status` de `"PENDING"` para `"COMPLETED"`.
4. Submeta o arquivo para validação via script offline `validate_human_annotations.py --mode completed`.
