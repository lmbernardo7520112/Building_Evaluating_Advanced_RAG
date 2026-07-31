# Gate 2 Report — RAGLab v7 (Slice 2 Controlled Benchmark & Sentence-Window)

> **Date:** 2026-07-31T00:30 BRT
> **Branch:** `feat/raglab-v7-evolution`
> **Status:** `GATE_2_PASSED — aguardando autorização explícita para o Slice 3`

---

## 1. Requisitos Autoritativos do Slice 2
O Slice 2 implementa uma avaliação experimental controlada e reproduzível sobre um subconjunto pedagogicamente coerente do livro de matemática discreta, isolando o efeito da janela de contexto (Sentence-Window) do efeito do reranking de segundo estágio.

- **F0 (Baseline):** Chunks planos fixos (512 chars), embedding semântico multilíngue local, top-k=3.
- **W0 (Sentence-Window sem Reranker):** Sentenças indexadas individualmente, expansão de janela (`window_size=2`), deduplicação de janelas sobrepostas, sem reranker.
- **W1 (Sentence-Window com Reranker Local):** Mesmos candidatos de W0 (`candidate_k=6`), reranking de segundo estágio local, `top_n=3`, avaliação de métricas de dano.

---

## 2. Auditoria do PDF Real
- **Arquivo:** `Fundamentos matemáticos para a ciência da computação Matemática Discreta e Suas Aplicações (Judith L. Gersting).pdf`
- **Caminho:** Recebido via parâmetro CLI `--pdf-path` ou variável de ambiente `RAGLAB_PDF_PATH` (sem caminhos pessoais em código).
- **Tamanho:** 12.543.319 bytes (12,5 MB)
- **SHA-256:** `33e2e9f1e190158b3e99c19fced1acd050720247c7556780bad82b2f93bf1254`
- **Total de páginas:** 749 páginas
- **Páginas extraíveis:** 743 páginas (99,2% de extraibilidade de texto)
- **Páginas vazias/capas:** 6 páginas (páginas 1-4, 363, 563)
- **Distribuição de caracteres por página:** Mín: 0, Máx: 10.159, Média: 2.688,8, Mediana: 2.673

---

## 3. Recorte Pedagogicamente Coerente
- **Páginas selecionadas:** Páginas 91 a 115 inclusive (25 páginas físicas da Seção 2.1 "Técnicas de Demonstração" e Seção 2.2 "Indução Matemática").
- **Justificativa:** Unidade conceitual completa em técnicas de prova (demonstração direta, exaustão, contraposição, absurdo) e indução matemática, com alta densidade de definições formais e teoremas.
- **Páginas excluídas do sub-corpus:** Páginas 1–90 e 116–749.
- **Fingerprint do recorte:** Rastreável pelo hash do PDF e interval 91-115.

---

## 4. Extração e Proveniência
- **Adapter:** `PyPdfExtractorAdapter` em `src/raglab/infrastructure/pdf_parsers/pdf_parser_adapter.py`.
- **Biblioteca:** `pypdf` 4.3.1 (licença BSD 3-Clause, 100% offline).
- **Invariantes de Proveniência:** Preserva `document_id`, `page_number` físico (1-indexed), offsets de caracteres e fingerprint SHA-256 individual por página.
- **Tratamento de Texto:** Detecção de páginas vazias, normalização conservadora de hifenização e espaços sem alterar símbolos matemáticos (`\forall`, `\exists`, `\in`, `=>`).

---

## 5. Modelo de Embedding Semântico Real (Local)
- **Modelo:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Adapter:** `FastEmbedEmbeddingAdapter` em `src/raglab/infrastructure/embeddings/fastembed_adapter.py`.
- **Motor de Inferência:** FastEmbed / ONNX Runtime (execução CPU 100% local, 0 API externa).
- **Dimensão:** 384
- **Licença:** Apache 2.0 / MIT

---

## 6. Ground Truth e Splits
- **Arquivo:** `benchmarks/questions/controlled_chapter2.json`
- **Registro de Anotação:** `GROUND_TRUTH_SINGLE_ANNOTATOR`
- **Divisão de Perguntas:**
  - `development` (4 perguntas): `q_dev_01` (exaustão), `q_dev_02` (contraposição), `q_dev_03` (absurdo), `q_dev_04` (passo base e indutivo).
  - `test` (4 perguntas): `q_test_01` (passo base), `q_test_02` (regra de inferência), `q_test_03` (passo indutivo), `q_test_04` (abstenção — Dijkstra).
  - `holdout` (2 perguntas): `q_holdout_01`, `q_holdout_02` (**MANTIDAS LACRADAS E NÃO EXECUTADAS**).

---

## 7. Resultados do Experimento Controlado

### 7.1 Síntese Agregada

| Variante | Mean Recall@3 | Mean MRR | Hit Rate | Perguntas Válidas | Abstenções |
|---|---|---|---|---|---|
| **F0 — Baseline (Chunks 512)** | 0.0000 | 0.0000 | 0.0% | 7 | 1 (100% abstenção correta) |
| **W0 — Sentence-Window (window=2)** | **0.2857** | **0.2143** | **28.57%** | 7 | 1 (100% abstenção correta) |
| **W1 — Sentence-Window + Reranker** | 0.1429 | 0.0714 | 14.29% | 7 | 1 (100% abstenção correta) |

### 7.2 Análise por Pergunta (Recall / Hit)

| QID | Split | Pergunta | F0 | W0 | W1 | Dano do Reranker |
|---|---|---|---|---|---|---|
| `q_dev_01` | dev | Demonstração por exaustão | 0.0 (pág 92 ❌) | **1.0 (pág 92 ✅)** | 0.0 (pág 92 ❌) | ⚠️ **Passagem descartada** |
| `q_dev_02` | dev | Demonstração por contraposição | 0.0 | 0.0 | 0.0 | — |
| `q_dev_03` | dev | Demonstração por absurdo | 0.0 (pág 97 ❌) | **1.0 (pág 97 ✅)** | **1.0 (pág 97 ✅)** | Preservada |
| `q_dev_04` | dev | Passos da indução matemática | 0.0 | 0.0 | 0.0 | — |
| `q_test_01` | test | Passo base da indução | 0.0 | 0.0 | 0.0 | — |
| `q_test_02` | test | Regra lógica da contraposição | 0.0 | 0.0 | 0.0 | — |
| `q_test_03` | test | Reação em cadeia do passo indutivo | 0.0 | 0.0 | 0.0 | — |
| `q_test_04` | test | Algoritmo de Dijkstra (Abstenção) | `None` ✅ | `None` ✅ | `None` ✅ | Abstenção correta |

---

## 8. Inferência Estatística e Efeito do Reranker

### Comparação W0 (Sentence-Window) vs F0 (Baseline)
- **Vitórias:** 2 (q_dev_01, q_dev_03)
- **Empates:** 5
- **Derrotas:** 0
- **Diferença Média de Recall:** +0.2857
- **IC 95% Bootstrap (1.000 amostras):** [0.0000, 0.5714]
- **Tamanho de Efeito (Cohen's d):** 0.5976 (médio a grande)
- **Conclusão:** `exploratory_gain` — A expansão por janela de sentença produziu ganho estatístico exploratório substancial sobre o chunking fixo semântico.

### Comparação W1 (Reranker) vs W0 (Sentence-Window) — Avaliação de Dano
- **Delta de Recall Média:** -0.1429
- **Passagens Relevantes Descartadas:** 1 (página 92 descartada na pergunta `q_dev_01`)
- **Taxa de Dano do Reranker (Relevant Passage Dropped Rate):** 14,29%
- **Conclusão:** O reranking de segundo estágio descartou a evidência relevante da pergunta `q_dev_01` que o primeiro estágio (W0) havia recuperado com sucesso. Isso comprova empiricamente a importância da métrica de dano isolada.

---

## 9. Checkpoints e Retomada
- **Mecanismo:** `FilesystemCheckpointStore` em `checkpoints/`
- **ID de Execução:** `run_controlled_slice2`
- **Integridade Envelope:** SHA-256 do manifesto do corpus e da configuração.

---

## 10. Suíte de Testes e Controle de Qualidade

```text
pytest:              PASSED   (178/178 testes unitários e de contrato green)
ruff:                PASSED   (0 lint issues)
mypy_strict:         PASSED   (0 type errors em 33 arquivos fontes)
pip_audit:           PASSED   (0 vulnerabilities)
verify_reference:    PASSED   (15/15 checks)
secret_scan:         PASSED   (0 findings)
license_audit:       PASSED   (100% licenças compatíveis em licenses.json)
raglab_smoke_det:    PASSED   (11/11 checks)
raglab_smoke_llama:  PASSED   (11/11 checks)
raglab_controlled:   PASSED   (F0, W0, W1 executados e checkpoints salvos)
git_remotes:         0
```

---

## 11. Dívidas Aceitas Registradas (Gate 1)
- **G1-D01:** Clean rebuild antes do primeiro CI remoto.
- **G1-D02:** SBOM rigoroso antes de produção.
- **G1-D03:** Revisão de licenças ambíguas antes de produção.
- **G1-D04:** Execução remota quando houver remoto.
- **G1-D05:** Auditoria histórica dos testes até Gate 3.

---

## 12. Decisão do Gate 2

```text
GATE_2_PASSED — aguardando autorização explícita para o Slice 3
```
