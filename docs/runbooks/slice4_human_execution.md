# Slice 4 Runbook — Execução Humana (Ambiente B)

> **CONFIDENCIAL: NÃO COMPARTILHAR**
> Este runbook autoriza acesso temporário a credenciais.
> Execute em terminal isolado, nunca no Antigravity IDE.

---

## Pré-requisitos

```bash
# Verificar que virtualenv está disponível
ls raglab-v7/.venv/bin/python

# Verificar PDF
export RAGLAB_PDF_PATH="/caminho/para/gersting.pdf"
sha256sum "$RAGLAB_PDF_PATH"
# Esperado: 33e2e9f1e190158b3e99c19fced1acd050720247c7556780bad82b2f93bf1254
```

---

## Protocolo de segurança de credencial

```bash
# 1. Descriptografar chave (comando seguro — somente você sabe o método)
export GEMINI_API_KEY="$(seu-comando-de-descriptografia)"

# 2. Verificar que a chave está presente (NÃO IMPRIMA O VALOR)
[ -n "$GEMINI_API_KEY" ] && echo "KEY_PRESENT" || echo "KEY_MISSING"
```

---

## Execução do benchmark

```bash
cd raglab-v7/

# Variáveis de ambiente obrigatórias
export RAGLAB_PDF_PATH="/caminho/para/gersting.pdf"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export LANGCHAIN_TRACING_V2=false
export LANGSMITH_ENDPOINT=""

# Executar benchmark Slice 4
.venv/bin/python benchmarks/run_slice4_benchmark.py
```

---

## Após execução

```bash
# 1. Remover credencial IMEDIATAMENTE
unset GEMINI_API_KEY
echo "CREDENTIAL_REMOVED"

# 2. Verificar que nenhuma chave vazou nos resultados
grep -r "GEMINI_API_KEY\|sk-\|AIzaSy" benchmarks/results/slice4_*.json && \
  echo "WARNING: CREDENTIAL FOUND IN RESULTS" || \
  echo "SANITIZATION_OK"

# 3. Verificar arquivo de resultado
ls -la benchmarks/results/slice4_results_*.json

# 4. Commit seguro dos resultados sanitizados
git add benchmarks/results/slice4_results_*.json checkpoints/
git commit -m "test(slice4): record RAG Triad benchmark results"

# 5. NUNCA executar git push
```

---

## Retomada após interrupção (idempotente)

O benchmark usa checkpoint em `checkpoints/slice4_gen_checkpoint_*.json`.

```bash
# Verificar progresso
cat checkpoints/slice4_gen_checkpoint_*.json | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print(f'Completed: {len(d[\"completed\"])} pairs')"

# Simplesmente re-executar — pares já completos serão pulados
.venv/bin/python benchmarks/run_slice4_benchmark.py
```

---

## Verificação de holdout

```bash
# Confirmar que holdout NÃO foi executado
grep -r "q_holdout" benchmarks/results/ && \
  echo "HOLDOUT_LEAK_DETECTED" || \
  echo "HOLDOUT_SEALED"
```

---

## Tratamento de erros

| Erro | Ação |
|---|---|
| `GEMINI_API_KEY not found` | Verificar que `export GEMINI_API_KEY=...` foi executado |
| `PDF SHA-256 mismatch` | Verificar `RAGLAB_PDF_PATH` aponta para o arquivo correto |
| `RetryExhaustedError` | Aguardar 1 minuto (quota) e re-executar |
| `NonRetryableError 403` | Verificar que a chave tem permissões corretas |
| `NonRetryableError 400` | Verificar parâmetros do modelo (model_id correto) |

---

## Quotas do Gemini Flash Lite (free tier)

| Dimensão | Limite |
|---|---|
| RPM | 15 requests/minuto |
| TPD | 1.500 requests/dia |
| TPM | 1.000.000 tokens/minuto |

O `QuotaManager` enforça RPM e TPD automaticamente antes de cada chamada.
O `RetryPolicy` aplica exponential backoff com jitter em caso de 429.

---

## Artefatos produzidos

| Arquivo | Conteúdo |
|---|---|
| `benchmarks/results/slice4_results_*.json` | Resultados sanitizados (RAG Triad) |
| `checkpoints/slice4_gen_checkpoint_*.json` | Estado de progresso |

> **NENHUM destes arquivos contém credenciais** — verificado por `sanitize_*_for_artifact()`.
