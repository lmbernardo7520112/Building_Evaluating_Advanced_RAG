# Slice 4 Runbook — Execução Humana (Ambiente B)

> **CONFIDENCIAL: NÃO COMPARTILHAR**
> Este runbook autoriza acesso temporário a credenciais Gemini.
> **Somente o operador humano deve executar estes comandos.**
> Execute em terminal isolado, **nunca no Antigravity IDE.**
>
> **Regras invioláveis:**
> - O Antigravity **não pode** executar, copiar, inspecionar ou registrar a chave.
> - Shell tracing deve permanecer desativado com `set +x` durante toda a sessão.
> - Nunca usar `echo "$GEMINI_API_KEY"` nem qualquer comando que imprima o valor.

---

## Protocolo de Duas Fases

> **FASE A** e **FASE B** são executadas em sessões separadas.
> **Nunca misture provisionamento de modelo e chave Gemini na mesma execução.**

---

## FASE A — Provisionamento (SEM Gemini)

### A.1 — Limpar credenciais

```bash
# Garantir que credenciais e tokens NÃO estão no ambiente antes do preflight
export LANGCHAIN_TRACING_V2=false
unset GEMINI_API_KEY
unset GOOGLE_API_KEY
unset LANGSMITH_API_KEY
unset HF_TOKEN
unset HUGGINGFACE_HUB_TOKEN
echo "CREDENTIALS_CLEARED"
```

### A.2 — Configurar cache persistente

```bash
cd raglab-v7/

# Opção 1: usar cache padrão (.model_cache/ na raiz do repo)
# Nada a fazer — o adapter usa .model_cache/ por padrão

# Opção 2: cache personalizado (recomendado para ambientes de produção)
export RAGLAB_MODEL_CACHE="/caminho/persistente/controlado"
```

### A.3 — Provisionar embedding (rede permitida)

```bash
# Este comando baixa o modelo ONNX para o cache persistente
# Rejeita GEMINI_API_KEY se presente — aborta antes de baixar
.venv/bin/python scripts/provision_embedding_model.py

# Saída esperada: PROVISION_OK
```

### A.4 — Desabilitar rede (opcional mas recomendado)

```bash
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
```

### A.5 — Preflight offline

```bash
export RAGLAB_PDF_PATH="/caminho/para/gersting.pdf"

.venv/bin/python benchmarks/run_slice4_benchmark.py --mode preflight

# Saída esperada: EMBEDDING_OFFLINE_READY
```

> **Somente prossiga para a FASE B se `EMBEDDING_OFFLINE_READY` for exibido.**

---

## FASE B — Execução com Gemini (credenciais temporárias)

### B.1 — Protocolo de segurança de credencial

> **Este bloco é executado exclusivamente pelo operador humano em terminal externo ao Antigravity.**
> O Antigravity não pode e não deve invocar `systemd-creds`, `credstore`, nem nenhum
> mecanismo de descriptografia de chaves.

```bash
# Desativar shell tracing para proteger o valor da chave
set +x

# Registrar limpeza garantida para qualquer saída do shell
cleanup_credentials() {
  unset GEMINI_API_KEY
  unset GOOGLE_API_KEY
}
trap cleanup_credentials EXIT INT TERM HUP

# Descriptografar chave via credstore isolado (systemd-creds)
export GEMINI_API_KEY="$(
  sudo systemd-creds decrypt \
    /home/lg-runner/.config/credstore.encrypted/GEMINI_API_KEY \
    -
)"

# Verificar presença sem revelar o valor
[ -n "${GEMINI_API_KEY:-}" ] || {
  echo "KEY_MISSING"
  exit 1
}
echo "KEY_PRESENT"
```

### B.2 — Variáveis de ambiente obrigatórias

```bash
cd raglab-v7/

export RAGLAB_PDF_PATH="/caminho/para/gersting.pdf"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export LANGCHAIN_TRACING_V2=false
export LANGSMITH_ENDPOINT=""
```

### B.3 — Smoke test (obrigatório antes do benchmark completo)

```bash
.venv/bin/python benchmarks/run_slice4_benchmark.py \
    --mode smoke \
    --smoke-strategy F0_baseline \
    --smoke-question q_dev_01

# Validar resultado sanitizado
SMOKE_FILE=$(ls -t benchmarks/results/slice4_results_smoke_*.json | head -1)
python3 -c "
import json, sys
d = json.load(open('$SMOKE_FILE'))
assert 'results' in d, 'schema inválido'
assert 'GEMINI_API_KEY' not in json.dumps(d), 'CREDENTIAL LEAKED'
print('SMOKE_OK: schema válido, sem credenciais')
"
```

> Somente prossiga para o benchmark completo se `SMOKE_OK` for exibido.

### B.4 — Benchmark completo

```bash
.venv/bin/python benchmarks/run_slice4_benchmark.py \
    --mode full \
    --confirm-full-benchmark
```

### B.5 — Remover credencial IMEDIATAMENTE

```bash
unset GEMINI_API_KEY
unset GOOGLE_API_KEY
unset LANGSMITH_API_KEY
unset HF_TOKEN
unset HUGGINGFACE_HUB_TOKEN
echo "CREDENTIAL_REMOVED"
```

---

## Retomada após interrupção

```bash
cd raglab-v7/

# Listar checkpoints disponíveis
ls checkpoints/slice4_gen_checkpoint_*.json

RUN_ID="raglab_v7_slice4_v1_20260731T1230UTC"    # ajuste ao RUN_ID real

# Re-executar com --mode resume + RUN_ID explícito
.venv/bin/python benchmarks/run_slice4_benchmark.py \
    --mode resume \
    --run-id "$RUN_ID"
```

---

## Após execução

```bash
# 1. Scanner autoritativo de segredos
.venv/bin/python scripts/scan_secrets.py
# Saída esperada: "findings_count": 0

# 2. Complementar (somente nomes de arquivos)
grep -rl "GEMINI_API_KEY\|sk-\|AIzaSy" benchmarks/results/ 2>/dev/null \
  && echo "WARNING: verifique os arquivos listados acima" \
  || echo "GREP_COMPLEMENT_OK"

# 3. Verificar holdout lacrado
grep -rl "q_holdout" benchmarks/results/ 2>/dev/null \
  && { echo "HOLDOUT_LEAK_DETECTED — NÃO COMMITAR"; exit 1; } \
  || echo "HOLDOUT_SEALED"

# 4. Verificar diff
git diff --check

# 5. Commit somente resultados sanitizados
git add benchmarks/results/slice4_results_*.json
git commit -m "test(slice4): record RAG Triad benchmark results"

# NUNCA executar git push
```

---

## Tratamento de erros

| Erro | Ação |
|---|---|
| `PROVISION_ERROR: GEMINI_API_KEY is set` | `unset GEMINI_API_KEY` e re-executar provisioning |
| `Transient directory '/tmp'...` | Definir `RAGLAB_MODEL_CACHE` com caminho persistente |
| `Embedding cache missing` | Executar `scripts/provision_embedding_model.py` primeiro |
| `PREFLIGHT_FAILED: Could not load` | Provisionar novamente com rede disponível |
| `GEMINI_API_KEY not found` | Confirmar que `systemd-creds decrypt` foi executado |
| `PDF SHA-256 mismatch` | Verificar `RAGLAB_PDF_PATH` |
| `RetryExhaustedError` | Aguardar quota e `--mode resume --run-id ...` |
| `Full benchmark requires --confirm` | Adicionar `--confirm-full-benchmark` |

---

## Quotas do Gemini (limites configurados, não garantias universais)

> **Atenção:** Os valores abaixo são os limites padrão **configurados neste projeto**
> para `gemini-3.1-flash-lite` no **free tier** no momento da implementação.
> Esses limites **dependem do projeto, modelo, plano e região** e podem ser alterados
> pela Google sem aviso. **Confirme os limites reais no console do projeto Gemini
> antes de executar** (`console.cloud.google.com → APIs & Services → Quotas`).

| Dimensão | Valor configurado | Fonte |
|---|---|---|
| RPM | 15 requests/minuto | `QuotaManager(rpm_limit=15)` |
| TPD | 1.500 requests/dia | `QuotaManager(tpd_limit=1_500)` |
| TPM (tokens) | 1.000.000 tokens/min | referência free tier |

---

## Artefatos produzidos

| Arquivo | Conteúdo | Versionado? |
|---|---|---|
| `benchmarks/results/slice4_results_*.json` | Resultados sanitizados (RAG Triad) | ✅ Sim |
| `checkpoints/slice4_gen_checkpoint_*.json` | Estado operacional local | ❌ Não (`.gitignore`) |
| `.model_cache/` | Pesos ONNX do embedding model | ❌ Não (`.gitignore`) |
| `benchmarks/provision_manifest.json` | Manifesto de provisionamento local | ❌ Não (`.gitignore`) |

> **NENHUM dos artefatos versionados contém credenciais** — verificado por
> `sanitize_*_for_artifact()` e pelo scanner `scripts/scan_secrets.py`.
