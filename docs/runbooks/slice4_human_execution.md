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

---

## Gate de smoke test (obrigatório antes do benchmark completo)

Execute este gate antes de autorizar as 7 estratégias × 8 perguntas.
Ele valida geração + julgamento + sanitização com **1 pergunta e 1 estratégia**.

```bash
cd raglab-v7/

# Variáveis de ambiente obrigatórias
export RAGLAB_PDF_PATH="/caminho/para/gersting.pdf"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export LANGCHAIN_TRACING_V2=false
export LANGSMITH_ENDPOINT=""
export RAGLAB_SMOKE_ONLY=1      # instrui o benchmark a rodar somente 1 pergunta × 1 estratégia

# Executar smoke test
.venv/bin/python benchmarks/run_slice4_benchmark.py

# Validar resultado sanitizado
SMOKE_FILE=$(ls -t benchmarks/results/slice4_results_*.json | head -1)
python3 -c "
import json, sys
d = json.load(open('$SMOKE_FILE'))
assert 'results' in d, 'schema inválido'
assert 'GEMINI_API_KEY' not in json.dumps(d), 'CREDENTIAL LEAKED'
print('SMOKE_OK: schema válido, sem credenciais')
"
```

> Somente prossiga para o benchmark completo se `SMOKE_OK` for exibido.

---

## Execução do benchmark completo

```bash
cd raglab-v7/

# Variáveis de ambiente obrigatórias (reutilizar se smoke já foi configurado)
export RAGLAB_PDF_PATH="/caminho/para/gersting.pdf"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export LANGCHAIN_TRACING_V2=false
export LANGSMITH_ENDPOINT=""
unset RAGLAB_SMOKE_ONLY   # garantir que o modo completo está ativo

# Executar benchmark Slice 4 (7 estratégias × 8 perguntas)
.venv/bin/python benchmarks/run_slice4_benchmark.py
```

---

## Após execução

```bash
# 1. Remover credencial IMEDIATAMENTE (trap já faz isso ao sair, mas execute agora)
unset GEMINI_API_KEY
unset GOOGLE_API_KEY
echo "CREDENTIAL_REMOVED"

# 2. Scanner autoritativo de segredos (usa scan_secrets.py, não grep de conteúdo)
.venv/bin/python scripts/scan_secrets.py
# Saída esperada: "findings_count": 0

# 3. (Complementar, somente nomes de arquivos — sem imprimir conteúdo)
grep -rl "GEMINI_API_KEY\|sk-\|AIzaSy" benchmarks/results/ 2>/dev/null \
  && echo "WARNING: verifique os arquivos listados acima" \
  || echo "GREP_COMPLEMENT_OK"

# 4. Verificar arquivos de resultado
ls -la benchmarks/results/slice4_results_*.json

# 5. Validar JSON e schema antes do commit
python3 -c "
import json, glob, sys
files = glob.glob('benchmarks/results/slice4_results_*.json')
for f in files:
    d = json.load(open(f))
    assert 'experiment_id' in d, f'schema inválido: {f}'
    assert 'results' in d, f'campo results ausente: {f}'
    assert 'GEMINI_API_KEY' not in json.dumps(d), f'CREDENTIAL LEAKED: {f}'
    print(f'JSON_OK: {f}')
"

# 6. Verificar holdout lacrado
grep -rl "q_holdout" benchmarks/results/ 2>/dev/null \
  && { echo "HOLDOUT_LEAK_DETECTED — NÃO COMMITAR"; exit 1; } \
  || echo "HOLDOUT_SEALED"

# 7. Verificar que nenhum checkpoint está staged (checkpoints são estado local, não versionados)
git diff --cached -- checkpoints/ | grep -q "." \
  && { echo "ERROR: checkpoints staged — remova com: git restore --staged checkpoints/"; exit 1; } \
  || echo "CHECKPOINTS_NOT_STAGED_OK"

# 8. Revisar diff antes de commitar
git diff --check
git diff --cached

# 9. CONFIRMAÇÃO HUMANA: revisar o diff acima antes de prosseguir
# Somente execute o próximo bloco após confirmação visual do operador.
```

### Commit dos resultados (somente resultados sanitizados — sem checkpoints)

> **Checkpoints (`checkpoints/*.json`) são estado operacional local.**
> Eles devem permanecer ignorados pelo Git (já listados no `.gitignore`).
> **Nunca** execute `git add checkpoints/`.

```bash
# Adicionar somente resultados sanitizados
git add benchmarks/results/slice4_results_*.json

# Commitar
git commit -m "test(slice4): record RAG Triad benchmark results"

# NUNCA executar git push
```

---

## Retomada após interrupção (idempotente)

O benchmark usa checkpoint em `checkpoints/slice4_gen_checkpoint_<RUN_ID>.json`.

```bash
# Listar checkpoints disponíveis
ls checkpoints/slice4_gen_checkpoint_*.json

# Inspecionar progresso de um checkpoint específico (selecione pelo RUN_ID)
RUN_ID="raglab_v7_slice4_v1_20260731T1230UTC"    # ajuste ao RUN_ID real
CKPT_FILE="checkpoints/slice4_gen_checkpoint_${RUN_ID}.json"

python3 - <<'EOF'
import json, os, sys
ckpt = os.environ.get("CKPT_FILE", "")
if not ckpt or not os.path.exists(ckpt):
    print("CKPT_FILE não encontrado — defina RUN_ID corretamente")
    sys.exit(1)
d = json.load(open(ckpt))
completed = d.get("completed", {})
print(f"Run ID:    {d.get('run_id')}")
print(f"Completed: {len(completed)} pares")
for k in sorted(completed)[:10]:
    print(f"  {k}")
EOF

# Simplesmente re-executar — pares já completos serão pulados automaticamente
.venv/bin/python benchmarks/run_slice4_benchmark.py
```

---

## Verificação de holdout

```bash
# Confirmar que holdout NÃO foi executado
grep -rl "q_holdout" benchmarks/results/ 2>/dev/null \
  && echo "HOLDOUT_LEAK_DETECTED" \
  || echo "HOLDOUT_SEALED"
```

---

## Tratamento de erros

| Erro | Ação |
|---|---|
| `GEMINI_API_KEY not found` | Confirmar que `systemd-creds decrypt` foi executado e KEY_PRESENT foi exibido |
| `PDF SHA-256 mismatch` | Verificar `RAGLAB_PDF_PATH` aponta para o arquivo correto |
| `RetryExhaustedError` | Aguardar 1–2 minutos (quota de RPM) e re-executar; o checkpoint garante retomada |
| `NonRetryableError 403` | Verificar permissões da chave no console do projeto Gemini |
| `NonRetryableError 400` | Verificar `model_id` correto (`gemini-3.1-flash-lite`) |
| `KEY_MISSING` | Verifique se o `credstore.encrypted` está no caminho correto e `sudo` disponível |

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

O `QuotaManager` enforça RPM e TPD automaticamente **antes** de cada chamada (pré-emptivo).
O `RetryPolicy` aplica exponential backoff com jitter em caso de 429 recebido (reativo).
Em caso de `RetryExhaustedError`, aguarde a janela de quota e re-execute — o checkpoint garante idempotência.

---

## Artefatos produzidos

| Arquivo | Conteúdo | Versionado? |
|---|---|---|
| `benchmarks/results/slice4_results_*.json` | Resultados sanitizados (RAG Triad) | ✅ Sim |
| `checkpoints/slice4_gen_checkpoint_*.json` | Estado operacional local de progresso | ❌ Não (`.gitignore`) |

> **NENHUM dos artefatos versionados contém credenciais** — verificado por
> `sanitize_*_for_artifact()` e pelo scanner `scripts/scan_secrets.py`.
