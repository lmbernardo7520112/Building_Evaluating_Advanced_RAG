# Runbook: Operational Procedures for Machine Silver Triage and Human Annotation

- **Target Audience**: Human Evaluators and Operations Engineers
- **Protocol**: `raglab_v7_slice4_v3` (Gate B2 - Hybrid Ground Truth v2)
- **Security Context**: Zero API keys inside IDE / git workspace. Credential handling outside IDE ONLY.
- **Judge Model Authorized**: `gemini-3.1-flash-lite` (`gemini-2.5-flash` is strictly prohibited).

---

## 1. Segredo e Gerenciamento de Credenciais (Fora da IDE)

Operadores humanos executando triagem automática com a API Gemini devem utilizar o seguinte procedimento seguro em terminal bash externo (fora da IDE e do ambiente Antigravity):

```bash
set +x

cleanup_credentials() {
  unset GEMINI_API_KEY
  unset GOOGLE_API_KEY
  echo "Credenciais limpas com sucesso."
}

trap cleanup_credentials EXIT INT TERM HUP

export GEMINI_API_KEY="$(
  sudo systemd-creds decrypt \
    /home/lg-runner/.config/credstore.encrypted/GEMINI_API_KEY \
    -
)"

[ -n "${GEMINI_API_KEY:-}" ] \
  || { echo "KEY_MISSING"; exit 1; }

echo "Chave GEMINI_API_KEY carregada na memória da sessão (NUNCA IMPRIMIR)."
```

### Regras de Segurança Estritas:
1. NUNCA utilize `echo $GEMINI_API_KEY` ou `printenv`.
2. NUNCA salve a chave descriptografada em arquivos no disco.
3. Ao término da sessão, confirme que o trap `cleanup_credentials` foi executado (`cleanup_credentials`).
4. Execute sempre a varredura de segredos ao final da sessão: `python scripts/scan_secrets.py`.

---

## 2. Execução da Triagem Silver (Máquina)

### Localização dos Artefatos e RUN_ID
Todas as execuções de triagem real geram diretórios isolados por `RUN_ID`:
`benchmarks/ground_truth/v2/hybrid/silver/runs/<RUN_ID>/`
- `silver_annotations.jsonl`
- `silver_manifest.json`
- `checkpoint.json`

### Comandos de Execução

```bash
# A. Validação sem chamada de rede (Modo offline dry, sem chave):
python scripts/run_silver_annotation.py --mode validate-only

# B. Smoke Real (Processa exatamente 1 item, exige GEMINI_API_KEY na sessão):
python scripts/run_silver_annotation.py --mode smoke

# C. Auditoria do Smoke Real (Verificar o manifesto gerado):
cat benchmarks/ground_truth/v2/hybrid/silver/runs/<RUN_ID>/silver_manifest.json

# D. Execução Completa (Processa 69 itens elegíveis, exige confirmação explícita):
python scripts/run_silver_annotation.py --mode full --confirm-full-silver-run

# E. Retomada Resiliente (Retoma exatamente um RUN_ID a partir do checkpoint):
python scripts/run_silver_annotation.py --mode resume --run-id <RUN_ID>
```

---

## 3. Geração de Filas Cegas para Anotadores A e B

```bash
# Execução definitiva (consumindo o resultado da triagem silver real):
python scripts/build_human_review_queues.py \
  --input-root benchmarks/ground_truth/v2/hybrid \
  --output-root benchmarks/ground_truth/v2/hybrid/human_queues \
  --silver-file benchmarks/ground_truth/v2/hybrid/silver/runs/<RUN_ID>/silver_annotations.jsonl

# Execução provisória (sem triagem silver prévia):
python scripts/build_human_review_queues.py \
  --input-root benchmarks/ground_truth/v2/hybrid \
  --output-root benchmarks/ground_truth/v2/hybrid/human_queues \
  --without-silver-execution
```

---

## 4. Anotação Humana Independente e Interface Local Offline

```bash
# A. Iniciar Anotação para Anotador A (Porta 8501):
python scripts/annotate_human_queue.py \
  --annotator-id annotator_a \
  --queue-file benchmarks/ground_truth/v2/hybrid/human_queues/annotator_a.jsonl \
  --questions-file benchmarks/questions/controlled_chapter2.json \
  --output-file benchmarks/ground_truth/v2/hybrid/human_annotations/work/annotator_a_work.jsonl

# B. Iniciar Anotação para Anotador B (Porta 8502):
python scripts/annotate_human_queue.py \
  --annotator-id annotator_b \
  --queue-file benchmarks/ground_truth/v2/hybrid/human_queues/annotator_b.jsonl \
  --questions-file benchmarks/questions/controlled_chapter2.json \
  --output-file benchmarks/ground_truth/v2/hybrid/human_annotations/work/annotator_b_work.jsonl \
  --port 8502

# C. Retomar Sessão Interrompida:
# Executar exatamente o mesmo comando anterior. O servidor lê o arquivo de trabalho e retoma atomicamente.

# D. Validar e Exportar Anotações Concluídas (Anotador A):
python scripts/validate_human_annotations.py \
  --annotator-id annotator_a \
  --queue-file benchmarks/ground_truth/v2/hybrid/human_queues/annotator_a.jsonl \
  --questions-file benchmarks/questions/controlled_chapter2.json \
  --work-file benchmarks/ground_truth/v2/hybrid/human_annotations/work/annotator_a_work.jsonl \
  --export-file benchmarks/ground_truth/v2/hybrid/human_annotations/export/annotator_a_final.jsonl

# E. Validar e Exportar Anotações Concluídas (Anotador B):
python scripts/validate_human_annotations.py \
  --annotator-id annotator_b \
  --queue-file benchmarks/ground_truth/v2/hybrid/human_queues/annotator_b.jsonl \
  --questions-file benchmarks/questions/controlled_chapter2.json \
  --work-file benchmarks/ground_truth/v2/hybrid/human_annotations/work/annotator_b_work.jsonl \
  --export-file benchmarks/ground_truth/v2/hybrid/human_annotations/export/annotator_b_final.jsonl
```

---

## 5. Calibração Humano–Máquina e Varredura de Segurança

```bash
# A. Calibração de concordância:
python scripts/calibrate_silver_against_human.py \
  --silver-file benchmarks/ground_truth/v2/hybrid/silver/runs/<RUN_ID>/silver_annotations.jsonl \
  --human-file benchmarks/ground_truth/v2/hybrid/human_queues/annotator_a.jsonl

# B. Varredura de segredos obrigatória antes de commit:
python scripts/scan_secrets.py

# C. Limpeza de credenciais na sessão:
cleanup_credentials
```
