# Runbook: Operational Procedures for Machine Silver Triage and Human Annotation

- **Target Audience**: Human Evaluators and Operations Engineers
- **Protocol**: `raglab_v7_slice4_v3` (Gate B2 - Hybrid Ground Truth v2)
- **Security Context**: Zero API keys inside IDE / git workspace. Credential handling outside IDE ONLY.

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
3. Ao término da sessão, confirme que o trap `cleanup_credentials` foi executado.

---

## 2. Execução da Triagem Silver (Máquina)

```bash
# Validação sem chamada de rede (Modo offline dry):
python scripts/run_silver_annotation.py --mode validate-only

# Execução completa (Exige confirmação explícita):
python scripts/run_silver_annotation.py --mode full --confirm-full-silver-run

# Retomada resiliente após interrupção:
python scripts/run_silver_annotation.py --mode resume --run-id <RUN_ID>
```

---

## 3. Geração de Filas Cegas para Anotadores A e B

```bash
# Gerar filas cegas orientadas por risco:
python scripts/build_human_review_queues.py --input-root benchmarks/ground_truth/v2/hybrid
```

---

## 4. Anotação Humana Independente e Blinded View

1. Os anotadores A e B devem abrir exclusivamente seus respectivos arquivos cegos:
   - `benchmarks/ground_truth/v2/hybrid/human_queues/annotator_a.jsonl`
   - `benchmarks/ground_truth/v2/hybrid/human_queues/annotator_b.jsonl`
2. Julgar as passagens segundo o manual `benchmarks/ground_truth/v2/annotation_guidelines.md`.
3. Preencher `relevance_grade` (0 a 3) e `evidence_role`.
4. Salvar os arquivos preenchidos.

---

## 5. Calibração Humano–Máquina

Após a conclusão das anotações humanas, execute a calibração de concordância:

```bash
python scripts/calibrate_silver_against_human.py \
  --silver-file benchmarks/ground_truth/v2/hybrid/silver/silver_annotations.jsonl \
  --human-file benchmarks/ground_truth/v2/hybrid/human_queues/annotator_a.jsonl
```
