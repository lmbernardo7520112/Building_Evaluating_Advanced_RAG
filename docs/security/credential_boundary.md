# Fronteira de Credenciais — RAGLab v7

> Documento de segurança e governança para o gerenciamento de credenciais no pipeline RAGLab v7.
> **Categoria:** SECURITY-DOC  
> **Status:** VIGENTE — revisão obrigatória antes de qualquer integração autenticada

---

## 1. Modelo de dois ambientes

### Ambiente A — Antigravity (IDE / agente)

- IDE e todos os processos do agente rodam neste ambiente.
- **Nenhuma credencial é exportada** neste ambiente.
- Atividades permitidas: desenvolvimento, testes, análise estática, execução offline.
- Para provedores autenticados: **uso obrigatório de fakes**.
- O Antigravity nunca deve executar `systemd-creds decrypt`, ler `credstore.encrypted`, nem acessar `/proc/*/environ`.

### Ambiente B — Terminal local independente

- Aberto **separadamente**, fora do IDE.
- **Não é** terminal integrado do Antigravity.
- **Não é** processo descendente do Antigravity.
- Operado manualmente pelo usuário.
- Único ambiente onde credenciais poderão futuramente ser descriptografadas e injetadas.
- Executa somente código previamente revisado e commitado.

### Por que variáveis exportadas no terminal B não chegam ao Antigravity

Variáveis de ambiente são herdadas apenas por **processos descendentes** (`fork/exec`).
Um terminal aberto separadamente e o processo do Antigravity **não são** pai e filho.
Portanto `export GEMINI_API_KEY=...` no terminal B nunca contamina o Ambiente A.

---

## 2. Proibições absolutas ao Antigravity

O Antigravity **nunca** deve:

- executar `systemd-creds decrypt`;
- ler arquivos em `credstore.encrypted`;
- solicitar ao usuário valores de chaves;
- ler `/proc/*/environ`;
- executar `env` ou `printenv` para procurar segredos;
- procurar ou imprimir valores de:
  - `GEMINI_API_KEY`
  - `GOOGLE_API_KEY`
  - `HF_TOKEN`
  - `LANGSMITH_API_KEY`
- criar `.env` ou persistir chaves em arquivos, banco, checkpoint ou log;
- adicionar segredos em testes, mocks ou fixtures;
- chamar Gemini, Hugging Face Inference API ou LangSmith;
- usar `git commit --amend`, `git reset`, `git rebase` ou force push.

Se qualquer segredo aparecer inadvertidamente em saída:
1. Interromper imediatamente.
2. Não reproduzir o valor.
3. Redigir completamente a evidência.
4. Emitir `CREDENTIAL_EXPOSURE_INCIDENT`.
5. Recomendar revogação e rotação humana.
6. Não prosseguir.

---

## 3. Execução humana autorizada (terminal B)

O procedimento abaixo é permitido **somente ao usuário**, manualmente, em terminal local independente:

```bash
cd /caminho/revisado/raglab-v7
source .venv/bin/activate

# Verificar o commit revisado ANTES de injetar credencial
git rev-parse HEAD
git status --short
git diff

# Descriptografar e injetar somente após conferir o estado acima
export GEMINI_API_KEY="$(
  sudo systemd-creds decrypt \
    /home/lg-runner/.config/credstore.encrypted/GEMINI_API_KEY -
)"

# Após uso: cleanup obrigatório
unset GEMINI_API_KEY
unset GOOGLE_API_KEY
unset HF_TOKEN
unset LANGSMITH_API_KEY
# Ou encerrar integralmente o terminal
```

> **Risco residual:** executar código não revisado com credencial ativa. Mitigação: conferir `git rev-parse HEAD` e `git status --short` imediatamente antes de injetar a chave.

---

## 4. Hugging Face e embedding multilíngue

```
HF_EMBEDDING: LOCAL_MULTILINGUAL
HF_TOKEN_REQUIRED_DEFAULT: NO
```

O embedding atual é:
- **Modelo:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Multilíngue:** sim (inclui português)
- **Dimensão:** 384
- **Execução:** local, via FastEmbed/ONNX, CPU
- **HF_TOKEN:** não necessário para uso normal (modelo público)

Após o modelo estar no cache, usar sempre:

```bash
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
```

Essas variáveis **não são segredos** e podem ser configuradas em qualquer ambiente.

Se o modelo não estiver em cache:
- **Não** procurar `HF_TOKEN`;
- Emitir `MULTILINGUAL_MODEL_CACHE_MISSING`;
- Fornecer ao usuário o comando de download público;
- Aguardar execução humana.

`HF_TOKEN` somente será considerado para recurso privado ou gated, com:
- token fine-grained, somente leitura;
- descriptografia somente em terminal humano independente;
- envio implícito desabilitado;
- nunca persistido no repositório.

---

## 5. Gemini — planejado, não executado

```
GEMINI_PROVIDER: PLANNED
GEMINI_MODEL_ALLOWLIST: gemini-3.1-flash-lite
```

Arquitetura futura planejada:
- **Chave:** `GEMINI_API_KEY` — injetada somente em terminal B pelo usuário
- **Generator e judge:** configurados separadamente
- **Rate limits:** retry com backoff, jitter e respeito a `RetryInfo`
- **Circuit breaker:** timeout e limite total de requisições
- **Allowlist rígida de modelos:** nenhuma seleção por entrada externa
- **Logs:** redaction obrigatório de credenciais
- **Checkpoint:** por pergunta, retomada idempotente
- **Chave:** nunca em exception, trace ou checkpoint

O Antigravity usa **FakeGeneratorAdapter** e **FakeJudgeAdapter** até autorização explícita.

---

## 6. LangSmith — permanentemente desabilitado

```
LANGSMITH_ENABLED=false
LANGSMITH_TRACING=false
```

LangSmith somente poderá ser introduzido por ADR específico futuro contendo:
- finalidade; dados enviados; política de redaction; retenção; jurisdição;
- custos; consentimento; risco de exposição; alternativa local; autorização humana.

---

## 7. Rotação e revogação

- Após uso acidental de credencial: revogar **imediatamente** no provedor antes de qualquer outra ação.
- Após exposure em log/commit: revogar, limpar histórico conforme política do provedor, reportar.
- Rotação preventiva: periodicidade definida pela política de segurança da equipe.

---

## 8. Opção futura de serviço systemd para produção

Para uso em produção, recomendado:
- `systemd` unit com `LoadCredential=GEMINI_API_KEY:/credstore/GEMINI_API_KEY`
- processo recebe a credencial como arquivo em `/run/credentials/`, não como variável de ambiente
- processo descarta a credencial da memória após autenticação inicial
- auditoria via `journald`

---

## 9. Tabela de estado atual

```
GEMINI_PROVIDER:           PLANNED
GEMINI_MODEL_ALLOWLIST:    gemini-3.1-flash-lite
HF_EMBEDDING:              LOCAL_MULTILINGUAL
HF_TOKEN_REQUIRED_DEFAULT: NO
LANGSMITH_DEFAULT:         DISABLED
CREDENTIAL_IN_REPO:        NEVER
CREDENTIAL_IN_TEST:        NEVER
CREDENTIAL_IN_LOG:         NEVER (redacted)
```
