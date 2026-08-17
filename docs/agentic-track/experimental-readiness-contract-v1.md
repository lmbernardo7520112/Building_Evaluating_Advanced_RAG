# Contrato de Aceitação — RAGLab V7 / Experimental Readiness Contract V1

> **Status do Contrato**: `EXPERIMENTAL_READINESS_CONTRACT_V1_FROZEN`  
> **Comportamento Requerido**: `ACCEPTANCE_TESTS_DEFINE_REQUIRED_BEHAVIOR`  
> **Estado da Implementação**: `PRODUCTION_IMPLEMENTATION_NOT_STARTED`  
> **Fase TDD**: `EXPECTED_RED_REQUIRED`  
> **Execução Científica**: `NO_EXPERIMENT_EXECUTED`

---

## 1. Problema que o Contrato Resolve

Em experimentos científicos de RAG Agentivo, garantias contratuais de governança não podem depender apenas de prompts, documentação em texto ou verificações manuais ad-hoc. 

O **Experimental Readiness Contract V1** transforma a governança científica do `RAGLab V7` em uma camada executável, invariante e imutável. Ele garante que qualquer futuro experimento (Slice 5B, Slice 5C, etc.):
1. Não execute em diretórios não isolados ou raízes proibidas (como `/tmp`, `benchmarks/`, `checkpoints/` ou a raiz do repositório).
2. Não utilize commits Git não verificados ou árvores de código com alterações não commitadas (`dirty tree`).
3. Não altere retrospectivamente o protocolo ou os arquivos de entrada (`inputs`) após a fase de preparação (`prepare`).
4. Mantenha uma cadeia de recibos histórica e append-only (`receipts/000_PREPARED.json`, `001_RUN_STARTED.json`, etc.) com hashes SHA-256 encadeados.
5. Não permita a adulteração ou ausência de qualquer artefato produzido durante a execução.
6. Não persista segredos, tokens ou credenciais em recibos ou logs.

---

## 2. Modelo de Ameaça (Threat Model)

O contrato protege o laboratório científico contra os seguintes vetores de falha ou adulteração:

| Vetor de Ameaça | Invariante de Proteção |
| :--- | :--- |
| **Execução em diretório compartilhado (`/tmp`, repo root)** | Rejeição estrita no preflight de `PathPolicy` (dispara erro e exit code $\neq 0$). |
| **Código modificado não commitado (`dirty git tree`)** | Preflight exige `git status` limpo no commit de implementação antes de preparar. |
| **Adulteração de protocolo pós-preparação** | Snapshot byte-a-byte do protocolo e hashes dos inputs congelados em `000_PREPARED.json`. |
| **Reescrita/Sobrescrita de histórico de recibos** | Estrutura append-only na subpasta `receipts/` com encadeamento hash do recibo anterior (`previous_receipt_sha256`). |
| **Adulteração ou exclusão de artefatos gerados** | Audit do `LineageVerifier` valida 100% dos hashes em `hashes.sha256` e inventário do recibo final. |
| **Concorrência/Race conditions em um mesmo run ID** | Lock exclusivo em nível de arquivo no diretório da execução. |
| **Escrita parcial de recibos por queda de energia ou crash** | Gravador atômico com escrita temporária no mesmo filesystem, `fsync()` explícito e `os.replace()`. |
| **Vazamento de credenciais em artefatos de execução** | Verificação automatizada que proíbe chaves de API, tokens e campos de segredo no recibo e nos logs. |

---

## 3. API Pública Esperada (Congelada)

### 3.1. Módulos Python

```text
raglab.agentic.experiments.contracts
raglab.agentic.experiments.hashing
raglab.agentic.experiments.path_policy
raglab.agentic.experiments.receipt_store
raglab.agentic.experiments.run_controller
raglab.agentic.experiments.lineage_verifier
```

### 3.2. Símbolos Obrigatórios

```python
from raglab.agentic.experiments import (
    ExperimentalPathPolicy,
    LineageAuditResult,
    LineageVerifier,
    PathPolicyError,
    ReceiptStoreError,
    RunController,
    RunControllerError,
    RunReceipt,
    RunReceiptStore,
    RunState,
    compute_canonical_json_sha256,
    compute_file_sha256,
)
```

### 3.3. Scripts CLI Esperados

- `scripts/prepare_agentic_run.py`
- `scripts/verify_agentic_run_lineage.py`

---

## 4. Máquina de Estados e Transições Governações

### Estados Válidos (`RunState`):
- `PREPARED`
- `RUN_STARTED`
- `RUN_COMPLETED`
- `AUDIT_COMPLETED` (Terminal de sucesso)
- `FAILED` (Terminal de falha)

### Transições Permitidas Exclusivamente:
- `PREPARED` $\rightarrow$ `RUN_STARTED`
- `RUN_STARTED` $\rightarrow$ `RUN_COMPLETED`
- `RUN_STARTED` $\rightarrow$ `FAILED`
- `RUN_COMPLETED` $\rightarrow$ `AUDIT_COMPLETED`

Todas as demais transições (ex: `PREPARED` $\rightarrow$ `AUDIT_COMPLETED`, `AUDIT_COMPLETED` $\rightarrow$ qualquer outro estado) são estritamente proibidas e devem disparar exceção `RunControllerError`.

---

## 5. Estrutura de Diretório Obrigatória da Execução

```text
<artifact_root>/<slice_id>/<run_id>/
├── protocol.snapshot.json
├── run_receipt.json
├── receipts/
│   ├── 000_PREPARED.json
│   ├── 001_RUN_STARTED.json
│   ├── 002_RUN_COMPLETED.json
│   └── 003_AUDIT_COMPLETED.json
├── raw/
├── derived/
├── logs/
└── hashes.sha256
```

---

## 6. Matriz de Cobertura dos 78 Testes de Aceitação

A suíte em [test_experimental_readiness_acceptance.py](file:///home/leonardomaximinobernardo/Downloads/huggingface_agents_course_bootcamp_AI_triggo_2026/Building_Evaluating_Advanced_RAG/raglab-v7/tests/agentic/test_experimental_readiness_acceptance.py) define 78 casos de teste divididos nas seguintes categorias:

1. **Validação de Paths (Casos 1 a 13)**: Rejeição de paths vazios, relativos, `/tmp`, `/var/tmp`, raiz do repositório, `benchmarks/`, `checkpoints/`, symlinks proibidos e run directories preexistentes.
2. **Git e Protocolo (Casos 14 a 22)**: Exigência de árvore Git limpa, ancestralidade válida dos commits, rastreamento do protocolo e verificação de SHA-256.
3. **Validação de Inputs (Casos 23 a 26)**: Verificação de existência, imutabilidade de hashes e registro completo de inputs no recibo.
4. **Cadeia Append-Only de Recibos (Casos 27 a 39)**: Validação do esquema JSON de `receipt.json`, encadeamento `previous_receipt_sha256`, recalculabilidade do hash, immutabilidade de recibos históricos e rejecção de transições proibidas.
5. **Snapshot e Artefatos (Casos 40 a 48)**: Preservação byte-a-byte do snapshot do protocolo, auditoria de inventário completo, detecção de artefatos ausentes, adulterados ou inesperados.
6. **Atomicidade e Lock (Casos 49 a 54)**: Aquisição de lock exclusivo por run ID, garantia de `fsync()` antes do rename/replace e prevenção contra escrita parcial.
7. **Verificador Read-Only (Casos 55 a 63)**: Verificação sem efeitos colaterais (não cria arquivos, não altera estado, não faz mutação em disco) e emissão de exit codes e status corretos.
8. **Segurança (Casos 64 a 67)**: Proibição estrita de persistência de segredos, chaves de API, tokens e valores confidenciais.
9. **CLIs da Camada (Casos 68 a 78)**: Requisitos de argumentos obrigatórios, suporte a `--input` repetível e retornos não-zero em caso de falha de preflight.

---

## 7. Critérios de Autorização para a Futura Implementação (Fase GREEN)

Para que a implementação futura seja considerada autorizada e aprovada:
1. NENHUM teste desta suíte congelada de aceitação poderá ser removido, enfraquecido ou alterado.
2. Todos os 78 casos de teste deverão passar com resultado `PASSED` (`ACCEPTANCE_GREEN`).
3. Não poderá haver regressão na suíte preexistente do `raglab-v7` (1278+ testes unitários/integrados).
4. O verificador e o controlador deverão respeitar 100% dos invariantes de imutabilidade e atomicidade aqui estabelecidos.
