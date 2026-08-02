# RAGLab v7 — Workspace Rules

## Preservação da V6.1

- O notebook v6.1 original **permanece fora** de `raglab-v7/`.
- A cópia em `reference/v6_1_reference.ipynb` é read-only (chmod 444).
- `reference/source_manifest.json` registra SHA-256 autoritativo.
- `scripts/verify_reference.py` valida integridade em todo commit.
- **Nunca** sobrescrever, excluir ou modificar o original ou a referência.

## Credenciais

- Nenhuma chave de API, token ou segredo pode ser commitado.
- Usar variáveis de ambiente ou cofre de segredos.
- `*.key`, `.env`, `service-account*.json` estão no `.gitignore`.

## Holdout

- Holdout não pode ser usado para tuning.
- Perguntas do holdout não podem ser parafraseadas no development set.
- `corpus_holdout` só é ingerido após congelamento do protocolo.

## Ações Destrutivas

- Não apagar checkpoints, índices ou bancos sem autorização.
- Não alterar RUN_ID histórico.
- Não preencher feedback ausente com zero.
- Não remover outlier sem regra congelada e aprovada.

## Dependências

- Toda dependência nova requer autorização.
- `trust_remote_code=True` proibido sem ADR.
- Não instalar pacote a partir de exceção de import.
- Lockfile com hashes obrigatório a partir do Slice 1.

## Commits

- Commits semânticos, pequenos e atômicos.
- Testes relevantes verdes antes de commit.
- Nenhum segredo ou saída pesada.
- Nenhum push sem autorização.

## Claims Científicos

- Nenhuma técnica declarada vencedora sem satisfazer gates estatísticos.
- Abstenção obrigatória quando evidência é insuficiente.
- RAG Triad não é ground truth.

## Autorização Externa

- Nenhum `git push`, `git remote add` ou publicação sem autorização.
- Nenhuma API externa chamada sem autorização por uso.
