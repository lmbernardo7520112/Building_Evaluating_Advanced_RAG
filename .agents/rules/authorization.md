# Authorization Rule

## External Actions

No action that reaches outside the local repository is permitted without
explicit human authorization. This includes, but is not limited to:

- `git push`, `git remote add`, or any remote Git operation
- `pip install` or any package installation
- API calls to Gemini, HuggingFace, or any model provider
- Publishing artifacts, reports, or notebooks
- Downloading corpus, models, or data not already in the workspace

## Implementation Gating

Each vertical slice requires separate authorization before execution.
The agent must stop and request approval at every gate boundary.

## Destructive Operations

The following are prohibited without explicit authorization and ADR:

- Deleting checkpoints, indices, or databases
- Altering historical RUN_IDs
- Overwriting the v6.1 reference or original
- Using `git reset`, `git rebase`, `git commit --amend`, or force push
- Installing packages with `trust_remote_code=True`
- Silencing warnings globally
- Converting exceptions into approvals
