# ADR 002: Knowledge Runtime vs. Ground Truth Evaluation Isolation

## Status
Accepted

## Date
2026-08-03

## Context & Problem Statement
In RAG (Retrieval-Augmented Generation) architectures, information leakage from evaluation datasets (gold answers, relevant pages, qrels) into the live retrieval and generation inference loop compromises scientific validity. 
Furthermore, citations to raw page numbers (e.g. `[p.92]`) create tight coupling between presentation formatting and corpus layout, preventing passage re-ranking and multi-document synthesis across different queries.

## Decision Drivers
- **Zero Gold Leakage**: Gold references, relevant pages, and qrels must NEVER enter the runtime inference path `query -> retriever -> evidence -> generator`.
- **Ephemeral Citation ID Mapping (Guard 1)**: Evidence citations presented to the LLM and generated in responses must use ephemeral identifiers (`E1`, `E2`, `E3`) mapped dynamically from retrieval rank. Persistent `passage_id` identity must remain unchanged across different query invocations.
- **Honest Legacy Metadata Migration (Guard 2)**: Binary legacy qrel data must not be converted into artificial numeric relevance grades. Absent metadata must be assigned `provenance_status = "LEGACY_METADATA_UNAVAILABLE"`, and metrics relying on missing fields (e.g. nDCG without grades) must explicitly return non-computable status codes rather than fake scores.
- **Fail-Closed JSON Output Contract**: Generation responses must produce structured JSON output (`{"status": "ANSWER", ...}` or `{"status": "ABSTAIN", ...}`). Citing an `evidence_id` outside the prompt snapshot raises `CITATION_PROVENANCE_MISMATCH`.

## Decisions

1. **Architectural Separation**:
   - Runtime modules (`raglab.domain`, `raglab.application`, `raglab.infrastructure`) must NEVER import `raglab.evaluation`. Enforced via automated AST static analysis tests.
   - Ground Truth data (`GroundTruthItemV2`) is strictly consumed post-generation by evaluation benchmarks and judges (`GeminiJudgeAdapter`).

2. **Ephemeral Presentation DTO (`PromptEvidence`)**:
   - `RetrievedEvidence` domain entities maintain persistent `passage_id`, `chunk_id`, `document_id`, `start_page`, `content_sha256`.
   - The presentation layer (`PromptEvidence`) maps rank 1 $\rightarrow$ `E1`, rank 2 $\rightarrow$ `E2`, rank 3 $\rightarrow$ `E3` per query.

3. **Untrusted Data Prompt Hardening**:
   - User queries and evidence passages are enclosed in strict `BEGIN_UNTRUSTED_...` / `END_UNTRUSTED_...` delimiters to prevent indirect prompt injection.

4. **Deterministic Evaluation Metrics**:
   - Passage Recall@k, MRR, Citation Precision/Recall, Abstention Confusion Matrix, and nDCG@k operate strictly post-generation.
   - Missing graded qrels return `"NOT_COMPUTABLE_MISSING_GRADED_QRELS"`.

## Consequences
- **Positive**: Complete scientific integrity of evaluation benchmarks; total isolation of gold data; robust protection against indirect prompt injection; clear provenance auditability.
- **Negative**: Legacy raw `[p.92]` strings in non-conforming evaluation pipelines must be mapped through `PromptEvidence` presentation DTOs.
