"""RAG prompts for Gemini generation and RAG Triad evaluation.

All prompts are:
- Deterministic (temperature=0.0 enforced in adapters)
- Structured for machine-readable extraction
- Language-adaptive (PT-BR context for Gersting corpus)
- Designed to request JSON-parseable sub-fields when needed

SECURITY:
- No credentials or keys are embedded here
- No network calls — pure string templates
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────
# Generation prompt
# ─────────────────────────────────────────────────────────────────

GENERATION_SYSTEM = """\
You are a precise academic assistant specializing in discrete mathematics \
and mathematical foundations of computer science.

Rules:
1. Answer ONLY from the provided context passages.
2. If the context does not contain sufficient information to answer, \
respond with exactly: ABSTAIN
3. Be concise but complete.
4. Do NOT hallucinate facts not present in the context.
5. Cite the passage numbers you relied on (e.g. [p.92]).
6. Respond in the same language as the question.
"""

GENERATION_USER_TEMPLATE = """\
QUESTION: {query}

CONTEXT PASSAGES:
{context}

Answer based solely on the context above. \
If you cannot answer from the context, respond: ABSTAIN
"""


def build_generation_prompt(query: str, context_passages: list[str]) -> str:
    """Build the user-turn prompt for answer generation."""
    numbered = "\n\n".join(
        f"[{i + 1}] {p.strip()}" for i, p in enumerate(context_passages)
    )
    return GENERATION_USER_TEMPLATE.format(query=query, context=numbered)


# ─────────────────────────────────────────────────────────────────
# RAG Triad evaluation prompts
# ─────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
You are a strict, impartial evaluator of RAG (Retrieval-Augmented Generation) \
pipeline outputs. Your job is to evaluate quality along specific dimensions.

Rules:
1. Evaluate ONLY the dimension specified in the user prompt.
2. Return ONLY valid JSON — no prose, no markdown, no code fences.
3. Scores must be floats in [0.0, 1.0].
4. Provide a brief reasoning before the score.
"""

# Context Relevance: Is the retrieved context relevant to the query?
CONTEXT_RELEVANCE_TEMPLATE = """\
DIMENSION: Context Relevance
QUESTION: Does the retrieved context contain information relevant to answering \
the query?

QUERY: {query}

RETRIEVED CONTEXT:
{context}

Evaluate whether the context is relevant and useful for answering the query.

Respond with ONLY this JSON:
{{
  "reasoning": "<one sentence explanation>",
  "score": <float 0.0 to 1.0>
}}

Where score=1.0 means fully relevant, 0.0 means completely irrelevant.
"""

# Groundedness: Is the answer supported by the retrieved context?
GROUNDEDNESS_TEMPLATE = """\
DIMENSION: Groundedness
QUESTION: Is the answer supported by (grounded in) the retrieved context?

QUERY: {query}

RETRIEVED CONTEXT:
{context}

ANSWER TO EVALUATE:
{answer}

Evaluate whether every claim in the answer is directly supported by the context.
Penalise any claim in the answer that cannot be traced to the context.

Respond with ONLY this JSON:
{{
  "reasoning": "<one sentence explanation>",
  "score": <float 0.0 to 1.0>
}}

Where score=1.0 means fully grounded, 0.0 means answer contains hallucinations.
"""

# Answer Relevance: Does the answer address the query?
ANSWER_RELEVANCE_TEMPLATE = """\
DIMENSION: Answer Relevance
QUESTION: Does the answer actually address the query?

QUERY: {query}

ANSWER TO EVALUATE:
{answer}

Evaluate whether the answer is responsive to the query, regardless of correctness.

Respond with ONLY this JSON:
{{
  "reasoning": "<one sentence explanation>",
  "score": <float 0.0 to 1.0>
}}

Where score=1.0 means directly addresses the query, 0.0 means off-topic.
"""

# Factual Correctness: Does the answer match a gold reference?
FACTUAL_CORRECTNESS_TEMPLATE = """\
DIMENSION: Factual Correctness
QUESTION: Does the answer agree with the reference answer?

QUERY: {query}

REFERENCE ANSWER (gold):
{gold_answer}

ANSWER TO EVALUATE:
{answer}

Evaluate whether the answer is factually consistent with the reference.

Respond with ONLY this JSON:
{{
  "reasoning": "<one sentence explanation>",
  "score": <float 0.0 to 1.0>
}}

Where score=1.0 means fully consistent, 0.0 means contradicts the reference.
"""


def build_context_relevance_prompt(query: str, context_passages: list[str]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {p.strip()}" for i, p in enumerate(context_passages))
    return CONTEXT_RELEVANCE_TEMPLATE.format(query=query, context=ctx)


def build_groundedness_prompt(
    query: str, context_passages: list[str], answer: str
) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {p.strip()}" for i, p in enumerate(context_passages))
    return GROUNDEDNESS_TEMPLATE.format(query=query, context=ctx, answer=answer)


def build_answer_relevance_prompt(query: str, answer: str) -> str:
    return ANSWER_RELEVANCE_TEMPLATE.format(query=query, answer=answer)


def build_factual_correctness_prompt(
    query: str, gold_answer: str, answer: str
) -> str:
    return FACTUAL_CORRECTNESS_TEMPLATE.format(
        query=query, gold_answer=gold_answer, answer=answer
    )
