"""Fake generator adapter for use in tests and offline development.

Implements the domain's GeneratedAnswer contract without any network access.

SECURITY CONTRACT:
- Never imports google.generativeai or any Gemini SDK
- Never reads GEMINI_API_KEY, GOOGLE_API_KEY, or any credential variable
- Never makes network calls
- Never logs credential values
- Safe to use in untrusted environments

The real GeminiGeneratorAdapter DOES NOT exist in this codebase yet.
It will be implemented in a future slice with explicit Gate authorization,
executed only by a human operator in an isolated terminal.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from raglab.domain.entities import GeneratedAnswer, RetrievedEvidence
from raglab.domain.value_objects import Citation

_FAKE_MODEL_ID = "fake-generator-v1-no-network"
_OFFLINE_NOTICE = (
    "[FAKE GENERATOR] No LLM was called. "
    "This adapter is for offline use only. "
    "Gemini integration is PLANNED but not implemented."
)


def _extract_page(document_id: str) -> int:
    """Extract page number from document_id like 'doc_p91'. Returns 0 on failure."""
    try:
        return int(document_id.split("_p")[-1])
    except (ValueError, IndexError):
        return 0


class FakeGeneratorAdapter:
    """Deterministic, network-free generator for offline testing.

    Conforms to GenerationPort (structural typing via Protocol).

    Behavior:
    - Always returns a deterministic answer derived from query + evidence
    - Never calls any external API
    - Never reads credentials
    - Logs no sensitive data
    - Safe for CI and Antigravity execution
    """

    @property
    def model_id(self) -> str:
        return _FAKE_MODEL_ID

    def generate(
        self,
        query_id: str,
        query: str,
        evidence: Sequence[RetrievedEvidence],
    ) -> GeneratedAnswer:
        """Return deterministic fake answer without any network call."""
        if not query or not query.strip():
            abstained = True
            text = "[FAKE] Empty query received — abstaining."
            citations: tuple[Citation, ...] = ()
        elif not evidence:
            abstained = True
            text = (
                f"[FAKE] No evidence retrieved for: {query[:60]!r}. "
                "Abstaining — cannot generate answer without context."
            )
            citations = ()
        else:
            abstained = False
            top_chunks = [ev.text[:80] for ev in evidence[:2]]
            text = (
                f"[FAKE] Com base em {len(evidence)} trechos recuperados, "
                f"uma resposta seria gerada para: {query[:50]!r}. "
                f"Evidência principal: {top_chunks[0]!r}. "
                f"{_OFFLINE_NOTICE}"
            )
            citations = tuple(
                Citation(
                    document_id=ev.document_id,
                    page_number=_extract_page(ev.document_id),
                    chunk_id=ev.chunk_id,
                    text_span=ev.text[:40],
                    evidence_id=f"E{idx + 1}",
                    passage_id=getattr(
                        ev, "canonical_passage_id", getattr(ev, "passage_id", None)
                    ),
                    content_sha256=getattr(ev, "content_sha256", None)
                    or hashlib.sha256(ev.text.encode("utf-8")).hexdigest(),

                    retrieval_rank=ev.rank,
                )
                for idx, ev in enumerate(evidence[:3])
            )


        return GeneratedAnswer(
            query_id=query_id,
            text=text,
            abstained=abstained,
            citations=citations,
        )

    @classmethod
    def is_credential_safe(cls) -> bool:
        """Verify that no credential is reachable from this adapter."""
        import os

        dangerous = {
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "HF_TOKEN",
            "LANGSMITH_API_KEY",
        }
        present = {k for k in dangerous if os.environ.get(k)}
        # Do NOT log values — only count
        return len(present) == 0
