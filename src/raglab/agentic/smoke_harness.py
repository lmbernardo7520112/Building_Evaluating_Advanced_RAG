"""Agentic conversational smoke harness.

Executes known canonical canaries against any GenerationPort implementation
using purely hermetic, synthetic evidence and the public ConversationSessionRunner API.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from raglab.agentic.runtime.conversation_session_runner import (
    ConversationSessionRunner,
)
from raglab.application.ports.generation import GenerationPort
from raglab.domain.entities import GeneratedAnswer, RetrievedEvidence
from raglab.domain.enums import PipelineStrategy
from raglab.domain.value_objects import ChunkId


@dataclass
class _SyntheticRetrievalPort:
    """Internal hermetic retrieval port for smoke canaries."""

    prefix: str
    empty: bool = False
    custom_passages: Mapping[str, Sequence[RetrievedEvidence]] | None = None
    captured_queries: list[str] = field(default_factory=list)

    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedEvidence]:
        self.captured_queries.append(query)
        if self.empty:
            return []
        if self.custom_passages is not None:
            matched_key = None
            max_pos = -1
            for query_key in self.custom_passages:
                pos = query.rfind(query_key)
                if pos > max_pos:
                    max_pos = pos
                    matched_key = query_key
            if matched_key is not None:
                return self.custom_passages[matched_key]
        text = f"Synthetic evidence for query: {query}"
        text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return [
            RetrievedEvidence(
                chunk_id=ChunkId(value=f"chk_{self.prefix}_001"),
                document_id=f"doc_{self.prefix}_001",
                text=text,
                rank=1,
                score=0.95,
                passage_id=f"ps_{self.prefix}_001",
                content_sha256=text_sha,
                page_number=10,
            )
        ]


class _GeneratorProxy:
    """Internal proxy recording invocations to the target GenerationPort."""

    def __init__(self, delegate: GenerationPort) -> None:
        self._delegate = delegate
        self.captured_calls: list[
            tuple[str, str, tuple[RetrievedEvidence, ...]]
        ] = []

    @property
    def model_id(self) -> str:
        return self._delegate.model_id

    def generate(
        self,
        query_id: str,
        query: str,
        evidence: Sequence[RetrievedEvidence],
    ) -> GeneratedAnswer:
        self.captured_calls.append((query_id, query, tuple(evidence)))
        return self._delegate.generate(query_id, query, evidence)


def _run_direct_supported_fact(generator: GenerationPort) -> None:
    """Canary 1: single turn direct fact retrieval and verified generation."""
    fact_text = (
        "A velocidade da luz no vácuo é exatamente 299.792.458 metros por segundo."
    )
    fact_sha = hashlib.sha256(fact_text.encode("utf-8")).hexdigest()

    evidence_item = RetrievedEvidence(
        chunk_id=ChunkId(value="chk_fact_001"),
        document_id="doc_physics_001",
        text=fact_text,
        rank=1,
        score=0.98,
        passage_id="ps_fact_001",
        content_sha256=fact_sha,
        page_number=42,
    )

    custom_mapping = {"velocidade da luz": [evidence_item]}
    ports = {
        PipelineStrategy.BASELINE: _SyntheticRetrievalPort(
            "base", custom_passages=custom_mapping
        ),
        PipelineStrategy.SENTENCE_WINDOW_RERANK: _SyntheticRetrievalPort(
            "window", custom_passages=custom_mapping
        ),
    }

    proxy = _GeneratorProxy(generator)
    runner = ConversationSessionRunner(ports=ports, generator=proxy)
    runner.create_conversation("smoke_canary_1")

    turn = runner.execute_turn(
        "smoke_canary_1", "Qual é a velocidade da luz no vácuo?", top_k=3
    )

    if turn.generated_answer is None or turn.generated_answer.abstained:
        raise ValueError("Canary 1 answer unexpectedly abstained or missing")
    if turn.generated_answer.query_id != "smoke_canary_1_turn_0_q":
        raise ValueError("Canary 1 query_id mismatch")
    if turn.generation_model_id != generator.model_id:
        raise ValueError("Canary 1 model_id mismatch")
    if len(proxy.captured_calls) != 1:
        raise ValueError("Canary 1 generator call count mismatch")
    if len(turn.generated_answer.citations) != 1:
        raise ValueError("Canary 1 expected exactly one citation")

    citation = turn.generated_answer.citations[0]
    citation_chunk_val = (
        citation.chunk_id.value
        if hasattr(citation.chunk_id, "value")
        else str(citation.chunk_id)
    )
    if (
        citation.passage_id != "ps_fact_001"
        or citation.document_id != "doc_physics_001"
        or citation_chunk_val != "chk_fact_001"
        or citation.page_number != 42
        or citation.content_sha256 != fact_sha
        or citation.retrieval_rank != 1
    ):
        raise ValueError("Canary 1 citation provenance mismatch")

    state = runner.get_conversation("smoke_canary_1")
    if len(state.turns) != 1 or state.turns[0].state_hash != turn.state_hash:
        raise ValueError("Canary 1 state persistence mismatch")


def _run_conversational_ellipsis(generator: GenerationPort) -> None:
    """Canary 2: multi-turn conversational ellipsis with contextual retrieval."""
    text_t0 = (
        "A relatividade geral é a teoria geométrica da gravitação"
        " publicada por Einstein."
    )
    sha_t0 = hashlib.sha256(text_t0.encode("utf-8")).hexdigest()
    ev_t0 = RetrievedEvidence(
        chunk_id=ChunkId(value="chk_relativity_001"),
        document_id="doc_relativity_001",
        text=text_t0,
        rank=1,
        score=0.92,
        passage_id="ps_relativity_001",
        content_sha256=sha_t0,
        page_number=1,
    )

    text_t1 = (
        "As comprovações da relatividade incluem o desvio da luz e a"
        " precessão do periélio."
    )
    sha_t1 = hashlib.sha256(text_t1.encode("utf-8")).hexdigest()
    ev_t1 = RetrievedEvidence(
        chunk_id=ChunkId(value="chk_proofs_001"),
        document_id="doc_proofs_001",
        text=text_t1,
        rank=1,
        score=0.96,
        passage_id="ps_proofs_001",
        content_sha256=sha_t1,
        page_number=25,
    )

    custom_mapping = {
        "relatividade geral": [ev_t0],
        "comprovações": [ev_t1],
    }

    port_base = _SyntheticRetrievalPort("base", custom_passages=custom_mapping)
    port_window = _SyntheticRetrievalPort("window", custom_passages=custom_mapping)
    ports = {
        PipelineStrategy.BASELINE: port_base,
        PipelineStrategy.SENTENCE_WINDOW_RERANK: port_window,
    }

    proxy = _GeneratorProxy(generator)
    runner = ConversationSessionRunner(ports=ports, generator=proxy)
    runner.create_conversation("smoke_canary_2")

    turn0 = runner.execute_turn(
        "smoke_canary_2", "O que é a teoria da relatividade geral?"
    )
    if turn0.generated_answer is None or turn0.generated_answer.abstained:
        raise ValueError("Canary 2 turn 0 unexpectedly abstained")

    queries_before_t1_base = len(port_base.captured_queries)
    queries_before_t1_window = len(port_window.captured_queries)

    turn1 = runner.execute_turn(
        "smoke_canary_2", "Quais são as suas principais comprovações?"
    )
    if turn1.generated_answer is None or turn1.generated_answer.abstained:
        raise ValueError("Canary 2 turn 1 unexpectedly abstained")

    if (
        "O que é a teoria da relatividade geral?" not in turn1.retrieval_query
        or "Quais são as suas principais comprovações?"
        not in turn1.retrieval_query
    ):
        raise ValueError("Canary 2 retrieval query missing context")

    t1_queries_base = port_base.captured_queries[queries_before_t1_base:]
    t1_queries_window = port_window.captured_queries[queries_before_t1_window:]
    new_queries = t1_queries_base + t1_queries_window
    if not new_queries:
        raise ValueError("Canary 2 turn 1 made zero retrieval calls")
    for q in new_queries:
        if q != turn1.retrieval_query:
            raise ValueError(
                "Canary 2 turn 1 retrieval call did not use contextual query"
            )

    if len(proxy.captured_calls) != 2:
        raise ValueError("Canary 2 generator call count mismatch")

    # Generator must receive ONLY raw user query for turn 1
    call1_query = proxy.captured_calls[1][1]
    if call1_query != "Quais são as suas principais comprovações?":
        raise ValueError("Canary 2 generator received context instead of raw query")

    call1_evidence = proxy.captured_calls[1][2]
    if len(call1_evidence) != 1 or call1_evidence[0].passage_id != "ps_proofs_001":
        raise ValueError("Canary 2 generator evidence mismatch on second call")

    if len(turn1.generated_answer.citations) != 1:
        raise ValueError("Canary 2 turn 1 expected exactly one citation")

    if turn1.generated_answer.citations[0].passage_id != "ps_proofs_001":
        raise ValueError("Canary 2 turn 1 citation reused turn 0 passage")

    state = runner.get_conversation("smoke_canary_2")
    if (
        len(state.turns) != 2
        or state.turns[0].state_hash == state.turns[1].state_hash
    ):
        raise ValueError("Canary 2 state persistence mismatch")


def _run_unsupported_abstention(generator: GenerationPort) -> None:
    """Canary 3: unsupported query with empty retrieval produces abstention."""
    ports = {
        PipelineStrategy.BASELINE: _SyntheticRetrievalPort("base", empty=True),
        PipelineStrategy.SENTENCE_WINDOW_RERANK: _SyntheticRetrievalPort(
            "window", empty=True
        ),
    }

    proxy = _GeneratorProxy(generator)
    runner = ConversationSessionRunner(ports=ports, generator=proxy)
    runner.create_conversation("smoke_canary_3")

    turn = runner.execute_turn(
        "smoke_canary_3", "Pergunta sobre fato inexistente no corpus"
    )

    if turn.generated_answer is None or not turn.generated_answer.abstained:
        raise ValueError("Canary 3 unexpectedly produced answer instead of abstaining")
    if turn.generated_answer.text != "":
        raise ValueError("Canary 3 abstained answer has non-empty text")
    if turn.generated_answer.citations != ():
        raise ValueError("Canary 3 abstained answer has citations")
    if turn.generation_model_id is not None:
        raise ValueError("Canary 3 model_id must be None")
    if len(proxy.captured_calls) != 0:
        raise ValueError("Canary 3 generator unexpectedly called on empty evidence")

    state = runner.get_conversation("smoke_canary_3")
    if len(state.turns) != 1 or state.turns[0].state_hash != turn.state_hash:
        raise ValueError("Canary 3 state persistence mismatch")


def run_known_canaries(
    generator: GenerationPort,
    *,
    backend: str,
) -> dict[str, object]:
    """Execute the three known conversational canaries and return structured report."""
    canaries: dict[str, dict[str, object]] = {}

    try:
        _run_direct_supported_fact(generator)
        canaries["direct_supported_fact"] = {"status": "PASS"}

        _run_conversational_ellipsis(generator)
        canaries["conversational_ellipsis"] = {"status": "PASS"}

        _run_unsupported_abstention(generator)
        canaries["unsupported_abstention"] = {"status": "PASS"}

        return {
            "backend": backend,
            "model_id": generator.model_id,
            "status": "PASS",
            "canaries": canaries,
            "error": None,
        }
    except Exception as exc:
        return {
            "backend": backend,
            "model_id": getattr(generator, "model_id", "unknown"),
            "status": "FAIL",
            "canaries": canaries,
            "error": {"type": type(exc).__name__},
        }
