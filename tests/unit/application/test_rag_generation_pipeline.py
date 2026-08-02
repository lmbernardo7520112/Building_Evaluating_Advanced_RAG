"""Tests for RagGenerationPipeline use case — offline with fakes."""

from __future__ import annotations

from raglab.domain.entities import RetrievedEvidence
from raglab.domain.value_objects import ChunkId


class _FakeRetriever:
    """Minimal fake retriever for pipeline tests."""

    def __init__(self, results: list) -> None:
        self._results = results

    def retrieve(self, query: str) -> list:
        return self._results


class _FakeGenerator:
    """Minimal fake generator for pipeline tests."""

    def __init__(self, abstained: bool = False) -> None:
        from raglab.domain.entities import GeneratedAnswer
        self._answer = GeneratedAnswer(
            query_id="placeholder",
            text="[FAKE] Test answer." if not abstained else "ABSTAIN",
            abstained=abstained,
            citations=(),
        )
        self._calls: list[tuple] = []

    @property
    def model_id(self) -> str:
        return "fake-generator-pipeline-test"

    def generate(self, query_id: str, query: str, evidence: list):
        from raglab.domain.entities import GeneratedAnswer
        self._calls.append((query_id, query))
        # Return answer with correct query_id
        return GeneratedAnswer(
            query_id=query_id,
            text=self._answer.text,
            abstained=self._answer.abstained,
            citations=(),
        )


def _make_evidence(n: int = 2) -> list[RetrievedEvidence]:
    return [
        RetrievedEvidence(
            chunk_id=ChunkId(f"chunk_{i}"),
            document_id=f"doc_p9{i}",
            text=f"Texto de evidência {i}.",
            rank=i + 1,
            score=0.9 - (i * 0.05),
        )
        for i in range(n)
    ]


class TestRagGenerationPipeline:
    def test_run_returns_generation_result(self):
        from raglab.application.use_cases.rag_generation_pipeline import (
            RagGenerationPipeline,
        )
        evidence = _make_evidence(2)
        pipeline = RagGenerationPipeline(
            retriever=_FakeRetriever(evidence),
            generator=_FakeGenerator(),
            retrieval_strategy="F0_baseline",
        )
        result = pipeline.run(query_id="q_dev_01", query="O que é indução?")

        assert result.query_id == "q_dev_01"
        assert result.query == "O que é indução?"
        assert result.retrieval_strategy == "F0_baseline"
        assert result.generator_model_id == "fake-generator-pipeline-test"
        assert not result.answer.abstained

    def test_run_passes_query_id_to_generator(self):
        from raglab.application.use_cases.rag_generation_pipeline import (
            RagGenerationPipeline,
        )
        gen = _FakeGenerator()
        pipeline = RagGenerationPipeline(
            retriever=_FakeRetriever(_make_evidence()),
            generator=gen,
        )
        result = pipeline.run(query_id="q_test_99", query="Test?")
        assert result.answer.query_id == "q_test_99"
        assert gen._calls[0][0] == "q_test_99"

    def test_run_with_empty_evidence_abstains(self):
        from raglab.application.use_cases.rag_generation_pipeline import (
            RagGenerationPipeline,
        )
        gen = _FakeGenerator(abstained=True)
        pipeline = RagGenerationPipeline(
            retriever=_FakeRetriever([]),
            generator=gen,
        )
        result = pipeline.run(query_id="q_abstain", query="Out of scope?")
        assert result.answer.abstained is True
        assert len(result.evidence) == 0

    def test_evidence_preserved_in_result(self):
        from raglab.application.use_cases.rag_generation_pipeline import (
            RagGenerationPipeline,
        )
        evidence = _make_evidence(3)
        pipeline = RagGenerationPipeline(
            retriever=_FakeRetriever(evidence),
            generator=_FakeGenerator(),
        )
        result = pipeline.run(query_id="q_dev_01", query="Test?")
        assert len(result.evidence) == 3
