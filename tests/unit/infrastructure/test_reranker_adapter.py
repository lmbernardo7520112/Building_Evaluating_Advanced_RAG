"""Unit tests for LocalRerankerAdapter and damage calculation."""

import unittest

from raglab.domain.entities import RetrievedEvidence
from raglab.domain.value_objects import ChunkId
from raglab.infrastructure.retrieval.reranker_adapter import LocalRerankerAdapter


class TestLocalRerankerAdapter(unittest.TestCase):

    def test_rerank_and_damage_metrics(self) -> None:
        reranker = LocalRerankerAdapter()
        evidences = [
            RetrievedEvidence(
                chunk_id=ChunkId("chunk_1"),
                document_id="doc1_p91",
                text="Texto sobre indução matemática.",
                rank=1,
                score=0.9,
            ),
            RetrievedEvidence(
                chunk_id=ChunkId("chunk_2"),
                document_id="doc1_p92",
                text="Texto sobre exaustão.",
                rank=2,
                score=0.8,
            ),
        ]

        reranked, dropped = reranker.rerank("exaustão", evidences, top_n=1)
        self.assertEqual(len(reranked), 1)

        damage = reranker.calculate_damage_metrics(
            candidates_pre=evidences,
            candidates_post=reranked,
            relevant_chunk_ids={"chunk_1"},
            candidate_k=2,
            top_n=1,
        )
        self.assertIsNotNone(damage)
        self.assertGreaterEqual(damage.relevant_passage_dropped_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
