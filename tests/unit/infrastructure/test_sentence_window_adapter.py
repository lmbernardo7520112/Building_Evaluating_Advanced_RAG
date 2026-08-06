"""Unit tests for SentenceWindowAdapter and sentence splitting."""

import unittest

from raglab.domain.value_objects import DocumentPage
from raglab.infrastructure.retrieval.sentence_window_adapter import (
    SentenceWindowAdapter,
    split_into_sentences,
)
from tests.unit.infrastructure.deterministic_embedding_double import (
    DeterministicTestEmbeddingAdapter,
)


class TestSentenceWindowAdapter(unittest.TestCase):

    def test_split_into_sentences_abbreviations(self) -> None:
        text = (
            "Veja a pág. 92 sobre demonstração por exaustão. "
            "Isso é um e.g. típico de prova. Fim do capítulo."
        )
        sents = split_into_sentences(text)
        self.assertEqual(len(sents), 3)
        self.assertIn("pág. 92", sents[0])
        self.assertIn("e.g. típico", sents[1])

    def test_sentence_window_retrieval(self) -> None:
        adapter = SentenceWindowAdapter(
            embedding_adapter=DeterministicTestEmbeddingAdapter(),
            window_size=1,
        )

        pages = [
            DocumentPage(
                document_id="doc1",
                page_number=91,
                text=(
                    "Primeira frase da página. "
                    "Segunda frase sobre exaustão. "
                    "Terceira frase final."
                ),
            )
        ]
        count = adapter.index_pages(pages)
        self.assertEqual(count, 3)

        results = adapter.retrieve("exaustão", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("Primeira frase", results[0].text)
        self.assertIn("Segunda frase sobre exaustão", results[0].text)
        self.assertIn("Terceira frase", results[0].text)


if __name__ == "__main__":
    unittest.main()
