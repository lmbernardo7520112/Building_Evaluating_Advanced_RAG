"""Tests for RAG prompt templates — pure strings, no network, no credentials."""

from __future__ import annotations

from raglab.infrastructure.gemini.prompts import (
    build_answer_relevance_prompt,
    build_context_relevance_prompt,
    build_factual_correctness_prompt,
    build_generation_prompt,
    build_groundedness_prompt,
)


class TestGenerationPrompts:
    def test_generation_prompt_includes_query(self):
        query = "O que é indução matemática?"
        passages = ["Indução é um método de prova.", "Etapa base e passo indutivo."]
        prompt = build_generation_prompt(query, passages)
        assert query in prompt
        assert "Indução é um método" in prompt
        assert "[1]" in prompt
        assert "[2]" in prompt

    def test_generation_prompt_empty_passages(self):
        prompt = build_generation_prompt("query?", [])
        assert "query?" in prompt
        assert "ABSTAIN" in prompt  # template mentions abstain

    def test_generation_prompt_has_no_credentials(self):
        prompt = build_generation_prompt("test query", ["some context"])
        assert "GEMINI_API_KEY" not in prompt
        assert "API_KEY" not in prompt
        assert "sk-" not in prompt


class TestContextRelevancePrompt:
    def test_includes_query_and_context(self):
        prompt = build_context_relevance_prompt(
            "O que é indução?", ["Contexto sobre indução matemática."]
        )
        assert "O que é indução?" in prompt
        assert "Contexto sobre indução" in prompt
        assert "Context Relevance" in prompt
        assert "score" in prompt

    def test_prompts_for_json_output(self):
        prompt = build_context_relevance_prompt("q?", ["ctx"])
        assert '{"' in prompt or '"score"' in prompt


class TestGroundednessPrompt:
    def test_includes_answer(self):
        prompt = build_groundedness_prompt(
            "query", ["evidence passage"], "My generated answer."
        )
        assert "My generated answer." in prompt
        assert "Groundedness" in prompt
        assert "grounded" in prompt


class TestAnswerRelevancePrompt:
    def test_includes_query_and_answer(self):
        prompt = build_answer_relevance_prompt(
            "O que é demonstração por exaustão?",
            "Demonstração por exaustão verifica todos os casos.",
        )
        assert "demonstração por exaustão" in prompt.lower()
        assert "Answer Relevance" in prompt


class TestFactualCorrectnessPrompt:
    def test_includes_gold_and_answer(self):
        prompt = build_factual_correctness_prompt(
            "Qual é o resultado?",
            gold_answer="O resultado é 42.",
            answer="A resposta é 42.",
        )
        assert "O resultado é 42." in prompt
        assert "A resposta é 42." in prompt
        assert "Factual Correctness" in prompt

    def test_no_credentials_in_any_prompt(self):
        for fn in [
            lambda: build_generation_prompt("q", ["c"]),
            lambda: build_context_relevance_prompt("q", ["c"]),
            lambda: build_groundedness_prompt("q", ["c"], "a"),
            lambda: build_answer_relevance_prompt("q", "a"),
            lambda: build_factual_correctness_prompt("q", "gold", "a"),
        ]:
            result = fn()
            assert "GEMINI_API_KEY" not in result
            assert "API_KEY" not in result
