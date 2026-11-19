"""Simple RAG with a stub retriever and a recording fake LLM (no network, no models)."""

import pytest
from langchain_core.messages import AIMessage

from pool_assistant.rag.prompts import NO_ANSWER, insufficient_evidence_message
from pool_assistant.rag.simple_rag import SimpleRAG, extract_cited_pages
from pool_assistant.schemas import RetrievalResult, ScoredChunk

from .test_vectorstore import make_chunk

EVIDENCE = [
    ScoredChunk(chunk=make_chunk("m", 33, 0, "3. La bomba hace ruido → Fijación incorrecta",
                                 lang="es"), score=0.53),
    ScoredChunk(chunk=make_chunk("m", 26, 0, "Desconectar la alimentación eléctrica",
                                 lang="es"), score=0.41),
]


class StubRetriever:
    def __init__(self, chunks, query_lang="es"):
        self.chunks, self.query_lang = chunks, query_lang

    def retrieve(self, question, *, lang=None):
        return RetrievalResult(question=question, query_lang=lang or self.query_lang,
                               search_lang="es", strategy="same_language", chunks=self.chunks)


class RecordingLLM:
    def __init__(self, reply: str):
        self.reply, self.calls = reply, []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.reply)


def rag(reply: str, chunks=EVIDENCE, query_lang="es") -> tuple[SimpleRAG, RecordingLLM]:
    llm = RecordingLLM(reply)
    return SimpleRAG(StubRetriever(chunks, query_lang), llm, model_name="fake"), llm


@pytest.mark.parametrize(
    ("text", "pages"),
    [
        ("Fija la bomba [p. 33]. Desconecta [p.26].", [33, 26]),
        ("Ver [P. 13] y de nuevo [p. 13].", [13]),
        ("Ver [p. 12, 13] y [p. 5; p. 6].", [12, 13, 5, 6]),
        ("Sin citas.", []),
    ],
)
def test_extract_cited_pages(text: str, pages: list[int]) -> None:
    assert extract_cited_pages(text) == pages


def test_answer_with_valid_citations() -> None:
    pipeline, _ = rag("Desconecta la alimentación [p. 26]. Fija la bomba [p. 33].")

    answer = pipeline.answer("La bomba hace ruido")

    assert [c.page_number for c in answer.citations] == [26, 33]
    assert answer.citations[0].source == "m.pdf"
    assert answer.invalid_citations == []
    assert not answer.insufficient_evidence
    assert answer.language == "es"
    assert answer.retrieval.strategy == "same_language"


def test_citations_outside_the_evidence_are_flagged() -> None:
    pipeline, _ = rag("Fija la bomba [p. 33]. Cambia el condensador [p. 99].")

    answer = pipeline.answer("La bomba hace ruido")

    assert [c.page_number for c in answer.citations] == [33]
    assert answer.invalid_citations == [99]


def test_no_answer_marker_returns_localized_fallback() -> None:
    pipeline, _ = rag(NO_ANSWER)

    answer = pipeline.answer("¿Cuál es el precio?")

    assert answer.insufficient_evidence
    assert answer.answer == insufficient_evidence_message("es")
    assert answer.citations == []


def test_empty_retrieval_does_not_call_the_llm() -> None:
    pipeline, llm = rag("should not be used", chunks=[])

    answer = pipeline.answer("La bomba hace ruido")

    assert llm.calls == []
    assert answer.insufficient_evidence


def test_prompt_contains_labelled_evidence_and_language() -> None:
    pipeline, llm = rag("ok [p. 33]")

    pipeline.answer("La bomba hace ruido", lang="es")

    system, user = llm.calls[0]
    assert "[p. N]" in system.content and NO_ANSWER in system.content
    assert "[Excerpt 1 | p. 33 | m.pdf | lang=es]" in user.content
    assert "<question>\nLa bomba hace ruido\n</question>" in user.content
    assert user.content.endswith("Answer in Spanish.")


def test_unknown_language_asks_to_mirror_the_question() -> None:
    pipeline, llm = rag("ok [p. 33]", query_lang=None)

    answer = pipeline.answer("E01")

    assert llm.calls[0][1].content.endswith("Answer in the same language as the question.")
    assert answer.language is None


def test_fallback_message_defaults_to_english_for_unsupported_languages() -> None:
    assert insufficient_evidence_message("ru") == insufficient_evidence_message("en")
    assert insufficient_evidence_message(None) == insufficient_evidence_message("en")
