"""Soft language filter of the retriever, on synthetic in-memory indexes."""

import pytest
from qdrant_client import QdrantClient

from pool_assistant.ingestion.language import detect_query_language
from pool_assistant.retrieval.retriever import Retriever
from pool_assistant.vectorstore import QdrantStore

from .test_vectorstore import make_chunk

EN = make_chunk("m", 13, 0, "The pump makes a noise: secure the pump correctly", lang="en")
ES = make_chunk("m", 33, 0, "La bomba hace ruido: fijar correctamente la bomba", lang="es")
DE = make_chunk("m", 53, 0, "Die Pumpe macht Geräusche: Pumpe korrekt befestigen", lang="de")
FIGURE = make_chunk("m", 95, 0, "Fan cover / Tapa ventilador / bomba ruido pump noise", lang=None)


def build(fake_embedder, *chunks) -> Retriever:
    store = QdrantStore(QdrantClient(":memory:"), "t", fake_embedder.dim)
    if chunks:
        vectors = fake_embedder.embed_documents([c.text for c in chunks])
        store.replace_document("m", list(chunks), vectors)
    return Retriever(fake_embedder, store, k=5, fallback_lang="en")


def pages(result) -> set[int]:
    return {c.chunk.page_number for c in result.chunks}


def test_same_language_when_available(fake_embedder) -> None:
    retriever = build(fake_embedder, EN, ES, DE)

    result = retriever.retrieve("La bomba hace ruido, ¿qué reviso?")

    assert (result.query_lang, result.search_lang, result.strategy) == ("es", "es", "same_language")
    assert pages(result) == {33}


def test_falls_back_to_english_when_question_language_is_missing(fake_embedder) -> None:
    """Manual only in EN (+DE), technician asks in Spanish."""
    retriever = build(fake_embedder, EN, DE)

    result = retriever.retrieve("La bomba hace ruido, ¿qué reviso?")

    assert (result.query_lang, result.search_lang, result.strategy) == (
        "es", "en", "fallback_language"
    )
    assert pages(result) == {13}


def test_unfiltered_when_neither_question_language_nor_fallback_exist(fake_embedder) -> None:
    retriever = build(fake_embedder, DE)

    result = retriever.retrieve("La bomba hace ruido, ¿qué reviso?")

    assert (result.search_lang, result.strategy) == (None, "unfiltered")
    assert pages(result) == {53}


def test_multilingual_chunks_are_included_with_language_filter(fake_embedder) -> None:
    retriever = build(fake_embedder, EN, ES, FIGURE)

    result = retriever.retrieve("La bomba hace ruido, ¿qué reviso?")

    assert pages(result) == {33, 95}  # ES chunk + lang=None figure, never the EN one


def test_explicit_language_overrides_detection(fake_embedder) -> None:
    retriever = build(fake_embedder, EN, ES)

    result = retriever.retrieve("bomba", lang="en")

    assert (result.query_lang, result.search_lang) == ("en", "en")


def test_undetermined_question_language_uses_fallback(fake_embedder) -> None:
    retriever = build(fake_embedder, EN, ES)

    result = retriever.retrieve("E01")

    assert (result.query_lang, result.strategy) == (None, "fallback_language")


def test_empty_index_returns_no_evidence(fake_embedder) -> None:
    result = build(fake_embedder).retrieve("La bomba hace ruido")

    assert (result.strategy, result.chunks) == ("unfiltered", [])


def test_k_limits_results(fake_embedder) -> None:
    many = [make_chunk("m", p, 0, f"pump noise number {p}", lang="en") for p in range(1, 9)]
    retriever = build(fake_embedder, *many)

    assert len(retriever.retrieve("pump noise", k=3).chunks) == 3


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("La bomba hace ruido, ¿qué debo revisar?", "es"),
        ("The pump makes a noise. What should I check?", "en"),
        ("La pompe fait du bruit", "fr"),
        ("bomba ruido", "es"),
        ("Fehlercode E01", None),  # low confidence -> undetermined rather than wrong
        ("¿", None),               # too short
    ],
)
def test_detect_query_language(question: str, expected: str | None) -> None:
    assert detect_query_language(question) == expected
