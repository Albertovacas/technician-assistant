"""End-to-end indexing of the sample manual with the real embedding model.

Slow (loads the ~1.2 GB ONNX model): run `pytest -m "not slow"` to skip.
Uses an in-memory Qdrant, so it never touches the real index on disk.
"""

import pytest
from qdrant_client import QdrantClient

from pool_assistant.config import get_settings
from pool_assistant.embeddings import get_embedder
from pool_assistant.ingestion.chunker import chunk_pages
from pool_assistant.ingestion.indexer import index_chunks
from pool_assistant.ingestion.loader import load_pdf
from pool_assistant.ingestion.tables import enrich_with_tables
from pool_assistant.ingestion.vision import enrich_with_vision
from pool_assistant.vectorstore import QdrantStore

pytestmark = pytest.mark.slow

SETTINGS = get_settings()
MANUAL = SETTINGS.data_dir / "manuals" / "victoria_plus_silent_manual.pdf"

# (question, language, expected pages in the English section, expected pages in the Spanish
# section). Pages checked by hand against the manual; also used to choose the embedding model.
QUERIES = [
    ("The pump makes a noise. What should I check?", "en", {13}, None),
    ("How often must the mechanical seal be replaced?", "en", {12}, None),
    ("What is the maximum water temperature for the pump?", "en", {5}, None),
    ("Which residual current device (RCD) rating is required?", "en", {6}, None),
    ("How do I prime the pump before the first start-up?", "en", {9, 11}, None),
    ("What should I do to protect the pump from freezing in winter?", "en", {11}, None),
    ("The motor has stopped and the thermal protection trips", "en", {13}, None),
    ("Can the pump be installed in zone 1 near the pool?", "en", {5}, None),
    ("La bomba hace ruido, ¿qué debo revisar?", "es", {13}, {33}),
    ("¿Cada cuánto hay que cambiar el condensador?", "es", {12}, {32}),
    ("¿Cuál es la temperatura máxima del agua?", "es", {5}, {25}),
    ("¿Qué diferencial necesito para la instalación eléctrica?", "es", {6}, {26, 30}),
    ("¿Cómo se ceba la bomba antes de la primera puesta en marcha?", "es", {9, 11}, {29, 31}),
    ("¿Qué hago si va a helar y la bomba va a estar parada?", "es", {11}, {32}),
    ("El motor gira en sentido contrario", "es", {13}, {33}),
    ("¿Se puede instalar la bomba en la zona 1?", "es", {5}, {26}),
]


@pytest.fixture(scope="module")
def indexed():
    if not MANUAL.exists():
        pytest.skip(f"Manual not available at {MANUAL}")
    pages = enrich_with_vision(
        enrich_with_tables(load_pdf(MANUAL), MANUAL), MANUAL, None,
        model_name=SETTINGS.vision_model, cache_dir=SETTINGS.vision_cache_dir,
    )
    chunks = chunk_pages(pages, doc_id="victoria-plus-silent")
    embedder = get_embedder(SETTINGS)
    store = QdrantStore(QdrantClient(":memory:"), "manuals", embedder.dim)
    index_chunks(chunks, embedder, store)
    return chunks, embedder, store


def test_every_chunk_is_indexed(indexed) -> None:
    chunks, _, store = indexed
    assert store.count() == len(chunks)


@pytest.mark.parametrize(
    ("question", "lang", "expected_page"),
    [
        ("The pump makes a noise. What should I check?", "en", 13),
        ("La bomba hace ruido, ¿qué debo revisar?", "es", 33),
        ("How often must the mechanical seal be replaced?", "en", 12),
        ("¿Cada cuánto hay que cambiar el condensador?", "es", 32),
    ],
)
def test_expected_page_in_top_5(indexed, question, lang, expected_page) -> None:
    _, embedder, store = indexed

    results = store.search(embedder.embed_query(question), k=5, lang=lang)

    assert expected_page in [r.chunk.page_number for r in results]


def test_retriever_hit_at_5_on_benchmark_questions(indexed) -> None:
    """The 16 EN/ES QUERIES through the retriever (language detected, not given)."""
    from pool_assistant.retrieval.retriever import Retriever

    _, embedder, store = indexed
    retriever = Retriever(embedder, store, k=5)
    hits = 0
    for question, lang, pages_en, pages_es in QUERIES:
        result = retriever.retrieve(question)  # language detected, not given
        assert result.strategy == "same_language"
        expected = pages_en if lang == "en" else pages_es
        hits += bool(expected & {c.chunk.page_number for c in result.chunks})

    assert hits / len(QUERIES) >= 0.9, f"hit@5 = {hits}/{len(QUERIES)}"


def test_cross_lingual_spanish_question_finds_english_chunk(indexed) -> None:
    """A manual that only exists in English must still answer a Spanish question."""
    _, embedder, store = indexed

    results = store.search(embedder.embed_query("La bomba hace ruido"), k=5, lang="en")

    assert 13 in [r.chunk.page_number for r in results]
