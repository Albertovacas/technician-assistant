"""Qdrant store in in-memory mode (same API as local/server mode, no files, no network)."""

import pytest
from qdrant_client import QdrantClient

from pool_assistant.schemas import Chunk
from pool_assistant.vectorstore import QdrantStore, VectorStoreError, point_id


def make_chunk(doc_id: str, page: int, idx: int, text: str, lang: str | None = "en") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}:p{page:04d}:c{idx:02d}", doc_id=doc_id, source=f"{doc_id}.pdf",
        page_number=page, chunk_index=idx, text=text, lang=lang,
        lang_confidence=1.0 if lang else 0.0,
        lang_method="detected" if lang else "undetermined",
    )


@pytest.fixture
def store(fake_embedder) -> QdrantStore:
    return QdrantStore(QdrantClient(":memory:"), "test", fake_embedder.dim)


def index(store, embedder, doc_id, chunks):
    vectors = embedder.embed_documents([c.text for c in chunks])
    return store.replace_document(doc_id, chunks, vectors)


def test_point_ids_are_deterministic_uuids() -> None:
    assert point_id("doc:p0001:c00") == point_id("doc:p0001:c00")
    assert point_id("doc:p0001:c00") != point_id("doc:p0001:c01")


def test_search_returns_closest_chunk_with_payload(store, fake_embedder) -> None:
    chunks = [
        make_chunk("pump", 13, 0, "The pump makes a noise: secure the pump correctly"),
        make_chunk("pump", 12, 0, "Replace the mechanical seal every year"),
    ]
    index(store, fake_embedder, "pump", chunks)

    (best,) = store.search(fake_embedder.embed_query("pump makes a noise"), k=1)

    assert best.chunk == chunks[0]  # full Chunk restored from the payload
    assert 0 < best.score <= 1.0


def test_reindexing_replaces_instead_of_duplicating(store, fake_embedder) -> None:
    v1 = [make_chunk("pump", 1, i, f"version one chunk {i}") for i in range(3)]
    v2 = [make_chunk("pump", 1, 0, "version two, a single chunk")]

    index(store, fake_embedder, "pump", v1)
    index(store, fake_embedder, "pump", v1)  # same version twice
    assert store.count(doc_id="pump") == 3

    index(store, fake_embedder, "pump", v2)  # new version with fewer chunks
    assert store.count(doc_id="pump") == 1  # stale chunks removed


def test_reindexing_one_document_keeps_the_others(store, fake_embedder) -> None:
    index(store, fake_embedder, "a", [make_chunk("a", 1, 0, "alpha")])
    index(store, fake_embedder, "b", [make_chunk("b", 1, 0, "beta")])

    index(store, fake_embedder, "a", [make_chunk("a", 1, 0, "alpha v2")])

    assert store.count() == 2
    assert store.count(doc_id="b") == 1


def test_filters_by_language_and_document(store, fake_embedder) -> None:
    index(store, fake_embedder, "pump", [
        make_chunk("pump", 13, 0, "the pump makes a noise", lang="en"),
        make_chunk("pump", 33, 0, "la bomba hace ruido", lang="es"),
        make_chunk("pump", 95, 0, "pump / bomba / pompe", lang=None),
    ])
    query = fake_embedder.embed_query("pump noise")

    assert {r.chunk.lang for r in store.search(query, k=10, lang="es")} == {"es"}
    assert store.count(lang="en") == 1
    assert store.count(lang="fr") == 0
    assert store.search(query, k=10, doc_id="other") == []


def test_dimension_change_fails_loudly(fake_embedder) -> None:
    client = QdrantClient(":memory:")
    QdrantStore(client, "manuals", 64).ensure_collection()

    with pytest.raises(VectorStoreError, match="embedding model changed"):
        QdrantStore(client, "manuals", 768).ensure_collection()


def test_compatibility_check_before_querying() -> None:
    client = QdrantClient(":memory:")
    QdrantStore(client, "manuals", 64).check_compatible()  # no collection yet: nothing to check
    QdrantStore(client, "manuals", 64).ensure_collection()

    QdrantStore(client, "manuals", 64).check_compatible()
    with pytest.raises(VectorStoreError, match="embedding model changed"):
        QdrantStore(client, "manuals", 768).check_compatible()


def test_empty_store_is_safe(store, fake_embedder) -> None:
    assert store.search(fake_embedder.embed_query("x")) == []
    assert store.count() == 0


def test_chunks_from_another_document_are_rejected(store, fake_embedder) -> None:
    with pytest.raises(ValueError):
        index(store, fake_embedder, "a", [make_chunk("b", 1, 0, "wrong doc")])
