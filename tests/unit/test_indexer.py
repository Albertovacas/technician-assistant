from qdrant_client import QdrantClient

from pool_assistant.ingestion.indexer import index_chunks
from pool_assistant.vectorstore import QdrantStore

from .test_vectorstore import make_chunk


def test_indexes_several_documents(fake_embedder) -> None:
    store = QdrantStore(QdrantClient(":memory:"), "t", fake_embedder.dim)
    chunks = [make_chunk("a", 1, i, f"alpha {i}") for i in range(3)] + [
        make_chunk("b", 1, 0, "beta")
    ]

    written = index_chunks(chunks, fake_embedder, store)

    assert written == 4
    assert store.count(doc_id="a") == 3
    assert store.count(doc_id="b") == 1


def test_reindexing_is_idempotent(fake_embedder) -> None:
    store = QdrantStore(QdrantClient(":memory:"), "t", fake_embedder.dim)
    chunks = [make_chunk("a", 1, i, f"alpha {i}") for i in range(3)]

    index_chunks(chunks, fake_embedder, store)
    index_chunks(chunks, fake_embedder, store)

    assert store.count() == 3
