"""Embedder wrapper with an injected fake model (no ONNX download)."""

import numpy as np

from pool_assistant.embeddings import FastEmbedEmbedder


class RecordingModel:
    """Stands in for fastembed.TextEmbedding and records what it is asked to embed."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed(self, texts, batch_size=None):
        for text in texts:
            self.seen.append(text)
            yield np.array([float(len(text)), 1.0, 0.0])


def test_gemma_prefixes_are_applied_to_queries_and_documents() -> None:
    model = RecordingModel()
    embedder = FastEmbedEmbedder("google/embeddinggemma-300m", model=model)

    embedder.embed_query("pump noise")
    embedder.embed_documents(["Secure the pump"])

    assert model.seen == [
        "task: search result | query: pump noise",
        "title: none | text: Secure the pump",
    ]


def test_symmetric_models_get_no_prefix() -> None:
    model = RecordingModel()
    embedder = FastEmbedEmbedder(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", model=model
    )

    embedder.embed_query("pump noise")

    assert model.seen == ["pump noise"]


def test_vectors_are_plain_lists_and_dim_is_cached() -> None:
    model = RecordingModel()
    embedder = FastEmbedEmbedder("any-model", model=model)

    assert embedder.embed_documents(["a", "bb"]) == [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]
    assert embedder.dim == 3
    calls = len(model.seen)
    assert embedder.dim == 3
    assert len(model.seen) == calls  # no second probe
