"""Text embeddings for retrieval (local ONNX models via FastEmbed: free, offline, no API).

Asymmetric models are trained with different instruction prefixes for queries and
documents, and FastEmbed does not add them for dense models. Forgetting them raises no
error, so they live here, next to the model choice. Measured effect on our 16-query set
(notebook 05): cross-lingual hit@3 0.75 -> 0.88, other metrics unchanged.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# model name -> (query prefix, document prefix)
PREFIXES: dict[str, tuple[str, str]] = {
    "google/embeddinggemma-300m": ("task: search result | query: ", "title: none | text: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
}


class Embedder(Protocol):
    """What the indexer and the retriever need from an embedding model."""

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """FastEmbed-backed embedder. The ONNX model is loaded lazily on first use."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: Path | None = None,
        batch_size: int = 32,
        model: Any | None = None,  # injectable for tests
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._model = model
        self._dim: int | None = None
        self.query_prefix, self.document_prefix = PREFIXES.get(model_name, ("", ""))

    @property
    def model(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            logger.info("Loading embedding model %s", self.model_name)
            self._model = TextEmbedding(
                self.model_name,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:  # computed once from a probe: works for any model
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        prefixed = [self.document_prefix + t for t in texts]
        return [v.tolist() for v in self.model.embed(prefixed, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.embed([self.query_prefix + text]))).tolist()


def get_embedder(settings=None) -> FastEmbedEmbedder:
    from pool_assistant.config import get_settings

    settings = settings or get_settings()
    return FastEmbedEmbedder(
        settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
        batch_size=settings.embedding_batch_size,
    )
