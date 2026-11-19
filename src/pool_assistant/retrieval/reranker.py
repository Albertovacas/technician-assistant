"""Cross-encoder reranking of the vector-search candidates (local ONNX model via FastEmbed).

Why: the embedding search compares two vectors computed separately; in a long chunk the
vector averages all its topics and a single fact is diluted. Measured case: "maximum water
temperature" -> the chunk with "35 ºC" (p.5, a long safety-warnings chunk) ranked 20th by
cosine. A cross-encoder reads question and chunk together and scores their relation, so
it is more precise but slower: it is applied only to a few candidates (top-20 -> top-5).

The model must be multilingual: questions and manuals come in any language (the ms-marco
models are English only).
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from pool_assistant.schemas import ScoredChunk

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    """What the retriever needs from a reranker."""

    def rerank(self, query: str, chunks: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
        ...


class FastEmbedReranker:
    """FastEmbed cross-encoder. The ONNX model is loaded lazily on first use."""

    def __init__(self, model_name: str, *, cache_dir: Path | None = None,
                 model: Any | None = None) -> None:  # `model` injectable for tests
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            logger.info("Loading reranker model %s", self.model_name)
            self._model = TextCrossEncoder(
                self.model_name, cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )
        return self._model

    def rerank(self, query: str, chunks: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
        """Top-k chunks by cross-encoder relevance (best first).

        The returned `score` stays the cosine similarity of the vector search, so scores keep
        one meaning everywhere; the order is the reranker's.
        """
        if not chunks:
            return []
        scores = list(self.model.rerank(query, [c.chunk.text for c in chunks]))
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        return [chunks[i] for i in order[:k]]


def get_reranker(settings=None) -> FastEmbedReranker | None:
    """The configured reranker, or None when RERANK_MODEL is empty (reranking disabled)."""
    from pool_assistant.config import get_settings

    settings = settings or get_settings()
    if not settings.rerank_model:
        return None
    return FastEmbedReranker(settings.rerank_model, cache_dir=settings.embedding_cache_dir)
