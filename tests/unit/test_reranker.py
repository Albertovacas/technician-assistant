"""Reranking step of the retriever, with fake cross-encoders (no model download)."""

from qdrant_client import QdrantClient

from pool_assistant.config import Settings
from pool_assistant.retrieval.reranker import FastEmbedReranker, get_reranker
from pool_assistant.retrieval.retriever import Retriever
from pool_assistant.schemas import ScoredChunk
from pool_assistant.vectorstore import QdrantStore

from .test_vectorstore import make_chunk

# Ten EN chunks about the pump; only page 5 holds the fact (like "35 ºC" ranked 20th).
CHUNKS = [make_chunk("m", p, 0, f"pump safety warning number {p}", lang="en")
          for p in range(6, 15)]
FACT = make_chunk("m", 5, 0, "General warnings. Maximum water temperature 35 ºC.", lang="en")


class KeywordReranker:
    """Scores 1.0 the chunks containing `keyword`, 0.0 the rest; records its calls."""

    def __init__(self, keyword: str) -> None:
        self.keyword = keyword
        self.calls: list[tuple[str, int, int]] = []  # (query, candidates, k)

    def rerank(self, query, chunks, k):
        self.calls.append((query, len(chunks), k))
        return sorted(chunks, key=lambda c: self.keyword in c.chunk.text, reverse=True)[:k]


class BrokenReranker:
    def rerank(self, query, chunks, k):
        raise RuntimeError("ONNX runtime error")


def build(fake_embedder, reranker, candidates: int = 20) -> Retriever:
    store = QdrantStore(QdrantClient(":memory:"), "t", fake_embedder.dim)
    chunks = [*CHUNKS, FACT]
    store.replace_document("m", chunks, fake_embedder.embed_documents([c.text for c in chunks]))
    return Retriever(fake_embedder, store, k=3, reranker=reranker, candidates=candidates)


def test_reranker_sees_the_candidates_and_returns_k(fake_embedder) -> None:
    reranker = KeywordReranker("35 ºC")

    result = build(fake_embedder, reranker).retrieve("pump safety warning", lang="en")

    assert reranker.calls == [("pump safety warning", 10, 3)]  # all 10 candidates, keep 3
    assert result.chunks[0].chunk.page_number == 5            # promoted to the top
    assert len(result.chunks) == 3 and result.reranked


def test_candidates_bound_what_the_reranker_can_promote(fake_embedder) -> None:
    reranker = KeywordReranker("35 ºC")

    build(fake_embedder, reranker, candidates=4).retrieve("pump safety warning", lang="en")

    assert reranker.calls[0][1] == 4  # only 4 candidates searched


def test_reranker_failure_falls_back_to_vector_order(fake_embedder) -> None:
    plain = build(fake_embedder, None).retrieve("pump safety warning", lang="en")

    result = build(fake_embedder, BrokenReranker()).retrieve("pump safety warning", lang="en")

    assert result.chunks == plain.chunks  # same top-k as without reranker
    assert not result.reranked


def test_without_reranker_nothing_changes(fake_embedder) -> None:
    result = build(fake_embedder, None).retrieve("pump safety warning", lang="en")

    assert len(result.chunks) == 3 and not result.reranked


def test_fastembed_reranker_orders_by_cross_encoder_score_and_keeps_cosine() -> None:
    class FakeCrossEncoder:
        def rerank(self, query, documents):
            return [float("35" in d) for d in documents]

    chunks = [ScoredChunk(chunk=CHUNKS[0], score=0.9), ScoredChunk(chunk=FACT, score=0.4)]
    reranker = FastEmbedReranker("fake", model=FakeCrossEncoder())

    (best,) = reranker.rerank("max water temperature", chunks, k=1)

    assert best.chunk == FACT and best.score == 0.4  # reranker order, cosine score kept
    assert reranker.rerank("q", [], k=5) == []


def test_empty_rerank_model_disables_reranking() -> None:
    assert get_reranker(Settings(_env_file=None, rerank_model="")) is None
    reranker = get_reranker(Settings(_env_file=None))
    assert reranker is not None and reranker._model is None  # lazy: nothing loaded yet
