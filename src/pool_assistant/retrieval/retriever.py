"""Retriever with a *soft* language filter.

Why a filter at all: multilingual manuals contain the same content in N languages, so an
unfiltered top-k fills up with translations of one passage (notebook 05, section 9).
Why *soft*: a strict filter returns nothing when the manual does not exist in the
technician's language - the most common real case (manual only in English).

    1. question language: given (Triage agent) -> detected (confidence >= 0.5) -> None
    2. chunks in that language?   -> search it                  (same_language)
    3. chunks in fallback (en)?   -> search fallback            (fallback_language)
    4. otherwise                  -> search without filter      (unfiltered)
    Language-filtered searches also include multilingual chunks (lang = null).

The answer is always written in the question's language by the LLM, whatever language
the evidence is in.

Optional reranking (reranker.py): search `candidates` chunks (default 20) with the same
filter, reorder them with a cross-encoder and keep the best k. If the reranker fails, the
cosine top-k is used (fail-open: same behaviour as without reranker).

Limitation: availability is checked on the whole index. With several manuals, one may
exist in Spanish and another only in English; the per-document check belongs with
product detection (planned improvement).
"""

import logging

from pool_assistant.embeddings import Embedder
from pool_assistant.ingestion.language import detect_query_language
from pool_assistant.retrieval.reranker import Reranker
from pool_assistant.schemas import RetrievalResult
from pool_assistant.vectorstore import QdrantStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, embedder: Embedder, store: QdrantStore, *, k: int = 5,
                 fallback_lang: str = "en", reranker: Reranker | None = None,
                 candidates: int = 20) -> None:
        self.embedder = embedder
        self.store = store
        self.k = k
        self.fallback_lang = fallback_lang
        self.reranker = reranker
        self.candidates = candidates

    def _choose_language(self, query_lang: str | None) -> tuple[str | None, str]:
        if query_lang and self.store.count(lang=query_lang) > 0:
            return query_lang, "same_language"
        if self.store.count(lang=self.fallback_lang) > 0:
            return self.fallback_lang, "fallback_language"
        return None, "unfiltered"

    def retrieve(self, question: str, *, lang: str | None = None,
                 k: int | None = None) -> RetrievalResult:
        """Top-k evidence chunks for a question. `lang` overrides language detection."""
        k = k or self.k
        query_lang = lang or detect_query_language(question)
        search_lang, strategy = self._choose_language(query_lang)
        chunks = self.store.search(
            self.embedder.embed_query(question),
            k=max(self.candidates, k) if self.reranker else k,
            lang=search_lang,
            include_undetermined=search_lang is not None,
        )
        reranked = False
        if self.reranker is not None and len(chunks) > 1:
            try:
                chunks = self.reranker.rerank(question, chunks, k)
                reranked = True
            except Exception:  # fail-open: cosine order, as without reranker
                logger.warning("Reranking failed: using the vector-search order", exc_info=True)
        chunks = chunks[:k]
        logger.info(
            "Retrieved %d chunks (query_lang=%s, search_lang=%s, strategy=%s, reranked=%s, "
            "pages=%s)", len(chunks), query_lang, search_lang, strategy, reranked,
            [c.chunk.page_number for c in chunks],
        )
        return RetrievalResult(
            question=question, query_lang=query_lang, search_lang=search_lang,
            strategy=strategy, chunks=chunks, reranked=reranked,
        )
