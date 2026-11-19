"""Index chunks: embed them and write them to the vector store, one document at a time."""

import logging
import time
from collections import defaultdict
from collections.abc import Sequence

from pool_assistant.embeddings import Embedder
from pool_assistant.schemas import Chunk
from pool_assistant.vectorstore import QdrantStore

logger = logging.getLogger(__name__)


def index_chunks(chunks: Sequence[Chunk], embedder: Embedder, store: QdrantStore) -> int:
    """Embed and (re)index chunks, grouped by document. Returns the number of points written.

    Re-running with the same document replaces its previous chunks (no duplicates).
    """
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.doc_id].append(chunk)

    written = 0
    for doc_id, doc_chunks in by_doc.items():
        start = time.perf_counter()
        vectors = embedder.embed_documents([c.text for c in doc_chunks])
        written += store.replace_document(doc_id, doc_chunks, vectors)
        logger.info(
            "Indexed %s: %d chunks in %.1fs", doc_id, len(doc_chunks), time.perf_counter() - start
        )
    return written
