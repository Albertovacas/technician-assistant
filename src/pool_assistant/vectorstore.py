"""Qdrant vector store: one collection, one point per chunk, chunk metadata as payload.

Local embedded mode (`QdrantClient(path=...)`) needs no server and uses the same API as
Qdrant Server/Cloud, so production is a config change (QDRANT_URL).
Caveat: embedded mode locks its folder, so only one process can open it at a time.
"""

import logging
import uuid
import warnings
from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from pool_assistant.config import Settings, get_settings
from pool_assistant.schemas import Chunk, ScoredChunk

logger = logging.getLogger(__name__)

# Fixed namespace: the same chunk_id always maps to the same Qdrant point id.
_POINT_NAMESPACE = uuid.UUID("6f1d2c3a-8b7e-4f0a-9c5d-2e4b6a8c0d1f")
_UPSERT_BATCH = 128
# Payload fields used in filters. Qdrant server/Cloud (strict mode) refuses to filter on
# fields without an index; embedded mode ignores indexes.
_FILTERABLE_FIELDS = ("doc_id", "lang")


class VectorStoreError(RuntimeError):
    pass


def point_id(chunk_id: str) -> str:
    """Deterministic UUID for a chunk (Qdrant ids must be UUIDs or unsigned ints)."""
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def _filter(**conditions: str | None) -> models.Filter | None:
    must = [
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in conditions.items()
        if value is not None
    ]
    return models.Filter(must=must) if must else None


def _search_filter(lang: str | None, doc_id: str | None,
                   include_undetermined: bool) -> models.Filter | None:
    """doc_id must match; lang must match, or (optionally) be null (multilingual chunks)."""
    if lang is None or not include_undetermined:
        return _filter(lang=lang, doc_id=doc_id)
    must = [] if doc_id is None else [
        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
    ]
    should = [
        models.FieldCondition(key="lang", match=models.MatchValue(value=lang)),
        models.IsNullCondition(is_null=models.PayloadField(key="lang")),
    ]
    return models.Filter(must=must or None, should=should)


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int) -> None:
        self.client = client
        self.collection = collection
        self.dim = dim

    @classmethod
    def from_settings(cls, dim: int, settings: Settings | None = None) -> "QdrantStore":
        settings = settings or get_settings()
        if settings.qdrant_url:
            key = settings.qdrant_api_key
            client = QdrantClient(
                url=settings.qdrant_url, api_key=key.get_secret_value() if key else None
            )
        else:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            client = QdrantClient(path=str(settings.qdrant_path))
        return cls(client, settings.qdrant_collection, dim)

    def check_compatible(self) -> None:
        """Fail loudly if the existing collection was built with another embedding model.

        Called before writing (ensure_collection) and before querying (`ask`): without it a
        query with the wrong vector size surfaces as a low-level shape error.
        """
        if not self.client.collection_exists(self.collection):
            return
        existing_dim = self.client.get_collection(self.collection).config.params.vectors.size
        if existing_dim != self.dim:
            raise VectorStoreError(
                f"Collection '{self.collection}' has vectors of size {existing_dim}, but the "
                f"embedding model produces {self.dim}. The embedding model changed: "
                "re-index with a new collection name or delete the index."
            )

    def ensure_collection(self) -> None:
        """Create the collection if missing; fail loudly if the vector size changed."""
        if self.client.collection_exists(self.collection):
            self.check_compatible()
        else:
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
            )
            logger.info("Created collection %s (dim=%d, cosine)", self.collection, self.dim)
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        """Keyword indexes for filtered fields (idempotent; no-op in embedded mode)."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Payload indexes have no effect.*")
            for field in _FILTERABLE_FIELDS:
                self.client.create_payload_index(
                    self.collection, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
                )

    def replace_document(self, doc_id: str, chunks: Sequence[Chunk],
                         vectors: Sequence[Sequence[float]]) -> int:
        """Idempotently (re)index a document: drop its old points, upsert the new ones.

        Deleting first also removes stale chunks when a new version has fewer chunks.
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if any(c.doc_id != doc_id for c in chunks):
            raise ValueError(f"all chunks must belong to doc_id={doc_id!r}")
        self.ensure_collection()

        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=_filter(doc_id=doc_id)),
        )
        points = [
            models.PointStruct(id=point_id(c.chunk_id), vector=list(v), payload=c.model_dump())
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), _UPSERT_BATCH):
            self.client.upsert(self.collection, points=points[start:start + _UPSERT_BATCH])
        return len(points)

    def search(self, vector: Sequence[float], *, k: int = 5, lang: str | None = None,
               doc_id: str | None = None,
               include_undetermined: bool = False) -> list[ScoredChunk]:
        """Top-k chunks by cosine similarity, optionally filtered by language / document.

        `include_undetermined=True` also returns chunks without a language (multilingual
        figure pages, cover...), which a strict language filter would always hide.
        """
        if not self.client.collection_exists(self.collection):
            return []
        response = self.client.query_points(
            self.collection,
            query=list(vector),
            query_filter=_search_filter(lang, doc_id, include_undetermined),
            limit=k,
            with_payload=True,
        )
        return [
            ScoredChunk(chunk=Chunk.model_validate(p.payload), score=p.score)
            for p in response.points
        ]

    def count(self, *, lang: str | None = None, doc_id: str | None = None) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(
            self.collection, count_filter=_filter(lang=lang, doc_id=doc_id), exact=True
        ).count

    def close(self) -> None:
        self.client.close()
