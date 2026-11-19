"""Ingestion pipeline: one PDF in, indexed chunks out.

    load_pdf ─► enrich_with_tables ─► enrich_with_vision ─► chunk_pages ─► index_chunks
    (text +     (deterministic        (figures / scans;      (boilerplate,   (embeddings
     signals)    vector tables)        cache + validation)    split, lang)    + Qdrant)

Kept out of the CLI so the API, scripts and tests can reuse it.
"""

import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from pool_assistant.config import Settings
from pool_assistant.embeddings import Embedder
from pool_assistant.ingestion.chunker import chunk_pages
from pool_assistant.ingestion.indexer import index_chunks
from pool_assistant.ingestion.loader import load_pdf
from pool_assistant.ingestion.tables import enrich_with_tables
from pool_assistant.ingestion.vision import enrich_with_vision
from pool_assistant.vectorstore import QdrantStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestReport:
    """What happened to one document (printed by the CLI, useful for monitoring)."""

    source: str
    doc_id: str
    pages: int
    parsers: dict[str, int] = field(default_factory=dict)    # text / table / vision pages
    chunks: int = 0
    languages: dict[str, int] = field(default_factory=dict)  # chunks per language
    seconds: float = 0.0


def make_doc_id(path: Path) -> str:
    """Stable, readable id from the file name: 'Victoria Plus_v2.pdf' -> 'victoria-plus-v2'."""
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    if not slug:
        raise ValueError(f"Cannot derive a doc_id from {path.name!r}; pass one explicitly")
    return slug


def ingest_pdf(
    pdf_path: Path,
    *,
    settings: Settings,
    embedder: Embedder,
    store: QdrantStore,
    vision_llm: BaseChatModel | None = None,
    use_vision: bool = True,
    explicit_lang: str | None = None,
    doc_id: str | None = None,
) -> IngestReport:
    """Run the full ingestion chain for one PDF and (re)index it idempotently."""
    start = time.perf_counter()
    doc_id = doc_id or make_doc_id(pdf_path)

    pages = load_pdf(pdf_path)
    pages = enrich_with_tables(pages, pdf_path)
    if use_vision:
        pages = enrich_with_vision(
            pages, pdf_path, vision_llm,
            model_name=settings.vision_model,
            cache_dir=settings.vision_cache_dir,
            render_scale=settings.vision_render_scale,
            min_text_coverage=settings.vision_min_text_coverage,
        )
    chunks = chunk_pages(pages, doc_id=doc_id, explicit_lang=explicit_lang)
    if not chunks:
        raise ValueError(f"No text could be extracted from {pdf_path.name}")
    index_chunks(chunks, embedder, store)

    return IngestReport(
        source=pdf_path.name,
        doc_id=doc_id,
        pages=len(pages),
        parsers=dict(Counter(p.parser for p in pages)),
        chunks=len(chunks),
        languages=dict(Counter(c.lang or "undetermined" for c in chunks)),
        seconds=round(time.perf_counter() - start, 1),
    )
