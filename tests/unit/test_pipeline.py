"""Ingestion pipeline on synthetic PDFs, with a fake embedder and in-memory Qdrant."""

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from pool_assistant.config import Settings
from pool_assistant.ingestion.pipeline import ingest_pdf, make_doc_id
from pool_assistant.vectorstore import QdrantStore


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, vision_cache_dir=tmp_path / "vision_cache")


@pytest.fixture
def store(fake_embedder) -> QdrantStore:
    return QdrantStore(QdrantClient(":memory:"), "manuals", fake_embedder.dim)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("victoria_plus_silent_manual.pdf", "victoria-plus-silent-manual"),
        ("Victoria Plus (v2) – ES.pdf", "victoria-plus-v2-es"),
    ],
)
def test_make_doc_id(name: str, expected: str) -> None:
    assert make_doc_id(Path(name)) == expected


def test_make_doc_id_rejects_unusable_names() -> None:
    with pytest.raises(ValueError):
        make_doc_id(Path("¿¿.pdf"))


def test_ingest_reports_and_indexes(make_pdf, settings, fake_embedder, store) -> None:
    pdf = make_pdf("Heat Pump Manual.pdf", ["text", "matrix", "text"])

    report = ingest_pdf(pdf, settings=settings, embedder=fake_embedder, store=store)

    assert report.doc_id == "heat-pump-manual"
    assert report.pages == 3
    assert report.parsers == {"text": 2, "table": 1}
    assert report.chunks == store.count(doc_id="heat-pump-manual") > 0
    assert report.languages == {"en": report.chunks}


def test_reingesting_is_idempotent(make_pdf, settings, fake_embedder, store) -> None:
    pdf = make_pdf("manual.pdf", ["text", "text"])

    first = ingest_pdf(pdf, settings=settings, embedder=fake_embedder, store=store)
    ingest_pdf(pdf, settings=settings, embedder=fake_embedder, store=store)

    assert store.count() == first.chunks


def test_explicit_language_and_doc_id(make_pdf, settings, fake_embedder, store) -> None:
    pdf = make_pdf("manual.pdf", ["text"])

    report = ingest_pdf(pdf, settings=settings, embedder=fake_embedder, store=store,
                        explicit_lang="es", doc_id="custom-id")

    assert report.doc_id == "custom-id"
    assert report.languages == {"es": report.chunks}


def test_scanned_pdf_without_llm_or_cache_fails_clearly(make_pdf, settings, fake_embedder,
                                                        store) -> None:
    pdf = make_pdf("scan.pdf", ["scanned"])

    with pytest.raises(ValueError, match="No text could be extracted"):
        ingest_pdf(pdf, settings=settings, embedder=fake_embedder, store=store)
