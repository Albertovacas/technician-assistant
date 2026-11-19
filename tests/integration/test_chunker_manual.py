"""Chunker against the real sample manual (no network, no LLM)."""

import pytest

from pool_assistant.config import get_settings
from pool_assistant.ingestion.chunker import chunk_pages, find_boilerplate
from pool_assistant.ingestion.loader import load_pdf
from pool_assistant.schemas import Chunk

MANUAL = get_settings().data_dir / "manuals" / "victoria_plus_silent_manual.pdf"
SECTIONS = {
    "en": range(5, 14), "fr": range(15, 24), "es": range(25, 34), "it": range(35, 44),
    "de": range(45, 54), "pt": range(55, 64), "el": range(65, 74), "ru": range(75, 84),
    "ar": range(85, 93),
}
TRUTH = {page: lang for lang, pages in SECTIONS.items() for page in pages}


@pytest.fixture(scope="module")
def pages():
    if not MANUAL.exists():
        pytest.skip(f"Manual not available at {MANUAL}")
    return load_pdf(MANUAL)


@pytest.fixture(scope="module")
def chunks(pages) -> list[Chunk]:
    return chunk_pages(pages, doc_id="victoria-plus-silent")


def test_boilerplate_is_exactly_running_titles_and_language_markers(pages) -> None:
    boilerplate = find_boilerplate(pages)

    assert len(boilerplate) == 18  # 9 languages x (running title + language marker)
    assert "Installation and general maintenance manual - POOL PUMPS" in boilerplate
    assert "ESPAÑOL" in boilerplate


def test_no_chunk_contains_running_header(chunks: list[Chunk]) -> None:
    assert not any("Installation and general maintenance manual" in c.text for c in chunks)


def test_chunk_sizes_are_bounded(chunks: list[Chunk]) -> None:
    sizes = sorted(len(c.text) for c in chunks)
    assert sizes[-1] <= 1200
    assert sizes[len(sizes) // 2] >= 700  # median close to the target size


def test_language_of_single_language_chunks(chunks: list[Chunk]) -> None:
    scored = [c for c in chunks if c.page_number in TRUTH]
    accuracy = sum(c.lang == TRUTH[c.page_number] for c in scored) / len(scored)

    assert accuracy >= 0.95, f"accuracy={accuracy:.2%}"


def test_troubleshooting_table_is_retrievable_text(chunks: list[Chunk]) -> None:
    page_13 = " ".join(c.text for c in chunks if c.page_number == 13)

    assert "TROUBLESHOOTING" in page_13
    assert "Pre-filter blocked" in page_13
