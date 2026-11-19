"""Language detection against the sample manual's known sections (ground truth
from the printed language markers, verified manually during task 1.1)."""

import pytest

from pool_assistant.config import get_settings
from pool_assistant.ingestion.language import detect_language
from pool_assistant.ingestion.loader import PageText, load_pdf

MANUAL = get_settings().data_dir / "manuals" / "victoria_plus_silent_manual.pdf"
SECTIONS = {
    "en": range(5, 14), "fr": range(15, 24), "es": range(25, 34), "it": range(35, 44),
    "de": range(45, 54), "pt": range(55, 64), "el": range(65, 74), "ru": range(75, 84),
    "ar": range(85, 93),
}
GROUND_TRUTH = {page: lang for lang, pages in SECTIONS.items() for page in pages}
MULTILINGUAL_PAGES = {1, 3, 4, 93, 94, 95, 96, 97, 100}


@pytest.fixture(scope="module")
def pages() -> dict[int, PageText]:
    if not MANUAL.exists():
        pytest.skip(f"Manual not available at {MANUAL}")
    return {p.page_number: p for p in load_pdf(MANUAL)}


def test_single_language_pages_accuracy(pages: dict[int, PageText]) -> None:
    hits = [detect_language(pages[n].text).lang == lang for n, lang in GROUND_TRUTH.items()]
    accuracy = sum(hits) / len(hits)

    assert accuracy >= 0.95, f"accuracy={accuracy:.2%}"


def test_multilingual_pages_are_undetermined(pages: dict[int, PageText]) -> None:
    detected = {n: detect_language(pages[n].text).lang for n in MULTILINGUAL_PAGES}

    assert all(lang is None for lang in detected.values()), detected
