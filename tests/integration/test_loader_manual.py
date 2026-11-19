"""Loader against the real Victoria Plus Silent sample manual (no network, no LLM).

Ground truth was established by manual inspection of the PDF:
- blank separator pages: 2, 14, 24, ..., 84, 98, 99
- drawn troubleshooting matrix: last page of each language section
- multilingual figure / exploded-view pages: 93-97
- text pages decorated with safety icons (must NOT be flagged): 8, 18, 86, 87
"""

import pytest

from pool_assistant.config import get_settings
from pool_assistant.ingestion.loader import PageText, load_pdf

MANUAL = get_settings().data_dir / "manuals" / "victoria_plus_silent_manual.pdf"

TROUBLESHOOTING_PAGES = {13, 23, 33, 43, 53, 63, 73, 83, 92}
FIGURE_PAGES = {93, 94, 95, 96, 97}
REGULAR_TEXT_PAGES = {5, 6, 7, 8, 9, 10, 11, 12, 18, 25, 45, 65, 75, 85, 86, 87}


@pytest.fixture(scope="module")
def pages() -> dict[int, PageText]:
    if not MANUAL.exists():
        pytest.skip(f"Manual not available at {MANUAL}")
    return {p.page_number: p for p in load_pdf(MANUAL)}


def test_blank_pages_are_dropped(pages: dict[int, PageText]) -> None:
    for blank in (14, 24, 34, 98, 99):
        assert blank not in pages


def test_page_numbers_match_printed_numbers(pages: dict[int, PageText]) -> None:
    assert "13" in pages[13].text.splitlines()[:3]


@pytest.mark.parametrize("page_number", sorted(TROUBLESHOOTING_PAGES | FIGURE_PAGES))
def test_tables_and_figures_are_flagged_for_vision(
    pages: dict[int, PageText], page_number: int
) -> None:
    assert pages[page_number].needs_vision


@pytest.mark.parametrize("page_number", sorted(REGULAR_TEXT_PAGES))
def test_regular_text_pages_are_trusted(pages: dict[int, PageText], page_number: int) -> None:
    assert not pages[page_number].needs_vision


def test_vision_is_needed_for_a_small_fraction_only(pages: dict[int, PageText]) -> None:
    flagged = sum(p.needs_vision for p in pages.values())
    assert flagged <= 0.25 * len(pages)  # keeps vision cost bounded
