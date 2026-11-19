"""Troubleshooting matrices of the sample manual, in all 9 languages, against ground truth.

Ground truth was read manually from the page image of p.13 (identical matrix in every
language; the Arabic table is mirrored right-to-left).
"""

import pytest

from pool_assistant.config import get_settings
from pool_assistant.ingestion.loader import PageText, load_pdf
from pool_assistant.ingestion.tables import MARK, enrich_with_tables

MANUAL = get_settings().data_dir / "manuals" / "victoria_plus_silent_manual.pdf"
TROUBLESHOOTING_PAGES = {13: "en", 23: "fr", 33: "es", 43: "it", 53: "de",
                         63: "pt", 73: "el", 83: "ru", 92: "ar"}
# Marked symptom columns per cause row, in table order.
MATRIX_TRUTH = [{1, 2}, {1}, {1, 2}, {1, 2, 4}, {2}, {2}, {3}, {5}, {6}, {6}, {6}]


@pytest.fixture(scope="module")
def pages() -> dict[int, PageText]:
    if not MANUAL.exists():
        pytest.skip(f"Manual not available at {MANUAL}")
    return {p.page_number: p for p in enrich_with_tables(load_pdf(MANUAL), MANUAL)}


def _marked_columns(text: str) -> list[set[int]]:
    rows = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in text.splitlines()
        if line.startswith("| ")
    ]
    header, body = rows[0], rows[2:]
    return [{int(header[i]) for i, cell in enumerate(row) if cell == MARK} for row in body]


@pytest.mark.parametrize("page_number", sorted(TROUBLESHOOTING_PAGES))
def test_troubleshooting_matrix_is_exact(pages: dict[int, PageText], page_number: int) -> None:
    page = pages[page_number]

    assert page.parser == "table"
    assert _marked_columns(page.text) == MATRIX_TRUTH


def test_relations_use_legend_labels(pages: dict[int, PageText]) -> None:
    text = pages[13].text

    assert "- 3. The pump makes a noise → Pump incorrectly secured (Secure the pump correctly)" in (
        text.splitlines()
    )


def test_spanish_relations_are_in_spanish(pages: dict[int, PageText]) -> None:
    assert "- 3. La bomba hace ruido → Fijación incorrecta de la bomba" in pages[33].text


def test_arabic_table_text_is_in_logical_order(pages: dict[int, PageText]) -> None:
    """RTL cells rebuilt glyph by glyph: ligatures and brackets intact."""
    assert "المحرك (الموتور) مسدود" in pages[92].text


def test_arabic_multiline_cells_are_complete(pages: dict[int, PageText]) -> None:
    """Regression: the 2nd line of RTL cells was dropped (glyphs touching the cell border)."""
    text = pages[92].text

    assert "المحرك (الموتور) يدور في الاتجاه الخاطئ" in text
    assert "فك المحرك واتصل بالخدمة الفنية" in text
    assert "(الفلتر) (افحص" not in text  # no glyphs leaking from the neighbouring row


def test_replacement_period_table_is_structured(pages: dict[int, PageText]) -> None:
    assert "| Mechanical seal | 1 year |" in pages[12].text
