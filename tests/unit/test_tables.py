"""Deterministic table reconstruction on synthetic PDFs (no dependency on the sample manual)."""

import pdfplumber

from pool_assistant.ingestion.loader import load_pdf
from pool_assistant.ingestion.tables import (
    MARK,
    enrich_with_tables,
    extract_tables,
    is_rtl,
    parse_legend,
    rtl_text_from_glyphs,
)


def test_parse_legend_handles_two_items_per_line() -> None:
    text = "7. TROUBLESHOOTING\n1. The pump is not primed 4. The pump will not start\n2. Low flow"

    legend = parse_legend(text)

    assert legend[1] == "1. The pump is not primed"
    assert legend[4] == "4. The pump will not start"
    assert legend[2] == "2. Low flow"


def _glyphs(texts: list[str]) -> list[dict]:
    """Fake pdfplumber chars laid out left-to-right (visual order) on one line."""
    return [{"text": t, "x0": float(i * 5), "top": 100.0} for i, t in enumerate(texts)]


def test_rtl_glyphs_are_rebuilt_in_logical_order_keeping_ligatures() -> None:
    # Real glyph sequence of "المحرك (الموتور) مسدود" on p.92; 'لم' is ONE ligature glyph.
    visual = ["د", "و", "د", "س", "م", " ", ")", "ر", "و", "ت", "و", "لم", "ا", "(", " ",
              "ك", "ر", "ح", "لم", "ا"]

    assert rtl_text_from_glyphs(_glyphs(visual)) == "المحرك (الموتور) مسدود"


def test_is_rtl() -> None:
    assert is_rtl("المحرك مسدود")
    assert not is_rtl("Motor blocked")
    assert not is_rtl("1 2 3")


def test_matrix_marks_and_relations_are_reconstructed(make_pdf) -> None:
    path = make_pdf("matrix.pdf", ["matrix"])
    page_text = load_pdf(path)[0].text

    with pdfplumber.open(path) as pdf:
        (table,) = extract_tables(pdf.pages[0], page_text)

    lines = table.markdown.splitlines()
    assert lines[0] == "| 1 | 2 | 3 | CAUSES | SOLUTIONS |"
    assert f"| {MARK} | {MARK} |  | Air entering the suction pipe | Check the seals |" in lines
    assert table.n_marks == 4
    assert (
        "- 2. Small flow of water → Air entering the suction pipe (Check the seals) "
        "| Pre-filter blocked (Clean the filter)"
    ) in lines
    expected = "- 3. The motor is making a noise → Motor blocked (Contact the technical service)"
    assert expected in lines


def test_grid_without_text_is_not_a_table(make_pdf) -> None:
    """Figure frames and decorative grids must not be emitted as tables."""
    path = make_pdf("grid.pdf", ["table"])  # drawn grid + marks, but no text in cells

    with pdfplumber.open(path) as pdf:
        assert extract_tables(pdf.pages[0], load_pdf(path)[0].text) == []


def test_enrich_keeps_original_text_and_appends_table(make_pdf) -> None:
    path = make_pdf("manual.pdf", ["text", "matrix"])
    pages = load_pdf(path)

    result = enrich_with_tables(pages, path)

    assert result[0] is pages[0]  # no table on the plain text page
    assert result[1].parser == "table"
    assert result[1].text.startswith(pages[1].text)
    assert "| CAUSES |" in result[1].text


def test_scanned_pages_are_left_for_vision(make_pdf) -> None:
    path = make_pdf("scan.pdf", ["scanned"])
    pages = load_pdf(path)

    assert enrich_with_tables(pages, path) == pages
