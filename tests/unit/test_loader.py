"""Loader on synthetic PDFs of different formats (no dependency on the sample manual)."""

from pathlib import Path

import pytest

from pool_assistant.ingestion.loader import load_pdf


def test_plain_text_pdf_is_trusted(make_pdf) -> None:
    pages = load_pdf(make_pdf("text.pdf", ["text", "text"]))

    assert [p.page_number for p in pages] == [1, 2]
    assert all(p.char_count > 100 and not p.needs_vision for p in pages)
    assert pages[0].source == "text.pdf"


def test_blank_pages_are_dropped_but_numbering_is_preserved(make_pdf) -> None:
    pages = load_pdf(make_pdf("blank.pdf", ["text", "blank", "text"]))

    assert [p.page_number for p in pages] == [1, 3]


def test_scanned_page_without_text_layer_needs_ocr(make_pdf) -> None:
    (page,) = load_pdf(make_pdf("scan.pdf", ["scanned"]))

    assert page.char_count == 0
    assert page.image_count == 1
    assert page.needs_ocr and not page.looks_complex


def test_drawn_table_page_looks_complex(make_pdf) -> None:
    (page,) = load_pdf(make_pdf("table.pdf", ["table"]))

    assert page.char_count > 0
    assert page.stroke_ops >= 100
    assert page.looks_complex and not page.needs_ocr


def test_text_with_filled_icons_is_not_complex(make_pdf) -> None:
    """Regression: safety-warning icons on text pages caused false positives."""
    (page,) = load_pdf(make_pdf("icons.pdf", ["icons"]))

    assert page.path_ops > 0
    assert not page.looks_complex


def test_thresholds_are_tunable(make_pdf) -> None:
    path = make_pdf("table.pdf", ["table"])

    (page,) = load_pdf(path, table_stroke_ops=100_000)

    assert not page.looks_complex


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_pdf(Path("does/not/exist.pdf"))
