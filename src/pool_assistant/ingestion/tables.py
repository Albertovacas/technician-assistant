"""Deterministic reconstruction of vector-drawn tables (digitally-born PDFs).

Why: the troubleshooting matrix relates causes and symptoms with filled dots. The text
layer loses that relation, and vision LLMs transcribe it unreliably (measured: 79/99
relations correct across the 9 languages of the sample manual, with columns shifted in
FR/IT/RU/EL/AR). In a digital PDF the grid lines and the dots are vector objects with
exact coordinates, so the table can be rebuilt exactly, for free, in milliseconds
(measured: 99/99 relations).

How:
1. pdfplumber finds tables from ruling lines ("lattice" strategy).
2. A body cell without text that contains a small filled shape is a mark ("●").
3. Columns made only of marks are "mark columns"; their header (e.g. "3") is linked to
   the page's numbered legend ("3. The pump makes a noise") when one exists.
4. Output: the Markdown table plus one explicit line per mark column
   ("<legend> → <row label> (<other cells>); ..."), which is what technicians ask about.

Columns are identified by their header text, not by position, so right-to-left tables
(Arabic, mirrored columns) work too. Scanned pages have no vector objects: they go to the
vision fallback instead.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

import pdfplumber

from pool_assistant.ingestion.loader import PageText

logger = logging.getLogger(__name__)

MARK = "●"
MAX_MARK_SIZE_PT = 15.0         # dots/ticks are small; bigger filled shapes are backgrounds
MIN_TABLE_ROWS = 3              # filters figure frames detected as 1-2 row "tables"
MIN_TABLE_COLS = 2
MIN_TEXT_CELLS = 4
_TABLE_SETTINGS = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
_LEGEND_ITEM = re.compile(r"(?<!\S)(\d{1,2})\s*[.)]\s*(.+?)(?=\s+\d{1,2}\s*[.)]\s|$)")


@dataclass(frozen=True)
class ExtractedTable:
    markdown: str
    n_rows: int
    n_marks: int
    rtl: bool


def _is_rtl_char(ch: str) -> bool:
    return unicodedata.bidirectional(ch) in {"R", "AL"}


def is_rtl(text: str) -> bool:
    """True when most letters belong to a right-to-left script (Arabic, Hebrew...)."""
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(map(_is_rtl_char, letters)) / len(letters) > 0.5


def rtl_text_from_glyphs(chars: list[dict], line_tolerance: float = 3.0) -> str:
    """Rebuild logical-order text from glyphs of an RTL cell.

    pdfplumber yields RTL text in visual order. Reversing the *string* breaks ligature
    glyphs whose text is already logical (e.g. one glyph 'لم'), so we sort *glyphs*
    right-to-left instead. Brackets come out correctly without mirroring.
    """
    lines: list[list[dict]] = []
    for char in sorted(chars, key=lambda c: c["top"]):
        if lines and abs(lines[-1][0]["top"] - char["top"]) <= line_tolerance:
            lines[-1].append(char)
        else:
            lines.append([char])
    text_lines = [
        "".join(c["text"] for c in sorted(line, key=lambda c: -c["x0"])) for line in lines
    ]
    return re.sub(r"\s+", " ", " ".join(text_lines)).strip()


def _glyphs_in_cell(page: "pdfplumber.page.Page", bbox) -> list[dict]:
    """Glyphs whose centre lies in the cell (same rule as pdfplumber's table.extract()).

    `within_bbox` would drop glyphs touching the cell border (losing whole lines) and
    `crop` would include glyphs of the neighbouring row that overlap slightly.
    """
    x0, top, x1, bottom = bbox
    return [
        c for c in page.chars
        if x0 <= (c["x0"] + c["x1"]) / 2 <= x1 and top <= (c["top"] + c["bottom"]) / 2 <= bottom
    ]


def _cell_text(page: "pdfplumber.page.Page", text: str | None, bbox) -> str:
    if not text:
        return ""
    if is_rtl(text) and bbox is not None:
        return rtl_text_from_glyphs(_glyphs_in_cell(page, bbox))
    return re.sub(r"\s+", " ", text).strip()


def _contains_mark(bbox: tuple[float, float, float, float], shapes: list[dict]) -> bool:
    x0, top, x1, bottom = bbox
    for shape in shapes:
        cx = (shape["x0"] + shape["x1"]) / 2
        cy = (shape["top"] + shape["bottom"]) / 2
        if x0 <= cx <= x1 and top <= cy <= bottom:
            return True
    return False


def parse_legend(page_text: str) -> dict[int, str]:
    """Numbered legend items such as '3. The pump makes a noise' -> {3: '3. The pump ...'}.

    Handles two items on the same line (two-column legends). First occurrence wins.
    """
    legend: dict[int, str] = {}
    for line in page_text.splitlines():
        for match in _LEGEND_ITEM.finditer(line.strip()):
            number = int(match.group(1))
            legend.setdefault(number, f"{number}. {match.group(2).strip()}")
    return legend


def _table_to_markdown(header: list[str], body: list[list[str]]) -> str:
    def row(cells: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", "/") for c in cells) + " |"

    return "\n".join([row(header), row(["---"] * len(header)), *map(row, body)])


def _relations(
    header: list[str], body: list[list[str]], mark_cols: list[int], legend: dict[int, str],
    reading_order: list[int],
) -> list[str]:
    label_cols = [c for c in reading_order if c not in mark_cols]
    lines = []
    for col in sorted(mark_cols, key=lambda c: int(header[c]) if header[c].isdigit() else c):
        column_label = legend.get(int(header[col]), header[col]) if header[col].isdigit() else (
            header[col]
        )
        items = []
        for cells in body:
            if cells[col] != MARK:
                continue
            main, *details = [cells[c] for c in label_cols if cells[c]]
            items.append(f"{main} ({'; '.join(details)})" if details else main)
        if items:
            lines.append(f"- {column_label} → " + " | ".join(items))
    return lines


def extract_tables(page: "pdfplumber.page.Page", page_text: str) -> list[ExtractedTable]:
    """Rebuild every ruled table of a pdfplumber page as Markdown (+ relations)."""
    shapes = [
        s for s in page.curves + page.rects
        if s.get("fill")
        and (s["x1"] - s["x0"]) <= MAX_MARK_SIZE_PT
        and (s["bottom"] - s["top"]) <= MAX_MARK_SIZE_PT
    ]
    legend = parse_legend(page_text)
    tables: list[ExtractedTable] = []

    for table in page.find_tables(_TABLE_SETTINGS):
        texts = table.extract()
        n_cols = max(len(r) for r in texts)
        text_cells = sum(1 for r in texts for c in r if c and c.strip())
        if len(texts) < MIN_TABLE_ROWS or n_cols < MIN_TABLE_COLS or text_cells < MIN_TEXT_CELLS:
            continue

        header = [
            _cell_text(page, text, bbox)
            for text, bbox in zip(texts[0], table.rows[0].cells, strict=True)
        ]
        body: list[list[str]] = []
        for row_texts, row in zip(texts[1:], table.rows[1:], strict=True):
            cells = []
            for text, bbox in zip(row_texts, row.cells, strict=True):
                cleaned = _cell_text(page, text, bbox)
                is_mark = not cleaned and bbox is not None and _contains_mark(bbox, shapes)
                cells.append(MARK if is_mark else cleaned)
            body.append(cells)

        mark_cols = [
            c for c in range(n_cols)
            if any(r[c] == MARK for r in body) and all(r[c] in ("", MARK) for r in body)
        ]
        rtl = is_rtl(" ".join(" ".join(r) for r in body))
        reading_order = list(range(n_cols))[::-1] if rtl else list(range(n_cols))

        parts = [_table_to_markdown(header, body)]
        if mark_cols:
            parts.append("\n".join(_relations(header, body, mark_cols, legend, reading_order)))
        tables.append(
            ExtractedTable(
                markdown="\n\n".join(parts),
                n_rows=len(body),
                n_marks=sum(r.count(MARK) for r in body),
                rtl=rtl,
            )
        )
    return tables


def enrich_with_tables(pages: list[PageText], pdf_path: Path) -> list[PageText]:
    """Append reconstructed tables to every page that has a text layer.

    Runs on all text pages, not only on pages flagged `looks_complex`: it is cheap
    (~10 ms/page) and small tables may stay below the loader's thresholds.
    The original text is kept (no information loss; correct RTL order from pypdf) and the
    Markdown tables are appended. Such pages are marked `parser="table"` so the vision
    fallback skips them.
    """
    candidates = {p.page_number for p in pages if not p.needs_ocr}
    if not candidates:
        return pages

    found: dict[int, list[ExtractedTable]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page in pages:
            if page.page_number in candidates:
                tables = extract_tables(pdf.pages[page.page_number - 1], page.text)
                if tables:
                    found[page.page_number] = tables

    enriched = []
    for page in pages:
        tables = found.get(page.page_number)
        if not tables:
            enriched.append(page)
            continue
        markdown = "\n\n".join(t.markdown for t in tables)
        enriched.append(replace(page, text=f"{page.text}\n\n{markdown}", parser="table"))

    logger.info(
        "Vector tables on %s: found on %d/%d text pages, marks=%d",
        pdf_path.name, len(found), len(candidates),
        sum(t.n_marks for ts in found.values() for t in ts),
    )
    return enriched
