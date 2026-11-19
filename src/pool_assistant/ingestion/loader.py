"""Generic PDF loader: text layer per page plus layout-independent quality signals.

The loader makes no assumption about the manual's template or language. For every
page it extracts the text layer and measures cheap structural signals that tell the
rest of the pipeline whether that text can be trusted:

* `needs_ocr`     - (almost) no text but graphical content: scanned page or drawing.
* `looks_complex` - has text, but the meaning is (partly) graphical, so the text
                    layer is probably incomplete. Detected from the vector drawing
                    operators in the page content stream:
                      - many stroked lines  -> drawn table / grid / matrix
                      - very many path ops  -> technical diagram / exploded view
                      - raster images       -> photo, scanned figure
                    Icons and logos are drawn as filled curves and trigger neither.

Pages flagged by either signal are candidates for the vision parser (task 1.4).
Language detection is a separate concern (see `language.py`).
"""

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pypdf import PageObject, PdfReader
from pypdf.generic import ContentStream

logger = logging.getLogger(__name__)

# PDF content-stream operators (ISO 32000-1, section 8.5).
_PATH_CONSTRUCTION_OPS = (b"m", b"l", b"c", b"v", b"y", b"re")
_STROKE_OPS = (b"S", b"s", b"B", b"B*", b"b", b"b*")

# Each sits near the geometric middle of the range that yields the correct pages:
#   strokes: text pages <=35, drawn troubleshooting tables ~212  -> 100
#   paths:   text/back-cover pages <=1,650, exploded views >=4,880 -> 3,000
DEFAULT_MIN_TEXT_CHARS = 50
DEFAULT_TABLE_STROKE_OPS = 100
DEFAULT_DIAGRAM_PATH_OPS = 3000


@dataclass(frozen=True)
class PageText:
    source: str          # manual file name, used in citations
    page_number: int     # 1-based physical page index
    text: str            # raw text layer (may be empty)
    char_count: int
    image_count: int     # embedded raster images
    stroke_ops: int      # stroked vector paths (table grids, line art)
    path_ops: int        # vector path construction operators
    needs_ocr: bool
    looks_complex: bool
    # Who produced `text`: pypdf text layer, + reconstructed vector tables, or vision LLM.
    parser: Literal["text", "table", "vision"] = "text"

    @property
    def needs_vision(self) -> bool:
        return self.needs_ocr or self.looks_complex


def _count_vector_ops(page: PageObject, reader: PdfReader) -> tuple[int, int]:
    """Return (stroke_ops, path_ops) from the page content stream."""
    contents = page.get_contents()
    if contents is None:
        return 0, 0
    ops = Counter(op for _, op in ContentStream(contents, reader).operations)
    return sum(ops[o] for o in _STROKE_OPS), sum(ops[o] for o in _PATH_CONSTRUCTION_OPS)


def _count_images(page: PageObject) -> int:
    try:
        return len(page.images)
    except Exception:  # malformed image streams must not break ingestion
        logger.warning("Could not inspect images on page", exc_info=True)
        return 0


def load_pdf(
    path: Path,
    *,
    min_text_chars: int = DEFAULT_MIN_TEXT_CHARS,
    table_stroke_ops: int = DEFAULT_TABLE_STROKE_OPS,
    diagram_path_ops: int = DEFAULT_DIAGRAM_PATH_OPS,
) -> list[PageText]:
    """Extract every non-empty page of a PDF with its quality signals.

    Pages with no text and no graphical content (blank separators) are dropped.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Manual not found: {path}")

    reader = PdfReader(path)
    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        char_count = len(text)
        image_count = _count_images(page)
        stroke_ops, path_ops = _count_vector_ops(page, reader)

        has_graphics = image_count > 0 or path_ops > 0
        if char_count == 0 and not has_graphics:
            continue  # blank page

        has_text = char_count >= min_text_chars
        is_graphical = (
            image_count > 0 or stroke_ops >= table_stroke_ops or path_ops >= diagram_path_ops
        )
        pages.append(
            PageText(
                source=path.name,
                page_number=index,
                text=text,
                char_count=char_count,
                image_count=image_count,
                stroke_ops=stroke_ops,
                path_ops=path_ops,
                needs_ocr=not has_text and has_graphics,
                looks_complex=has_text and is_graphical,
            )
        )

    logger.info(
        "Loaded %s: %d/%d non-blank pages (needs_ocr=%d, looks_complex=%d)",
        path.name,
        len(pages),
        len(reader.pages),
        sum(p.needs_ocr for p in pages),
        sum(p.looks_complex for p in pages),
    )
    return pages
