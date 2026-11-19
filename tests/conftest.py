"""Shared fixtures: synthetic PDFs in different formats, generated on the fly."""

import hashlib
import math
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from fpdf import FPDF
from PIL import Image

PdfFactory = Callable[[str, list[str]], Path]


class FakeEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar vectors.

    Lets indexer/vector-store tests run offline in milliseconds, with meaningful ranking.
    """

    dim = 64

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()

LOREM = (
    "Check the condition of the seals before start-up. Disconnect the unit from the "
    "power supply before any maintenance operation. "
) * 4


def _add_text_page(pdf: FPDF, text: str = LOREM) -> None:
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, text)


def _add_scanned_page(pdf: FPDF, tmp_path: Path) -> None:
    """A page that is a single raster image and has no text layer."""
    image_path = tmp_path / "scan.png"
    Image.new("RGB", (400, 560), "white").save(image_path)
    pdf.add_page()
    pdf.image(str(image_path), x=10, y=10, w=180)


def _add_drawn_table_page(pdf: FPDF, rows: int = 25, cols: int = 8) -> None:
    """Text plus a grid of drawn cells with filled marks (like a troubleshooting matrix)."""
    _add_text_page(pdf, "TROUBLESHOOTING\n" + LOREM)
    for r in range(rows):
        for c in range(cols):
            x, y = 10 + c * 22, 80 + r * 7
            pdf.rect(x, y, 22, 7)
            if (r + c) % 3 == 0:
                pdf.rect(x + 8, y + 2, 3, 3, style="F")


def _add_text_with_icons_page(pdf: FPDF, icons: int = 60) -> None:
    """Regular text decorated with filled curved icons (like safety warning symbols)."""
    _add_text_page(pdf)
    for i in range(icons):
        pdf.circle(x=15 + (i % 10) * 18, y=120 + (i // 10) * 18, radius=4, style="F")


MATRIX_ROWS = [  # (cause, solution, marked symptom columns)
    ("Air entering the suction pipe", "Check the seals", {1, 2}),
    ("Pre-filter blocked", "Clean the filter", {2}),
    ("Motor blocked", "Contact the technical service", {3}),
]


def _add_matrix_table_page(pdf: FPDF) -> None:
    """Ruled table relating causes (rows) to symptoms (columns 1-3) with filled dots."""
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    for legend_line in ("1. The pump is not primed  2. Small flow of water",
                        "3. The motor is making a noise"):
        pdf.cell(0, 6, legend_line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    widths = [10, 10, 10, 70, 70]
    for text, width in zip(["1", "2", "3", "CAUSES", "SOLUTIONS"], widths, strict=True):
        pdf.cell(width, 8, text, border=1)
    pdf.ln()
    for cause, solution, marks in MATRIX_ROWS:
        y = pdf.get_y()
        x = pdf.l_margin
        for col in (1, 2, 3):
            pdf.cell(10, 8, "", border=1)
            if col in marks:
                pdf.circle(x=x + (col - 1) * 10 + 5, y=y + 4, radius=1.5, style="F")
        pdf.cell(70, 8, cause, border=1)
        pdf.cell(70, 8, solution, border=1)
        pdf.ln()


@pytest.fixture
def make_pdf(tmp_path: Path) -> PdfFactory:
    """Build a PDF from page kinds: 'text', 'blank', 'scanned', 'table', 'icons', 'matrix'."""

    def factory(name: str, kinds: list[str]) -> Path:
        pdf = FPDF()
        for kind in kinds:
            if kind == "text":
                _add_text_page(pdf)
            elif kind == "blank":
                pdf.add_page()
            elif kind == "scanned":
                _add_scanned_page(pdf, tmp_path)
            elif kind == "table":
                _add_drawn_table_page(pdf)
            elif kind == "icons":
                _add_text_with_icons_page(pdf)
            elif kind == "matrix":
                _add_matrix_table_page(pdf)
            else:
                raise ValueError(f"Unknown page kind: {kind}")
        path = tmp_path / name
        pdf.output(str(path))
        return path

    return factory
