"""Template-agnostic chunking: pages -> cleaned, size-bounded, language-tagged chunks.

Steps:
1. Boilerplate removal: lines that repeat in the header/footer zone of several pages
   (running titles, language markers) and bare page numbers. Learned from the document
   itself, so it works for any layout.
2. Recursive splitting per page (paragraph > line > sentence > word), so every chunk
   belongs to exactly one page and citations stay exact.
3. Tiny trailing pieces are merged into the previous chunk: language detection and
   retrieval both degrade on very short texts.
4. Language resolved per chunk (explicit -> detected -> None).
"""

import logging
import re
from collections import Counter
from collections.abc import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from pool_assistant.ingestion.language import resolve_language
from pool_assistant.ingestion.loader import PageText
from pool_assistant.schemas import Chunk

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1000       # characters (~250 tokens): fits multilingual embedders
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_MIN_CHUNK_CHARS = 200   # shorter trailing pieces are merged into the previous one
EDGE_LINES = 3                  # header/footer zone: first and last N lines of a page
MIN_BOILERPLATE_REPEATS = 3     # a line repeated in >= N page edges is boilerplate
MAX_BOILERPLATE_CHARS = 120     # running headers/footers are short; long repeated lines are content
_PAGE_NUMBER = re.compile(r"^\d{1,4}$")


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def find_boilerplate(
    pages: Sequence[PageText], min_repeats: int = MIN_BOILERPLATE_REPEATS
) -> set[str]:
    """Short lines that repeat in the header/footer zone of at least `min_repeats` pages."""
    counts: Counter[str] = Counter()
    for page in pages:
        lines = _lines(page.text)
        edges = lines[:EDGE_LINES] + lines[-EDGE_LINES:]
        counts.update({line for line in edges if len(line) <= MAX_BOILERPLATE_CHARS})
    return {line for line, n in counts.items() if n >= min_repeats}


def clean_page_text(text: str, boilerplate: set[str]) -> str:
    """Drop boilerplate lines and bare page numbers in the header/footer zone."""
    lines = _lines(text)
    last = len(lines) - 1
    kept = [
        line
        for i, line in enumerate(lines)
        if line not in boilerplate
        and not (_PAGE_NUMBER.match(line) and (i < EDGE_LINES or i > last - EDGE_LINES))
    ]
    return "\n".join(kept)


def _merge_small_tail(pieces: list[str], min_chars: int) -> list[str]:
    if len(pieces) > 1 and len(pieces[-1]) < min_chars:
        pieces = [*pieces[:-2], f"{pieces[-2]}\n{pieces[-1]}"]
    return pieces


def chunk_pages(
    pages: Sequence[PageText],
    *,
    doc_id: str,
    explicit_lang: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
) -> list[Chunk]:
    """Turn loaded pages into retrieval chunks with citation and language metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",
    )
    boilerplate = find_boilerplate(pages)
    logger.debug("Boilerplate lines for %s: %s", doc_id, sorted(boilerplate))

    chunks: list[Chunk] = []
    for page in pages:
        text = clean_page_text(page.text, boilerplate)
        if not text:
            continue
        pieces = _merge_small_tail(splitter.split_text(text), min_chunk_chars)
        for index, piece in enumerate(pieces):
            language = resolve_language(piece, explicit_lang)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:p{page.page_number:04d}:c{index:02d}",
                    doc_id=doc_id,
                    source=page.source,
                    page_number=page.page_number,
                    chunk_index=index,
                    text=piece,
                    lang=language.lang,
                    lang_confidence=round(language.confidence, 3),
                    lang_method=language.method,
                    parser=page.parser,
                )
            )

    langs = Counter(c.lang for c in chunks)
    logger.info(
        "Chunked %s: %d pages -> %d chunks, %d boilerplate lines removed, languages=%s",
        doc_id, len(pages), len(chunks), len(boilerplate), dict(langs),
    )
    return chunks
