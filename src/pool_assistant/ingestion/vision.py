"""Vision fallback: transcribe pages the text layer cannot represent faithfully.

Pages flagged by the loader (`needs_ocr` or `looks_complex`) are rendered to PNG and
transcribed to Markdown by a multimodal LLM. Typical wins: tables whose meaning is
graphical (the troubleshooting matrix relates causes and symptoms with dots), scanned
pages and labelled diagrams.

Safeguards, because a wrong transcription is worse than none:
1. Prompt forbids inventing structure (e.g. turning a figure into a table).
2. Coverage check: if the page has a text layer, the transcription must contain most of
   its words; otherwise it is rejected and the original text is kept.
3. Any LLM/render error degrades gracefully to the original text.
4. Disk cache keyed by (file hash, page, model, prompt version): re-ingestion costs no
   LLM calls, and committing the cache makes ingestion reproducible without an API key.
"""

import base64
import hashlib
import io
import logging
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pypdfium2 as pdfium
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from pool_assistant.ingestion.loader import DEFAULT_MIN_TEXT_CHARS, PageText

logger = logging.getLogger(__name__)

# Bump the version whenever the prompt changes: it invalidates cached transcriptions.
VISION_PROMPT_VERSION = "v2"
VISION_PROMPT = """\
You are transcribing one page of a technical product manual so it can be searched by a RAG system.

Transcribe ALL the content of the page image into clean Markdown, following these rules:
- Keep the original language(s) of the page. Do not translate. Do not summarise.
- Use Markdown headings for titles.
- Tables: ONLY when the page contains a real table (visible grid of rows and columns),
  reproduce it as a Markdown table. Never turn a figure or a diagram into a table.
- If a table relates rows and columns with symbols (dots, checkmarks, crosses), add right after
  it a bullet list stating each relation in words: "- <row label> → <column label>; <column label>".
  Use the full labels, not just numbers.
- Figures and diagrams: write "[Figure <number if printed>: <one-sentence description of what it
  shows>]", then a bullet list with every visible label exactly as printed (all languages) and
  any measurement, arrow or CORRECT/INCORRECT annotation together with what it refers to.
- Never add information that is not visible on the page.
- Omit running headers, footers and page numbers.
- Output only the Markdown, without code fences or comments."""

_WORD = re.compile(r"\w{4,}", re.UNICODE)  # words of 4+ letters: ignores numbers/particles


def render_page_png(pdf_path: Path, page_number: int, scale: float = 2.0) -> bytes:
    """Render a 1-based page of a PDF to PNG bytes."""
    document = pdfium.PdfDocument(pdf_path)
    try:
        image = document[page_number - 1].render(scale=scale).to_pil()
    finally:
        document.close()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def transcribe_page_image(llm: BaseChatModel, png: bytes) -> str:
    """Ask a multimodal LLM to transcribe a page image into Markdown."""
    image_url = f"data:image/png;base64,{base64.b64encode(png).decode()}"
    message = HumanMessage(
        content=[
            {"type": "text", "text": VISION_PROMPT},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )
    return llm.invoke([message]).text.strip()


def text_coverage(reference: str, candidate: str) -> float:
    """Share of the reference's distinct words (4+ letters) that appear in the candidate."""
    reference_words = set(_WORD.findall(reference.lower()))
    if not reference_words:
        return 1.0
    candidate_words = set(_WORD.findall(candidate.lower()))
    return len(reference_words & candidate_words) / len(reference_words)


def file_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _cache_path(cache_dir: Path, fingerprint: str, page_number: int, model: str) -> Path:
    safe_model = re.sub(r"[^\w.-]", "_", model)
    name = f"{fingerprint}_p{page_number:04d}_{safe_model}_{VISION_PROMPT_VERSION}.md"
    return cache_dir / name


def enrich_with_vision(
    pages: list[PageText],
    pdf_path: Path,
    llm: BaseChatModel | None,
    *,
    model_name: str,
    cache_dir: Path | None = None,
    render_scale: float = 2.0,
    min_text_coverage: float = 0.6,
) -> list[PageText]:
    """Replace the text of flagged pages with a validated vision transcription.

    Pages that are not flagged are returned unchanged. On any failure or a rejected
    transcription, the original text-layer page is kept. With `llm=None` (no API key),
    only cached transcriptions are used: ingestion stays reproducible offline.
    """
    fingerprint = file_fingerprint(pdf_path)
    stats: Counter[str] = Counter()
    enriched: list[PageText] = []

    for page in pages:
        # Skip pages that need no help, or that the deterministic table extractor handled.
        if not page.needs_vision or page.parser != "text":
            enriched.append(page)
            continue

        cache_file = None
        if cache_dir is not None:
            cache_file = _cache_path(cache_dir, fingerprint, page.page_number, model_name)

        if cache_file and cache_file.exists():
            transcription = cache_file.read_text(encoding="utf-8")
            stats["cache_hit"] += 1
        elif llm is None:
            stats["skipped_no_llm"] += 1
            enriched.append(page)
            continue
        else:
            try:
                png = render_page_png(pdf_path, page.page_number, render_scale)
                transcription = transcribe_page_image(llm, png)
            except Exception:  # LLM/network/render errors must not break ingestion
                logger.warning("Vision failed on p.%d, keeping text layer", page.page_number,
                               exc_info=True)
                stats["failed"] += 1
                enriched.append(page)
                continue

            coverage = text_coverage(page.text, transcription)
            has_text_layer = page.char_count >= DEFAULT_MIN_TEXT_CHARS
            if not transcription or (has_text_layer and coverage < min_text_coverage):
                logger.warning(
                    "Rejected vision transcription of p.%d (coverage=%.2f < %.2f)",
                    page.page_number, coverage, min_text_coverage,
                )
                stats["rejected"] += 1
                enriched.append(page)
                continue

            stats["transcribed"] += 1
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(transcription, encoding="utf-8")

        enriched.append(replace(page, text=transcription, parser="vision"))

    logger.info(
        "Vision fallback on %s: candidates=%d %s",
        pdf_path.name, sum(p.needs_vision and p.parser == "text" for p in pages), dict(stats),
    )
    return enriched
