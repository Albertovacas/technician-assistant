"""Vision fallback with fake LLMs (no network)."""

from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from pool_assistant.ingestion.loader import load_pdf
from pool_assistant.ingestion.vision import (
    enrich_with_vision,
    render_page_png,
    text_coverage,
)

GOOD_TRANSCRIPTION = (
    "# TROUBLESHOOTING\n\nCheck the condition of the seals before start-up. "
    "Disconnect the unit from the power supply before any maintenance operation.\n\n"
    "| 1 | 2 | CAUSES |\n|---|---|---|\n| ● | | Air entering |"
)


class RaisingLLM:
    """Stands in for a chat model whose provider is down."""

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        raise RuntimeError("503 UNAVAILABLE")


def fake_llm(*responses: str) -> FakeListChatModel:
    return FakeListChatModel(responses=list(responses))


@pytest.fixture
def table_pdf(make_pdf) -> Path:
    return make_pdf("manual.pdf", ["text", "table"])  # p.1 plain text, p.2 drawn table


def test_render_page_png_returns_png(table_pdf: Path) -> None:
    png = render_page_png(table_pdf, 2, scale=1.0)

    assert png.startswith(b"\x89PNG")


def test_text_coverage() -> None:
    assert text_coverage("pump seals filter", "the PUMP seals and filter") == 1.0
    assert text_coverage("pump seals filter basket", "pump") == 0.25
    assert text_coverage("", "anything") == 1.0  # no text layer to compare against


def test_only_flagged_pages_are_transcribed(table_pdf: Path) -> None:
    pages = load_pdf(table_pdf)

    result = enrich_with_vision(pages, table_pdf, fake_llm(GOOD_TRANSCRIPTION), model_name="fake")

    assert result[0] is pages[0]  # plain text page untouched
    assert (result[1].parser, result[1].text) == ("vision", GOOD_TRANSCRIPTION)
    assert result[1].page_number == 2


def test_llm_failure_keeps_text_layer(table_pdf: Path) -> None:
    pages = load_pdf(table_pdf)

    result = enrich_with_vision(pages, table_pdf, RaisingLLM(), model_name="fake")

    assert result == pages


def test_low_coverage_transcription_is_rejected(table_pdf: Path) -> None:
    pages = load_pdf(table_pdf)
    hallucinated = "| Fan cover | CORRECT |\n|---|---|\n| ✓ | |"

    result = enrich_with_vision(pages, table_pdf, fake_llm(hallucinated), model_name="fake")

    assert result[1].parser == "text"
    assert result[1].text == pages[1].text


def test_cache_avoids_second_llm_call(table_pdf: Path, tmp_path: Path) -> None:
    pages = load_pdf(table_pdf)
    cache = tmp_path / "vision_cache"
    enrich_with_vision(
        pages, table_pdf, fake_llm(GOOD_TRANSCRIPTION), model_name="m", cache_dir=cache
    )

    offline = RaisingLLM()
    result = enrich_with_vision(pages, table_pdf, offline, model_name="m", cache_dir=cache)

    assert offline.calls == 0
    assert result[1].text == GOOD_TRANSCRIPTION
    assert len(list(cache.glob("*.md"))) == 1


def test_without_llm_only_cache_is_used(table_pdf: Path, tmp_path: Path) -> None:
    """Offline ingestion (no API key): cached pages enriched, others keep the text layer."""
    pages = load_pdf(table_pdf)
    cache = tmp_path / "vision_cache"

    assert enrich_with_vision(pages, table_pdf, None, model_name="m", cache_dir=cache) == pages

    enrich_with_vision(
        pages, table_pdf, fake_llm(GOOD_TRANSCRIPTION), model_name="m", cache_dir=cache
    )
    result = enrich_with_vision(pages, table_pdf, None, model_name="m", cache_dir=cache)
    assert result[1].parser == "vision"


def test_cache_key_depends_on_model(table_pdf: Path, tmp_path: Path) -> None:
    pages = load_pdf(table_pdf)
    cache = tmp_path / "vision_cache"
    enrich_with_vision(
        pages, table_pdf, fake_llm(GOOD_TRANSCRIPTION), model_name="a", cache_dir=cache
    )

    other_model = RaisingLLM()
    enrich_with_vision(pages, table_pdf, other_model, model_name="b", cache_dir=cache)

    assert other_model.calls == 1  # different model -> cache miss


def test_pages_handled_by_table_extractor_are_skipped(table_pdf: Path) -> None:
    pages = [replace(p, parser="table") if p.needs_vision else p for p in load_pdf(table_pdf)]
    llm = RaisingLLM()

    result = enrich_with_vision(pages, table_pdf, llm, model_name="fake")

    assert llm.calls == 0
    assert result == pages


def test_rejected_transcriptions_are_not_cached(table_pdf: Path, tmp_path: Path) -> None:
    pages = load_pdf(table_pdf)
    cache = tmp_path / "vision_cache"

    enrich_with_vision(pages, table_pdf, fake_llm("nonsense"), model_name="m", cache_dir=cache)

    assert not cache.exists() or not list(cache.glob("*.md"))
