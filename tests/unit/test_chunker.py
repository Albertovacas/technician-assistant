from pool_assistant.ingestion.chunker import chunk_pages, clean_page_text, find_boilerplate
from pool_assistant.ingestion.loader import PageText

HEADER = "Service Manual - HEAT PUMPS"
EN_SENTENCE = "Disconnect the unit from the power supply before any maintenance operation. "
ES_SENTENCE = "Desconecte el aparato de la red eléctrica antes de cualquier mantenimiento. "


def make_page(number: int, body: str, *, header: str = HEADER, source: str = "m.pdf") -> PageText:
    text = f"{header}\n{number}\n{body}"
    return PageText(
        source=source, page_number=number, text=text, char_count=len(text),
        image_count=0, stroke_ops=0, path_ops=0, needs_ocr=False, looks_complex=False,
    )


def test_running_header_repeated_on_page_edges_is_boilerplate() -> None:
    pages = [make_page(n, f"Unique body text number {n}.") for n in range(1, 5)]

    assert find_boilerplate(pages) == {HEADER}


def test_long_content_line_repeated_on_page_edges_is_not_boilerplate() -> None:
    """Regression: identical long content lines (e.g. a repeated safety warning) were removed."""
    warning = "WARNING: " + EN_SENTENCE * 3
    pages = [make_page(n, warning) for n in range(1, 5)]

    assert find_boilerplate(pages) == {HEADER}


def test_header_on_too_few_pages_is_kept() -> None:
    pages = [make_page(1, "a"), make_page(2, "b", header="Other title")]

    assert find_boilerplate(pages) == set()


def test_clean_removes_boilerplate_and_edge_page_numbers_only() -> None:
    text = f"{HEADER}\n12\nSet the timer to 8 hours.\n12\nmore text\nend\nlast\n12"
    cleaned = clean_page_text(text, {HEADER})

    assert HEADER not in cleaned
    assert cleaned.splitlines() == ["Set the timer to 8 hours.", "12", "more text", "end", "last"]


def test_long_page_is_split_into_bounded_overlapping_chunks() -> None:
    pages = [make_page(n, EN_SENTENCE * 40) for n in (1, 2, 3)]

    chunks = chunk_pages(pages, doc_id="doc", chunk_size=500, chunk_overlap=80)
    page_1 = [c for c in chunks if c.page_number == 1]

    assert len(page_1) > 1
    assert all(len(c.text) <= 500 + 200 for c in chunks)  # size + possible merged tail
    assert [c.chunk_index for c in page_1] == list(range(len(page_1)))
    assert page_1[0].text[-40:] in page_1[1].text or page_1[1].text[:40] in page_1[0].text


def test_chunks_never_span_pages_and_keep_citation_metadata() -> None:
    pages = [make_page(n, EN_SENTENCE * 5, source="victoria.pdf") for n in (7, 9, 11)]

    chunks = chunk_pages(pages, doc_id="victoria")

    assert {c.page_number for c in chunks} == {7, 9, 11}
    assert all(c.source == "victoria.pdf" and c.doc_id == "victoria" for c in chunks)
    assert all(HEADER not in c.text for c in chunks)


def test_small_trailing_piece_is_merged_into_previous_chunk() -> None:
    # Distinct tails per page: an identical short tail on every page would be a footer.
    pages = [make_page(n, EN_SENTENCE * 12 + f"\nEnd of section {n}.") for n in (1, 2, 3)]

    chunks = chunk_pages(pages, doc_id="doc", chunk_size=500, chunk_overlap=0, min_chunk_chars=200)

    assert all(len(c.text) >= 200 for c in chunks)
    assert sum(c.text.endswith("End of section 1.") for c in chunks) == 1


def test_language_is_resolved_per_chunk() -> None:
    pages = [make_page(1, EN_SENTENCE * 3), make_page(2, ES_SENTENCE * 3), make_page(3, "x")]

    by_page = {c.page_number: c for c in chunk_pages(pages, doc_id="doc")}

    assert (by_page[1].lang, by_page[1].lang_method) == ("en", "detected")
    assert by_page[2].lang == "es"


def test_explicit_language_overrides_detection() -> None:
    pages = [make_page(n, EN_SENTENCE * 3) for n in (1, 2, 3)]

    chunks = chunk_pages(pages, doc_id="doc", explicit_lang="es")

    assert all((c.lang, c.lang_method) == ("es", "explicit") for c in chunks)


def test_chunk_ids_are_deterministic() -> None:
    pages = [make_page(n, EN_SENTENCE * 30) for n in (1, 2, 3)]

    first = [c.chunk_id for c in chunk_pages(pages, doc_id="doc")]
    second = [c.chunk_id for c in chunk_pages(pages, doc_id="doc")]

    assert first == second
    assert first[0] == "doc:p0001:c00"
    assert len(set(first)) == len(first)


def test_page_with_only_boilerplate_produces_no_chunks() -> None:
    pages = [make_page(n, EN_SENTENCE * 3) for n in (1, 2)] + [make_page(3, "")]

    chunks = chunk_pages(pages, doc_id="doc")

    assert 3 not in {c.page_number for c in chunks}
