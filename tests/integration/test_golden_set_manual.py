"""The golden set itself is checked against the manual: ground truth must be true.

For every 'answer' item, each key fact must appear (any alternative) in at least one of
its expected pages, using the same enriched text the system indexes (tables included).
"""

from collections import Counter

import pytest

from pool_assistant.config import get_settings
from pool_assistant.evaluation.golden import load_golden_set
from pool_assistant.ingestion.loader import load_pdf
from pool_assistant.ingestion.tables import enrich_with_tables

MANUAL = get_settings().data_dir / "manuals" / "victoria_plus_silent_manual.pdf"
GOLDEN = load_golden_set()
ANSWER_ITEMS = [item for item in GOLDEN.items if item.expected == "answer"]


@pytest.fixture(scope="module")
def page_texts() -> dict[int, str]:
    if not MANUAL.exists():
        pytest.skip(f"Manual not available at {MANUAL}")
    return {p.page_number: p.text.lower() for p in enrich_with_tables(load_pdf(MANUAL), MANUAL)}


@pytest.mark.parametrize("item", ANSWER_ITEMS, ids=lambda item: item.id)
def test_key_facts_are_in_the_expected_pages(item, page_texts) -> None:
    text = " ".join(page_texts.get(page, "") for page in item.pages)
    for fact in item.key_facts:
        assert any(alt.lower() in text for alt in fact), f"{item.id}: {fact} not in {item.pages}"


def test_golden_set_covers_every_behaviour_and_several_languages() -> None:
    expected = Counter(item.expected for item in GOLDEN.items)
    languages = {item.lang for item in GOLDEN.items}

    assert expected["answer"] >= 20
    assert expected["not_in_manual"] >= 3 and expected["out_of_scope"] >= 3
    assert {"en", "es", "fr", "de", "it", "pt"} <= languages
