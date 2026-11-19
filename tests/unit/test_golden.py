import pytest
from pydantic import ValidationError

from pool_assistant.evaluation.golden import GoldenItem, GoldenSet, load_golden_set


def test_repository_golden_set_is_valid() -> None:
    golden = load_golden_set()

    assert golden.version >= 1 and len(golden.items) >= 25


def test_answer_items_require_pages_and_facts() -> None:
    with pytest.raises(ValidationError, match="needs pages and key_facts"):
        GoldenItem(id="x", question="q", lang="en", category="c", expected="answer")


def test_negative_items_cannot_have_ground_truth() -> None:
    with pytest.raises(ValidationError, match="only 'answer' items"):
        GoldenItem(id="x", question="q", lang="en", category="c", expected="out_of_scope",
                   pages=[5])


def test_ids_must_be_unique() -> None:
    item = GoldenItem(id="dup", question="q", lang="en", category="c", expected="not_in_manual")
    with pytest.raises(ValidationError, match="duplicate"):
        GoldenSet(version=1, items=[item, item])
