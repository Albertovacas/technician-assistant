"""Golden set: questions with verified expected behaviour, pages and key facts."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from pool_assistant.config import PROJECT_ROOT

DEFAULT_GOLDEN_SET = PROJECT_ROOT / "eval" / "golden_set.json"

Expected = Literal["answer", "not_in_manual", "out_of_scope"]


class GoldenItem(BaseModel):
    id: str
    question: str
    lang: str
    category: str
    expected: Expected
    pages: list[int] = Field(default_factory=list)      # pages that contain the answer
    # Each fact = list of acceptable alternatives (case-insensitive substrings).
    key_facts: list[list[str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _answers_need_ground_truth(self) -> "GoldenItem":
        if self.expected == "answer" and not (self.pages and self.key_facts):
            raise ValueError(f"{self.id}: an 'answer' item needs pages and key_facts")
        if self.expected != "answer" and (self.pages or self.key_facts):
            raise ValueError(f"{self.id}: only 'answer' items have pages/key_facts")
        return self


class GoldenSet(BaseModel):
    description: str = ""
    version: int
    items: list[GoldenItem]

    @model_validator(mode="after")
    def _unique_ids(self) -> "GoldenSet":
        ids = [item.id for item in self.items]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate golden ids: {sorted(duplicates)}")
        return self


def load_golden_set(path: Path = DEFAULT_GOLDEN_SET) -> GoldenSet:
    return GoldenSet.model_validate(json.loads(path.read_text(encoding="utf-8")))
