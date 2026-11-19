"""Run a pipeline over the golden set. One failing question never stops the evaluation."""

import logging
import time
from collections.abc import Callable, Sequence
from typing import Protocol

from pool_assistant.evaluation.golden import GoldenItem
from pool_assistant.evaluation.metrics import ItemResult, error_result, score
from pool_assistant.schemas import Answer

logger = logging.getLogger(__name__)


class Pipeline(Protocol):
    def answer(self, question: str, *, lang: str | None = None) -> Answer: ...


def run_mode(pipeline: Pipeline, items: Sequence[GoldenItem], mode: str,
             on_result: Callable[[ItemResult], None] | None = None) -> list[ItemResult]:
    """Ask every golden question (language NOT given: detection is part of what we test)."""
    results = []
    for i, item in enumerate(items, start=1):
        start = time.perf_counter()
        try:
            result = score(item, pipeline.answer(item.question), mode)
        except Exception as error:  # record and continue
            logger.exception("Evaluation of %s failed", item.id)
            result = error_result(item, mode, error)
        results.append(result)
        if on_result:
            on_result(result)
        logger.info("[%s %d/%d] %s correct=%s (%.1fs)", mode, i, len(items), item.id,
                    result.correct, time.perf_counter() - start)
    return results
