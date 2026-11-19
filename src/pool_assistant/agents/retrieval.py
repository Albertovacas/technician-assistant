"""Retrieval Agent: decide WHICH evidence the answer will use.

Nodes:
- `retrieve`: vector search with the triage's query and language (soft language filter).
- `grade`: one structured LLM call selects the excerpts that really help (all at once).
- `rewrite`: if nothing helps, one different query, phrased like a manual would.
Edge:
- `route_after_grade`: answer if there is evidence, otherwise rewrite while attempts remain
  (bounded loop: `max_rewrites`, default 1), then answer (-> "not in the manual").

Why: the baseline answers from the raw top-5 in a single pass. Grading removes noise
(e.g. table rows without headers, notebook 07) and gives the system a second chance with
a better query when the first search misses.

Fail-open: a grading error keeps all retrieved chunks (= baseline behaviour); a rewrite
error falls back to the original question.
"""

import logging
import time
from typing import Literal, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from pool_assistant.agents.models import EvidenceGrade, RewrittenQuery
from pool_assistant.agents.prompts import (
    GRADE_SYSTEM,
    REWRITE_SYSTEM,
    grade_user_prompt,
    rewrite_user_prompt,
)
from pool_assistant.agents.state import AgentState, over_budget
from pool_assistant.rag.prompts import format_context
from pool_assistant.schemas import RetrievalResult, TraceStep

logger = logging.getLogger(__name__)


class SupportsRetrieve(Protocol):
    def retrieve(self, question: str, *, lang: str | None = None) -> RetrievalResult: ...


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


class RetrievalAgent:
    def __init__(self, retriever: SupportsRetrieve, llm: BaseChatModel, *,
                 max_rewrites: int = 1) -> None:
        self.retriever = retriever
        self.grader = llm.with_structured_output(EvidenceGrade)
        self.rewriter = llm.with_structured_output(RewrittenQuery)
        self.max_rewrites = max_rewrites

    # --- node: retrieve -----------------------------------------------------------------
    def retrieve(self, state: AgentState) -> dict:
        start = time.perf_counter()
        triage = state.get("triage")
        lang = (triage.language if triage else None) or state.get("lang_hint")
        query = state.get("search_query") or state["question"]
        result = self.retriever.retrieve(query, lang=lang or None)
        pages = [c.chunk.page_number for c in result.chunks]
        return {
            "retrieval": result,
            "rewrites": state.get("rewrites", 0),
            "trace": [TraceStep(
                agent="retrieval", node="retrieve", latency_ms=_ms(start),
                summary=(f"query={query!r} strategy={result.strategy}"
                         f"{' +rerank' if result.reranked else ''} pages={pages}"),
            )],
        }

    # --- node: grade --------------------------------------------------------------------
    def grade(self, state: AgentState) -> dict:
        start = time.perf_counter()
        chunks = state["retrieval"].chunks
        if not chunks:
            return {"relevant_chunks": [], "trace": [TraceStep(
                agent="retrieval", node="grade", summary="no chunks retrieved: nothing to grade",
            )]}

        try:
            grade: EvidenceGrade = self.grader.invoke([
                SystemMessage(GRADE_SYSTEM),
                HumanMessage(grade_user_prompt(state["question"], format_context(chunks))),
            ])
            numbers = list(dict.fromkeys(grade.relevant_excerpts))  # dedupe, keep order
            kept = [chunks[n - 1] for n in numbers if 1 <= n <= len(chunks)]
            summary = (f"kept {len(kept)}/{len(chunks)} excerpts {numbers} "
                       f"pages={[c.chunk.page_number for c in kept]} | {grade.reason}")
        except Exception:
            logger.warning("Grading failed: keeping all retrieved chunks", exc_info=True)
            kept = list(chunks)
            summary = f"kept {len(kept)}/{len(chunks)} [fallback: grading unavailable]"

        update: dict = {"relevant_chunks": kept}
        if not kept and over_budget(state):  # no time for an optional rewrite: close the loop
            update["rewrites"] = self.max_rewrites
            summary += " | time budget exhausted: no rewrite"
        logger.info("Grade: %s", summary)
        return update | {"trace": [TraceStep(
            agent="retrieval", node="grade", summary=summary, llm_calls=1,
            latency_ms=_ms(start),
        )]}

    # --- node: rewrite ------------------------------------------------------------------
    def rewrite(self, state: AgentState) -> dict:
        start = time.perf_counter()
        question, previous = state["question"], state.get("search_query") or state["question"]
        triage = state.get("triage")
        language = (triage.language if triage else None) or state.get("lang_hint")
        try:
            result: RewrittenQuery = self.rewriter.invoke([
                SystemMessage(REWRITE_SYSTEM),
                HumanMessage(rewrite_user_prompt(question, previous, language)),
            ])
            new_query = result.search_query.strip() or question
            note = ""
        except Exception:
            logger.warning("Rewrite failed: retrying with the original question", exc_info=True)
            new_query, note = question, " [fallback]"

        return {
            "search_query": new_query,
            "rewrites": state.get("rewrites", 0) + 1,
            "trace": [TraceStep(
                agent="retrieval", node="rewrite", llm_calls=1, latency_ms=_ms(start),
                summary=f"{previous!r} -> {new_query!r}{note}",
            )],
        }

    # --- edge ---------------------------------------------------------------------------
    def route_after_grade(self, state: AgentState) -> Literal["answer", "rewrite"]:
        if state.get("relevant_chunks"):
            return "answer"
        if state.get("rewrites", 0) < self.max_rewrites:
            return "rewrite"
        return "answer"  # no evidence after all attempts -> answer agent says "not in manual"
