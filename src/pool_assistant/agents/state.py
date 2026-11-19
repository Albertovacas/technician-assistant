"""Shared state of the LangGraph workflow.

Each node receives the whole state and returns only the keys it changes. LangGraph merges
them: plain keys are overwritten, `trace` is *appended* to (reducer `operator.add`), so
every agent adds its step without knowing about the others.
"""

import operator
import time
from typing import Annotated, TypedDict

from pool_assistant.agents.models import TriageResult
from pool_assistant.schemas import Answer, RetrievalResult, ScoredChunk, TraceStep


class AgentState(TypedDict, total=False):
    # --- input ---
    question: str
    lang_hint: str | None                 # language given by the caller (e.g. --lang)
    deadline: float | None                # time.monotonic() after which optional steps are skipped
    # --- triage agent ---
    triage: TriageResult
    # --- retrieval agent ---
    search_query: str                     # current query (triage's, or a rewrite)
    retrieval: RetrievalResult            # last search
    relevant_chunks: list[ScoredChunk]    # evidence kept after grading
    rewrites: int                         # rewrite attempts used (bounded loop)
    # --- answer agent ---
    answer: Answer
    # --- verifier agent ---
    feedback: str | None                  # corrections for the answer's next attempt
    answer_retries: int                   # regenerations requested so far (bounded loop)
    retry_answer: bool                    # verifier's routing decision
    # --- observability ---
    trace: Annotated[list[TraceStep], operator.add]


def over_budget(state: "AgentState") -> bool:
    """True when the question's time budget is spent (no deadline = unlimited)."""
    deadline = state.get("deadline")
    return deadline is not None and time.monotonic() > deadline
