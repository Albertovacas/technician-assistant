"""Triage Agent: understand the question before spending any retrieval effort.

Nodes:
- `triage`: one structured LLM call -> in scope? language, intent, search query.
- `reject`: fixed, translated out-of-scope answer (no search, no further LLM calls).
Edge:
- `route_after_triage`: "retrieve" or "reject", read from the typed TriageResult.

Why an LLM here: measured in phase 1-2, neither the similarity score (notebook 05: "heat
pump" scored higher than a legitimate question) nor fast-langdetect on short questions
("Fehlercode E01") are reliable for these decisions.

Fail-open: if the LLM call fails, the question is treated as in scope with a detected
language and the raw question as query - the system degrades to the baseline instead of
refusing a technician's question because of a transient error.
"""

import logging
import time
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from pool_assistant.agents.models import TriageResult
from pool_assistant.agents.prompts import TRIAGE_SYSTEM, out_of_scope_message, triage_user_prompt
from pool_assistant.agents.state import AgentState
from pool_assistant.ingestion.language import detect_query_language, normalize_lang_code
from pool_assistant.schemas import Answer, TraceStep

logger = logging.getLogger(__name__)


def _valid_lang(code: str | None) -> str | None:
    try:
        return normalize_lang_code(code) if code else None
    except ValueError:
        return None


class TriageAgent:
    def __init__(self, llm: BaseChatModel, *, model_name: str) -> None:
        self.llm = llm.with_structured_output(TriageResult)
        self.model_name = model_name

    # --- node: triage -------------------------------------------------------------------
    def triage(self, state: AgentState) -> dict:
        start = time.perf_counter()
        question, hint = state["question"], _valid_lang(state.get("lang_hint"))
        try:
            result: TriageResult = self.llm.invoke(
                [SystemMessage(TRIAGE_SYSTEM), HumanMessage(triage_user_prompt(question))]
            )
            fallback = False
        except Exception:
            logger.warning("Triage LLM call failed: falling back to in-scope", exc_info=True)
            result = TriageResult(in_scope=True, language="", intent="other",
                                  search_query=question, reason="triage unavailable (fallback)")
            fallback = True

        # Explicit hint > valid LLM code > fast-langdetect > None.
        language = hint or _valid_lang(result.language) or detect_query_language(question)
        search_query = result.search_query.strip() or question
        result = result.model_copy(
            update={"language": language or "", "search_query": search_query}
        )

        summary = (f"in_scope={result.in_scope} lang={language} intent={result.intent}"
                   + (" [fallback]" if fallback else "") + f" | {result.reason}")
        logger.info("Triage: %s | query=%r", summary, search_query)
        return {
            "triage": result,
            "search_query": search_query,
            "trace": [TraceStep(agent="triage", node="triage", summary=summary, llm_calls=1,
                                latency_ms=int((time.perf_counter() - start) * 1000))],
        }

    # --- node: reject -------------------------------------------------------------------
    def reject(self, state: AgentState) -> dict:
        triage = state["triage"]
        language = triage.language or None
        answer = Answer(
            question=state["question"], answer=out_of_scope_message(language), citations=[],
            out_of_scope=True, language=language, retrieval=None,
            model=self.model_name, latency_ms=0,
        )
        return {
            "answer": answer,
            "trace": [TraceStep(agent="triage", node="reject",
                                summary="out of scope: no search performed")],
        }


# --- edge ---------------------------------------------------------------------------------
def route_after_triage(state: AgentState) -> Literal["retrieve", "reject"]:
    return "retrieve" if state["triage"].in_scope else "reject"
