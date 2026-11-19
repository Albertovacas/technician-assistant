"""Verifier Agent: is every claim of the answer supported by the evidence text?

Node:
- `verify`: checks the answer against the excerpts it was written from.
Edge:
- `route_after_verify`: back to `answer` once with the list of unsupported claims, or end.

Why: citation validation only checks that a cited page was retrieved, not that the page
says what the answer claims. Notebook 08 showed "replace the capacitor every 5 years
[p. 32]" passing citation validation while the manual says 1 year.

Policy:
- nothing to verify (not in the manual / out of scope)       -> skipped, no LLM call
- citations to pages outside the evidence                    -> unsupported, no LLM call
- supported                                                  -> answer as is
- unsupported, retries left                                  -> regenerate with feedback
- unsupported after the retry                                -> safe fallback: no claims,
                                                                point to the evidence pages
- verifier call fails                                        -> keep the answer, marked
                                                                "unavailable" (= baseline)
"""

import logging
import time
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END

from pool_assistant.agents.models import VerificationResult
from pool_assistant.agents.prompts import (
    VERIFY_SYSTEM,
    correction_feedback,
    unverified_message,
    verify_user_prompt,
)
from pool_assistant.agents.state import AgentState, over_budget
from pool_assistant.rag.prompts import format_context
from pool_assistant.schemas import Citation, TraceStep, Verification

logger = logging.getLogger(__name__)


class VerifierAgent:
    def __init__(self, llm: BaseChatModel, *, max_answer_retries: int = 1) -> None:
        self.llm = llm.with_structured_output(VerificationResult)
        self.max_answer_retries = max_answer_retries

    def _step(self, summary: str, start: float, llm_calls: int) -> list[TraceStep]:
        return [TraceStep(agent="verifier", node="verify", summary=summary, llm_calls=llm_calls,
                          latency_ms=int((time.perf_counter() - start) * 1000))]

    def verify(self, state: AgentState) -> dict:
        start = time.perf_counter()
        answer = state["answer"]
        retries = state.get("answer_retries", 0)

        def done(verification: Verification, summary: str, calls: int, **answer_updates) -> dict:
            final = answer.model_copy(update={"verification": verification, **answer_updates})
            return {"answer": final, "retry_answer": False, "feedback": None,
                    "trace": self._step(summary, start, calls)}

        if answer.insufficient_evidence or answer.out_of_scope or answer.degraded:
            # degraded = verbatim manual excerpts: nothing generated, nothing to verify
            return done(Verification(status="skipped", answer_retries=retries),
                        "skipped: no generated claims to verify", 0)

        evidence = state.get("relevant_chunks", [])
        if answer.invalid_citations:  # deterministic: cites pages it was not given
            verdict = VerificationResult(
                supported=False, unsupported_claims=[],
                explanation=f"cites pages outside the evidence: {answer.invalid_citations}")
            calls = 0
        else:
            try:
                verdict = self.llm.invoke([
                    SystemMessage(VERIFY_SYSTEM),
                    HumanMessage(verify_user_prompt(state["question"], format_context(evidence),
                                                    answer.answer)),
                ])
                calls = 1
            except Exception:
                logger.warning("Verification failed: returning the answer unverified",
                               exc_info=True)
                return done(Verification(status="unavailable", answer_retries=retries),
                            "verifier unavailable: answer returned unverified", 1)

        if verdict.supported:
            return done(Verification(status="supported", answer_retries=retries),
                        f"supported | {verdict.explanation}", calls)

        claims = verdict.unsupported_claims
        out_of_time = over_budget(state)
        if retries < self.max_answer_retries and not out_of_time:
            logger.info("Answer not supported (%s): requesting a corrected answer", claims)
            return {
                "retry_answer": True,
                "answer_retries": retries + 1,
                "feedback": correction_feedback(claims, answer.invalid_citations),
                "trace": self._step(f"NOT supported {claims} -> retry answer | "
                                    f"{verdict.explanation}", start, calls),
            }

        # Still unsupported after the retry: never present unverified claims as facts.
        pages = list(dict.fromkeys(c.chunk.page_number for c in evidence))
        sources = {c.chunk.page_number: c.chunk.source for c in evidence}
        reason = "time budget exhausted" if out_of_time else f"after {retries} retries"
        logger.warning("Answer not supported (%s): safe fallback", reason)
        return done(
            Verification(status="unsupported", unsupported_claims=claims, answer_retries=retries),
            f"NOT supported ({reason}) -> safe fallback | {verdict.explanation}",
            calls,
            answer=unverified_message(answer.language, pages),
            citations=[Citation(source=sources[p], page_number=p) for p in pages],
            invalid_citations=[],
        )

    def route_after_verify(self, state: AgentState) -> Literal["answer", "__end__"]:
        return "answer" if state.get("retry_answer") else END
