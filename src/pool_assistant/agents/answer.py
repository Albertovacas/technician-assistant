"""Answer Agent: write the answer from the graded evidence only.

Node:
- `answer`: grounded, cited answer in the technician's language, built with the same
  `compose_answer` as the baseline. The difference with the baseline is the input: only
  the excerpts the Retrieval Agent kept, not the raw top-k.

`Answer.retrieval` exposes the evidence actually given to the LLM; the excerpts discarded
by grading remain visible in the trace (`grade` step).

Graceful degradation: if the LLM call fails after the provider's retries (503, timeout,
quota), the technician gets the graded excerpts with their pages instead of an error -
the evidence is already retrieved and filtered, and it is useful without an LLM.
"""

import logging
import time

from langchain_core.language_models import BaseChatModel

from pool_assistant.agents.prompts import degraded_message
from pool_assistant.agents.state import AgentState
from pool_assistant.rag.simple_rag import Generation, compose_answer
from pool_assistant.schemas import Answer, Citation, TraceStep

logger = logging.getLogger(__name__)


class AnswerAgent:
    def __init__(self, llm: BaseChatModel, *, model_name: str) -> None:
        self.llm = llm
        self.model_name = model_name

    def answer(self, state: AgentState) -> dict:
        start = time.perf_counter()
        question, retrieval = state["question"], state["retrieval"]
        triage = state.get("triage")
        language = ((triage.language if triage else None) or state.get("lang_hint")
                    or retrieval.query_lang)
        evidence = state.get("relevant_chunks", [])
        feedback = state.get("feedback")  # set by the Verifier for a corrected second attempt

        degraded = False
        try:
            generation = compose_answer(self.llm, question, evidence, language,
                                        feedback=feedback)
        except Exception:
            logger.error("Answer generation failed: returning the relevant excerpts",
                         exc_info=True)
            degraded = True
            generation = Generation(
                text=degraded_message(language,
                                      [(c.chunk.page_number, c.chunk.text) for c in evidence]),
                citations=list(dict.fromkeys(
                    Citation(source=c.chunk.source, page_number=c.chunk.page_number)
                    for c in evidence)),
                llm_calls=1,
            )

        answer = Answer(
            question=question, answer=generation.text, citations=generation.citations,
            invalid_citations=generation.invalid_citations,
            insufficient_evidence=generation.insufficient_evidence, degraded=degraded,
            language=language,
            retrieval=retrieval.model_copy(update={"chunks": list(evidence)}),
            model=self.model_name, latency_ms=0,  # total latency is set by the graph runner
        )
        summary = (("[retry with verifier feedback] " if feedback else "")
                   + ("[DEGRADED: LLM failed, excerpts returned] " if degraded else "")
                   + f"citations={[c.page_number for c in generation.citations]} "
                   f"invalid={generation.invalid_citations} "
                   f"insufficient={generation.insufficient_evidence} "
                   f"evidence={len(evidence)} chunks")
        return {"answer": answer, "trace": [TraceStep(
            agent="answer", node="answer", summary=summary, llm_calls=generation.llm_calls,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )]}
