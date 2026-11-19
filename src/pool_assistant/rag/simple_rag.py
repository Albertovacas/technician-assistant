"""Simple RAG baseline: retrieve -> prompt -> LLM -> citations.

Single pass, no agents. It is the reference the multi-agent system (phase 3) is compared
against, and it already enforces the non-negotiables: answers grounded in the manual,
page citations, the technician's language, and an explicit "not in the manual" outcome.
"""

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from pool_assistant.observability import log_request_completed, request_context, track_usage
from pool_assistant.rag.prompts import (
    NO_ANSWER,
    SYSTEM_PROMPT,
    build_user_prompt,
    insufficient_evidence_message,
)
from pool_assistant.schemas import Answer, Citation, RetrievalResult, ScoredChunk

logger = logging.getLogger(__name__)

# [p. 13], [p.13], [P. 13], [p. 12, 13] -> page numbers
_CITATION = re.compile(r"\[\s*p\.?\s*(\d+(?:\s*[,;]\s*(?:p\.?\s*)?\d+)*)\s*\]", re.IGNORECASE)


class SupportsRetrieve(Protocol):
    def retrieve(self, question: str, *, lang: str | None = None) -> RetrievalResult: ...


def extract_cited_pages(text: str) -> list[int]:
    """Pages cited as [p. N] (also [p. 12, 13]), in order of first appearance."""
    pages: list[int] = []
    for match in _CITATION.finditer(text):
        for number in re.findall(r"\d+", match.group(1)):
            if int(number) not in pages:
                pages.append(int(number))
    return pages


@dataclass(frozen=True)
class Generation:
    """Outcome of grounded generation (shared by the baseline and the Answer Agent)."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    insufficient_evidence: bool = False
    llm_calls: int = 0


def compose_answer(llm: BaseChatModel, question: str, chunks: Sequence[ScoredChunk],
                   lang: str | None, feedback: str | None = None) -> Generation:
    """Write a cited answer from the given evidence only.

    No evidence -> "not in the manual" without calling the LLM. The LLM can also decline
    with NO_ANSWER; the user-facing text is then ours (fixed, translated).
    `feedback` (from the Verifier) asks for a corrected second attempt.
    """
    if not chunks:
        return Generation(insufficient_evidence_message(lang), insufficient_evidence=True)

    messages = [SystemMessage(SYSTEM_PROMPT),
                HumanMessage(build_user_prompt(question, chunks, lang, feedback))]
    text = llm.invoke(messages).text.strip()
    if text.startswith(NO_ANSWER):
        return Generation(insufficient_evidence_message(lang), insufficient_evidence=True,
                          llm_calls=1)

    sources = {c.chunk.page_number: c.chunk.source for c in chunks}
    cited = extract_cited_pages(text)
    valid = [Citation(source=sources[p], page_number=p) for p in cited if p in sources]
    invalid = [p for p in cited if p not in sources]
    if invalid:
        logger.warning("Answer cites pages not present in the evidence: %s", invalid)
    return Generation(text, valid, invalid, llm_calls=1)


class SimpleRAG:
    def __init__(self, retriever: SupportsRetrieve, llm: BaseChatModel, *, model_name: str):
        self.retriever = retriever
        self.llm = llm
        self.model_name = model_name

    def answer(self, question: str, *, lang: str | None = None,
               request_id: str | None = None) -> Answer:
        with request_context(request_id) as rid, track_usage() as usage:
            start = time.perf_counter()
            retrieval = self.retriever.retrieve(question, lang=lang)
            answer_lang = lang or retrieval.query_lang
            generation = compose_answer(self.llm, question, retrieval.chunks, answer_lang)
            answer = Answer(
                question=question, answer=generation.text, citations=generation.citations,
                invalid_citations=generation.invalid_citations,
                insufficient_evidence=generation.insufficient_evidence, language=answer_lang,
                retrieval=retrieval, model=self.model_name,
                latency_ms=int((time.perf_counter() - start) * 1000),
                request_id=rid, usage=usage.summary(),
            )
            log_request_completed(answer, mode="simple")
            return answer
