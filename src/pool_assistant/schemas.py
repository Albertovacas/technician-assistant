"""Domain models shared across ingestion, retrieval, agents and API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """A retrievable unit of manual text, with everything needed to cite it."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str                 # deterministic: "<doc_id>:p<page>:c<index>"
    doc_id: str
    source: str                   # manual file name shown in citations
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)  # position within the page
    text: str
    lang: str | None              # ISO 639-1, or None if undetermined
    lang_confidence: float = Field(ge=0.0, le=1.0)
    lang_method: Literal["explicit", "detected", "undetermined"]
    parser: Literal["text", "table", "vision"] = "text"  # see PageText.parser


class ScoredChunk(BaseModel):
    """A chunk returned by vector search, with its similarity score (cosine, higher=closer)."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float


class RetrievalResult(BaseModel):
    """Evidence for a question, plus how the language filter was resolved (for tracing)."""

    model_config = ConfigDict(frozen=True)

    question: str
    query_lang: str | None          # language of the question (None = undetermined)
    search_lang: str | None         # language actually searched (None = no filter)
    strategy: Literal["same_language", "fallback_language", "unfiltered"]
    chunks: list[ScoredChunk]
    reranked: bool = False          # order set by the cross-encoder (else cosine order)


class TraceStep(BaseModel):
    """One step of the agent workflow (what ran, what it decided, at what cost)."""

    model_config = ConfigDict(frozen=True)

    agent: Literal["triage", "retrieval", "answer", "verifier"]
    node: str                       # graph node name
    summary: str                    # human-readable decision, e.g. "in_scope=True lang=es"
    llm_calls: int = 0
    latency_ms: int = 0


class Verification(BaseModel):
    """Verifier Agent verdict attached to the final answer."""

    model_config = ConfigDict(frozen=True)

    # supported: every claim backed by the evidence · unsupported: still unsupported after the
    # retry (a safe fallback was returned) · skipped: nothing to verify (not in manual / out of
    # scope) · unavailable: the verifier failed, the answer is shown unverified.
    status: Literal["supported", "unsupported", "skipped", "unavailable"]
    unsupported_claims: list[str] = []
    answer_retries: int = 0           # regenerations requested by the verifier


class Usage(BaseModel):
    """LLM consumption of one question (the basis for cost monitoring)."""

    model_config = ConfigDict(frozen=True)

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_by_node: dict[str, int] = {}   # multi-agent: which agent spent the tokens


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    page_number: int


class Answer(BaseModel):
    """Final answer returned to the technician (CLI, API)."""

    model_config = ConfigDict(frozen=True)

    question: str
    answer: str
    citations: list[Citation]               # valid citations, in order of appearance
    invalid_citations: list[int] = []       # pages cited but not in the evidence (red flag)
    insufficient_evidence: bool = False     # in scope, but the manual does not cover it
    out_of_scope: bool = False              # rejected by triage (not a technical question)
    degraded: bool = False                  # LLM failed: relevant excerpts returned instead
    language: str | None = None             # language the answer was requested in
    retrieval: RetrievalResult | None = None  # None when no search was run (out of scope)
    model: str
    latency_ms: int
    request_id: str | None = None           # correlates the answer with its log lines
    usage: Usage | None = None
    # Multi-agent runs only (None / empty for the single-pass baseline).
    verification: Verification | None = None
    agent_trace: list[TraceStep] = []
