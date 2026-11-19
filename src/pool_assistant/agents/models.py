"""Structured outputs of the agents.

The LLM fills these via `with_structured_output(...)`, so the graph's edges read typed
fields (`in_scope`, `relevant_excerpts`) instead of parsing free text. Field descriptions
are part of the prompt: the model sees them when producing the object.
"""

from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal[
    "troubleshooting", "installation", "maintenance", "specifications", "safety", "other"
]


class TriageResult(BaseModel):
    """Triage Agent: understand the question before spending any retrieval effort."""

    in_scope: bool = Field(
        description="True if the question is about installing, operating, maintaining, "
        "troubleshooting or the specifications of pool equipment (pumps, filters, etc.). "
        "False for anything else (prices, orders, unrelated topics, chit-chat)."
    )
    language: str = Field(
        description="ISO 639-1 code of the language the question is written in, e.g. 'es'."
    )
    intent: Intent = Field(description="Main type of information the technician needs.")
    search_query: str = Field(
        description="A self-contained search query in the SAME language as the question, "
        "rephrased to find the relevant passage of a technical manual (expand terse "
        "questions, keep technical terms and error codes)."
    )
    reason: str = Field(description="One short sentence justifying the decision.")


class EvidenceGrade(BaseModel):
    """Retrieval Agent: which retrieved excerpts actually help answer the question."""

    relevant_excerpts: list[int] = Field(
        description="Numbers of the excerpts that contain information needed to answer "
        "the question. Empty list if none of them is useful."
    )
    reason: str = Field(description="One short sentence justifying the selection.")


class VerificationResult(BaseModel):
    """Verifier Agent: is every claim of the answer supported by the evidence text?"""

    supported: bool = Field(
        description="True only if every factual claim of the answer (values, steps, conditions, "
        "safety instructions) is stated in the excerpts, possibly in another language."
    )
    unsupported_claims: list[str] = Field(
        description="Each claim of the answer that the excerpts do not support, quoted briefly. "
        "Empty if supported."
    )
    explanation: str = Field(description="One short sentence justifying the verdict.")


class RewrittenQuery(BaseModel):
    """Retrieval Agent: a second attempt when the first search found nothing useful."""

    search_query: str = Field(
        description="A different search query, in the same language as the question, using "
        "other words or the likely technical terms of the manual."
    )
