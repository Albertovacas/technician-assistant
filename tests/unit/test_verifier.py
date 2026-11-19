"""Verifier Agent inside the full graph (scripted LLM and retriever, no network)."""

from pool_assistant.agents.graph import MultiAgentRAG
from pool_assistant.agents.models import EvidenceGrade, TriageResult, VerificationResult
from pool_assistant.agents.prompts import VERIFY_SYSTEM, unverified_message
from pool_assistant.schemas import ScoredChunk
from tests.fakes import FakeAgentLLM

from .test_retrieval_agent import ScriptedRetriever
from .test_vectorstore import make_chunk

CAPACITOR = ScoredChunk(chunk=make_chunk("m", 32, 0, "| Condensador | 1 año |", lang="es"),
                        score=0.4)
TRIAGE = TriageResult(in_scope=True, language="es", intent="maintenance",
                      search_query="sustitución condensador", reason="t")
KEEP = EvidenceGrade(relevant_excerpts=[1], reason="t")
SUPPORTED = VerificationResult(supported=True, unsupported_claims=[], explanation="ok")
WRONG = "El condensador debe cambiarse cada 5 años [p. 32]."   # the notebook 08 example
RIGHT = "El condensador debe cambiarse cada 1 año [p. 32]."


def unsupported(*claims: str) -> VerificationResult:
    return VerificationResult(supported=False, unsupported_claims=list(claims),
                              explanation="value differs from the excerpt")


def run(*responses):
    retriever = ScriptedRetriever({"sustitución condensador": [CAPACITOR]})
    llm = FakeAgentLLM(TRIAGE, KEEP, *responses)
    answer = MultiAgentRAG(retriever, llm, model_name="fake").answer("¿Cada cuánto el condensador?")
    return answer, llm


def nodes(answer) -> list[str]:
    return [s.node for s in answer.agent_trace]


def test_supported_answer_is_returned_as_is() -> None:
    answer, llm = run(RIGHT, SUPPORTED)

    assert answer.answer == RIGHT
    assert answer.verification.status == "supported"
    system, user = llm.calls[-1][1]
    assert system.content == VERIFY_SYSTEM
    assert "| Condensador | 1 año |" in user.content and RIGHT in user.content


def test_wrong_value_is_corrected_on_retry() -> None:
    answer, llm = run(WRONG, unsupported("cada 5 años"), RIGHT, SUPPORTED)

    assert answer.answer == RIGHT
    assert answer.verification.status == "supported"
    assert answer.verification.answer_retries == 1
    assert nodes(answer)[-4:] == ["answer", "verify", "answer", "verify"]
    retry_prompt = llm.calls[4][1][1].content          # second answer call
    assert "Not supported by the excerpts: cada 5 años" in retry_prompt


def test_still_unsupported_after_retry_returns_safe_fallback() -> None:
    answer, _ = run(WRONG, unsupported("cada 5 años"), WRONG, unsupported("cada 5 años"))

    assert answer.answer == unverified_message("es", [32])
    assert "5 años" not in answer.answer                # the false claim never reaches the user
    assert [c.page_number for c in answer.citations] == [32]
    assert answer.verification.status == "unsupported"
    assert answer.verification.unsupported_claims == ["cada 5 años"]


def test_invalid_citation_is_unsupported_without_llm_call() -> None:
    answer, llm = run("Cambiar cada año [p. 99].", RIGHT, SUPPORTED)

    assert answer.answer == RIGHT
    assert llm.schemas_called.count("VerificationResult") == 1  # only the second verification
    assert "Cited pages that are not in the excerpts: [99]" in llm.calls[3][1][1].content


def test_verifier_failure_returns_answer_marked_unavailable() -> None:
    answer, _ = run(RIGHT)  # no scripted verdict -> the verifier call raises

    assert answer.answer == RIGHT
    assert answer.verification.status == "unavailable"


def test_nothing_to_verify_is_skipped() -> None:
    retriever = ScriptedRetriever({})  # no evidence at all -> "not in the manual"
    llm = FakeAgentLLM(TRIAGE)  # grading skipped (no chunks), rewrite fails open
    answer = MultiAgentRAG(retriever, llm, model_name="fake").answer("E01")

    assert answer.insufficient_evidence
    assert answer.verification.status == "skipped"


def test_degraded_answer_is_not_verified() -> None:
    answer, llm = run()  # answer call raises -> degraded excerpts

    assert answer.degraded
    assert answer.verification.status == "skipped"
    assert "VerificationResult" not in llm.schemas_called


def test_over_budget_skips_the_retry_and_returns_safe_fallback() -> None:
    retriever = ScriptedRetriever({"sustitución condensador": [CAPACITOR]})
    llm = FakeAgentLLM(TRIAGE, KEEP, WRONG, unsupported("cada 5 años"))
    rag = MultiAgentRAG(retriever, llm, model_name="fake", time_budget_s=1e-9)  # already spent

    answer = rag.answer("¿Cada cuánto el condensador?")

    assert answer.verification.status == "unsupported"
    assert answer.verification.answer_retries == 0       # no time for a retry
    assert "5 años" not in answer.answer                  # safety preserved
    assert "time budget exhausted" in answer.agent_trace[-1].summary
