"""Answer Agent: grounded generation from graded evidence only (scripted LLM, no network)."""

from pool_assistant.agents.answer import AnswerAgent
from pool_assistant.agents.models import TriageResult
from pool_assistant.rag.prompts import NO_ANSWER, insufficient_evidence_message
from pool_assistant.schemas import RetrievalResult, ScoredChunk
from tests.fakes import FakeAgentLLM

from .test_vectorstore import make_chunk

KEPT = ScoredChunk(chunk=make_chunk("m", 33, 0, "3. La bomba hace ruido → Fijación incorrecta",
                                    lang="es"), score=0.53)
DISCARDED = ScoredChunk(chunk=make_chunk("m", 29, 0, "Instalar la tubería con pendiente",
                                         lang="es"), score=0.45)


def state(relevant, language="es", lang_hint=None) -> dict:
    triage = TriageResult(in_scope=True, language=language, intent="troubleshooting",
                          search_query="bomba ruido", reason="t")
    retrieval = RetrievalResult(question="bomba ruido", query_lang="es", search_lang="es",
                                strategy="same_language", chunks=[KEPT, DISCARDED])
    return {"question": "La bomba hace ruido, ¿qué reviso?", "lang_hint": lang_hint,
            "triage": triage, "retrieval": retrieval, "relevant_chunks": relevant}


def test_answers_from_graded_evidence_only() -> None:
    llm = FakeAgentLLM("Fija correctamente la bomba [p. 33].")

    out = AnswerAgent(llm, model_name="fake").answer(state([KEPT]))

    answer = out["answer"]
    assert [c.page_number for c in answer.citations] == [33]
    assert answer.language == "es"
    user_prompt = llm.calls[0][1][1].content
    assert "p. 33" in user_prompt and "p. 29" not in user_prompt  # discarded chunk not sent
    assert [c.chunk.page_number for c in answer.retrieval.chunks] == [33]  # evidence used
    assert out["trace"][0].llm_calls == 1


def test_no_evidence_answers_not_in_manual_without_llm() -> None:
    llm = FakeAgentLLM()  # any call would raise

    out = AnswerAgent(llm, model_name="fake").answer(state([]))

    assert out["answer"].insufficient_evidence
    assert out["answer"].answer == insufficient_evidence_message("es")
    assert llm.calls == [] and out["trace"][0].llm_calls == 0


def test_llm_can_still_decline() -> None:
    out = AnswerAgent(FakeAgentLLM(NO_ANSWER), model_name="fake").answer(state([KEPT]))

    assert out["answer"].insufficient_evidence


def test_invalid_citations_are_flagged() -> None:
    llm = FakeAgentLLM("Fija la bomba [p. 33]. Ver también [p. 29].")  # 29 was discarded

    answer = AnswerAgent(llm, model_name="fake").answer(state([KEPT]))["answer"]

    assert answer.invalid_citations == [29]


def test_language_comes_from_triage() -> None:
    llm = FakeAgentLLM("Secure the pump [p. 33].")

    AnswerAgent(llm, model_name="fake").answer(state([KEPT], language="en"))

    assert llm.calls[0][1][1].content.endswith("Answer in English.")


def test_llm_failure_degrades_to_the_graded_excerpts() -> None:
    out = AnswerAgent(FakeAgentLLM(), model_name="fake").answer(state([KEPT]))  # call raises

    answer = out["answer"]
    assert answer.degraded and not answer.insufficient_evidence
    assert answer.answer.startswith("No he podido generar una respuesta")
    assert "[p. 33] 3. La bomba hace ruido → Fijación incorrecta" in answer.answer
    assert [c.page_number for c in answer.citations] == [33]
    assert "DEGRADED" in out["trace"][0].summary
