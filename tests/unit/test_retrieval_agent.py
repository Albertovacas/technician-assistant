"""Retrieval Agent (retrieve -> grade -> rewrite loop) inside a small LangGraph, no network."""

from langgraph.graph import END, START, StateGraph

from pool_assistant.agents.models import EvidenceGrade, RewrittenQuery, TriageResult
from pool_assistant.agents.prompts import GRADE_SYSTEM
from pool_assistant.agents.retrieval import RetrievalAgent
from pool_assistant.agents.state import AgentState
from pool_assistant.schemas import RetrievalResult, ScoredChunk
from tests.fakes import FakeAgentLLM

from .test_vectorstore import make_chunk

NOISE = ScoredChunk(chunk=make_chunk("m", 31, 0, "Comprobar el sentido de giro", lang="es"),
                    score=0.4)
ROW = ScoredChunk(chunk=make_chunk("m", 33, 1, "|  |  | ● | Fijación incorrecta |", lang="es"),
                  score=0.38)
ANSWER = ScoredChunk(chunk=make_chunk("m", 33, 0, "3. La bomba hace ruido → Fijación incorrecta",
                                      lang="es"), score=0.53)


class ScriptedRetriever:
    """Returns the chunks scripted for each query; records the calls."""

    def __init__(self, by_query: dict[str, list[ScoredChunk]]):
        self.by_query, self.calls = by_query, []

    def retrieve(self, question, *, lang=None):
        self.calls.append((question, lang))
        return RetrievalResult(question=question, query_lang=lang, search_lang=lang,
                               strategy="same_language", chunks=self.by_query.get(question, []))


def grade(*numbers: int) -> EvidenceGrade:
    return EvidenceGrade(relevant_excerpts=list(numbers), reason="test")


def run(retriever, llm, max_rewrites: int = 1) -> dict:
    """START -> retrieve -> grade -> (answer stub | rewrite -> retrieve)"""
    agent = RetrievalAgent(retriever, llm, max_rewrites=max_rewrites)
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", agent.retrieve)
    graph.add_node("grade", agent.grade)
    graph.add_node("rewrite", agent.rewrite)
    graph.add_node("answer", lambda state: {})  # stand-in for task 3.4
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", agent.route_after_grade, ["answer", "rewrite"])
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("answer", END)
    triage = TriageResult(in_scope=True, language="es", intent="troubleshooting",
                          search_query="bomba ruido", reason="t")
    return graph.compile().invoke({"question": "La bomba hace ruido, ¿qué reviso?",
                                   "triage": triage, "search_query": "bomba ruido",
                                   "trace": []})


def nodes(final) -> list[str]:
    return [step.node for step in final["trace"]]


def test_relevant_evidence_goes_straight_to_answer() -> None:
    retriever = ScriptedRetriever({"bomba ruido": [ANSWER, NOISE, ROW]})
    llm = FakeAgentLLM(grade(1))

    final = run(retriever, llm)

    assert final["relevant_chunks"] == [ANSWER]   # noise and headerless row dropped
    assert nodes(final) == ["retrieve", "grade"]
    assert retriever.calls == [("bomba ruido", "es")]  # triage query + triage language


def test_nothing_relevant_triggers_one_rewrite_then_succeeds() -> None:
    retriever = ScriptedRetriever({"bomba ruido": [NOISE],
                                   "la bomba hace ruido: causas": [ANSWER]})
    llm = FakeAgentLLM(grade(), RewrittenQuery(search_query="la bomba hace ruido: causas"),
                       grade(1))

    final = run(retriever, llm)

    assert final["relevant_chunks"] == [ANSWER]
    assert final["rewrites"] == 1
    assert nodes(final) == ["retrieve", "grade", "rewrite", "retrieve", "grade"]
    assert llm.schemas_called == ["EvidenceGrade", "RewrittenQuery", "EvidenceGrade"]
    rewrite_prompt = llm.calls[1][1][1].content
    assert rewrite_prompt.endswith("Write the new query in Spanish.")  # explicit target language


def test_loop_is_bounded_when_nothing_is_ever_relevant() -> None:
    retriever = ScriptedRetriever({"bomba ruido": [NOISE], "otra consulta": [NOISE]})
    llm = FakeAgentLLM(grade(), RewrittenQuery(search_query="otra consulta"), grade())

    final = run(retriever, llm, max_rewrites=1)

    assert final["relevant_chunks"] == []
    assert nodes(final) == ["retrieve", "grade", "rewrite", "retrieve", "grade"]  # then answer
    assert len(llm.calls) == 3


def test_empty_retrieval_is_not_sent_to_the_llm() -> None:
    retriever = ScriptedRetriever({})  # every query returns nothing
    llm = FakeAgentLLM(RewrittenQuery(search_query="otra"))

    final = run(retriever, llm)

    assert llm.schemas_called == ["RewrittenQuery"]  # grade skipped twice, only the rewrite
    assert final["relevant_chunks"] == []


def test_grading_failure_keeps_all_chunks() -> None:
    retriever = ScriptedRetriever({"bomba ruido": [ANSWER, NOISE]})

    final = run(retriever, FakeAgentLLM())  # no scripted response -> grading raises

    assert final["relevant_chunks"] == [ANSWER, NOISE]
    assert "fallback" in final["trace"][1].summary


def test_out_of_range_and_duplicate_excerpt_numbers_are_ignored() -> None:
    retriever = ScriptedRetriever({"bomba ruido": [ANSWER, NOISE]})

    final = run(retriever, FakeAgentLLM(grade(1, 1, 7, 0)))

    assert final["relevant_chunks"] == [ANSWER]


def test_rewrite_failure_retries_with_original_question() -> None:
    question = "La bomba hace ruido, ¿qué reviso?"
    retriever = ScriptedRetriever({"bomba ruido": [NOISE], question: [ANSWER]})
    llm = FakeAgentLLM(grade())  # rewrite call will raise, then grading raises -> keep all

    final = run(retriever, llm)

    assert retriever.calls[1] == (question, "es")
    assert final["relevant_chunks"] == [ANSWER]


def test_grade_prompt_uses_original_question_and_labelled_excerpts() -> None:
    retriever = ScriptedRetriever({"bomba ruido": [ANSWER]})
    llm = FakeAgentLLM(grade(1))

    run(retriever, llm)

    system, user = llm.calls[0][1]
    assert system.content == GRADE_SYSTEM
    assert "<question>\nLa bomba hace ruido, ¿qué reviso?\n</question>" in user.content
    assert "[Excerpt 1 | p. 33 | m.pdf | lang=es]" in user.content


def test_over_budget_skips_the_rewrite() -> None:
    from pool_assistant.agents.graph import MultiAgentRAG

    retriever = ScriptedRetriever({"bomba ruido": [NOISE]})
    triage = TriageResult(in_scope=True, language="es", intent="troubleshooting",
                          search_query="bomba ruido", reason="t")
    llm = FakeAgentLLM(triage, grade())  # no rewrite response scripted: must not be needed
    rag = MultiAgentRAG(retriever, llm, model_name="fake", time_budget_s=1e-9)

    answer = rag.answer("bomba ruido")

    assert answer.insufficient_evidence
    assert "rewrite" not in [s.node for s in answer.agent_trace]
    assert "time budget exhausted: no rewrite" in answer.agent_trace[2].summary
