"""The assembled multi-agent graph, end to end, with a scripted LLM and a scripted retriever."""

from pool_assistant.agents.graph import MultiAgentRAG
from pool_assistant.agents.models import (
    EvidenceGrade,
    RewrittenQuery,
    TriageResult,
    VerificationResult,
)
from pool_assistant.rag.simple_rag import SimpleRAG
from pool_assistant.schemas import Answer
from tests.fakes import FakeAgentLLM

from .test_retrieval_agent import ANSWER, NOISE, ScriptedRetriever


def triage(**overrides) -> TriageResult:
    values = dict(in_scope=True, language="es", intent="troubleshooting",
                  search_query="la bomba hace ruido", reason="avería")
    return TriageResult(**(values | overrides))


def grade(*numbers: int) -> EvidenceGrade:
    return EvidenceGrade(relevant_excerpts=list(numbers), reason="t")


SUPPORTED = VerificationResult(supported=True, unsupported_claims=[], explanation="ok")


def path(answer: Answer) -> list[str]:
    return [step.node for step in answer.agent_trace]


def test_happy_path_four_llm_calls() -> None:
    retriever = ScriptedRetriever({"la bomba hace ruido": [ANSWER, NOISE]})
    llm = FakeAgentLLM(triage(), grade(1), "Fija correctamente la bomba [p. 33].", SUPPORTED)

    answer = MultiAgentRAG(retriever, llm, model_name="fake").answer("bomba ruido")

    assert path(answer) == ["triage", "retrieve", "grade", "answer", "verify"]
    assert sum(s.llm_calls for s in answer.agent_trace) == 4
    assert answer.verification.status == "supported"
    assert [c.page_number for c in answer.citations] == [33]
    assert answer.language == "es"
    assert answer.latency_ms >= 0 and answer.model == "fake"


def test_out_of_scope_one_llm_call_no_search() -> None:
    retriever = ScriptedRetriever({})
    llm = FakeAgentLLM(triage(in_scope=False, intent="other", search_query=""))

    answer = MultiAgentRAG(retriever, llm, model_name="fake").answer("¿Precio?")

    assert path(answer) == ["triage", "reject"]
    assert answer.out_of_scope and retriever.calls == []


def test_rewrite_path_six_llm_calls() -> None:
    retriever = ScriptedRetriever({"la bomba hace ruido": [NOISE],
                                   "la bomba hace ruido: causas": [ANSWER]})
    rewrite = RewrittenQuery(search_query="la bomba hace ruido: causas")
    llm = FakeAgentLLM(triage(), grade(), rewrite, grade(1), "Fija la bomba [p. 33].", SUPPORTED)

    answer = MultiAgentRAG(retriever, llm, model_name="fake").answer("bomba ruido")

    assert path(answer) == ["triage", "retrieve", "grade", "rewrite", "retrieve", "grade",
                            "answer", "verify"]
    assert sum(s.llm_calls for s in answer.agent_trace) == 6


def test_no_evidence_after_rewrite_answers_not_in_manual() -> None:
    retriever = ScriptedRetriever({"la bomba hace ruido": [NOISE], "otra": [NOISE]})
    llm = FakeAgentLLM(triage(), grade(), RewrittenQuery(search_query="otra"), grade())

    answer = MultiAgentRAG(retriever, llm, model_name="fake").answer("E01")

    assert answer.insufficient_evidence
    assert path(answer)[-2:] == ["answer", "verify"]
    assert answer.verification.status == "skipped"
    assert sum(s.llm_calls for s in answer.agent_trace) == 4  # answer + verify skip the LLM


def test_same_interface_and_output_type_as_the_baseline() -> None:
    retriever = ScriptedRetriever({"bomba ruido": [ANSWER], "la bomba hace ruido": [ANSWER]})
    baseline = SimpleRAG(retriever, FakeAgentLLM("Fija la bomba [p. 33].", "Otra vez [p. 33]."),
                         model_name="fake")
    agents = MultiAgentRAG(
        retriever, FakeAgentLLM(triage(), grade(1), "Fija la bomba [p. 33].", SUPPORTED),
        model_name="fake")

    for pipeline in (baseline, agents):
        answer = pipeline.answer("bomba ruido", lang="es")
        assert isinstance(answer, Answer)
        assert [c.page_number for c in answer.citations] == [33]
    assert baseline.answer("bomba ruido").agent_trace == []  # baseline has no agent trace


def test_mermaid_diagram_matches_the_graph() -> None:
    llm = FakeAgentLLM()
    diagram = MultiAgentRAG(ScriptedRetriever({}), llm, model_name="fake").mermaid()

    for edge in ("__start__ --> triage", "retrieve --> grade", "rewrite --> retrieve",
                 "triage -.-> reject", "grade -.-> rewrite", "answer --> verify",
                 "verify -.-> answer", "verify -.-> __end__"):
        assert edge in diagram
