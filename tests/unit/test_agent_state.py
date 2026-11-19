"""State reducers and agent models (no LLM): the contract the agents will build on."""

import pytest
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from pool_assistant.agents.models import EvidenceGrade, TriageResult
from pool_assistant.agents.state import AgentState
from pool_assistant.schemas import TraceStep
from tests.fakes import FakeAgentLLM


def test_trace_is_appended_and_other_keys_are_overwritten() -> None:
    def first(state: AgentState) -> dict:
        return {"search_query": "q1", "rewrites": 0,
                "trace": [TraceStep(agent="triage", node="first", summary="one")]}

    def second(state: AgentState) -> dict:
        return {"search_query": "q2", "rewrites": state["rewrites"] + 1,
                "trace": [TraceStep(agent="retrieval", node="second", summary="two")]}

    graph = StateGraph(AgentState)
    graph.add_node("first", first)
    graph.add_node("second", second)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)

    final = graph.compile().invoke({"question": "La bomba hace ruido", "trace": []})

    assert final["search_query"] == "q2"          # overwritten
    assert final["rewrites"] == 1
    assert [s.node for s in final["trace"]] == ["first", "second"]  # appended


def test_triage_result_rejects_unknown_intent() -> None:
    with pytest.raises(ValidationError):
        TriageResult(in_scope=True, language="es", intent="pricing", search_query="x", reason="y")


def test_fake_llm_serves_structured_and_text_responses_in_order() -> None:
    triage = TriageResult(in_scope=True, language="es", intent="troubleshooting",
                          search_query="bomba hace ruido", reason="avería")
    llm = FakeAgentLLM(triage, "respuesta [p. 33]")

    assert llm.with_structured_output(TriageResult).invoke(["msg"]) is triage
    assert llm.invoke(["msg"]).text == "respuesta [p. 33]"
    assert llm.schemas_called == ["TriageResult", "text"]


def test_fake_llm_detects_schema_mismatch() -> None:
    llm = FakeAgentLLM(EvidenceGrade(relevant_excerpts=[1], reason="ok"))

    with pytest.raises(AssertionError, match="expected TriageResult"):
        llm.with_structured_output(TriageResult).invoke(["msg"])
