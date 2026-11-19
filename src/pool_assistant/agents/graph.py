"""The multi-agent workflow, assembled in one place.

    START ─► triage ─┬─ out of scope ─► reject ───────────────────────────────────► END
                     └─ in scope ────► retrieve ─► grade ─┬─ evidence ─► answer ─► verify ─┬─► END
                                          ▲               ├─ none, retry ─► rewrite ─┐  ▲    │
                                          └───────────────┼──────────────────────────┘  │    │
                                                          └─ none, exhausted ─► answer  └────┘
                                                                         unsupported: retry once

Four agents with distinct responsibilities (triage, retrieval, answer, verifier)
collaborating through a typed shared state. Orchestration is deterministic: edges read
structured fields produced by the agents; there is no LLM supervisor. Cost is bounded:
1 call (out of scope), 4 (normal), up to 8 (one query rewrite + one answer retry).
"""

import time

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from pool_assistant.agents.answer import AnswerAgent
from pool_assistant.agents.retrieval import RetrievalAgent, SupportsRetrieve
from pool_assistant.agents.state import AgentState
from pool_assistant.agents.triage import TriageAgent, route_after_triage
from pool_assistant.agents.verifier import VerifierAgent
from pool_assistant.observability import log_request_completed, request_context, track_usage
from pool_assistant.schemas import Answer


def build_agent_graph(triage: TriageAgent, retrieval: RetrievalAgent, answer: AnswerAgent,
                      verifier: VerifierAgent):
    graph = StateGraph(AgentState)
    graph.add_node("triage", triage.triage)
    graph.add_node("reject", triage.reject)
    graph.add_node("retrieve", retrieval.retrieve)
    graph.add_node("grade", retrieval.grade)
    graph.add_node("rewrite", retrieval.rewrite)
    graph.add_node("answer", answer.answer)
    graph.add_node("verify", verifier.verify)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges("triage", route_after_triage, ["retrieve", "reject"])
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", retrieval.route_after_grade, ["answer", "rewrite"])
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("answer", "verify")
    graph.add_conditional_edges("verify", verifier.route_after_verify, ["answer", END])
    graph.add_edge("reject", END)
    return graph.compile()


class MultiAgentRAG:
    """Same interface and output as `SimpleRAG`, so CLI, API and evaluation treat both alike."""

    def __init__(self, retriever: SupportsRetrieve, llm: BaseChatModel, *, model_name: str,
                 max_rewrites: int = 1, max_answer_retries: int = 1,
                 time_budget_s: float | None = None) -> None:
        self.model_name = model_name
        self.time_budget_s = time_budget_s  # None = no budget (optional steps always run)
        self.graph = build_agent_graph(
            TriageAgent(llm, model_name=model_name),
            RetrievalAgent(retriever, llm, max_rewrites=max_rewrites),
            AnswerAgent(llm, model_name=model_name),
            VerifierAgent(llm, max_answer_retries=max_answer_retries),
        )

    def answer(self, question: str, *, lang: str | None = None,
               request_id: str | None = None) -> Answer:
        with request_context(request_id) as rid, track_usage() as usage:
            start = time.perf_counter()
            deadline = (time.monotonic() + self.time_budget_s) if self.time_budget_s else None
            final = self.graph.invoke({"question": question, "lang_hint": lang,
                                       "deadline": deadline, "trace": []})
            answer = final["answer"].model_copy(update={
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "agent_trace": final["trace"],
                "request_id": rid,
                "usage": usage.summary(),
            })
            log_request_completed(answer, mode="agents")
            return answer

    def mermaid(self) -> str:
        """Diagram generated from the compiled graph itself (cannot drift from the code)."""
        return self.graph.get_graph().draw_mermaid()
