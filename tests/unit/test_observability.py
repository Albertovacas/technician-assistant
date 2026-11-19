"""Request correlation, LLM usage accounting and structured (JSON) logs."""

import json
import logging

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from pool_assistant.logging_config import JsonFormatter
from pool_assistant.observability import (
    RequestIdFilter,
    current_request_id,
    request_context,
    track_usage,
)
from pool_assistant.rag.simple_rag import SimpleRAG
from pool_assistant.schemas import RetrievalResult, ScoredChunk

from .test_vectorstore import make_chunk


def fake_llm(*texts: str, tokens: tuple[int, int] = (100, 20)) -> GenericFakeChatModel:
    """A real LangChain chat model (callbacks fire) that reports token usage."""
    messages = [AIMessage(content=t, usage_metadata={
        "input_tokens": tokens[0], "output_tokens": tokens[1],
        "total_tokens": sum(tokens)}) for t in texts]
    return GenericFakeChatModel(messages=iter(messages))


def test_request_context_sets_and_restores_the_id() -> None:
    assert current_request_id() is None
    with request_context("abc123") as rid:
        assert rid == current_request_id() == "abc123"
    assert current_request_id() is None
    with request_context() as generated:
        assert len(generated) == 12


def test_every_log_record_gets_the_request_id() -> None:
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "msg", (), None)
    with request_context("req-1"):
        RequestIdFilter().filter(record)
    assert record.request_id == "req-1"

    outside = logging.LogRecord("x", logging.INFO, __file__, 1, "msg", (), None)
    RequestIdFilter().filter(outside)
    assert outside.request_id == "-"


def test_json_formatter_includes_structured_fields() -> None:
    record = logging.LogRecord("pool_assistant.x", logging.WARNING, __file__, 1,
                               "done in %d ms", (42,), None)
    record.request_id, record.event, record.llm_calls = "req-1", "request_completed", 4

    payload = json.loads(JsonFormatter().format(record))

    assert payload["severity"] == "WARNING"
    assert payload["message"] == "done in 42 ms"
    assert payload["request_id"] == "req-1"
    assert payload["event"] == "request_completed" and payload["llm_calls"] == 4


def test_usage_of_every_llm_call_is_collected() -> None:
    llm = fake_llm("a", "b", tokens=(100, 20))

    with track_usage() as usage:
        llm.invoke("hello")
        llm.invoke("hello again")

    summary = usage.summary()
    assert (summary.llm_calls, summary.input_tokens, summary.output_tokens) == (2, 200, 40)


def test_collector_is_hashable() -> None:
    """Regression: a @dataclass collector was unhashable and broke every structured call.

    LangChain merges callback handlers in a set() for chained runnables (what
    `with_structured_output` does); plain-text calls did not exercise that path, and the
    fakes used in the agent tests are not LangChain models. Found in a live run.
    """
    from pool_assistant.observability import UsageCollector

    assert hash(UsageCollector()) is not None


def test_calls_outside_the_block_are_not_collected() -> None:
    llm = fake_llm("a", "b")
    with track_usage() as usage:
        llm.invoke("inside")
    llm.invoke("outside")

    assert usage.summary().llm_calls == 1


def test_usage_is_attributed_to_the_graph_node() -> None:
    llm = fake_llm("x", tokens=(50, 10))
    graph = StateGraph(dict)
    graph.add_node("grade", lambda state: {"out": llm.invoke("grade this").text})
    graph.add_edge(START, "grade")
    graph.add_edge("grade", END)

    with track_usage() as usage:
        graph.compile().invoke({"in": 1})

    assert usage.summary().tokens_by_node == {"grade": 60}


def test_answers_carry_request_id_usage_and_a_completion_event(caplog) -> None:
    class OneChunkRetriever:
        def retrieve(self, question, *, lang=None):
            chunk = ScoredChunk(chunk=make_chunk("m", 33, 0, "Fijar la bomba", lang="es"),
                                score=0.5)
            return RetrievalResult(question=question, query_lang="es", search_lang="es",
                                   strategy="same_language", chunks=[chunk])

    rag = SimpleRAG(OneChunkRetriever(), fake_llm("Fija la bomba [p. 33]."), model_name="fake")

    with caplog.at_level(logging.INFO, logger="pool_assistant"):
        answer = rag.answer("La bomba hace ruido", request_id="req-42")

    assert answer.request_id == "req-42"
    assert (answer.usage.llm_calls, answer.usage.input_tokens) == (1, 100)
    event = next(r for r in caplog.records if getattr(r, "event", "") == "request_completed")
    assert event.mode == "simple" and event.citations == [33] and event.llm_calls == 1
