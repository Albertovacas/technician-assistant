"""Request correlation and LLM usage accounting.

- `request_context()`: sets a request id in a ContextVar; a logging filter stamps it on
  every log record emitted while handling that request, from any module.
- `track_usage()`: collects tokens and latency of every LLM call made inside the block.
  The callback is registered with LangChain's `register_configure_hook` (the mechanism
  behind `get_openai_callback`), so it attaches itself to all chat-model runs - agents,
  baseline, structured output - without passing callbacks around. Inside LangGraph, the
  run metadata names the node, so usage is attributed per agent node.

Note: latency per call includes waiting for the client-side rate limiter.
"""

import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.tracers.context import register_configure_hook

from pool_assistant.schemas import Usage

logger = logging.getLogger(__name__)

_request_id: ContextVar[str | None] = ContextVar("pool_assistant_request_id", default=None)


def current_request_id() -> str | None:
    return _request_id.get()


@contextmanager
def request_context(request_id: str | None = None) -> Iterator[str]:
    """Bind a request id (new one by default) to everything that runs inside the block."""
    rid = request_id or uuid.uuid4().hex[:12]
    token = _request_id.set(rid)
    try:
        yield rid
    finally:
        _request_id.reset(token)


class RequestIdFilter(logging.Filter):
    """Adds `request_id` to every record ("-" outside a request)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get() or "-"
        return True


# --- LLM usage ------------------------------------------------------------------------------
@dataclass
class _CallStart:
    started: float
    node: str
    model: str


class UsageCollector(BaseCallbackHandler):
    """LangChain callback: one entry per chat-model call (tokens, latency, graph node).

    A plain class on purpose: LangChain merges callback handlers in a `set()` when runnables
    are chained (e.g. `with_structured_output`), so handlers must stay hashable. A
    `@dataclass` defines `__eq__` and therefore sets `__hash__ = None`; that broke every
    structured call ("unhashable type") while plain text calls still worked.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._running: dict = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None,
                            **kwargs: Any) -> None:
        metadata = metadata or {}
        model = ((serialized or {}).get("kwargs", {}).get("model")
                 or metadata.get("ls_model_name") or "unknown")
        self._running[run_id] = _CallStart(time.perf_counter(),
                                           metadata.get("langgraph_node", "-"), str(model))

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs: Any) -> None:
        start = self._running.pop(run_id, None)
        usage = {}
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) or usage
        call = {
            "node": start.node if start else "-",
            "model": start.model if start else "unknown",
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "latency_ms": int((time.perf_counter() - start.started) * 1000) if start else 0,
        }
        self.calls.append(call)
        logger.info("llm_call node=%s in=%d out=%d %d ms", call["node"], call["input_tokens"],
                    call["output_tokens"], call["latency_ms"], extra={"event": "llm_call", **call})

    def on_llm_error(self, error: BaseException, *, run_id, **kwargs: Any) -> None:
        start = self._running.pop(run_id, None)
        logger.warning("llm_error node=%s %s: %s", start.node if start else "-",
                       type(error).__name__, error,
                       extra={"event": "llm_error", "node": start.node if start else "-"})

    def summary(self) -> Usage:
        by_node: dict[str, int] = {}
        for call in self.calls:
            by_node[call["node"]] = (by_node.get(call["node"], 0)
                                     + call["input_tokens"] + call["output_tokens"])
        return Usage(
            llm_calls=len(self.calls),
            input_tokens=sum(c["input_tokens"] for c in self.calls),
            output_tokens=sum(c["output_tokens"] for c in self.calls),
            tokens_by_node=by_node,
        )


_usage_collector: ContextVar[UsageCollector | None] = ContextVar(
    "pool_assistant_usage", default=None
)
register_configure_hook(_usage_collector, inheritable=True)


@contextmanager
def track_usage() -> Iterator[UsageCollector]:
    """Collect usage of every LLM call made inside the block (nested runs included)."""
    collector = UsageCollector()
    token = _usage_collector.set(collector)
    try:
        yield collector
    finally:
        _usage_collector.reset(token)


def log_request_completed(answer, mode: str) -> None:
    """One structured event per answered question: the basis for metrics and alerts."""
    usage = answer.usage
    fields = {
        "event": "request_completed",
        "mode": mode,
        "latency_ms": answer.latency_ms,
        "language": answer.language,
        "strategy": answer.retrieval.strategy if answer.retrieval else None,
        "citations": [c.page_number for c in answer.citations],
        "insufficient_evidence": answer.insufficient_evidence,
        "out_of_scope": answer.out_of_scope,
        "degraded": answer.degraded,
        "verification": answer.verification.status if answer.verification else None,
        "llm_calls": usage.llm_calls if usage else None,
        "input_tokens": usage.input_tokens if usage else None,
        "output_tokens": usage.output_tokens if usage else None,
    }
    logger.info("request_completed mode=%s %d ms calls=%s tokens=%s", mode, answer.latency_ms,
                fields["llm_calls"],
                (usage.input_tokens + usage.output_tokens) if usage else None, extra=fields)
