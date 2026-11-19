"""Test doubles shared by several test modules."""

from collections import deque

from langchain_core.messages import AIMessage
from pydantic import BaseModel


class FakeAgentLLM:
    """Scripted chat model supporting `with_structured_output`.

    LangChain's FakeListChatModel cannot return structured objects, which the agents need.
    Responses are consumed in order: a Pydantic object answers a structured call (its type
    must match the requested schema), a string answers a plain `invoke`. Every call is
    recorded so tests can assert on the prompts.
    """

    def __init__(self, *responses: BaseModel | str) -> None:
        self.responses = deque(responses)
        self.calls: list[tuple[type | None, list]] = []  # (schema or None, messages)

    def _next(self, schema: type | None, messages):
        self.calls.append((schema, messages))
        if not self.responses:
            raise AssertionError(f"FakeAgentLLM: no scripted response left for {schema}")
        response = self.responses.popleft()
        if schema is not None and not isinstance(response, schema):
            raise AssertionError(f"FakeAgentLLM: expected {schema.__name__}, got {response!r}")
        return response

    def invoke(self, messages):
        return AIMessage(content=self._next(None, messages))

    def with_structured_output(self, schema: type[BaseModel]):
        llm = self

        class _Structured:
            def invoke(self, messages):
                return llm._next(schema, messages)

        return _Structured()

    @property
    def schemas_called(self) -> list[str]:
        return [schema.__name__ if schema else "text" for schema, _ in self.calls]
