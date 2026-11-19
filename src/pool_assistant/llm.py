"""LLM factory: the only place that knows which chat model provider is in use.

Agents depend on LangChain's `BaseChatModel` interface, never on a concrete provider,
so swapping Gemini for another model (or a fake in tests) is a config change.
"""

import logging
from collections.abc import Sequence
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from pool_assistant.config import Settings, get_settings

logger = logging.getLogger(__name__)

DEFAULT_FAKE_RESPONSES = ["[fake-llm] This is a canned response for offline runs."]


class LLMConfigError(RuntimeError):
    """Raised when the LLM cannot be built from the current configuration."""


@lru_cache
def shared_rate_limiter(requests_per_minute: int) -> InMemoryRateLimiter:
    """One limiter per process and rate, shared by every model built here.

    The provider quota is per project and model (free tier: 15 requests/min), not per agent,
    so triage, grading, rewrite, answer and vision must draw from the same budget.
    """
    return InMemoryRateLimiter(
        requests_per_second=requests_per_minute / 60,
        check_every_n_seconds=0.1,
        max_bucket_size=1,  # no bursts: spread calls evenly
    )


def get_llm(
    settings: Settings | None = None,
    *,
    model: str | None = None,
    thinking_budget: int | None = None,
    max_output_tokens: int | None = None,
    timeout_s: float | None = None,
    fake_responses: Sequence[str] | None = None,
) -> BaseChatModel:
    """Build the chat model selected by `settings.llm_provider`.

    Args:
        settings: Configuration to use; defaults to the process-wide settings.
        model, thinking_budget, max_output_tokens, timeout_s: per-use overrides
            (e.g. the vision parser needs a thinking budget and longer outputs).
        fake_responses: Canned responses for the fake provider (cycled in order).
    """
    settings = settings or get_settings()
    model = model or settings.llm_model
    logger.info("Building LLM provider=%s model=%s", settings.llm_provider, model)

    if settings.llm_provider == "fake":
        return FakeListChatModel(responses=list(fake_responses or DEFAULT_FAKE_RESPONSES))

    if settings.llm_provider == "gemini":
        if settings.google_api_key is None:
            raise LLMConfigError(
                "LLM_PROVIDER=gemini requires GOOGLE_API_KEY (set it in .env) "
                "or use LLM_PROVIDER=fake for offline runs."
            )
        # Imported lazily so the fake provider works without Google SDK side effects.
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=settings.llm_temperature,
            timeout=timeout_s or settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
            max_output_tokens=max_output_tokens or settings.llm_max_output_tokens,
            thinking_budget=(
                settings.llm_thinking_budget if thinking_budget is None else thinking_budget
            ),
            google_api_key=settings.google_api_key,
            rate_limiter=(shared_rate_limiter(settings.llm_requests_per_minute)
                          if settings.llm_requests_per_minute else None),
        )

    # Unreachable thanks to the Literal type in Settings, but keeps the function total.
    raise LLMConfigError(f"Unsupported LLM provider: {settings.llm_provider}")


def get_vision_llm(settings: Settings | None = None) -> BaseChatModel:
    """Multimodal model used by the ingestion vision fallback."""
    settings = settings or get_settings()
    return get_llm(
        settings,
        model=settings.vision_model,
        thinking_budget=settings.vision_thinking_budget,
        max_output_tokens=settings.vision_max_output_tokens,
        timeout_s=settings.vision_timeout_s,
    )
