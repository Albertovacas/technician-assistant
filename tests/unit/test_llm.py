import pytest

from pool_assistant.config import Settings
from pool_assistant.llm import LLMConfigError, get_llm


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_fake_provider_returns_canned_responses_in_order() -> None:
    llm = get_llm(make_settings(llm_provider="fake"), fake_responses=["first", "second"])

    assert llm.invoke("hi").content == "first"
    assert llm.invoke("hi").content == "second"


def test_gemini_without_api_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(LLMConfigError, match="GOOGLE_API_KEY"):
        get_llm(make_settings(llm_provider="gemini"))


def test_gemini_is_built_from_settings_without_network_call() -> None:
    settings = make_settings(
        llm_provider="gemini",
        google_api_key="dummy-key",
        llm_model="gemini-3.1-flash-lite",
        llm_timeout_s=12,
    )

    llm = get_llm(settings)

    assert type(llm).__name__ == "ChatGoogleGenerativeAI"
    assert llm.model.endswith("gemini-3.1-flash-lite")
    assert llm.timeout == 12


def test_all_models_share_one_rate_limiter() -> None:
    """The provider quota is per project/model, so every agent must draw from one budget."""
    settings = make_settings(llm_provider="gemini", google_api_key="dummy-key",
                             llm_requests_per_minute=14)

    chat, vision = get_llm(settings), get_llm(settings, model="gemini-3.1-flash-lite",
                                              thinking_budget=2048)

    assert chat.rate_limiter is not None
    assert chat.rate_limiter is vision.rate_limiter


def test_rate_limiter_can_be_disabled() -> None:
    settings = make_settings(llm_provider="gemini", google_api_key="dummy-key",
                             llm_requests_per_minute=0)

    assert get_llm(settings).rate_limiter is None


def test_provider_retries_and_timeout_come_from_settings() -> None:
    """Retries with exponential backoff on 408/429/5xx are done by the google-genai SDK."""
    settings = make_settings(llm_provider="gemini", google_api_key="dummy-key",
                             llm_max_retries=3, llm_timeout_s=20)

    llm = get_llm(settings)

    assert llm.max_retries == 3
    assert llm.timeout == 20
