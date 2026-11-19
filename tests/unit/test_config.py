import pytest
from pydantic import ValidationError

from pool_assistant.config import Settings


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "fake"
    assert settings.llm_temperature == 0.2


def test_api_key_is_not_leaked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "super-secret")

    settings = Settings(_env_file=None)

    assert "super-secret" not in repr(settings)
    assert settings.google_api_key.get_secret_value() == "super-secret"


def test_empty_secrets_mean_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: `GOOGLE_API_KEY=` from .env.example produced a raw SDK traceback."""
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("QDRANT_URL", "  ")

    settings = Settings(_env_file=None)

    assert settings.google_api_key is None
    assert settings.qdrant_url is None


def test_invalid_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
