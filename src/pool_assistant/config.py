"""Centralised, typed configuration loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    llm_provider: Literal["gemini", "fake"] = "gemini"
    llm_model: str = "gemini-3.1-flash-lite"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_timeout_s: float = Field(default=30.0, gt=0)
    # Total attempts per call (google-genai SDK): exponential backoff with jitter (1s, 2s, 4s...)
    # on 408/429/500/502/503/504 and transient network errors. 2 = one retry.
    llm_max_retries: int = Field(default=2, ge=0)
    # Client-side throttle shared by all agents (0 = off). Gemini free tier allows 15/min per
    # model: exceeding it returns 429 RESOURCE_EXHAUSTED (hit while running notebook 09).
    llm_requests_per_minute: int = Field(default=14, ge=0)
    llm_max_output_tokens: int = Field(default=1024, gt=0)
    # Gemini 2.5 "thinking" tokens: 0 disables it (lower latency/cost), -1 = dynamic.
    llm_thinking_budget: int = Field(default=0, ge=-1)
    google_api_key: SecretStr | None = None

    # --- Vision fallback (ingestion of complex / scanned pages) ---
    # Measured on the troubleshooting matrix (p.13, 5 runs): flash-lite without thinking
    # 8/11 relations correct every run; with a 2048-token thinking budget 11/11 every run.
    vision_model: str = "gemini-3.1-flash-lite"
    vision_thinking_budget: int = Field(default=2048, ge=-1)
    vision_max_output_tokens: int = Field(default=8192, gt=0)
    vision_timeout_s: float = Field(default=120.0, gt=0)
    vision_render_scale: float = Field(default=2.0, gt=0)  # 2.0 ~ 144 dpi
    # Reject a transcription that misses too many words of the page's own text layer.
    vision_min_text_coverage: float = Field(default=0.6, ge=0.0, le=1.0)

    # embeddinggemma-300m: best MRR same-language (0.86) and cross-lingual ES->EN (0.78);
    # half the size of multilingual-e5-large (1.2 vs 2.2 GB), 2.6x faster to index.
    embedding_model: str = "google/embeddinggemma-300m"
    embedding_batch_size: int = Field(default=32, gt=0)

    # --- Multi-agent ---
    # Time budget per question: once exceeded, optional steps (query rewrite, answer retry
    # after a failed verification) are skipped. Bounds worst-case latency; safety is kept
    # (an unverified answer still falls back to the safe message).
    agent_time_budget_s: float = Field(default=45.0, gt=0)

    # --- Retrieval ---
    # top-k = 5: in the embedding benchmark every miss of embeddinggemma was at rank 4.
    retrieval_k: int = Field(default=5, gt=0)
    # Language searched when the question's language has no chunks (most manuals exist in EN).
    retrieval_fallback_lang: str = "en"
    # Cross-encoder reranking: search `rerank_candidates` chunks, keep the best `retrieval_k`.
    rerank_model: str | None = "jinaai/jina-reranker-v2-base-multilingual"
    rerank_candidates: int = Field(default=20, gt=0)

    # --- Vector store (Qdrant) ---
    # Local embedded mode by default (no server); set QDRANT_URL to use a Qdrant server.
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "manuals"

    # --- Paths ---
    data_dir: Path = PROJECT_ROOT / "data"
    vision_cache_dir: Path = PROJECT_ROOT / "data" / "vision_cache"
    embedding_cache_dir: Path = PROJECT_ROOT / "data" / ".cache" / "fastembed"
    qdrant_path: Path = PROJECT_ROOT / "data" / "index" / "qdrant"

    # --- Observability ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["text", "json"] = "text"   # json: one object per line (Cloud Logging)

    @field_validator("google_api_key", "qdrant_api_key", "qdrant_url", "rerank_model",
                     mode="before")
    @classmethod
    def _empty_means_unset(cls, value):
        """`GOOGLE_API_KEY=` (as copied from .env.example) must mean "not configured".

        Otherwise an empty string reaches the provider SDK and the user gets a raw validation
        traceback instead of our clear "set GOOGLE_API_KEY" message (found in a clean-copy run).
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (one per process)."""
    return Settings()
