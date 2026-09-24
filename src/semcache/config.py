"""
Application configuration loaded from environment variables.

Uses pydantic-settings to:
  1. Read from a .env file (if present)
  2. Override with actual environment variables
  3. Validate types at startup — a missing GEMINI_API_KEY fails immediately
     with a clear error, not a cryptic 401 five minutes later.

Usage:
    from semcache.config import settings
    print(settings.gemini_api_key)

Why pydantic-settings?
    Regular os.getenv() returns strings and never validates. You'd write
    `int(os.getenv("PORT", "8000"))` everywhere and pray nothing is wrong.
    pydantic-settings does that once, at startup, with proper error messages.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated configuration for the semantic cache."""

    # ── Gemini API ──────────────────────────────────────────
    gemini_api_key: str  # required — no default, forces you to set it

    # ── Redis ───────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"

    # ── Embedding ───────────────────────────────────────────
    embedding_model: str = "gemini-embedding-001"
    embedding_dims: int = 768

    # ── Cache Behavior ──────────────────────────────────────
    default_similarity_threshold: float = 0.95
    default_ttl_seconds: int = 86400  # 24 hours

    # ── Server ──────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",          # auto-load .env file from project root
        env_file_encoding="utf-8",
        case_sensitive=False,      # GEMINI_API_KEY == gemini_api_key
    )


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Why lru_cache?
        Settings reads from disk (.env file) and env vars. We only need to
        do that once. lru_cache ensures every call to get_settings() returns
        the same instance — no repeated I/O, no risk of inconsistent config
        if env vars change mid-request.
    """
    return Settings()  # type: ignore[call-arg]


# Convenience alias — import this directly in modules that need config.
# Note: This evaluates at import time. If .env is missing GEMINI_API_KEY,
# the app crashes immediately with a validation error. That's intentional.
# Fail fast > fail mysteriously.
settings = get_settings()
