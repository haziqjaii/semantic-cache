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

    # ── Classifier ─────────────────────────────────────────
    classifier_model: str = "gemini-3.5-flash-lite"
    classifier_timeout_seconds: float = 5.0

    # ── Server ──────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Admin ───────────────────────────────────────────────
    # When set, cache invalidation requires "Authorization: Bearer <token>".
    # Unset (the default) leaves it open, which is fine for local use only.
    admin_token: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",          # auto-load .env file from project root
        env_file_encoding="utf-8",
        case_sensitive=False,      # GEMINI_API_KEY == gemini_api_key
    )

# Static pricing table used for all cost figures (/v1/analytics, /metrics).
# Model prices are in USD per million tokens, as the providers publish them.
# Every cost we report is converted to Malaysian ringgit (MYR) using
# usd_to_myr, so update that rate along with the prices.
PRICING_TABLE = {
    "as_of": "2026-09-28",
    "currency": "MYR",
    "usd_to_myr": 4.07,
    "usd_to_myr_as_of": "2026-09-25",
    "models": {
        "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
        "gemini-3.5-flash-lite": {"input": 0.075, "output": 0.30},
        "gemini-embedding-001": {"input": 0.15, "output": 0.0},
        "text-embedding-004": {"input": 0.15, "output": 0.0},
    },
}


def estimate_cost_myr(model: str, input_tokens: int, output_tokens: int = 0) -> float | None:
    """
    Estimated cost of a call in MYR, from PRICING_TABLE's USD prices.

    Returns None for models missing from the table, so callers can report
    them as unpriced instead of silently pricing them as some other model.
    """
    price = PRICING_TABLE["models"].get(model)
    if price is None:
        return None
    usd = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
    return usd * PRICING_TABLE["usd_to_myr"]


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

