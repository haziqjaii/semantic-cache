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

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelPrice(BaseModel):
    """What a model costs, in USD per million tokens."""

    input: float = Field(ge=0)
    output: float = Field(default=0.0, ge=0)


class Settings(BaseSettings):
    """Typed, validated configuration for the semantic cache."""

    # ── Gemini API ──────────────────────────────────────────
    gemini_api_key: str  # required — no default, forces you to set it

    # ── Redis ───────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"

    # ── Embedding ───────────────────────────────────────────
    embedding_model: str = "gemini-embedding-001"
    embedding_dims: int = 768
    # How long the embedding of a text is remembered, so the same text
    # isn't sent to the embedding API twice (see embeddings/memory.py).
    # 0 turns the memory off. Default: 7 days.
    embedding_cache_ttl_seconds: int = 7 * 24 * 3600

    # ── Cache Behavior ──────────────────────────────────────
    default_similarity_threshold: float = 0.95
    default_ttl_seconds: int = 86400  # 24 hours

    # ── Classifier ─────────────────────────────────────────
    # A Gemini/Gemma model, or one of the second provider's models (then
    # asked through it, off the Gemini quota).
    classifier_model: str = "gemini-3.5-flash-lite"
    # The user never waits for the classifier (it finishes after the
    # response), so this can be generous enough for slower models.
    classifier_timeout_seconds: float = 30.0

    # ── Server ──────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Admin ───────────────────────────────────────────────
    # When set, cache invalidation requires "Authorization: Bearer <token>".
    # Unset (the default) leaves it open, which is fine for local use only.
    admin_token: str | None = None

    # ── A second model provider (optional) ──────────────────
    # Any service with an OpenAI-style chat completions API: OpenAI itself,
    # or a host serving open models (Mistral, Qwen, gpt-oss, ...) behind one
    # address and one key. With the key set, requests for a model that isn't
    # Gemini's go there; Gemini models and embeddings still use GEMINI_API_KEY,
    # and so does the intent classifier unless CLASSIFIER_MODEL is one of
    # this provider's models.
    openai_compatible_api_key: str | None = None
    # The API's address up to the version, e.g. https://host.example/v1
    # ("/chat/completions" is added to it). OpenAI's own if unset.
    openai_compatible_base_url: str | None = None
    # The models to offer from that provider, comma separated. Normally
    # unset: the list is then read from the provider itself (GET /models).
    openai_compatible_models: str | None = None

    # ── Prices for more models (optional) ───────────────────
    # Adds to (or overrides) PRICING_TABLE below without editing code, as
    # JSON in USD per million tokens:
    #   EXTRA_MODEL_PRICES={"some-model": {"input": 0.15, "output": 0.60}}
    # A model without a price still works; only its cost figures are left out.
    extra_model_prices: dict[str, ModelPrice] = {}

    # ── Tracing (optional) ──────────────────────────────────
    # Set both keys to record every request as a trace in Langfuse
    # (see tracing.py). Unset (the default) leaves tracing off.
    # With tracing on, questions and answers are sent to the Langfuse server.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    # Langfuse Cloud unless set (another region, or a self-hosted server).
    langfuse_base_url: str | None = None

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


def add_model_prices(prices: dict[str, ModelPrice]) -> None:
    """Add prices to PRICING_TABLE (from EXTRA_MODEL_PRICES, at startup)."""
    for model, price in prices.items():
        PRICING_TABLE["models"][model] = {"input": price.input, "output": price.output}


def estimate_cost_usd(
    model: str, input_tokens: int, output_tokens: int = 0
) -> dict[str, float] | None:
    """
    Estimated cost of a call in USD, split into input, output and total.

    Returns None for models missing from PRICING_TABLE, so callers can
    report them as unpriced instead of silently pricing them as some other
    model.
    """
    price = PRICING_TABLE["models"].get(model)
    if price is None:
        return None
    input_cost = input_tokens * price["input"] / 1_000_000
    output_cost = output_tokens * price["output"] / 1_000_000
    return {"input": input_cost, "output": output_cost, "total": input_cost + output_cost}


def estimate_cost_myr(model: str, input_tokens: int, output_tokens: int = 0) -> float | None:
    """Estimated cost of a call in MYR (None for a model without a price)."""
    usd = estimate_cost_usd(model, input_tokens, output_tokens)
    if usd is None:
        return None
    return usd["total"] * PRICING_TABLE["usd_to_myr"]


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

