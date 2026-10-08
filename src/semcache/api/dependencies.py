"""
FastAPI dependencies for injecting singletons (Engine, Provider, Classifier).
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from semcache import tracing
from semcache.cache.classifier import IntentClassifier
from semcache.cache.engine import CacheEngine
from semcache.cache.lookup_log import RedisLookupLog
from semcache.cache.policy import CachePolicy, TTLTier
from semcache.cache.store.redis_store import RedisVectorStore
from semcache.config import add_model_prices, get_settings
from semcache.embeddings.base import Embedder
from semcache.embeddings.gemini import GeminiEmbedder
from semcache.embeddings.memory import CachedEmbedder, RedisEmbeddingStore
from semcache.metrics import CacheMetricsCollector
from semcache.providers.base import LLMProvider
from semcache.providers.gemini import GeminiProvider
from semcache.providers.openai_compatible import (
    DEFAULT_BASE_URL,
    OpenAICompatibleProvider,
)
from semcache.providers.router import RoutingProvider, is_google_model
from semcache.providers.traced import TracedProvider

logger = logging.getLogger(__name__)

GAUGE_REFRESH_SECONDS = 15

# Global references for our singletons
_engine: CacheEngine | None = None
_provider: LLMProvider | None = None
_classifier: IntentClassifier | None = None


async def refresh_cache_gauges(engine: CacheEngine) -> None:
    """Copy live store stats into the Prometheus gauges."""
    stats = await engine.stats()
    CacheMetricsCollector._cache_entries = stats.get("total_entries", 0)
    CacheMetricsCollector._evicted_keys = stats.get("evicted_keys", 0)
    CacheMetricsCollector._expired_keys = stats.get("expired_keys", 0)
    CacheMetricsCollector._thresholds = engine.thresholds_in_use()


async def gauge_refresh_loop(engine: CacheEngine) -> None:
    """
    Refresh the gauges and learned thresholds forever (until cancelled on shutdown).

    Re-learning here keeps every server process in step, even though
    feedback arrives at only one of them.

    Failures are logged once when they start and once when they recover,
    so a long Redis outage doesn't flood the log every 15 seconds.
    """
    failing = False
    while True:
        try:
            await engine.refresh_learned_thresholds()
            await refresh_cache_gauges(engine)
        except Exception:
            if not failing:
                logger.exception(
                    "Failed to refresh cache gauges; retrying every %ds",
                    GAUGE_REFRESH_SECONDS,
                )
            failing = True
        else:
            if failing:
                logger.info("Cache gauge refresh recovered")
            failing = False
        await asyncio.sleep(GAUGE_REFRESH_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.
    Runs on startup, yields control to the app, then runs cleanup on shutdown.
    """
    global _engine, _provider, _classifier

    settings = get_settings()
    if settings.admin_token is None:
        logger.warning(
            "ADMIN_TOKEN is not set: /v1/cache/invalidate is open to anyone who can "
            "reach this server. Set it before exposing the proxy beyond localhost."
        )

    # 1. Initialize Vector Store
    store = RedisVectorStore(
        redis_url=settings.redis_url,
        dims=settings.embedding_dims,
    )
    await store.initialize()

    # 2. Initialize Embedder, wrapped in the embedding memory so a text
    # seen before isn't sent to the embedding API again.
    embedder: Embedder = GeminiEmbedder(
        api_key=settings.gemini_api_key,
        model=settings.embedding_model,
        dims=settings.embedding_dims,
    )
    embedding_store: RedisEmbeddingStore | None = None
    if settings.embedding_cache_ttl_seconds > 0:
        embedding_store = RedisEmbeddingStore(redis_url=settings.redis_url)
        await embedding_store.initialize()
        embedder = CachedEmbedder(
            embedder,
            embedding_store,
            identity=f"{settings.embedding_model}:{settings.embedding_dims}",
            ttl_seconds=settings.embedding_cache_ttl_seconds,
        )

    # Map the configured TTL seconds to the closest TTLTier
    tier = next(
        (t for t in TTLTier if t.value == settings.default_ttl_seconds),
        TTLTier.LONG,
    )

    # 3. Build Engine with default policy from settings
    default_policy = CachePolicy(
        similarity_threshold=settings.default_similarity_threshold,
        ttl_tier=tier,
    )

    # Records every lookup for the near-miss analyzer, threshold tuner,
    # and thresholds learned from feedback.
    lookup_log = RedisLookupLog(redis_url=settings.redis_url)
    await lookup_log.initialize()

    _engine = CacheEngine(
        embedder=embedder,
        store=store,
        default_policy=default_policy,
        lookup_log=lookup_log,
    )
    learned = await _engine.refresh_learned_thresholds()
    if learned:
        logger.info(
            "Using thresholds learned from feedback: %s",
            {intent: lt.threshold for intent, lt in learned.items()},
        )

    # Prices for models missing from PRICING_TABLE (EXTRA_MODEL_PRICES).
    add_model_prices(settings.extra_model_prices)

    # Optional Langfuse tracing: on only when both keys are set.
    tracing_on = tracing.configure(
        settings.langfuse_public_key, settings.langfuse_secret_key, settings.langfuse_base_url
    )

    # 4. Initialize LLM Provider (recording each call on the request's trace
    # when tracing is on)
    _provider = GeminiProvider(api_key=settings.gemini_api_key)
    # Optional second provider: with its key set, every model that isn't
    # Gemini's is answered by it (see providers/router.py). Embeddings stay
    # on Gemini either way; the intent classifier does unless CLASSIFIER_MODEL
    # names one of this provider's models.
    openai_compatible: OpenAICompatibleProvider | None = None
    if settings.openai_compatible_api_key:
        listed = settings.openai_compatible_models
        openai_compatible = OpenAICompatibleProvider(
            api_key=settings.openai_compatible_api_key,
            base_url=settings.openai_compatible_base_url or DEFAULT_BASE_URL,
            models=[m.strip() for m in listed.split(",") if m.strip()] if listed else None,
        )
        _provider = RoutingProvider(google=_provider, other=openai_compatible)
        logger.info(
            "Non-Gemini models are served by %s",
            settings.openai_compatible_base_url or DEFAULT_BASE_URL,
        )
    if tracing_on:
        _provider = TracedProvider(_provider)

    # 5. Initialize Intent Classifier (on the second provider when
    # CLASSIFIER_MODEL isn't a Gemini model)
    if not is_google_model(settings.classifier_model) and openai_compatible is None:
        logger.warning(
            "CLASSIFIER_MODEL=%r isn't a Gemini model, but OPENAI_COMPATIBLE_API_KEY "
            "is not set: every classification will fall back to the default policy.",
            settings.classifier_model,
        )
    _classifier = IntentClassifier(
        api_key=settings.gemini_api_key,
        model=settings.classifier_model,
        timeout_seconds=settings.classifier_timeout_seconds,
        default_policy=default_policy,
        other_provider=openai_compatible,
    )

    # 6. Start background task for Prometheus gauges
    task = asyncio.create_task(gauge_refresh_loop(_engine))

    yield  # App runs here

    # Cleanup on shutdown
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    if embedding_store is not None:
        await embedding_store.close()
    await lookup_log.close()
    await store.close()
    if openai_compatible is not None:
        await openai_compatible.close()
    tracing.shutdown()  # sends any traces still waiting


def get_engine() -> CacheEngine:
    if _engine is None:
        raise RuntimeError("CacheEngine not initialized")
    return _engine


def get_provider() -> LLMProvider:
    if _provider is None:
        raise RuntimeError("LLMProvider not initialized")
    return _provider


def get_classifier() -> IntentClassifier:
    if _classifier is None:
        raise RuntimeError("IntentClassifier not initialized")
    return _classifier
