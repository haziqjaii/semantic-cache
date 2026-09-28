"""
FastAPI dependencies for injecting singletons (Engine, Provider, Classifier).
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from semcache.cache.classifier import IntentClassifier
from semcache.cache.engine import CacheEngine
from semcache.cache.policy import CachePolicy, TTLTier
from semcache.cache.store.redis_store import RedisVectorStore
from semcache.config import get_settings
from semcache.embeddings.gemini import GeminiEmbedder
from semcache.metrics import CacheMetricsCollector
from semcache.providers.base import LLMProvider
from semcache.providers.gemini import GeminiProvider

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


async def gauge_refresh_loop(engine: CacheEngine) -> None:
    """
    Refresh the gauges forever (until cancelled on shutdown).

    Failures are logged once when they start and once when they recover,
    so a long Redis outage doesn't flood the log every 15 seconds.
    """
    failing = False
    while True:
        try:
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

    # 1. Initialize Vector Store
    store = RedisVectorStore(
        redis_url=settings.redis_url,
        dims=settings.embedding_dims,
    )
    await store.initialize()

    # 2. Initialize Embedder
    embedder = GeminiEmbedder(
        api_key=settings.gemini_api_key,
        model=settings.embedding_model,
        dims=settings.embedding_dims,
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

    _engine = CacheEngine(
        embedder=embedder,
        store=store,
        default_policy=default_policy,
    )

    # 4. Initialize LLM Provider
    _provider = GeminiProvider(api_key=settings.gemini_api_key)

    # 5. Initialize Intent Classifier
    _classifier = IntentClassifier(
        api_key=settings.gemini_api_key,
        model=settings.classifier_model,
        timeout_seconds=settings.classifier_timeout_seconds,
        default_policy=default_policy,
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
    await store.close()


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
