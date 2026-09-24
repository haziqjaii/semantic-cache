"""
FastAPI dependencies for injecting singletons (Engine, Provider, Config).
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from semcache.cache.engine import CacheEngine
from semcache.cache.policy import CachePolicy
from semcache.cache.store.redis_store import RedisVectorStore
from semcache.config import get_settings
from semcache.embeddings.gemini import GeminiEmbedder
from semcache.providers.base import LLMProvider
from semcache.providers.gemini import GeminiProvider

# Global references for our singletons
_engine: CacheEngine | None = None
_provider: LLMProvider | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.
    Runs on startup, yields control to the app, then runs cleanup on shutdown.
    """
    global _engine, _provider

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
        TTLTier.LONG
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

    yield  # App runs here

    # Cleanup on shutdown
    await store.close()


def get_engine() -> CacheEngine:
    if _engine is None:
        raise RuntimeError("CacheEngine not initialized")
    return _engine


def get_provider() -> LLMProvider:
    if _provider is None:
        raise RuntimeError("LLMProvider not initialized")
    return _provider
