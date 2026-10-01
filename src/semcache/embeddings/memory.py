"""
Embedding memory — remembers the embedding of every text it has seen.

WHY
    Every cacheable request starts by turning the question into an
    embedding, which is a call to the Gemini Embedding API. But the same
    text keeps coming back: in the load test, 2,000 requests contain only
    419 different wordings. Embedding "What is the capital of Japan?" a
    second time returns exactly the same vector, so the second call is
    pure waste:
      - it uses API quota (the free tier allows 1,000 embedding calls a day)
      - it adds latency (~0.3 s), even when the answer is then a cache hit

HOW
    CachedEmbedder wraps the real embedder and looks just like one (it is
    an Embedder), so the cache engine doesn't know it's there:

        embed("What is the capital of Japan?")
          1. key = hash of (model, dimensions, exact text)
          2. Redis has that key?  → return the stored vector (no API call)
          3. otherwise            → call Gemini, store the vector, return it

    The key includes the model and dimensions, so vectors from different
    embedding models are never mixed up. The text must match exactly;
    a reworded question is a new text (that's what the semantic cache
    itself is for). Entries expire after EMBEDDING_CACHE_TTL_SECONDS
    (7 days by default) to bound memory: about 3 KB per text at 768
    dimensions.

    Like the rest of the cache, it never breaks a request: if Redis fails,
    the text is simply embedded by the API.

    This is not the same as caching answers: an embedding is a fixed
    function of its text, so a remembered vector is exactly the one the
    API would return. Search results, and so hits and misses, don't change.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod

import numpy as np
from redis.asyncio import Redis

from semcache.embeddings.base import Embedder
from semcache.metrics import metrics

logger = logging.getLogger(__name__)

KEY_PREFIX = "semcache:embedding:"


class EmbeddingStore(ABC):
    """Where remembered vectors are kept."""

    @abstractmethod
    async def get(self, key: str) -> list[float] | None:
        """The vector stored under `key`, or None."""

    @abstractmethod
    async def set(self, key: str, vector: list[float], ttl_seconds: int) -> None:
        """Store `vector` under `key` for `ttl_seconds`."""


class RedisEmbeddingStore(EmbeddingStore):
    """Vectors as raw float32 bytes, one Redis key each (outside the search index)."""

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._redis: Redis | None = None

    async def initialize(self) -> None:
        self._redis = Redis.from_url(self._redis_url)

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()

    @property
    def _client(self) -> Redis:
        if self._redis is None:
            raise RuntimeError("Embedding store not initialized. Call initialize() first.")
        return self._redis

    async def get(self, key: str) -> list[float] | None:
        data = await self._client.get(key)
        return None if data is None else np.frombuffer(data, dtype=np.float32).tolist()

    async def set(self, key: str, vector: list[float], ttl_seconds: int) -> None:
        await self._client.set(key, np.array(vector, dtype=np.float32).tobytes(), ex=ttl_seconds)


class CachedEmbedder(Embedder):
    """
    An embedder that remembers what it has embedded (see the module docstring).

    Args:
        inner: The real embedder, called for texts not seen before.
        store: Where vectors are remembered.
        identity: What makes vectors comparable, e.g. "gemini-embedding-001:768".
            Part of every key, so different models never share vectors.
        ttl_seconds: How long a remembered vector is kept.
    """

    def __init__(self, inner: Embedder, store: EmbeddingStore, *, identity: str, ttl_seconds: int) -> None:
        self._inner = inner
        self._store = store
        self._identity = identity
        self._ttl = ttl_seconds

    def _key(self, text: str) -> str:
        digest = hashlib.sha256(f"{self._identity}\n{text}".encode()).hexdigest()
        return f"{KEY_PREFIX}{digest[:32]}"

    async def embed(self, text: str) -> list[float]:
        key = self._key(text)
        try:
            remembered = await self._store.get(key)
        except Exception:
            logger.warning("Embedding memory unavailable; calling the embedding API", exc_info=True)
            remembered = None
        if remembered is not None:
            metrics.embedding_cache_hits += 1
            return remembered

        vector = await self._inner.embed(text)
        try:
            await self._store.set(key, vector, self._ttl)
        except Exception:
            logger.warning("Could not remember an embedding", exc_info=True)
        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]
