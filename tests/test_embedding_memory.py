"""
Tests for the embedding memory (embeddings/memory.py).
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from semcache.cache.engine import CacheEngine
from semcache.embeddings.base import Embedder
from semcache.embeddings.gemini import GeminiEmbedder
from semcache.embeddings.memory import KEY_PREFIX, CachedEmbedder, EmbeddingStore
from semcache.metrics import metrics
from tests.conftest import MockEmbedder


class MemoryStore(EmbeddingStore):
    """An EmbeddingStore in a dict, recording the TTLs it was given."""

    def __init__(self) -> None:
        self.data: dict[str, list[float]] = {}
        self.ttls: dict[str, int] = {}

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, vector, ttl_seconds):
        self.data[key] = vector
        self.ttls[key] = ttl_seconds


class CountingEmbedder(MockEmbedder):
    """The test embedder, counting the texts it was really asked to embed."""

    def __init__(self) -> None:
        super().__init__(dims=8)
        self.calls: list[str] = []

    async def embed(self, text):
        self.calls.append(text)
        return await super().embed(text)


def _memory(inner=None, store=None, identity="gemini-embedding-001:768", ttl=3600):
    inner = inner or CountingEmbedder()
    store = store if store is not None else MemoryStore()
    return CachedEmbedder(inner, store, identity=identity, ttl_seconds=ttl), inner, store


class TestCachedEmbedder:
    @pytest.mark.asyncio
    async def test_same_text_is_embedded_only_once(self):
        memory, inner, store = _memory()

        first = await memory.embed("What is the capital of Japan?")
        second = await memory.embed("What is the capital of Japan?")

        assert first == second
        assert inner.calls == ["What is the capital of Japan?"]
        assert metrics.embedding_cache_hits == 1
        (key,) = store.data
        assert key.startswith(KEY_PREFIX)
        assert store.ttls[key] == 3600

    @pytest.mark.asyncio
    async def test_different_or_reworded_text_is_embedded(self):
        """Only exact repeats are remembered; rewording is the semantic cache's job."""
        memory, inner, _ = _memory()

        await memory.embed("What is the capital of Japan?")
        await memory.embed("Which city is the capital of Japan?")
        await memory.embed("what is the capital of japan?")

        assert len(inner.calls) == 3
        assert metrics.embedding_cache_hits == 0

    @pytest.mark.asyncio
    async def test_different_models_never_share_vectors(self):
        store = MemoryStore()
        model_a, inner_a, _ = _memory(store=store, identity="gemini-embedding-001:768")
        model_b, inner_b, _ = _memory(store=store, identity="gemini-embedding-001:256")

        await model_a.embed("hello")
        await model_b.embed("hello")

        assert inner_a.calls == inner_b.calls == ["hello"]
        assert len(store.data) == 2

    @pytest.mark.asyncio
    async def test_memory_failure_falls_back_to_the_api(self):
        """Like the rest of the cache, a Redis problem never breaks a request."""
        store = MagicMock(spec=EmbeddingStore)
        store.get = AsyncMock(side_effect=ConnectionError("Redis down"))
        store.set = AsyncMock(side_effect=ConnectionError("Redis down"))
        memory, inner, _ = _memory(store=store)

        vector = await memory.embed("hello")

        assert vector == await MockEmbedder(dims=8).embed("hello")
        assert inner.calls == ["hello"]

    @pytest.mark.asyncio
    async def test_batch(self):
        memory, inner, _ = _memory()
        await memory.embed("a")

        vectors = await memory.embed_batch(["a", "b"])

        assert len(vectors) == 2
        assert inner.calls == ["a", "b"]

    @pytest.mark.asyncio
    async def test_engine_results_are_unchanged(self, memory_store):
        """A remembered vector is the same vector, so hits and misses don't change."""
        memory, inner, _ = _memory()
        engine = CacheEngine(embedder=memory, store=memory_store)

        miss = await engine.lookup(prompt="What is Python?", model="m")
        await engine.store(miss, prompt="What is Python?", response="A language.", model="m")
        hit = await engine.lookup(prompt="What is Python?", model="m")

        assert not miss.hit
        assert hit.hit
        assert hit.similarity == pytest.approx(1.0)
        # One API call: the second lookup used the memory. The engine embeds
        # the question cleaned for matching (cache/text.py), not as typed.
        assert inner.calls == ["what is python"]


class TestEmbeddingApiAccounting:
    @pytest.mark.asyncio
    async def test_only_real_api_calls_are_counted(self):
        """embedding_calls means calls to the API: remembered texts don't count."""
        gemini = GeminiEmbedder(api_key="fake-key", dims=4)
        response = MagicMock()
        response.embeddings = [MagicMock(values=[1.0, 0.0, 0.0, 0.0])]
        gemini._client.aio.models.embed_content = AsyncMock(return_value=response)
        memory = CachedEmbedder(gemini, MemoryStore(), identity="m:4", ttl_seconds=60)

        for _ in range(5):
            await memory.embed("12345678")  # 8 characters ≈ 2 tokens

        assert metrics.embedding_calls == 1
        assert metrics.embedding_tokens_total == 2
        assert metrics.embedding_cache_hits == 4
        assert gemini._client.aio.models.embed_content.await_count == 1


def test_stored_vectors_survive_the_float32_round_trip():
    """Vectors are stored as float32 bytes; embedders already produce float32 values."""
    vector = (np.random.default_rng(0).random(768, dtype=np.float32)).tolist()
    restored = np.frombuffer(np.array(vector, dtype=np.float32).tobytes(), dtype=np.float32).tolist()
    assert restored == vector


def test_embedder_interface_is_kept():
    assert issubclass(CachedEmbedder, Embedder)
