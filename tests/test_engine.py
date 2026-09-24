"""
Tests for the cache engine — the full lookup → miss → store → hit cycle.

These tests use the MockEmbedder and InMemoryVectorStore (no Redis needed).
They verify the ENGINE LOGIC, not the Redis integration.

THE CRITICAL TEST SCENARIO:
  1. lookup("What is Python?") → MISS (nothing cached yet)
  2. store(response="Python is a language...") → entry stored
  3. lookup("What is Python?") → HIT (exact same text, similarity=1.0)
  4. lookup("Completely unrelated query") → MISS (different meaning)
"""

import pytest

from semcache.cache.engine import CacheEngine
from semcache.cache.policy import CachePolicy, TTLTier


class TestCacheEngine:
    """Test the CacheEngine orchestration logic."""

    @pytest.mark.asyncio
    async def test_miss_on_empty_cache(self, mock_embedder, memory_store):
        """First lookup on an empty cache should be a miss."""
        engine = CacheEngine(embedder=mock_embedder, store=memory_store)

        result = await engine.lookup(
            prompt="What is Python?",
            model="gemini-3.5-flash",
        )

        assert result.hit is False
        assert result.entry is None
        assert result.embedding is not None  # embedding should be computed
        assert result.namespace != ""  # namespace should be computed

    @pytest.mark.asyncio
    async def test_hit_after_store(self, mock_embedder, memory_store):
        """After storing a response, the same prompt should be a hit."""
        engine = CacheEngine(embedder=mock_embedder, store=memory_store)

        # Step 1: Lookup (miss).
        miss_result = await engine.lookup(
            prompt="What is Python?",
            model="gemini-3.5-flash",
        )
        assert miss_result.hit is False

        # Step 2: Store the response.
        await engine.store(
            lookup_result=miss_result,
            prompt="What is Python?",
            response="Python is a high-level programming language.",
            model="gemini-3.5-flash",
        )

        # Step 3: Lookup again (hit).
        # With identical text, our mock embedder gives similarity=1.0,
        # so it easily clears the default 0.95 threshold.
        hit_result = await engine.lookup(
            prompt="What is Python?",
            model="gemini-3.5-flash",
        )
        assert hit_result.hit is True
        assert hit_result.entry is not None
        assert hit_result.entry.response == "Python is a high-level programming language."
        assert hit_result.similarity == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.asyncio
    async def test_miss_for_different_text(self, mock_embedder, memory_store):
        """Different text should not hit a cached entry (at high threshold)."""
        engine = CacheEngine(embedder=mock_embedder, store=memory_store)

        # Store one entry.
        miss_result = await engine.lookup(
            prompt="What is Python?",
            model="gemini-3.5-flash",
        )
        await engine.store(
            lookup_result=miss_result,
            prompt="What is Python?",
            response="Python is a programming language.",
            model="gemini-3.5-flash",
        )

        # Lookup with completely different text.
        result = await engine.lookup(
            prompt="Recipe for chocolate cake",
            model="gemini-3.5-flash",
        )
        # With the mock embedder, different texts produce different hashes,
        # so cosine similarity will be low → miss.
        assert result.hit is False

    @pytest.mark.asyncio
    async def test_namespace_isolation(self, mock_embedder, memory_store):
        """Same prompt but different system prompts → different namespaces → miss."""
        engine = CacheEngine(embedder=mock_embedder, store=memory_store)

        # Store with system prompt A.
        miss_a = await engine.lookup(
            prompt="What is Python?",
            model="gemini-3.5-flash",
            system_prompt="You are a teacher.",
        )
        await engine.store(
            lookup_result=miss_a,
            prompt="What is Python?",
            response="Python is great for learning!",
            model="gemini-3.5-flash",
        )

        # Lookup with system prompt B — should MISS despite same user prompt.
        result_b = await engine.lookup(
            prompt="What is Python?",
            model="gemini-3.5-flash",
            system_prompt="You are a code reviewer.",
            policy=CachePolicy(ttl_tier=TTLTier.LONG, similarity_threshold=0.0),
        )
        assert result_b.hit is False

    @pytest.mark.asyncio
    async def test_no_cache_policy_skips_lookup(self, mock_embedder, memory_store):
        """NO_CACHE policy should skip the cache entirely."""
        engine = CacheEngine(embedder=mock_embedder, store=memory_store)

        result = await engine.lookup(
            prompt="Write me a poem",
            model="gemini-3.5-flash",
            policy=CachePolicy(ttl_tier=TTLTier.NO_CACHE),
        )
        assert result.hit is False
        assert result.embedding is None  # didn't even compute the embedding

    @pytest.mark.asyncio
    async def test_store_without_embedding_raises(self, mock_embedder, memory_store):
        """Storing a NO_CACHE result (no embedding) should raise ValueError."""
        engine = CacheEngine(embedder=mock_embedder, store=memory_store)

        result = await engine.lookup(
            prompt="Write me a poem",
            model="gemini-3.5-flash",
            policy=CachePolicy(ttl_tier=TTLTier.NO_CACHE),
        )

        with pytest.raises(ValueError, match="no embedding"):
            await engine.store(
                lookup_result=result,
                prompt="Write me a poem",
                response="Roses are red...",
                model="gemini-3.5-flash",
            )

    @pytest.mark.asyncio
    async def test_invalidate_namespace(self, mock_embedder, memory_store):
        """Invalidation should remove all entries in a namespace."""
        engine = CacheEngine(embedder=mock_embedder, store=memory_store)

        # Store an entry.
        miss = await engine.lookup(
            prompt="What is Python?",
            model="gemini-3.5-flash",
            system_prompt="You are helpful.",
        )
        await engine.store(
            lookup_result=miss,
            prompt="What is Python?",
            response="Python is a language.",
            model="gemini-3.5-flash",
        )

        # Invalidate that namespace.
        deleted = await engine.invalidate_namespace(
            model="gemini-3.5-flash",
            system_prompt="You are helpful.",
        )
        assert deleted == 1

        # Lookup should now miss.
        result = await engine.lookup(
            prompt="What is Python?",
            model="gemini-3.5-flash",
            system_prompt="You are helpful.",
            policy=CachePolicy(ttl_tier=TTLTier.LONG, similarity_threshold=0.0),
        )
        assert result.hit is False

    @pytest.mark.asyncio
    async def test_stats(self, mock_embedder, memory_store):
        """Stats should return the total entry count."""
        engine = CacheEngine(embedder=mock_embedder, store=memory_store)

        stats = await engine.stats()
        assert stats["total_entries"] == 0

        # Store one entry.
        miss = await engine.lookup(prompt="test", model="test-model")
        await engine.store(
            lookup_result=miss,
            prompt="test",
            response="test response",
            model="test-model",
        )

        stats = await engine.stats()
        assert stats["total_entries"] == 1
