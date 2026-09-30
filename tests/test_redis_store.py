"""
Integration tests for RedisVectorStore.

Requires a running Redis instance with RedisStack (for RediSearch/RedisVL).
Run with:
    docker run -d -p 6379:6379 redis/redis-stack-server:latest
    pytest -m integration
"""

import asyncio
import os

import pytest
import pytest_asyncio

from semcache.cache.store.base import CacheEntry, EntryFilter
from semcache.cache.store.redis_store import RedisVectorStore


@pytest.fixture
def redis_url():
    # These tests FLUSH the database. Point them at a throwaway Redis with
    # SEMCACHE_TEST_REDIS_URL to keep a dev cache on the default port intact.
    return os.environ.get("SEMCACHE_TEST_REDIS_URL", "redis://localhost:6379")


@pytest_asyncio.fixture
async def store(redis_url):
    """Provide an initialized RedisVectorStore and clean it up after."""
    import redis.asyncio as redis
    
    # Flush first
    client = redis.from_url(redis_url)
    await client.flushdb()
    await client.aclose()
    
    store = RedisVectorStore(redis_url=redis_url, dims=4)
    try:
        await store.initialize()
        yield store
    finally:
        if store._redis:
            await store._redis.flushdb()
        await store.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_store_and_search(store: RedisVectorStore):
    """Test basic store and similarity search."""
    namespace = "test_ns_1"
    
    # Store an entry
    entry = CacheEntry(
        prompt="hello",
        response="world",
        model="test-model",
        namespace=namespace,
    )
    # Unit vector for testing
    emb = [1.0, 0.0, 0.0, 0.0]
    entry_id = await store.store(emb, entry)
    
    assert entry_id is not None

    # Search with exact same vector
    results = await store.search(emb, namespace, threshold=0.99)
    assert results
    hit_entry, similarity = results[0]
    
    assert similarity >= 0.99
    assert hit_entry.prompt == "hello"
    assert hit_entry.response == "world"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_namespace_filtering(store: RedisVectorStore):
    """Test that search respects namespaces."""
    emb = [1.0, 0.0, 0.0, 0.0]
    
    await store.store(
        emb,
        CacheEntry(prompt="A", response="A", model="m", namespace="ns1")
    )
    
    # Search in ns1 -> hit
    res_ns1 = await store.search(emb, "ns1", threshold=0.9)
    assert len(res_ns1) > 0
    
    # Search in ns2 -> miss
    res_ns2 = await store.search(emb, "ns2", threshold=0.9)
    assert len(res_ns2) == 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_delete_and_count(store: RedisVectorStore):
    """Test count and delete_matching by namespace."""
    emb = [1.0, 0.0, 0.0, 0.0]
    
    # Store 3 in ns1, 2 in ns2
    for _ in range(3):
        await store.store(emb, CacheEntry(prompt="A", response="A", model="m", namespace="ns1"))
    for _ in range(2):
        await store.store(emb, CacheEntry(prompt="A", response="A", model="m", namespace="ns2"))
        
    # Test count
    assert await store.count(EntryFilter(namespace="ns1")) == 3
    assert await store.count(EntryFilter(namespace="ns2")) == 2
    assert await store.count() == 5
    
    # Test delete
    deleted = await store.delete_matching(EntryFilter(namespace="ns1"))
    assert deleted == 3
    assert await store.count(EntryFilter(namespace="ns1")) == 0
    assert await store.count(EntryFilter(namespace="ns2")) == 2


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ttl_expiry(store: RedisVectorStore):
    """Test that entries actually expire from Redis."""
    emb = [1.0, 0.0, 0.0, 0.0]
    
    # Store with 1 second TTL
    entry = CacheEntry(prompt="A", response="A", model="m", namespace="ttl_ns", ttl_seconds=1)
    await store.store(emb, entry)
    
    # Verify it's there
    assert await store.count(EntryFilter(namespace="ttl_ns")) == 1
    
    # Wait for TTL to expire
    await asyncio.sleep(1.1)
    
    # Verify it's gone
    assert await store.count(EntryFilter(namespace="ttl_ns")) == 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_hit_counts(store: RedisVectorStore):
    """Test that recording a hit increments the hit count."""
    emb = [1.0, 0.0, 0.0, 0.0]
    entry = CacheEntry(prompt="A", response="A", model="m", namespace="hit_ns")
    
    entry_id = await store.store(emb, entry)
    
    # We now explicitly record hits
    await store.record_hit(entry_id)
    await store.record_hit(entry_id)
    
    # Search to inspect the returned entry's hit count.
    result = await store.search(emb, "hit_ns", threshold=0.9)
    assert result
    hit_entry, _ = result[0]
    assert hit_entry.hit_count == 2


@pytest.mark.asyncio
@pytest.mark.integration
async def test_backend_stats(store: RedisVectorStore):
    """backend_stats reports Redis eviction/expiry counters as ints."""
    stats = await store.backend_stats()

    assert set(stats) == {"evicted_keys", "expired_keys"}
    assert all(isinstance(v, int) and v >= 0 for v in stats.values())


@pytest.mark.asyncio
@pytest.mark.integration
async def test_response_metadata_round_trip(store: RedisVectorStore):
    """Usage and finish reason stored on a miss come back intact on a hit."""
    metadata = {
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46},
    }
    emb = [1.0, 0.0, 0.0, 0.0]
    await store.store(
        emb,
        CacheEntry(prompt="A", response="A", model="m", namespace="meta_ns", response_metadata=metadata),
    )

    (entry, _), = await store.search(emb, "meta_ns", threshold=0.9)
    assert entry.response_metadata == metadata


def _entry(namespace="ns", **fields) -> CacheEntry:
    return CacheEntry(prompt="p", response="r", model=fields.pop("model", "m"), namespace=namespace, **fields)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_filters_match_literally_on_redis(store: RedisVectorStore):
    """Model names and tags with '-', '.', ':' are matched exactly (escaped)."""
    emb = [1.0, 0.0, 0.0, 0.0]
    await store.store(emb, _entry(model="gemini-3.5-flash", system_prompt_hash="aaaa000000000000", tags=["billing:v1", "support"]))
    await store.store(emb, _entry(model="gemini-3.5-flash-lite", system_prompt_hash="aaaa000000000000", tags=["billing:v2"]))
    await store.store(emb, _entry(model="gemini-3.5-flash", system_prompt_hash="bbbb000000000000", tags=[]))

    assert await store.count(EntryFilter(model="gemini-3.5-flash")) == 2
    assert await store.count(EntryFilter(system_prompt_hash="aaaa000000000000")) == 2
    assert await store.count(EntryFilter(tag="support")) == 1
    assert await store.count(EntryFilter(tag="billing:v2")) == 1
    assert await store.count(EntryFilter(tag_prefix="billing:")) == 2
    assert await store.count(EntryFilter(model="gemini-3.5-flash", tag_prefix="billing:")) == 1
    assert await store.count() == 3

    # Tags survive the round trip.
    results = await store.search(emb, "ns", threshold=0.9)
    assert sorted(tuple(e.tags) for e, _ in results) == [(), ("billing:v1", "support"), ("billing:v2",)]

    assert await store.delete_matching(EntryFilter(model="gemini-3.5-flash")) == 2
    assert await store.count() == 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_delete_matching_goes_past_one_batch(store: RedisVectorStore):
    """The old delete stopped at 10,000; batching must remove every match."""
    emb = [1.0, 0.0, 0.0, 0.0]
    pipe = store._redis.pipeline()
    for i in range(2500):
        pipe.hset(f"semcache:entry:bulk-{i}", mapping={
            "namespace": "bulk", "model": "m", "prompt": "p", "response": "r",
            "embedding": bytes(bytearray(4 * 4)), "created_at": "2026-01-01T00:00:00+00:00",
            "ttl_seconds": 0, "hit_count": 0, "required_similarity": 0.95,
            "response_metadata": "", "system_prompt_hash": "", "tags": "",
        })
    await pipe.execute()
    await store.store(emb, _entry(namespace="keep"))

    # Wait for RediSearch to index the bulk-loaded hashes.
    for _ in range(50):
        if await store.count(EntryFilter(namespace="bulk")) == 2500:
            break
        await asyncio.sleep(0.1)

    assert await store.delete_matching(EntryFilter(namespace="bulk")) == 2500
    assert await store.count() == 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_old_index_is_upgraded_and_entries_are_kept(redis_url):
    """An index from before system_prompt_hash/tags is rebuilt on startup."""
    import redis.asyncio as redis
    from redisvl.index import AsyncSearchIndex
    from redisvl.schema import IndexSchema

    from semcache.cache.store import redis_store

    client = redis.from_url(redis_url)
    await client.flushdb()
    old_schema = redis_store._build_schema(4)
    old_schema["fields"] = [
        f for f in old_schema["fields"] if f["name"] not in ("system_prompt_hash", "tags")
    ]
    old_index = AsyncSearchIndex(IndexSchema.from_dict(old_schema), redis_client=client)
    await old_index.create()

    # An entry cached by the old version (no system_prompt_hash / tags fields).
    await client.hset("semcache:entry:old-1", mapping={
        "namespace": "ns", "model": "gemini-3.5-flash", "prompt": "p", "response": "r",
        "embedding": bytes(bytearray(4 * 4)), "created_at": "2026-01-01T00:00:00+00:00",
        "ttl_seconds": 3600, "hit_count": 0, "required_similarity": 0.95, "response_metadata": "",
    })

    store = RedisVectorStore(redis_url=redis_url, dims=4)
    try:
        await store.initialize()
        await store.store([1.0, 0.0, 0.0, 0.0], _entry(tags=["new"]))

        # The new fields are queryable, and the old entry survived the rebuild.
        for _ in range(50):
            if await store.count() == 2:
                break
            await asyncio.sleep(0.1)
        assert await store.count() == 2
        assert await store.count(EntryFilter(tag="new")) == 1
        assert await store.count(EntryFilter(model="gemini-3.5-flash")) == 1

        # Restarting again with the same schema doesn't rebuild or lose anything.
        again = RedisVectorStore(redis_url=redis_url, dims=4)
        await again.initialize()
        assert await again.count() == 2
        await again.close()
    finally:
        await client.flushdb()
        await client.aclose()
        await store.close()
