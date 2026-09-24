"""
Integration tests for RedisVectorStore.

Requires a running Redis instance with RedisStack (for RediSearch/RedisVL).
Run with:
    docker run -d -p 6379:6379 redis/redis-stack-server:latest
    pytest -m integration
"""

import asyncio
import numpy as np
import pytest

import pytest_asyncio

from semcache.cache.store.base import CacheEntry
from semcache.cache.store.redis_store import RedisVectorStore


@pytest.fixture
def redis_url():
    return "redis://localhost:6379"


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
    result = await store.search(emb, namespace, threshold=0.99)
    assert result is not None
    hit_entry, similarity = result
    
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
    assert res_ns1 is not None
    
    # Search in ns2 -> miss
    res_ns2 = await store.search(emb, "ns2", threshold=0.9)
    assert res_ns2 is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_delete_and_count(store: RedisVectorStore):
    """Test count and delete_by_namespace."""
    emb = [1.0, 0.0, 0.0, 0.0]
    
    # Store 3 in ns1, 2 in ns2
    for _ in range(3):
        await store.store(emb, CacheEntry(prompt="A", response="A", model="m", namespace="ns1"))
    for _ in range(2):
        await store.store(emb, CacheEntry(prompt="A", response="A", model="m", namespace="ns2"))
        
    # Test count
    assert await store.count("ns1") == 3
    assert await store.count("ns2") == 2
    assert await store.count() == 5
    
    # Test delete
    deleted = await store.delete_by_namespace("ns1")
    assert deleted == 3
    assert await store.count("ns1") == 0
    assert await store.count("ns2") == 2


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ttl_expiry(store: RedisVectorStore):
    """Test that entries actually expire from Redis."""
    emb = [1.0, 0.0, 0.0, 0.0]
    
    # Store with 1 second TTL
    entry = CacheEntry(prompt="A", response="A", model="m", namespace="ttl_ns", ttl_seconds=1)
    await store.store(emb, entry)
    
    # Verify it's there
    assert await store.count("ttl_ns") == 1
    
    # Wait for TTL to expire
    await asyncio.sleep(1.1)
    
    # Verify it's gone
    assert await store.count("ttl_ns") == 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_hit_counts(store: RedisVectorStore):
    """Test that searching increments the hit count."""
    emb = [1.0, 0.0, 0.0, 0.0]
    entry = CacheEntry(prompt="A", response="A", model="m", namespace="hit_ns")
    
    entry_id = await store.store(emb, entry)
    
    # Search twice
    await store.search(emb, "hit_ns", threshold=0.9)
    await store.search(emb, "hit_ns", threshold=0.9)
    
    # Search one more time to inspect the returned entry's hit count.
    # Note: the returned hit_count is the state BEFORE the increment
    # because our search method retrieves the document and THEN increments.
    # So on the 3rd search, we should see hit_count == 2.
    result = await store.search(emb, "hit_ns", threshold=0.9)
    assert result is not None
    hit_entry, _ = result
    assert hit_entry.hit_count == 2
