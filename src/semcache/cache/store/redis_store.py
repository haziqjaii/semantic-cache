"""
Redis-backed vector store using RedisVL.

HOW THIS WORKS (the full picture):
    Redis is an in-memory key-value store — insanely fast (~0.1ms reads).
    But plain Redis can't do "find me the most similar vector." That's where
    RedisVL comes in.

    RedisVL adds a vector search index ON TOP of Redis. Under the hood:
      1. We define a "schema" — what fields each entry has (namespace,
         embedding vector, prompt text, response text, metadata).
      2. RedisVL creates an HNSW index on the embedding field.
      3. When we search, Redis uses HNSW to find approximate nearest
         neighbors in O(log n) time.

    HNSW (Hierarchical Navigable Small World):
      Think of it like a skip list for vectors. Instead of comparing your
      query against every stored vector (O(n) — slow at scale), HNSW builds
      a multi-level graph where each level has fewer nodes. You start at the
      top (coarse) and drill down to the bottom (precise). This gives you
      ~99% accuracy at O(log n) speed.

REDIS KEYS LAYOUT:
    Each cache entry is stored as a Redis Hash:
      semcache:entry:{uuid}  →  {
        "namespace": "a3f2c1...",
        "prompt": "What is Python?",
        "response": "Python is a programming language...",
        "model": "gemini-3.5-flash",
        "embedding": <binary float32 blob>,
        "created_at": "2024-01-15T10:30:00",
        "ttl_seconds": 86400,
        "hit_count": 0,
        "response_metadata": "{...json...}"
      }

    The vector index spans all keys matching "semcache:entry:*" and
    enables filtered similarity search (by namespace).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

import numpy as np
from redis.asyncio import Redis
from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from redisvl.schema import IndexSchema

from semcache.cache.store.base import CacheEntry, VectorStore

# ── Index Schema ────────────────────────────────────────────
# This tells RedisVL what fields to index and how.
# We define it as a dict (RedisVL also accepts YAML files, but inline
# is more explicit and version-controllable).

INDEX_NAME = "semcache_idx"
KEY_PREFIX = "semcache:entry:"


def _build_schema(dims: int) -> dict:
    """
    Build the RedisVL index schema.

    Why a function, not a constant?
        The embedding dimensions are configurable (768 by default, but
        the user might use output_dimensionality=256 for faster lookups).
        The schema must match the actual vector size.
    """
    return {
        "index": {
            "name": INDEX_NAME,
            "prefix": KEY_PREFIX,
        },
        "fields": [
            # TAG field = exact match filter. We filter by namespace
            # BEFORE doing vector search, so Redis only compares vectors
            # within the same namespace. This is crucial for isolation.
            {"name": "namespace", "type": "tag"},
            {"name": "model", "type": "tag"},
            {"name": "ttl_seconds", "type": "numeric"},
            {"name": "hit_count", "type": "numeric"},
            {"name": "required_similarity", "type": "numeric"},
            # The embedding vector — this is what we search against.
            # HNSW = the index algorithm. COSINE = the distance metric.
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "hnsw",
                    "dims": dims,
                    "distance_metric": "cosine",
                    "datatype": "float32",
                    # HNSW tuning parameters:
                    # M = max connections per node. Higher = more accurate but more memory.
                    # EF_CONSTRUCTION = search width during index building. Higher = better
                    # index quality but slower inserts.
                    "m": 16,
                    "ef_construction": 200,
                },
            },
        ],
    }


class RedisVectorStore(VectorStore):
    """
    Redis + RedisVL implementation of the vector store.

    Usage:
        store = RedisVectorStore(redis_url="redis://localhost:6379", dims=768)
        await store.initialize()  # creates the index if it doesn't exist
    """

    def __init__(self, redis_url: str, dims: int = 768) -> None:
        self._redis_url = redis_url
        self._dims = dims
        self._redis: Redis | None = None
        self._index: AsyncSearchIndex | None = None

    async def initialize(self) -> None:
        """
        Connect to Redis and create the vector index.

        We separate __init__ from initialize because:
          - __init__ should be fast and never fail (no I/O)
          - initialize() does I/O (connects to Redis) and can fail
          - This lets us handle connection errors gracefully at startup
          - In tests, we can create the object without a real Redis connection
        """
        self._redis = Redis.from_url(self._redis_url)

        schema_dict = _build_schema(self._dims)
        schema = IndexSchema.from_dict(schema_dict)
        self._index = AsyncSearchIndex(schema, redis_client=self._redis)

        # Create the index. If it already exists (from a previous run),
        # this is a no-op. We don't drop and recreate — that would destroy
        # all cached data on every restart.
        await self._index.create(overwrite=False)

    async def search(
        self,
        embedding: list[float],
        namespace: str,
        threshold: float = 0.95,
        top_k: int = 5,
    ) -> list[tuple[CacheEntry, float]]:
        """
        Find the most similar cached entries in the given namespace.
        """
        if self._index is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        # Convert to numpy float32 bytes — the format Redis expects.
        query_bytes = np.array(embedding, dtype=np.float32).tobytes()

        query = VectorQuery(
            vector=query_bytes,
            vector_field_name="embedding",
            return_fields=[
                "prompt", "response", "model", "namespace",
                "created_at", "ttl_seconds", "hit_count",
                "required_similarity", "response_metadata",
            ],
            filter_expression=f"@namespace:{{{namespace}}}",
            num_results=top_k,
        )

        results = await self._index.query(query)

        candidates = []
        for result in results:
            distance = float(result.get("vector_distance", 1.0))
            similarity = 1.0 - distance

            if similarity < threshold:
                continue

            entry = CacheEntry(
                prompt=result["prompt"],
                response=result["response"],
                model=result["model"],
                namespace=result["namespace"],
                created_at=datetime.fromisoformat(result["created_at"]),
                ttl_seconds=int(result["ttl_seconds"]),
                hit_count=int(result["hit_count"]),
                required_similarity=float(result.get("required_similarity", 0.95)),
                response_metadata=(
                    json.loads(result["response_metadata"])
                    if result.get("response_metadata")
                    else None
                ),
                id=result.get("id", ""),
            )
            candidates.append((entry, similarity))

        return candidates

    async def record_hit(self, entry_id: str) -> None:
        if self._redis and entry_id:
            await self._redis.hincrby(entry_id, "hit_count", 1)

    async def store(
        self,
        embedding: list[float],
        entry: CacheEntry,
    ) -> str:
        """
        Store a new cache entry with its embedding vector.

        Each entry gets a unique UUID key. We store all fields as a Redis
        Hash (like a Python dict inside Redis). The embedding is stored as
        raw float32 bytes for efficient vector indexing.
        """
        if self._redis is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        entry_id = str(uuid.uuid4())
        key = f"{KEY_PREFIX}{entry_id}"

        # Convert embedding to float32 bytes for Redis vector storage.
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

        # Build the hash fields.
        fields = {
            "namespace": entry.namespace,
            "prompt": entry.prompt,
            "response": entry.response,
            "model": entry.model,
            "embedding": embedding_bytes,
            "created_at": entry.created_at.isoformat(),
            "ttl_seconds": entry.ttl_seconds,
            "hit_count": entry.hit_count,
            "required_similarity": entry.required_similarity,
            "response_metadata": (
                json.dumps(entry.response_metadata)
                if entry.response_metadata
                else ""
            ),
        }

        # hset = "hash set" — sets multiple fields on a Redis Hash in one call.
        await self._redis.hset(key, mapping=fields)

        # Set TTL on the Redis key itself. After ttl_seconds, Redis
        # automatically deletes the key — no background cleanup needed.
        if entry.ttl_seconds > 0:
            await self._redis.expire(key, entry.ttl_seconds)

        return entry_id

    async def delete_by_namespace(self, namespace: str) -> int:
        """
        Delete all entries in a namespace.
        """
        if self._redis is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        # RedisVL provides a clean way to fetch keys matching a filter
        from redisvl.query import FilterQuery
        
        # We can just fetch the ids
        query = FilterQuery(
            filter_expression=f"@namespace:{{{namespace}}}",
            return_fields=["id"],
            num_results=10000,
        )
        
        results = await self._index.query(query)
        if not results:
            return 0
            
        keys_to_delete = [res["id"] for res in results if "id" in res]
        if keys_to_delete:
            await self._redis.delete(*keys_to_delete)
            return len(keys_to_delete)
        return 0

    async def count(self, namespace: str | None = None) -> int:
        """Count entries, optionally filtered by namespace."""
        if self._index is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        if namespace is None:
            # Count all entries with a wildcard query.
            info = await self._index.info()
            return int(info.get("num_docs", 0))

        # We use a direct FT.SEARCH with LIMIT 0 0 to get the count
        # without fetching any documents.
        from redis.commands.search.query import Query
        
        q = Query(f"@namespace:{{{namespace}}}").paging(0, 0)
        res = await self._redis.ft(INDEX_NAME).search(q)
        return res.total

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis:
            await self._redis.aclose()
