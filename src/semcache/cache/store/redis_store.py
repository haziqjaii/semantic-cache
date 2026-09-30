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
        "required_similarity": 0.95,
        "response_metadata": "{...json...}",
        "system_prompt_hash": "9f86d081884c7d65",
        "tags": "support,billing:v2"
      }

    The vector index spans all keys matching "semcache:entry:*" and
    enables filtered similarity search (by namespace).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime

import numpy as np
from redis.asyncio import Redis
from redisvl.index import AsyncSearchIndex
from redisvl.query import CountQuery, FilterQuery, VectorQuery
from redisvl.query.filter import FilterExpression, Tag
from redisvl.schema import IndexSchema

from semcache.cache.store.base import CacheEntry, EntryFilter, VectorStore

logger = logging.getLogger(__name__)

# How many keys each invalidation round trip fetches and deletes.
_DELETE_BATCH = 1000

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
            # Invalidation handles: every entry for one system prompt, and
            # client-supplied labels (comma-separated, one TAG value each).
            {"name": "system_prompt_hash", "type": "tag"},
            {"name": "tags", "type": "tag", "attrs": {"separator": ","}},
            {"name": "ttl_seconds", "type": "numeric"},
            {"name": "hit_count", "type": "numeric"},
            {"name": "required_similarity", "type": "numeric"},
            # Creation time as a Unix timestamp, sortable so list_entries()
            # can return the newest entries first.
            {"name": "created_ts", "type": "numeric", "attrs": {"sortable": True}},
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


# The hash fields read back into a CacheEntry (everything but the vector).
_ENTRY_FIELDS = [
    "prompt", "response", "model", "namespace",
    "created_at", "ttl_seconds", "hit_count",
    "required_similarity", "response_metadata",
    "system_prompt_hash", "tags", "intent",
]


def _entry_from_result(result: dict) -> CacheEntry:
    """Build a CacheEntry from a query result containing _ENTRY_FIELDS."""
    return CacheEntry(
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
        # Absent on entries cached before these fields existed.
        system_prompt_hash=result.get("system_prompt_hash", ""),
        tags=[t for t in result.get("tags", "").split(",") if t],
        intent=result.get("intent") or None,
        id=result.get("id", ""),
    )


def _to_redis_filter(entry_filter: EntryFilter) -> FilterExpression | str:
    """
    Translate an EntryFilter into a RediSearch filter ("*" when empty).

    RedisVL escapes the values, so model names like "gemini-3.5-flash" and
    tags like "billing:v2" are matched literally.
    """
    clauses: list[FilterExpression] = []
    if entry_filter.namespace is not None:
        clauses.append(Tag("namespace") == entry_filter.namespace)
    if entry_filter.model is not None:
        clauses.append(Tag("model") == entry_filter.model)
    if entry_filter.system_prompt_hash is not None:
        clauses.append(Tag("system_prompt_hash") == entry_filter.system_prompt_hash)
    if entry_filter.tag is not None:
        clauses.append(Tag("tags") == entry_filter.tag)
    if entry_filter.tag_prefix is not None:
        clauses.append(Tag("tags") % f"{entry_filter.tag_prefix}*")

    if not clauses:
        return "*"
    expression = clauses[0]
    for clause in clauses[1:]:
        expression = expression & clause
    return expression


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

        if not await self._index.exists():
            await self._index.create()
            return

        # The index survives restarts (dropping it with its data would wipe
        # the cache). But if this version indexes different fields, rebuild
        # the index definition only: drop=False keeps every cached entry,
        # and Redis re-indexes them in the background.
        existing = await AsyncSearchIndex.from_existing(INDEX_NAME, redis_client=self._redis)
        if set(existing.schema.field_names) != set(schema.field_names):
            logger.warning(
                "Cache index fields changed (%s → %s); rebuilding the index. "
                "Cached entries are kept.",
                sorted(existing.schema.field_names),
                sorted(schema.field_names),
            )
            await self._index.create(overwrite=True, drop=False)

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
            return_fields=_ENTRY_FIELDS,
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

            candidates.append((_entry_from_result(result), similarity))

        return candidates

    async def list_entries(self, limit: int = 50) -> list[CacheEntry]:
        """The most recently cached entries, newest first."""
        if self._index is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        query = FilterQuery(
            filter_expression="*",
            return_fields=_ENTRY_FIELDS,
            num_results=limit,
            sort_by=("created_ts", "DESC"),
        )
        return [_entry_from_result(result) for result in await self._index.query(query)]

    async def record_hit(self, entry_id: str) -> None:
        if self._redis and entry_id:
            key = entry_id if entry_id.startswith(KEY_PREFIX) else f"{KEY_PREFIX}{entry_id}"
            await self._redis.hincrby(key, "hit_count", 1)

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
            "created_ts": entry.created_at.timestamp(),
            "ttl_seconds": entry.ttl_seconds,
            "hit_count": entry.hit_count,
            "required_similarity": entry.required_similarity,
            "response_metadata": (
                json.dumps(entry.response_metadata)
                if entry.response_metadata
                else ""
            ),
            "system_prompt_hash": entry.system_prompt_hash,
            "tags": ",".join(entry.tags),
            "intent": entry.intent or "",
        }

        # hset = "hash set" — sets multiple fields on a Redis Hash in one call.
        await self._redis.hset(key, mapping=fields)

        # Set TTL on the Redis key itself. After ttl_seconds, Redis
        # automatically deletes the key — no background cleanup needed.
        if entry.ttl_seconds > 0:
            await self._redis.expire(key, entry.ttl_seconds)

        return entry_id

    async def delete_matching(self, entry_filter: EntryFilter) -> int:
        """
        Delete every entry matching the filter, in batches.

        Deleted keys drop out of the index immediately, so each round trip
        fetches the next batch of matches until none are left.
        """
        if self._redis is None or self._index is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        query_filter = _to_redis_filter(entry_filter)
        deleted = 0
        while True:
            query = FilterQuery(
                filter_expression=query_filter,
                return_fields=["id"],
                num_results=_DELETE_BATCH,
            )
            keys = [res["id"] for res in await self._index.query(query) if "id" in res]
            if not keys:
                return deleted
            removed = await self._redis.delete(*keys)
            deleted += removed
            if removed == 0:
                # Every key in the batch was already gone (e.g. expired a
                # moment ago). Stop rather than risk looping on a stale index.
                return deleted

    async def count(self, entry_filter: EntryFilter | None = None) -> int:
        """Count entries, optionally only those matching a filter."""
        if self._index is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        query_filter = _to_redis_filter(entry_filter or EntryFilter())
        return await self._index.query(CountQuery(filter_expression=query_filter))

    async def backend_stats(self) -> dict:
        """
        Eviction and expiry counters from Redis INFO.

        These are server-wide: they include any non-semcache keys in the
        same Redis instance.
        """
        if self._redis is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        info = await self._redis.info("stats")
        return {
            "evicted_keys": int(info.get("evicted_keys", 0)),
            "expired_keys": int(info.get("expired_keys", 0)),
        }

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis:
            await self._redis.aclose()
