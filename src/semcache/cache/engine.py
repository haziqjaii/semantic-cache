"""
Cache engine — the orchestrator that ties embeddings, keys, store, and policy together.

This is the SINGLE ENTRY POINT for all cache operations. Route handlers
(Phase 2) never talk to the embedder or vector store directly — they
talk to the engine.

THE FLOW:
    ┌─────────────────────────────────────────────────────────┐
    │  1. Request comes in (prompt + model + system prompt)   │
    │  2. Generate namespace hash (keys.py)                   │
    │  3. Embed the user prompt (embeddings/gemini.py)        │
    │  4. Search vector store for similar entry (redis_store) │
    │  5a. HIT  → return cached response + bump hit counter   │
    │  5b. MISS → return None                                 │
    │      → caller sends to LLM, gets response               │
    │      → caller calls engine.store() to cache it           │
    └─────────────────────────────────────────────────────────┘

WHY A SEPARATE ENGINE CLASS?
    Without it, the route handler in chat.py would have to:
      - Call build_namespace() with the right params
      - Call embedder.embed() on the prompt
      - Call store.search() with the embedding and namespace
      - Handle hit/miss logic
      - On miss, later call store.store() with the right metadata

    That's 5 concerns in one function. The engine encapsulates all of it
    behind two clean methods: lookup() and store().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from semcache.cache.keys import build_namespace
from semcache.cache.policy import DEFAULT_POLICY, CachePolicy
from semcache.cache.store.base import CacheEntry, VectorStore
from semcache.embeddings.base import Embedder

logger = logging.getLogger(__name__)


@dataclass
class LookupResult:
    """
    The result of a cache lookup.

    Why a dataclass instead of returning Optional[CacheEntry]?
        Because we want to return metadata about the lookup itself:
        - Was it a hit or miss?
        - What was the similarity score? (for monitoring)
        - How long did the lookup take? (for latency tracking)
        This lets the caller (route handler) set response headers and
        emit Prometheus metrics without re-computing anything.
    """

    hit: bool
    entry: CacheEntry | None = None
    similarity: float = 0.0
    namespace: str = ""
    embedding: list[float] | None = None  # cached for later store() call
    policy: CachePolicy | None = None


class CacheEngine:
    """
    The main cache orchestrator.

    Coordinates between the embedder, vector store, and policy system
    to provide a simple lookup/store interface.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        default_policy: CachePolicy | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._default_policy = default_policy or DEFAULT_POLICY

    async def lookup(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        policy: CachePolicy | None = None,
    ) -> LookupResult:
        """
        Look up a prompt in the cache.

        This is the HOT PATH — every single request goes through here.
        Performance matters. The steps are:
          1. Build namespace (microseconds — just hashing)
          2. Embed the prompt (5-50ms — API call to Gemini)
          3. Search Redis (0.1-1ms — sub-millisecond with HNSW)

        The embedding step dominates latency. On a cache miss, we've
        "wasted" that embedding call. But on a hit, we save the full
        LLM generation call (500-5000ms). The math works out as long
        as hit rate > ~5%.

        Args:
            prompt: The user's message text.
            model: LLM model name (affects namespace).
            system_prompt: System instruction (affects namespace).
            temperature: Sampling temperature (affects namespace).
            max_tokens: Max response tokens (affects namespace).
            policy: Override the default cache policy.

        Returns:
            LookupResult with hit=True and the cached entry,
            or hit=False and the embedding (saved for the store() call).
        """
        policy = policy or self._default_policy

        # Skip cache entirely for NO_CACHE policies.
        if policy.ttl_seconds == 0:
            logger.debug("Cache skip: NO_CACHE policy for prompt: %s", prompt[:50])
            return LookupResult(hit=False, namespace="", embedding=None, policy=policy)

        # Step 1: Build the namespace hash.
        namespace = build_namespace(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Step 2: Embed the prompt.
        embedding = await self._embedder.embed(prompt)

        # Step 3: Search the vector store.
        result = await self._store.search(
            embedding=embedding,
            namespace=namespace,
            threshold=policy.similarity_threshold,
        )

        if result is not None:
            entry, similarity = result
            logger.info(
                "Cache HIT (similarity=%.4f) for prompt: %s",
                similarity,
                prompt[:50],
            )
            return LookupResult(
                hit=True,
                entry=entry,
                similarity=similarity,
                namespace=namespace,
                embedding=embedding,
                policy=policy,
            )

        logger.info("Cache MISS for prompt: %s", prompt[:50])
        return LookupResult(
            hit=False,
            namespace=namespace,
            embedding=embedding,
            policy=policy,
        )

    async def store(
        self,
        lookup_result: LookupResult,
        prompt: str,
        response: str,
        model: str,
        response_metadata: dict | None = None,
        ttl_seconds: int | None = None,
    ) -> str:
        """
        Store a new response in the cache after a miss.

        Why pass the LookupResult back?
            It contains the embedding and namespace we already computed
            during lookup(). No need to re-embed or re-hash — we reuse them.
            This saves one API call per cache miss.

        Args:
            lookup_result: The LookupResult from the lookup() call.
            prompt: The original user prompt.
            response: The full LLM response text.
            model: The model that generated the response.
            response_metadata: Optional metadata (token counts, etc.).
            ttl_seconds: Override TTL (or use policy default).

        Returns:
            The unique ID of the stored cache entry.

        Raises:
            ValueError: If lookup_result has no embedding (was a NO_CACHE skip).
        """
        if lookup_result.embedding is None:
            raise ValueError(
                "Cannot store: lookup_result has no embedding. "
                "This happens when the policy was NO_CACHE."
            )

        # Determine TTL: explicitly provided > policy attached to lookup > engine default
        if ttl_seconds is not None:
            final_ttl = ttl_seconds
        elif lookup_result.policy is not None:
            final_ttl = lookup_result.policy.ttl_seconds
        else:
            final_ttl = self._default_policy.ttl_seconds

        entry = CacheEntry(
            prompt=prompt,
            response=response,
            model=model,
            namespace=lookup_result.namespace,
            created_at=datetime.now(timezone.utc),
            ttl_seconds=final_ttl,
            hit_count=0,
            response_metadata=response_metadata,
        )

        entry_id = await self._store.store(
            embedding=lookup_result.embedding,
            entry=entry,
        )

        logger.info("Cached response (id=%s) for namespace: %s", entry_id, entry.namespace)
        return entry_id

    async def invalidate_namespace(
        self,
        model: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> int:
        """
        Invalidate all cache entries for a specific configuration.

        Use case: When you update a system prompt, all cached responses
        for that system prompt are stale. Call this to clear them.

        Returns:
            Number of entries deleted.
        """
        namespace = build_namespace(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        deleted = await self._store.delete_by_namespace(namespace)
        logger.info("Invalidated %d entries in namespace: %s", deleted, namespace)
        return deleted

    async def stats(self) -> dict:
        """
        Get cache statistics for monitoring.

        Returns a dict suitable for JSON serialization and the /stats endpoint.
        """
        total = await self._store.count()
        return {
            "total_entries": total,
        }
