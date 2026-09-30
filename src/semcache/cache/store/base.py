"""
Abstract base class for the vector store.

The vector store is responsible for:
  1. Storing embeddings alongside their cached responses
  2. Finding the most similar embedding to a query (nearest neighbor search)
  3. Respecting namespace boundaries (different contexts = different search spaces)

WHY ABSTRACT?
    Same reason as the Embedder interface — we might want to swap Redis for
    Qdrant, Pinecone, or an in-memory store for testing. The cache engine
    codes against this interface, not a specific database.

WHAT'S A VECTOR STORE?
    A regular database indexes data by keys (user_id=42) or text (LIKE '%python%').
    A vector store indexes data by SIMILARITY. You give it a vector and ask:
    "What's the closest vector you've seen before?"

    Under the hood, it uses algorithms like HNSW (Hierarchical Navigable Small
    World) to do this efficiently — O(log n) instead of O(n) brute force.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime


def _now_utc() -> datetime:
    return datetime.now(UTC)


@dataclass
class CacheEntry:
    """
    A single cached response with its metadata.

    This is what we store in the vector store alongside the embedding.
    When we get a cache hit, we return this entire object.
    """

    # The original user prompt that generated this response
    prompt: str

    # The full LLM response text
    response: str

    # Which model generated this response
    model: str

    # The namespace this entry belongs to (from keys.py)
    namespace: str

    # Metadata for cache management
    created_at: datetime = field(default_factory=_now_utc)
    ttl_seconds: int = 86400  # When this entry expires
    hit_count: int = 0  # How many times this entry has been served

    # Per-entry adaptive similarity threshold.
    # Set by the classifier at store time. During lookup, the engine
    # compares the candidate's similarity against THIS value, not
    # a global default. This is what makes thresholds adaptive:
    #   "The threshold is a property of the cached entry, not of the request."
    required_similarity: float = 0.95

    # Optional: full response metadata (token counts, finish reason, etc.)
    # Stored as a JSON-serializable dict so we can return it to the client.
    response_metadata: dict | None = None

    # Invalidation handles (see EntryFilter). The system prompt hash
    # (keys.hash_system_prompt) selects every entry for one system prompt,
    # across models and parameters.
    system_prompt_hash: str = ""

    # Labels supplied by the client (X-Cache-Tags header), lowercased.
    # They group entries for invalidation; they don't partition the cache.
    tags: list[str] = field(default_factory=list)

    # The unique ID of this entry in the store (e.g., Redis key)
    id: str = ""


@dataclass(frozen=True)
class EntryFilter:
    """
    Selects cache entries to invalidate or count.

    Set fields are combined with AND. An empty filter matches every entry,
    so callers must opt in to that explicitly (see CacheEngine.invalidate).
    Matching is case-insensitive, as Redis TAG fields are.
    """

    namespace: str | None = None
    model: str | None = None
    system_prompt_hash: str | None = None
    tag: str | None = None  # entry has this tag
    tag_prefix: str | None = None  # entry has a tag starting with this

    def __post_init__(self) -> None:
        # An empty value would silently match everything in Redis
        # (RedisVL turns `field == ""` into a wildcard), so reject it.
        for name, value in self.as_dict().items():
            if not value.strip():
                raise ValueError(f"EntryFilter.{name} must not be empty")

    def as_dict(self) -> dict[str, str]:
        """The fields that are set."""
        return {
            name: value
            for name in ("namespace", "model", "system_prompt_hash", "tag", "tag_prefix")
            if (value := getattr(self, name)) is not None
        }

    def is_empty(self) -> bool:
        return not self.as_dict()

    def matches(self, entry: CacheEntry) -> bool:
        """Reference semantics, used by stores that filter in Python."""

        def same(a: str, b: str) -> bool:
            return a.lower() == b.lower()

        tags = [t.lower() for t in entry.tags]
        return (
            (self.namespace is None or same(entry.namespace, self.namespace))
            and (self.model is None or same(entry.model, self.model))
            and (
                self.system_prompt_hash is None
                or same(entry.system_prompt_hash, self.system_prompt_hash)
            )
            and (self.tag is None or self.tag.lower() in tags)
            and (
                self.tag_prefix is None
                or any(t.startswith(self.tag_prefix.lower()) for t in tags)
            )
        )


class VectorStore(ABC):
    """Interface for vector similarity search backends."""

    @abstractmethod
    async def search(
        self,
        embedding: list[float],
        namespace: str,
        threshold: float = 0.95,
        top_k: int = 5,
    ) -> list[tuple[CacheEntry, float]]:
        """
        Find the most similar cached entries in the given namespace.

        Args:
            embedding: The query vector (from the user's prompt).
            namespace: Only search within this namespace (from keys.py).
            threshold: Minimum cosine similarity to return.
            top_k: Max number of candidates to return.

        Returns:
            A list of tuples (CacheEntry, similarity_score) sorted by
            similarity descending.
        """

    @abstractmethod
    async def record_hit(self, entry_id: str) -> None:
        """
        Increment the hit count for a specific entry.
        """

    @abstractmethod
    async def store(
        self,
        embedding: list[float],
        entry: CacheEntry,
    ) -> str:
        """
        Store a new cache entry with its embedding.

        Args:
            embedding: The vector for this entry's prompt.
            entry: The full cache entry (prompt, response, metadata).

        Returns:
            The unique ID assigned to this entry in the store.
        """

    @abstractmethod
    async def delete_matching(self, entry_filter: EntryFilter) -> int:
        """
        Delete every entry matching the filter (an empty filter deletes all).

        Used for cache invalidation — e.g., when a system prompt changes or
        a model is upgraded, the affected cached responses are stale.

        Returns:
            The number of entries deleted.
        """

    @abstractmethod
    async def count(self, entry_filter: EntryFilter | None = None) -> int:
        """
        Count entries, optionally only those matching a filter.

        Used for monitoring and admin endpoints.
        """

    async def backend_stats(self) -> dict:
        """
        Backend-specific counters for monitoring (e.g. evicted/expired keys).

        Optional: stores with nothing to report keep this default.
        """
        return {}
