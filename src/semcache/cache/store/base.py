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


class VectorStore(ABC):
    """Interface for vector similarity search backends."""

    @abstractmethod
    async def search(
        self,
        embedding: list[float],
        namespace: str,
        threshold: float = 0.95,
    ) -> tuple[CacheEntry, float] | None:
        """
        Find the most similar cached entry in the given namespace.

        Args:
            embedding: The query vector (from the user's prompt).
            namespace: Only search within this namespace (from keys.py).
            threshold: Minimum cosine similarity to consider a "hit".

        Returns:
            A tuple of (CacheEntry, similarity_score) if a hit is found,
            or None if no entry exceeds the threshold.

        Why return the score too?
            For monitoring and threshold tuning. We want to track the
            distribution of similarity scores to find the optimal threshold.
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
    async def delete_by_namespace(self, namespace: str) -> int:
        """
        Delete all entries in a namespace.

        Used for cache invalidation — e.g., when a system prompt changes,
        all cached responses for that system prompt are stale.

        Returns:
            The number of entries deleted.
        """

    @abstractmethod
    async def count(self, namespace: str | None = None) -> int:
        """
        Count entries, optionally filtered by namespace.

        Used for monitoring and admin endpoints.
        """
