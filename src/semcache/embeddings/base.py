"""
Abstract base class for embedding providers.

WHY AN ABSTRACT CLASS?
    We want the cache engine to work with ANY embedding provider — Gemini,
    OpenAI, a local sentence-transformers model, or something that doesn't
    exist yet. By coding against this interface (not a concrete class), the
    engine never needs to change when you swap providers.

    This is the Strategy Pattern: define a family of algorithms (embedding
    providers), make them interchangeable, and let the client (cache engine)
    pick one at runtime.

    In practice:
        engine = CacheEngine(embedder=GeminiEmbedder(...))   # production
        engine = CacheEngine(embedder=MockEmbedder(...))     # tests
"""

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Interface that all embedding providers must implement."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """
        Embed a single text string into a vector.

        Args:
            text: The input text to embed (e.g., a user prompt).

        Returns:
            A list of floats representing the embedding vector.
            Length must equal the configured embedding dimensions.
        """

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts in a single call.

        Why batch?
            API calls have overhead (network round-trip, auth, etc.).
            Sending 10 texts in 1 call is much faster than 10 separate calls.
            For cache warm-up or bulk operations, this matters.

        Args:
            texts: List of input texts to embed.

        Returns:
            A list of embedding vectors, one per input text.
            Order matches the input order.
        """
