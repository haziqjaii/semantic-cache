"""
Gemini embedding provider using the google-genai SDK.

This is the PRIMARY embedder for the semantic cache. It calls the Gemini
Embedding API to convert text into dense vectors that capture meaning.

HOW EMBEDDINGS WORK (conceptual):
    An embedding model maps text → a fixed-size list of floats (a "vector").
    Texts with similar MEANING get vectors that are close together in space.

    Example:
        embed("What is Python?")    → [0.12, -0.34, 0.56, ...]
        embed("Explain Python")     → [0.11, -0.33, 0.57, ...]  ← very close!
        embed("Recipe for cake")    → [0.89, 0.23, -0.67, ...]  ← far away

    "Close together" is measured by cosine similarity (we'll use that in the
    vector store). Two identical texts have cosine similarity = 1.0. Two
    completely unrelated texts approach 0.0.

WHY gemini-embedding-001?
    - Free tier: 1,500 requests/day (more than enough for development)
    - 768 dimensions by default (good balance of quality vs. speed)
    - Configurable output_dimensionality (can shrink for faster lookups)
    - Same SDK (google-genai) we use for generation — one dependency
"""

from google import genai
from google.genai import types

from semcache.embeddings.base import Embedder


class GeminiEmbedder(Embedder):
    """
    Embeds text using the Gemini Embedding API.

    Args:
        api_key: Your Gemini API key.
        model: Model name (default: gemini-embedding-001).
        dims: Output dimensionality. Lower = faster lookups but less precise.
              768 is the sweet spot for our use case.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        dims: int = 768,
    ) -> None:
        # The genai Client handles auth, retries, and connection pooling.
        # We create it once and reuse it for all embedding calls.
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dims = dims

    async def embed(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Note: The Gemini SDK's embed_content is synchronous under the hood.
        We wrap it here to match our async interface. In Phase 2, when we're
        inside FastAPI's async request handlers, this prevents blocking the
        event loop (FastAPI runs sync functions in a thread pool automatically
        when called from async routes via Depends()).
        """
        response = self._client.models.embed_content(
            model=self._model,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=self._dims,
            ),
        )
        # response.embeddings is a list (one per input content).
        # We sent one text, so we get one embedding back.
        return response.embeddings[0].values

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts in a single API call.

        The Gemini API accepts a list of contents and returns embeddings
        for all of them in one round-trip. This is 5-10x faster than
        calling embed() in a loop for bulk operations.
        """
        if not texts:
            return []

        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=types.EmbedContentConfig(
                output_dimensionality=self._dims,
            ),
        )
        return [emb.values for emb in response.embeddings]
