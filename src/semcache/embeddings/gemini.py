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
    - 768 dimensions by default (good balance of quality vs. speed)
    - Configurable output_dimensionality (can shrink for faster lookups)
    - Same SDK (google-genai) we use for generation — one dependency
"""

import numpy as np
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
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dims = dims

    def _normalize(self, values: list[float]) -> list[float]:
        """Normalize vector to unit length so cosine similarity works."""
        vec = np.array(values, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    async def embed(self, text: str) -> list[float]:
        """
        Embed a single text string using the async client.
        """
        response = await self._client.aio.models.embed_content(
            model=self._model,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=self._dims,
                task_type="SEMANTIC_SIMILARITY",
            ),
        )
        return self._normalize(response.embeddings[0].values)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts in a single API call using the async client.
        """
        if not texts:
            return []

        response = await self._client.aio.models.embed_content(
            model=self._model,
            contents=texts,
            config=types.EmbedContentConfig(
                output_dimensionality=self._dims,
                task_type="SEMANTIC_SIMILARITY",
            ),
        )
        return [self._normalize(emb.values) for emb in response.embeddings]
