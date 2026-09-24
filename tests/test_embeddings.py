"""
Tests for the embedding providers.

These tests verify:
  1. MockEmbedder produces correct dimensions
  2. Same text → same embedding (deterministic)
  3. Different texts → different embeddings
  4. Embeddings are normalized (unit length)
  5. Batch embedding works correctly
"""

import numpy as np
import pytest


class TestMockEmbedder:
    """Test the MockEmbedder from conftest."""

    @pytest.mark.asyncio
    async def test_correct_dimensions(self, mock_embedder):
        """Embedding should have the configured number of dimensions."""
        embedding = await mock_embedder.embed("test text")
        assert len(embedding) == 768

    @pytest.mark.asyncio
    async def test_deterministic(self, mock_embedder):
        """Same input must produce the same embedding every time."""
        emb1 = await mock_embedder.embed("What is Python?")
        emb2 = await mock_embedder.embed("What is Python?")
        assert emb1 == emb2

    @pytest.mark.asyncio
    async def test_different_texts_different_embeddings(self, mock_embedder):
        """Different inputs must produce different embeddings."""
        emb1 = await mock_embedder.embed("What is Python?")
        emb2 = await mock_embedder.embed("Recipe for cake")
        assert emb1 != emb2

    @pytest.mark.asyncio
    async def test_embeddings_are_normalized(self, mock_embedder):
        """Embeddings should be unit vectors (length ≈ 1.0)."""
        embedding = await mock_embedder.embed("test text")
        vec = np.array(embedding)
        norm = np.linalg.norm(vec)
        # Allow small floating point tolerance.
        assert abs(norm - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_batch_embedding(self, mock_embedder):
        """Batch embedding should return one vector per input."""
        texts = ["Hello", "World", "Python"]
        embeddings = await mock_embedder.embed_batch(texts)
        assert len(embeddings) == 3
        # Each should have correct dimensions.
        for emb in embeddings:
            assert len(emb) == 768

    @pytest.mark.asyncio
    async def test_batch_matches_individual(self, mock_embedder):
        """Batch results should match individual embed() calls."""
        texts = ["Hello", "World"]
        batch = await mock_embedder.embed_batch(texts)
        individual = [await mock_embedder.embed(t) for t in texts]
        assert batch == individual
