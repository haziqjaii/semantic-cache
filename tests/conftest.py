"""
Shared test fixtures.

WHAT ARE FIXTURES?
    pytest fixtures are reusable setup/teardown functions. Instead of
    repeating "create a mock embedder" in every test, you define it once
    here and pytest injects it automatically into any test that asks for it.

WHY MOCK THE EMBEDDER?
    We don't want tests to call the real Gemini API because:
      1. Tests should run offline (no internet required)
      2. Tests should be fast (no API latency)
      3. Tests should be free (no API quota burned)
      4. Tests should be deterministic (same input = same output, always)

    The MockEmbedder returns predictable vectors based on a simple hash,
    so we can reason about similarity in tests.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from semcache.cache.store.base import CacheEntry, VectorStore
from semcache.embeddings.base import Embedder


# ── Mock Embedder ───────────────────────────────────────────

class MockEmbedder(Embedder):
    """
    A deterministic embedder for testing.

    Instead of calling an API, it generates a vector by hashing the text.
    Same text → same vector (deterministic).

    The vectors are normalized (unit length) so cosine similarity works
    correctly. Similar texts DON'T get similar vectors here — that's fine,
    we test similarity logic separately.
    """

    def __init__(self, dims: int = 768) -> None:
        self._dims = dims

    async def embed(self, text: str) -> list[float]:
        # Hash the text to get deterministic bytes.
        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Expand the hash to fill our vector dimensions.
        # We cycle through the hash bytes to fill the vector.
        expanded = (hash_bytes * (self._dims // len(hash_bytes) + 1))[:self._dims]
        vec = np.frombuffer(bytearray(expanded), dtype=np.uint8).astype(np.float32)
        # Normalize to unit length (required for meaningful cosine similarity).
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


# ── In-Memory Vector Store ──────────────────────────────────

class InMemoryVectorStore(VectorStore):
    """
    In-memory vector store for testing (no Redis required).

    Stores entries in a plain Python list and does brute-force cosine
    similarity search. Not efficient, but perfect for tests because:
      - No Docker/Redis needed
      - Deterministic and inspectable
      - Tests the LOGIC, not the Redis integration
    """

    def __init__(self) -> None:
        self._entries: list[tuple[list[float], CacheEntry]] = []

    async def search(
        self,
        embedding: list[float],
        namespace: str,
        threshold: float = 0.95,
    ) -> tuple[CacheEntry, float] | None:
        best_entry = None
        best_score = -1.0

        query_vec = np.array(embedding, dtype=np.float32)

        for stored_emb, entry in self._entries:
            if entry.namespace != namespace:
                continue
            stored_vec = np.array(stored_emb, dtype=np.float32)
            # Cosine similarity = dot(a, b) / (|a| * |b|)
            dot = np.dot(query_vec, stored_vec)
            norm_q = np.linalg.norm(query_vec)
            norm_s = np.linalg.norm(stored_vec)
            if norm_q == 0 or norm_s == 0:
                continue
            similarity = dot / (norm_q * norm_s)

            if similarity > best_score:
                best_score = similarity
                best_entry = entry

        if best_entry is not None and best_score >= threshold:
            return best_entry, float(best_score)
        return None

    async def store(self, embedding: list[float], entry: CacheEntry) -> str:
        import uuid
        self._entries.append((embedding, entry))
        return str(uuid.uuid4())

    async def delete_by_namespace(self, namespace: str) -> int:
        before = len(self._entries)
        self._entries = [
            (emb, entry) for emb, entry in self._entries
            if entry.namespace != namespace
        ]
        return before - len(self._entries)

    async def count(self, namespace: str | None = None) -> int:
        if namespace is None:
            return len(self._entries)
        return sum(1 for _, entry in self._entries if entry.namespace == namespace)


# ── pytest Fixtures ─────────────────────────────────────────

@pytest.fixture
def mock_embedder() -> MockEmbedder:
    """Provide a mock embedder for tests."""
    return MockEmbedder(dims=768)


@pytest.fixture
def memory_store() -> InMemoryVectorStore:
    """Provide an in-memory vector store for tests."""
    return InMemoryVectorStore()
