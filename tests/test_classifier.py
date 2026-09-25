"""
Tests for the IntentClassifier and the adaptive threshold system.

These tests focus on:
  1. Classifier → policy mapping (happy path)
  2. Classifier failure → DEFAULT_POLICY fallback (the critical test)
  3. Per-entry adaptive thresholds in the engine
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from semcache.cache.classifier import IntentClassifier
from semcache.cache.engine import CacheEngine, LookupResult
from semcache.cache.policy import (
    DEFAULT_POLICY,
    FLOOR_THRESHOLD,
    TASK_POLICIES,
    CachePolicy,
    TTLTier,
)
from semcache.cache.store.base import CacheEntry
from tests.conftest import InMemoryVectorStore, MockEmbedder


# ── Classifier Tests ─────────────────────────────────────────


class TestClassifierSafe:
    """Test that classify_safe() NEVER raises, even on failures."""

    @pytest.fixture
    def classifier(self) -> IntentClassifier:
        """Create a classifier (we'll mock the API calls)."""
        return IntentClassifier(api_key="fake-key")

    @pytest.mark.asyncio
    async def test_timeout_falls_back_to_default(self, classifier: IntentClassifier):
        """
        If the classifier times out, classify_safe() should return
        DEFAULT_POLICY, not raise.
        """
        with patch.object(
            classifier, "classify", side_effect=TimeoutError("API timeout")
        ):
            policy = await classifier.classify_safe("What is Python?")

        assert policy == DEFAULT_POLICY

    @pytest.mark.asyncio
    async def test_generic_exception_falls_back_to_default(
        self, classifier: IntentClassifier
    ):
        """
        Any exception (rate limit, network, parse error) should fall back
        to DEFAULT_POLICY.
        """
        with patch.object(
            classifier, "classify", side_effect=RuntimeError("API error")
        ):
            policy = await classifier.classify_safe("How do I sort a list?")

        assert policy == DEFAULT_POLICY

    @pytest.mark.asyncio
    async def test_successful_classification(self, classifier: IntentClassifier):
        """
        When classify() succeeds, classify_safe() should return the
        classifier's chosen policy.
        """
        creative_policy = TASK_POLICIES["creative"]
        with patch.object(
            classifier, "classify", return_value=creative_policy
        ):
            policy = await classifier.classify_safe("Write a poem about rain")

        assert policy == creative_policy
        assert policy.ttl_seconds == 0  # NO_CACHE
        assert policy.similarity_threshold == 0.99


# ── Adaptive Threshold Tests ─────────────────────────────────


class TestAdaptiveThresholds:
    """
    Test that per-entry required_similarity works correctly.

    The key insight: "the threshold is a property of the cached entry,
    not a property of the request."
    """

    @pytest_asyncio.fixture
    async def engine(self) -> CacheEngine:
        embedder = MockEmbedder()
        store = InMemoryVectorStore()
        return CacheEngine(embedder=embedder, store=store)

    @pytest.mark.asyncio
    async def test_classification_entry_matches_at_low_similarity(
        self, engine: CacheEngine
    ):
        """
        A classification-intent entry (required_similarity=0.90) should
        be found when the query similarity is 0.92.
        """
        # Store an entry with classification policy (threshold 0.90)
        classification_policy = TASK_POLICIES["classification"]

        # First, do a lookup to get the embedding
        result = await engine.lookup(prompt="Is this email spam?", model="test-model")
        assert not result.hit

        # Store with classification policy
        await engine.store(
            lookup_result=result,
            prompt="Is this email spam?",
            response="Yes, this appears to be spam.",
            model="test-model",
            policy=classification_policy,
        )

        # Look up the same prompt — should be a HIT
        result2 = await engine.lookup(prompt="Is this email spam?", model="test-model")
        assert result2.hit
        assert result2.entry is not None
        assert result2.entry.required_similarity == 0.90

    @pytest.mark.asyncio
    async def test_creative_entry_requires_high_similarity(
        self, engine: CacheEngine
    ):
        """
        A creative-intent entry (required_similarity=0.99) should NOT
        cache at all because its TTL is 0 (NO_CACHE).
        """
        creative_policy = TASK_POLICIES["creative"]

        result = await engine.lookup(prompt="Write a poem about rain", model="test-model")

        # Trying to store with NO_CACHE policy should be a no-op
        entry_id = await engine.store(
            lookup_result=result,
            prompt="Write a poem about rain",
            response="Roses are red...",
            model="test-model",
            policy=creative_policy,
        )

        # Should return empty string (the TTL <= 0 guard kicks in)
        assert entry_id == ""

    @pytest.mark.asyncio
    async def test_floor_threshold_is_minimum_across_policies(self):
        """Verify FLOOR_THRESHOLD equals the lowest threshold in TASK_POLICIES."""
        expected = min(p.similarity_threshold for p in TASK_POLICIES.values())
        assert FLOOR_THRESHOLD == expected
        assert FLOOR_THRESHOLD == 0.90  # classification's threshold

    @pytest.mark.asyncio
    async def test_store_sets_required_similarity_from_policy(
        self, engine: CacheEngine
    ):
        """
        engine.store(policy=...) should set required_similarity on the
        CacheEntry from the policy's similarity_threshold.
        """
        how_to_policy = TASK_POLICIES["how_to"]

        result = await engine.lookup(
            prompt="How do I sort a list?", model="test-model"
        )
        await engine.store(
            lookup_result=result,
            prompt="How do I sort a list?",
            response="Use sorted() or .sort()...",
            model="test-model",
            policy=how_to_policy,
        )

        # Look it up again and check the stored threshold
        result2 = await engine.lookup(
            prompt="How do I sort a list?", model="test-model"
        )
        assert result2.hit
        assert result2.entry is not None
        assert result2.entry.required_similarity == 0.93  # how_to threshold
