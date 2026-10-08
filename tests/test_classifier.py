"""
Tests for the IntentClassifier and the adaptive threshold system.

These tests focus on:
  1. Classifier → policy mapping (happy path)
  2. Classifier failure → DEFAULT_POLICY fallback (the critical test)
  3. Per-entry adaptive thresholds in the engine
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import pytest_asyncio

from semcache.cache.classifier import IntentClassifier
from semcache.cache.engine import CacheEngine
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
        a fallback ClassifierResult, not raise.
        """
        with patch.object(
            classifier, "classify", side_effect=TimeoutError("API timeout")
        ):
            result = await classifier.classify_safe("What is Python?")

        assert result.policy == DEFAULT_POLICY
        assert result.tokens == 0
        assert result.is_fallback is True

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
            result = await classifier.classify_safe("How do I sort a list?")

        assert result.policy == DEFAULT_POLICY
        assert result.tokens == 0
        assert result.is_fallback is True

    @pytest.mark.asyncio
    async def test_fallback_uses_configured_default_policy(self):
        """A configured default policy (from settings) wins over DEFAULT_POLICY."""
        configured = CachePolicy(ttl_tier=TTLTier.MEDIUM, similarity_threshold=0.97)
        classifier = IntentClassifier(api_key="fake-key", default_policy=configured)

        with patch.object(
            classifier, "classify", side_effect=RuntimeError("API error")
        ):
            result = await classifier.classify_safe("What is Python?")

        assert result.policy is configured
        assert result.is_fallback is True

    @pytest.mark.asyncio
    async def test_successful_classification(self, classifier: IntentClassifier):
        """
        When classify() succeeds, classify_safe() should return the
        classifier's chosen policy.
        """
        from semcache.cache.classifier import ClassifierResult
        creative_policy = TASK_POLICIES["creative"]
        with patch.object(
            classifier, "classify", return_value=ClassifierResult(policy=creative_policy, tokens=15, is_fallback=False)
        ):
            result = await classifier.classify_safe("Write a poem about rain")

        assert result.policy == creative_policy
        assert result.policy.ttl_seconds == 0  # NO_CACHE
        assert result.policy.similarity_threshold == 0.99
        assert result.tokens == 15
        assert result.is_fallback is False


# ── Adaptive Threshold Tests ─────────────────────────────────

class TestAdaptiveThresholds:
    """
    Test that per-entry required_similarity works correctly.

    The key insight: "the threshold is a property of the cached entry,
    not a property of the request."
    """

    @pytest.mark.asyncio
    async def test_two_entries_top_k_shadowing(self):
        """
        A strict entry (required=0.95) might be the nearest neighbor (similarity=0.93)
        but fail its own threshold. A loose entry (required=0.90) might be the 
        second nearest neighbor (similarity=0.91). 
        
        If we only check top-1, we miss the loose entry.
        If we check top-k, we correctly hit the loose entry.
        """

        from semcache.cache.engine import CacheEngine
        from tests.conftest import InMemoryVectorStore, MockEmbedder

        store = InMemoryVectorStore()
        embedder = MockEmbedder(dims=2)  # 2D for simple math
        engine = CacheEngine(embedder=embedder, store=store)

        # Get the namespace engine will actually use
        lookup_res = await engine.lookup(prompt="Query", model="test-model")
        namespace = lookup_res.namespace
        
        from datetime import UTC, datetime
        
        strict_entry = CacheEntry(
            prompt="Factual prompt",
            response="Factual answer",
            model="test-model",
            namespace=namespace,
            created_at=datetime.now(UTC),
            ttl_seconds=3600,
            hit_count=0,
            required_similarity=0.95
        )
        
        loose_entry = CacheEntry(
            prompt="Classification prompt",
            response="Classification answer",
            model="test-model",
            namespace=namespace,
            created_at=datetime.now(UTC),
            ttl_seconds=3600,
            hit_count=0,
            required_similarity=0.90
        )
        
        # We need normalized vectors for dot product to work correctly in InMemoryVectorStore
        # Query: [1.0, 0.0]
        # Vec 1 (sim 0.93): [0.93, sqrt(1 - 0.93^2)] -> [0.93, 0.367695]
        # Vec 2 (sim 0.91): [0.91, sqrt(1 - 0.91^2)] -> [0.91, 0.414608]
        await store.store([0.93, 0.367695], strict_entry)
        await store.store([0.91, 0.414608], loose_entry)

        # Mock the embedder to return our query vector
        with patch.object(embedder, "embed", return_value=[1.0, 0.0]):
            result = await engine.lookup(prompt="Query", model="test-model")
            
        assert result.hit is True
        assert result.entry is not None
        assert result.entry.prompt == "Classification prompt"
        assert result.similarity == pytest.approx(0.91)

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


# ── Gemma classifier (plain text, instructions in the message) ──

class TestParseCategory:
    @pytest.mark.parametrize(("reply", "category"), [
        ('{"category": "how_to"}', "how_to"),  # Gemini's forced JSON
        ("FACTUAL", "factual"),  # Gemma's plain text
        ("**HOW_TO**", "how_to"),
        ("How-to", "how_to"),
        ("time sensitive", "time_sensitive"),
        ("Creative.", "creative"),
        ("classification\n", "classification"),
    ])
    def test_reads_each_reply_shape(self, reply, category):
        from semcache.cache.classifier import _parse_category

        assert _parse_category(reply) == category

    @pytest.mark.parametrize("reply", ["banana", "", "factual or creative", '{"category": "poem"}'])
    def test_no_single_category_is_none(self, reply):
        from semcache.cache.classifier import _parse_category

        assert _parse_category(reply) is None


def _fake_response(text: str, tokens: int = 400):
    from unittest.mock import MagicMock

    response = MagicMock()
    response.text = text
    response.usage_metadata.total_token_count = tokens
    return response


class TestGemmaClassifier:
    @pytest.mark.asyncio
    async def test_gemma_gets_instructions_in_the_message_and_no_options(self):
        """Gemma rejects system instructions, JSON mode and temperature (500 errors)."""
        from unittest.mock import AsyncMock

        from semcache.cache.classifier import _CLASSIFIER_SYSTEM_PROMPT

        classifier = IntentClassifier(api_key="fake-key", model="gemma-4-26b-a4b-it")
        call = AsyncMock(return_value=_fake_response("HOW_TO"))
        classifier._client.aio.models.generate_content = call

        result = await classifier.classify("How do I sort a list?")

        kwargs = call.await_args.kwargs
        assert kwargs["model"] == "gemma-4-26b-a4b-it"
        assert kwargs["contents"].startswith(_CLASSIFIER_SYSTEM_PROMPT)
        assert kwargs["contents"].endswith("Prompt: How do I sort a list?")
        assert "config" not in kwargs
        assert result.policy == TASK_POLICIES["how_to"]
        assert (result.intent, result.tokens, result.is_fallback) == ("how_to", 400, False)

    @pytest.mark.asyncio
    async def test_gemini_still_uses_structured_output(self):
        from unittest.mock import AsyncMock

        classifier = IntentClassifier(api_key="fake-key", model="gemini-3.5-flash-lite")
        call = AsyncMock(return_value=_fake_response('{"category": "factual"}'))
        classifier._client.aio.models.generate_content = call

        result = await classifier.classify("What is Python?")

        config = call.await_args.kwargs["config"]
        assert config.response_mime_type == "application/json"
        assert config.system_instruction
        assert result.intent == "factual"


def _provider_reply(text: str, tokens: int = 150):
    """An OpenAI-compatible provider's ChatCompletionResponse carrying `text`."""
    from semcache.schemas import (
        ChatCompletionChoice,
        ChatCompletionChoiceMessage,
        ChatCompletionResponse,
        UsageInfo,
    )

    return ChatCompletionResponse(
        id="chatcmpl-test",
        created=0,
        model="mistral-small",
        choices=[ChatCompletionChoice(message=ChatCompletionChoiceMessage(content=text))],
        usage=UsageInfo(prompt_tokens=tokens - 3, completion_tokens=3, total_tokens=tokens),
    )


class TestOtherProviderClassifier:
    @pytest.mark.asyncio
    async def test_non_gemini_model_is_asked_through_the_other_provider(self):
        from unittest.mock import AsyncMock, MagicMock

        from semcache.cache.classifier import _CLASSIFIER_SYSTEM_PROMPT

        provider = MagicMock()
        provider.generate = AsyncMock(return_value=_provider_reply("TIME_SENSITIVE"))
        classifier = IntentClassifier(
            api_key="fake-key", model="Mistral Small 3.2 24B Instruct 2506", other_provider=provider
        )
        gemini = AsyncMock()
        classifier._client.aio.models.generate_content = gemini

        result = await classifier.classify("What's the weather today?")

        request = provider.generate.await_args.args[0]
        assert request.model == "Mistral Small 3.2 24B Instruct 2506"
        assert [(m.role, m.content) for m in request.messages] == [
            ("system", _CLASSIFIER_SYSTEM_PROMPT),
            ("user", "What's the weather today?"),
        ]
        assert request.temperature == 0.0
        gemini.assert_not_awaited()
        assert result.policy == TASK_POLICIES["time_sensitive"]
        assert (result.intent, result.tokens, result.is_fallback) == ("time_sensitive", 150, False)

    @pytest.mark.asyncio
    async def test_gemini_model_stays_on_gemini_with_a_provider_set(self):
        from unittest.mock import AsyncMock, MagicMock

        provider = MagicMock()
        provider.generate = AsyncMock()
        classifier = IntentClassifier(api_key="fake-key", model="gemini-3.1-flash-lite", other_provider=provider)
        classifier._client.aio.models.generate_content = AsyncMock(
            return_value=_fake_response('{"category": "factual"}')
        )

        result = await classifier.classify("What is Python?")

        provider.generate.assert_not_awaited()
        assert result.intent == "factual"

    @pytest.mark.asyncio
    async def test_without_the_provider_it_falls_back(self):
        classifier = IntentClassifier(api_key="fake-key", model="Mistral Small 3.2 24B Instruct 2506")

        result = await classifier.classify_safe("What is Python?")

        assert result.is_fallback is True
        assert result.policy == DEFAULT_POLICY

    @pytest.mark.asyncio
    async def test_unreadable_reply_is_a_fallback(self):
        from unittest.mock import AsyncMock, MagicMock

        provider = MagicMock()
        provider.generate = AsyncMock(return_value=_provider_reply("It could be factual or how_to."))
        classifier = IntentClassifier(api_key="fake-key", model="qwen-qwen3-8-27b", other_provider=provider)

        result = await classifier.classify_safe("What is Python?")

        assert result.is_fallback is True

    @pytest.mark.asyncio
    async def test_rate_limit_from_the_provider_is_retried(self):
        from unittest.mock import AsyncMock, MagicMock

        from semcache.providers.openai_compatible import ProviderHTTPError

        provider = MagicMock()
        provider.generate = AsyncMock(
            side_effect=[ProviderHTTPError(429, "slow down"), _provider_reply("FACTUAL")]
        )
        classifier = IntentClassifier(
            api_key="fake-key", model="qwen-qwen3-8-27b", other_provider=provider, retry_delays=(0, 0, 0)
        )

        result = await classifier.classify_safe("What is Python?")

        assert provider.generate.await_count == 2
        assert (result.intent, result.is_fallback) == ("factual", False)

    @pytest.mark.asyncio
    async def test_unreadable_reply_is_a_fallback_not_a_guess(self):
        from unittest.mock import AsyncMock

        classifier = IntentClassifier(api_key="fake-key", model="gemma-4-26b-a4b-it")
        classifier._client.aio.models.generate_content = AsyncMock(
            return_value=_fake_response("I think this might be a poem or a fact.")
        )

        result = await classifier.classify_safe("Write about the sea")

        assert result.is_fallback is True
        assert result.policy == DEFAULT_POLICY
        assert result.intent is None


# ── Retrying brief Google errors ──

class TestClassifierRetry:
    @staticmethod
    def _google_error(code: int) -> Exception:
        from google.genai import errors

        cls = errors.ClientError if code < 500 else errors.ServerError
        return cls(code, {"error": {"code": code, "message": "brief", "status": "X"}})

    @staticmethod
    def _ok():
        from semcache.cache.classifier import ClassifierResult

        return ClassifierResult(policy=TASK_POLICIES["factual"], tokens=200, is_fallback=False, intent="factual")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("code", [429, 500, 503])
    async def test_brief_errors_are_retried(self, code):
        classifier = IntentClassifier(api_key="fake-key", retry_delays=(0, 0, 0))
        with patch.object(
            classifier, "classify", side_effect=[self._google_error(code), self._google_error(code), self._ok()]
        ) as classify:
            result = await classifier.classify_safe("What is Python?")

        assert classify.call_count == 3
        assert result.is_fallback is False
        assert result.intent == "factual"

    @pytest.mark.asyncio
    async def test_gives_up_after_the_last_retry(self):
        classifier = IntentClassifier(api_key="fake-key", retry_delays=(0, 0, 0))
        with patch.object(classifier, "classify", side_effect=self._google_error(503)) as classify:
            result = await classifier.classify_safe("What is Python?")

        assert classify.call_count == 4  # the first try and three retries
        assert result.is_fallback is True
        assert result.policy == DEFAULT_POLICY

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exc", [
        TimeoutError("already waited the whole timeout"),
        ValueError("reply names no category"),
        "bad request",
    ])
    async def test_other_failures_are_not_retried(self, exc):
        classifier = IntentClassifier(api_key="fake-key", retry_delays=(0, 0, 0))
        if exc == "bad request":
            exc = self._google_error(400)
        with patch.object(classifier, "classify", side_effect=exc) as classify:
            result = await classifier.classify_safe("What is Python?")

        assert classify.call_count == 1
        assert result.is_fallback is True

    @pytest.mark.asyncio
    async def test_waits_between_retries(self):
        classifier = IntentClassifier(api_key="fake-key")
        with (
            patch.object(classifier, "classify", side_effect=[self._google_error(503), self._google_error(503), self._ok()]),
            patch("semcache.cache.classifier.asyncio.sleep") as sleep,
        ):
            await classifier.classify_safe("What is Python?")

        assert [c.args[0] for c in sleep.await_args_list] == [1.0, 2.0]
