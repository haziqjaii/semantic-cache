from unittest.mock import AsyncMock, MagicMock

import pytest

from semcache.api.chat import chat_completions
from semcache.cache.classifier import ClassifierResult, IntentClassifier
from semcache.cache.engine import CacheEngine, LookupResult
from semcache.cache.policy import DEFAULT_POLICY, CachePolicy
from semcache.metrics import metrics
from semcache.providers.base import LLMProvider
from semcache.schemas import ChatCompletionRequest, ChatMessage


@pytest.fixture
def base_request():
    return ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Hello")]
    )

@pytest.fixture
def mock_provider():
    provider = MagicMock(spec=LLMProvider)
    from semcache.schemas import ChatCompletionChoice, ChatCompletionResponse
    from semcache.schemas import ChatCompletionChoiceMessage as RespMessage
    provider.generate = AsyncMock(return_value=ChatCompletionResponse(
        id="test_id",
        object="chat.completion",
        created=123,
        model="test-model",
        choices=[ChatCompletionChoice(index=0, message=RespMessage(role="assistant", content="Response"))]
    ))
    return provider

@pytest.fixture
def mock_engine():
    engine = MagicMock(spec=CacheEngine)
    engine.lookup = AsyncMock(return_value=LookupResult(
        hit=False,
        entry=None,
        similarity=0.0,
        namespace="ns",
        embedding=[0.0],
        policy=None
    ))
    engine.store = AsyncMock()
    return engine


@pytest.mark.asyncio
async def test_chat_completions_fallback_classifier(base_request, mock_provider, mock_engine):
    """Test that classifier_calls_fallback increments on fallback."""
    classifier = MagicMock(spec=IntentClassifier)
    classifier.classify_safe = AsyncMock(return_value=ClassifierResult(
        policy=DEFAULT_POLICY,
        tokens=0,
        is_fallback=True
    ))

    await chat_completions(base_request, mock_engine, mock_provider, classifier)
    
    assert metrics.classifier_calls_fallback == 1
    assert metrics.classifier_calls_success == 0
    assert metrics.classifier_tokens_total == 0

@pytest.mark.asyncio
async def test_chat_completions_success_classifier_none_tokens(base_request, mock_provider, mock_engine):
    """Test that classifier_calls_success increments on valid policy with zero tokens."""
    classifier = MagicMock(spec=IntentClassifier)
    from semcache.cache.policy import TTLTier
    policy = CachePolicy(ttl_tier=TTLTier.LONG, similarity_threshold=0.95)
    classifier.classify_safe = AsyncMock(return_value=ClassifierResult(
        policy=policy,
        tokens=0,
        is_fallback=False
    ))

    await chat_completions(base_request, mock_engine, mock_provider, classifier)
    
    assert metrics.classifier_calls_success == 1
    assert metrics.classifier_calls_fallback == 0
    assert metrics.classifier_tokens_total == 0

@pytest.mark.asyncio
async def test_chat_completions_success_classifier_with_tokens(base_request, mock_provider, mock_engine):
    """Test that tokens are tracked."""
    classifier = MagicMock(spec=IntentClassifier)
    from semcache.cache.policy import TTLTier
    policy = CachePolicy(ttl_tier=TTLTier.LONG, similarity_threshold=0.95)
    classifier.classify_safe = AsyncMock(return_value=ClassifierResult(
        policy=policy,
        tokens=50,
        is_fallback=False
    ))

    await chat_completions(base_request, mock_engine, mock_provider, classifier)
    
    assert metrics.classifier_calls_success == 1
    assert metrics.classifier_tokens_total == 50

@pytest.mark.asyncio
async def test_chat_completions_cache_hit(base_request, mock_provider, mock_engine):
    """Test that metrics correctly track a cache hit, and classifier is NOT called."""
    classifier = MagicMock(spec=IntentClassifier)
    
    from semcache.cache.store.base import CacheEntry
    entry = CacheEntry(prompt="Hello", response="Response", model="test-model", namespace="ns")
    
    mock_engine.lookup = AsyncMock(return_value=LookupResult(
        hit=True,
        entry=entry,
        similarity=0.99,
        namespace="ns",
        embedding=[0.0],
        policy=DEFAULT_POLICY
    ))
    
    await chat_completions(base_request, mock_engine, mock_provider, classifier)
    
    # We should have one hit, 0 misses, and the classifier should never have been invoked
    assert metrics.cache_hits == 1
    assert metrics.cache_misses == 0
    assert metrics.classifier_calls_skipped == 1  # Skipped because of hit
    assert metrics.classifier_calls_success == 0
    assert metrics.classifier_calls_fallback == 0
    
    classifier.classify_safe.assert_not_called()
