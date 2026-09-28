from unittest.mock import AsyncMock, MagicMock

import pytest

from semcache.api.chat import chat_completions
from semcache.cache.classifier import ClassifierResult, IntentClassifier
from semcache.cache.engine import CacheEngine, LookupResult
from semcache.cache.policy import DEFAULT_POLICY, CachePolicy
from semcache.metrics import metrics
from semcache.providers.base import LLMProvider
from semcache.schemas import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    UsageInfo,
)


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


def _response(content: str, prompt_tokens: int = 0, completion_tokens: int = 0):
    return ChatCompletionResponse(
        id="test_id",
        created=123,
        model="test-model",
        choices=[ChatCompletionChoice(message=ChatCompletionChoiceMessage(content=content))],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _classifier(policy: CachePolicy = DEFAULT_POLICY) -> MagicMock:
    classifier = MagicMock(spec=IntentClassifier)
    classifier.classify_safe = AsyncMock(
        return_value=ClassifierResult(policy=policy, tokens=0, is_fallback=False)
    )
    return classifier


# ── Item 1: empty responses are never cached ────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "   \n"])
async def test_empty_response_is_not_cached(base_request, mock_provider, mock_engine, content):
    mock_provider.generate = AsyncMock(return_value=_response(content))

    response = await chat_completions(base_request, mock_engine, mock_provider, _classifier())

    assert response.choices[0].message.content == content
    mock_engine.store.assert_not_called()


# ── Item 2: cache failures never break the request ──────────

@pytest.mark.asyncio
async def test_lookup_failure_serves_uncached(base_request, mock_provider, mock_engine):
    mock_engine.lookup = AsyncMock(side_effect=ConnectionError("Redis down"))
    mock_provider.generate = AsyncMock(return_value=_response("Answer", 10, 20))
    classifier = _classifier()

    response = await chat_completions(base_request, mock_engine, mock_provider, classifier)

    assert response.choices[0].message.content == "Answer"
    assert response.x_cache_status == "BYPASS (Cache error)"
    assert metrics.cache_lookup_errors == 1
    assert metrics.cache_misses == 0
    assert metrics.classifier_calls_skipped == 1
    assert metrics.llm_calls == 1
    # Single-turn, so it still informs the per-hit savings average.
    assert metrics.llm_calls_cacheable == 1
    classifier.classify_safe.assert_not_called()
    mock_engine.store.assert_not_called()


@pytest.mark.asyncio
async def test_store_failure_still_returns_response(base_request, mock_provider, mock_engine):
    mock_engine.store = AsyncMock(side_effect=ConnectionError("Redis down"))

    response = await chat_completions(base_request, mock_engine, mock_provider, _classifier())

    assert response.choices[0].message.content == "Response"
    assert metrics.cache_store_errors == 1
    assert metrics.cache_misses == 1
    assert metrics.llm_calls == 1


# ── Item 3: fallbacks use the engine's configured default ───

@pytest.mark.asyncio
async def test_classifier_exception_uses_engine_default_policy(
    base_request, mock_provider, mock_engine
):
    from semcache.cache.policy import TTLTier

    configured = CachePolicy(ttl_tier=TTLTier.MEDIUM, similarity_threshold=0.97)
    mock_engine.default_policy = configured
    classifier = MagicMock(spec=IntentClassifier)
    classifier.classify_safe = AsyncMock(side_effect=RuntimeError("boom"))

    await chat_completions(base_request, mock_engine, mock_provider, classifier)

    assert mock_engine.store.call_args.kwargs["policy"] is configured
    assert metrics.classifier_calls_fallback == 1


# ── Item 4: bypassed requests count toward LLM usage ────────

@pytest.mark.asyncio
async def test_bypass_records_llm_usage(mock_provider, mock_engine):
    multi_turn = ChatCompletionRequest(
        model="test-model",
        messages=[
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello!"),
            ChatMessage(role="user", content="What is Python?"),
        ],
    )
    mock_provider.generate = AsyncMock(return_value=_response("Answer", 100, 40))

    await chat_completions(multi_turn, mock_engine, mock_provider, _classifier())

    assert metrics.cache_bypasses == 1
    assert metrics.llm_calls == 1
    assert metrics.llm_tokens_prompt == 100
    assert metrics.llm_tokens_completion == 40
    # Multi-turn: excluded from the per-hit savings average.
    assert metrics.llm_calls_cacheable == 0
    assert metrics.llm_tokens_prompt_cacheable == 0


@pytest.mark.asyncio
async def test_miss_records_cacheable_llm_usage(base_request, mock_provider, mock_engine):
    mock_provider.generate = AsyncMock(return_value=_response("Answer", 10, 20))

    await chat_completions(base_request, mock_engine, mock_provider, _classifier())

    assert metrics.llm_calls == 1
    assert metrics.llm_calls_cacheable == 1
    assert metrics.llm_tokens_prompt_cacheable == 10
    assert metrics.llm_tokens_completion_cacheable == 20
