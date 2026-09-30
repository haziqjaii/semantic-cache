from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Response

from semcache.api.chat import chat_completions
from semcache.cache.classifier import ClassifierResult, IntentClassifier
from semcache.cache.engine import CacheEngine, LookupResult
from semcache.cache.policy import DEFAULT_POLICY, CachePolicy
from semcache.cache.store.base import CacheEntry
from semcache.config import PRICING_TABLE
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

USD_TO_MYR = PRICING_TABLE["usd_to_myr"]


@pytest.fixture
def base_request():
    return ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Hello")]
    )

@pytest.fixture
def mock_provider():
    provider = MagicMock(spec=LLMProvider)
    provider.generate = AsyncMock(return_value=ChatCompletionResponse(
        id="test_id",
        object="chat.completion",
        created=123,
        model="test-model",
        choices=[ChatCompletionChoice(
            index=0, message=ChatCompletionChoiceMessage(role="assistant", content="Response")
        )]
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

    await chat_completions(base_request, Response(), mock_engine, mock_provider, classifier)
    
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

    await chat_completions(base_request, Response(), mock_engine, mock_provider, classifier)
    
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

    await chat_completions(base_request, Response(), mock_engine, mock_provider, classifier)
    
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
    
    await chat_completions(base_request, Response(), mock_engine, mock_provider, classifier)
    
    # We should have one hit, 0 misses, and the classifier should never have been invoked
    assert metrics.cache_hits == 1
    assert metrics.cache_misses == 0
    assert metrics.classifier_calls_skipped == 1  # Skipped because of hit
    assert metrics.classifier_calls_success == 0
    assert metrics.classifier_calls_fallback == 0
    
    classifier.classify_safe.assert_not_called()


def _response(
    content: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    model: str = "gemini-3.5-flash",
    finish_reason: str = "stop",
):
    return ChatCompletionResponse(
        id="test_id",
        created=123,
        model=model,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionChoiceMessage(content=content),
                finish_reason=finish_reason,
            )
        ],
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

    response = await chat_completions(base_request, Response(), mock_engine, mock_provider, _classifier())

    assert response.choices[0].message.content == content
    mock_engine.store.assert_not_called()


# ── Item 2: cache failures never break the request ──────────

@pytest.mark.asyncio
async def test_lookup_failure_serves_uncached(base_request, mock_provider, mock_engine):
    mock_engine.lookup = AsyncMock(side_effect=ConnectionError("Redis down"))
    mock_provider.generate = AsyncMock(return_value=_response("Answer", 10, 20))
    classifier = _classifier()

    response = await chat_completions(base_request, Response(), mock_engine, mock_provider, classifier)

    assert response.choices[0].message.content == "Answer"
    assert response.x_cache_status == "BYPASS (Cache error)"
    assert metrics.cache_lookup_errors == 1
    assert metrics.cache_misses == 0
    assert metrics.classifier_calls_skipped == 1
    assert metrics.llm_calls == 1
    classifier.classify_safe.assert_not_called()
    mock_engine.store.assert_not_called()


@pytest.mark.asyncio
async def test_store_failure_still_returns_response(base_request, mock_provider, mock_engine):
    mock_engine.store = AsyncMock(side_effect=ConnectionError("Redis down"))

    response = await chat_completions(base_request, Response(), mock_engine, mock_provider, _classifier())

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

    await chat_completions(base_request, Response(), mock_engine, mock_provider, classifier)

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

    await chat_completions(multi_turn, Response(), mock_engine, mock_provider, _classifier())

    assert metrics.cache_bypasses == 1
    assert metrics.llm_calls == 1
    assert metrics.llm_tokens_prompt == 100
    assert metrics.llm_tokens_completion == 40


# ── Savings: tokens and ringgit, priced by the cached answer's model ─

def _hit_on(entry_model, prompt_tokens, completion_tokens):
    entry = CacheEntry(
        prompt="Hello", response="Cached", model=entry_model, namespace="ns",
        response_metadata={"usage": {
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }},
    )
    return LookupResult(hit=True, entry=entry, similarity=0.99, namespace="ns", embedding=[0.0])


@pytest.mark.asyncio
async def test_savings_are_priced_by_each_entry_model(base_request, mock_provider, mock_engine):
    mock_engine.lookup = AsyncMock(side_effect=[
        _hit_on("gemini-3.5-flash", 1_000_000, 0),       # $1.50
        _hit_on("gemini-3.5-flash-lite", 1_000_000, 0),  # $0.075
    ])

    await chat_completions(base_request, Response(), mock_engine, mock_provider, _classifier())
    await chat_completions(base_request, Response(), mock_engine, mock_provider, _classifier())

    assert metrics.tokens_saved == 2_000_000
    assert metrics.cost_saved_myr == pytest.approx((1.50 + 0.075) * USD_TO_MYR)
    assert metrics.cache_hits_unpriced == 0


@pytest.mark.asyncio
async def test_hit_on_unknown_model_saves_tokens_but_is_unpriced(
    base_request, mock_provider, mock_engine
):
    mock_engine.lookup = AsyncMock(return_value=_hit_on("gemini-9-ultra", 10, 20))

    await chat_completions(base_request, Response(), mock_engine, mock_provider, _classifier())

    assert metrics.tokens_saved == 30
    assert metrics.cache_hits_unpriced == 1
    assert metrics.unpriced_models == {"gemini-9-ultra"}
    # Not priced as some other model.
    assert metrics.cost_saved_myr == 0


# ── Only normally-finished responses are cached ─────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "other"])
async def test_unfinished_response_is_not_cached(
    base_request, mock_provider, mock_engine, finish_reason
):
    mock_provider.generate = AsyncMock(
        return_value=_response("Partial answer", finish_reason=finish_reason)
    )

    response = await chat_completions(base_request, Response(), mock_engine, mock_provider, _classifier())

    assert response.choices[0].finish_reason == finish_reason
    mock_engine.store.assert_not_called()


# ── Cache key uses the same system prompt the model gets ────

@pytest.mark.asyncio
async def test_all_system_messages_form_the_cache_key(mock_provider, mock_engine):
    request = ChatCompletionRequest(
        model="test-model",
        messages=[
            ChatMessage(role="system", content="You are a teacher."),
            ChatMessage(role="system", content="Answer in French."),
            ChatMessage(role="user", content="What is Python?"),
        ],
    )

    await chat_completions(request, Response(), mock_engine, mock_provider, _classifier())

    assert (
        mock_engine.lookup.call_args.kwargs["system_prompt"]
        == "You are a teacher.\n\nAnswer in French."
    )


# ── Cache status is reported in HTTP headers ────────────────

@pytest.mark.asyncio
async def test_miss_sets_miss_header(base_request, mock_provider, mock_engine):
    http_response = Response()

    await chat_completions(base_request, http_response, mock_engine, mock_provider, _classifier())

    assert http_response.headers["X-Cache-Status"] == "MISS"
    assert "X-Cache-Similarity" not in http_response.headers


@pytest.mark.asyncio
async def test_hit_sets_hit_and_similarity_headers(base_request, mock_provider, mock_engine):
    entry = CacheEntry(prompt="Hello", response="Cached", model="gemini-3.5-flash", namespace="ns")
    mock_engine.lookup = AsyncMock(return_value=LookupResult(
        hit=True, entry=entry, similarity=0.97321, namespace="ns", embedding=[0.0],
    ))
    http_response = Response()

    await chat_completions(base_request, http_response, mock_engine, mock_provider, _classifier())

    assert http_response.headers["X-Cache-Status"] == "HIT"
    assert http_response.headers["X-Cache-Similarity"] == "0.9732"


@pytest.mark.asyncio
async def test_bypass_headers_give_the_reason(mock_provider, mock_engine):
    multi_turn = ChatCompletionRequest(
        model="test-model",
        messages=[
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello!"),
            ChatMessage(role="user", content="What is Python?"),
        ],
    )
    uncacheable = Response()
    await chat_completions(multi_turn, uncacheable, mock_engine, mock_provider, _classifier())

    mock_engine.lookup = AsyncMock(side_effect=ConnectionError("Redis down"))
    cache_down = Response()
    await chat_completions(
        ChatCompletionRequest(model="test-model", messages=[ChatMessage(role="user", content="Hi")]),
        cache_down, mock_engine, mock_provider, _classifier(),
    )

    assert uncacheable.headers["X-Cache-Status"] == "BYPASS"
    assert uncacheable.headers["X-Cache-Bypass-Reason"] == "uncacheable"
    assert cache_down.headers["X-Cache-Status"] == "BYPASS"
    assert cache_down.headers["X-Cache-Bypass-Reason"] == "cache-error"


def test_headers_reach_the_http_client(mock_provider, mock_engine):
    """End to end through FastAPI: the header is on the real HTTP response."""
    from fastapi.testclient import TestClient

    from semcache.api.dependencies import get_classifier, get_engine, get_provider
    from semcache.main import app

    app.dependency_overrides[get_engine] = lambda: mock_engine
    app.dependency_overrides[get_provider] = lambda: mock_provider
    app.dependency_overrides[get_classifier] = lambda: _classifier()
    try:
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]},
        )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.headers["X-Cache-Status"] == "MISS"
    assert resp.json()["choices"][0]["message"]["content"] == "Response"


# ── Usage and finish reason are stored, replayed, and priced ─

@pytest.mark.asyncio
async def test_miss_stores_usage_and_finish_reason(base_request, mock_provider, mock_engine):
    mock_provider.generate = AsyncMock(return_value=_response("Answer", 12, 34))

    await chat_completions(base_request, Response(), mock_engine, mock_provider, _classifier())

    assert mock_engine.store.call_args.kwargs["response_metadata"] == {
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46},
    }


@pytest.mark.asyncio
async def test_hit_returns_stored_usage_and_credits_exact_saving(
    base_request, mock_provider, mock_engine
):
    entry = CacheEntry(
        prompt="Hello", response="Cached", model="gemini-3.5-flash", namespace="ns",
        response_metadata={
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1000, "completion_tokens": 2000, "total_tokens": 3000},
        },
    )
    mock_engine.lookup = AsyncMock(return_value=LookupResult(
        hit=True, entry=entry, similarity=0.99, namespace="ns", embedding=[0.0],
    ))

    response = await chat_completions(
        base_request, Response(), mock_engine, mock_provider, _classifier()
    )

    assert response.usage == UsageInfo(prompt_tokens=1000, completion_tokens=2000, total_tokens=3000)
    assert response.choices[0].finish_reason == "stop"
    # The hit saved what the original gemini-3.5-flash call cost.
    expected = (1000 * 1.50 + 2000 * 9.00) / 1e6 * USD_TO_MYR
    assert metrics.cost_saved_myr == pytest.approx(expected)
    assert metrics.tokens_saved == 3000
    assert metrics.cache_hits_unpriced == 0
    mock_provider.generate.assert_not_called()


@pytest.mark.asyncio
async def test_hit_without_stored_usage_is_unpriced(base_request, mock_provider, mock_engine):
    """Entries cached before usage was stored still serve, but can't be priced."""
    entry = CacheEntry(prompt="Hello", response="Cached", model="gemini-3.5-flash", namespace="ns")
    mock_engine.lookup = AsyncMock(return_value=LookupResult(
        hit=True, entry=entry, similarity=0.99, namespace="ns", embedding=[0.0],
    ))

    response = await chat_completions(
        base_request, Response(), mock_engine, mock_provider, _classifier()
    )

    assert response.usage is None
    assert response.choices[0].message.content == "Cached"
    assert metrics.cost_saved_myr == 0
    assert metrics.cache_hits_unpriced == 1


@pytest.mark.asyncio
async def test_miss_then_hit_round_trip(mock_provider, mock_embedder, memory_store):
    """With a real engine: the hit replays the miss's usage and saves its cost."""
    engine = CacheEngine(embedder=mock_embedder, store=memory_store)
    # The provider answers with the requested model, as GeminiProvider does.
    request = ChatCompletionRequest(
        model="gemini-3.5-flash", messages=[ChatMessage(role="user", content="Hello")]
    )
    mock_provider.generate = AsyncMock(return_value=_response("Answer", 100, 200))

    miss = Response()
    await chat_completions(request, miss, engine, mock_provider, _classifier())
    hit = Response()
    cached = await chat_completions(request, hit, engine, mock_provider, _classifier())

    assert miss.headers["X-Cache-Status"] == "MISS"
    assert hit.headers["X-Cache-Status"] == "HIT"
    assert cached.usage == UsageInfo(prompt_tokens=100, completion_tokens=200, total_tokens=300)
    # The hit saved exactly what the miss spent.
    assert metrics.tokens_saved == metrics.llm_tokens_prompt + metrics.llm_tokens_completion == 300
    assert metrics.cost_saved_myr == pytest.approx((100 * 1.50 + 200 * 9.00) / 1e6 * USD_TO_MYR)
    assert mock_provider.generate.await_count == 1
