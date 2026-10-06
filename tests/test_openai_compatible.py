"""
The OpenAI-compatible provider and the router that picks a provider by model.

The provider talks to a fake HTTP server (httpx's MockTransport), so these
check the real requests it sends and how it reads real response shapes.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException, Response

from semcache.api.chat import chat_completions
from semcache.cache.classifier import ClassifierResult, IntentClassifier
from semcache.cache.engine import CacheEngine
from semcache.cache.policy import TASK_POLICIES
from semcache.providers.base import LLMProvider, StreamChunk
from semcache.providers.openai_compatible import (
    OpenAICompatibleProvider,
    ProviderHTTPError,
)
from semcache.providers.router import RoutingProvider, is_google_model
from semcache.schemas import ChatCompletionRequest, ChatMessage
from tests.conftest import InMemoryLookupLog, InMemoryVectorStore, MockEmbedder

MODEL = "openai-gpt-oss-120b"
BASE_URL = "https://models.example/v1"


# ── Helpers ─────────────────────────────────────────────────

def _request(model=MODEL, stream=False, **kwargs) -> ChatCompletionRequest:
    messages = [
        ChatMessage(role="system", content="Be brief."),
        ChatMessage(role="user", content="What is the capital of Malaysia?"),
    ]
    return ChatCompletionRequest(model=model, stream=stream, messages=messages, **kwargs)


def _provider(handler) -> tuple[OpenAICompatibleProvider, list[httpx.Request]]:
    """A provider wired to `handler` instead of the network, and the requests it sent."""
    sent: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return handler(request)

    # Built the way the provider builds its own client, so the URL joining is tested too.
    client = httpx.AsyncClient(
        base_url=BASE_URL.rstrip("/") + "/",
        headers={"Authorization": "Bearer secret-key"},
        transport=httpx.MockTransport(record),
    )
    return OpenAICompatibleProvider(api_key="secret-key", base_url=BASE_URL, client=client), sent


def _completion(content="Kuala Lumpur.", finish_reason="stop", **message_extra) -> dict:
    return {
        "id": "abc",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content, **message_extra},
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }


def _sse(*events) -> httpx.Response:
    body = "".join(f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in events)
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def _delta(content=None, finish_reason=None, **delta_extra) -> dict:
    delta = {**delta_extra}
    if content is not None:
        delta["content"] = content
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


async def _collect(stream) -> list[StreamChunk]:
    return [chunk async for chunk in stream]


# ── A normal (non-streamed) answer ──────────────────────────

@pytest.mark.asyncio
async def test_sends_an_openai_request_and_reads_the_answer():
    provider, sent = _provider(lambda request: httpx.Response(200, json=_completion()))

    response = await provider.generate(_request(temperature=0.2, max_tokens=100))

    [request] = sent
    assert str(request.url) == "https://models.example/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret-key"
    assert json.loads(request.content) == {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "What is the capital of Malaysia?"},
        ],
        "temperature": 0.2,
        "max_tokens": 100,
    }

    assert response.choices[0].message.content == "Kuala Lumpur."
    assert response.choices[0].finish_reason == "stop"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (20, 5)
    assert response.model == MODEL


@pytest.mark.asyncio
async def test_reasoning_is_not_part_of_the_answer():
    """Thinking models send their reasoning separately; only the answer is returned."""
    body = _completion(content="Kuala Lumpur.", reasoning_content="The user asks about Malaysia...")
    provider, _ = _provider(lambda request: httpx.Response(200, json=body))

    response = await provider.generate(_request())

    assert response.choices[0].message.content == "Kuala Lumpur."


@pytest.mark.asyncio
async def test_an_answer_with_no_text_is_empty_not_a_crash():
    body = _completion(content=None, finish_reason="length", reasoning_content="Still thinking...")
    provider, _ = _provider(lambda request: httpx.Response(200, json=body))

    response = await provider.generate(_request())

    assert response.choices[0].message.content == ""
    assert response.choices[0].finish_reason == "length"  # so chat.py won't cache it


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "body", "expected_message"), [
    (429, {"error": {"message": "Rate limit reached"}}, "Rate limit reached"),
    (503, {"error": "model is loading"}, "model is loading"),
    (401, "Unauthorized", "Unauthorized"),
])
async def test_error_statuses_keep_their_code_and_message(status, body, expected_message):
    def handler(request):
        return httpx.Response(status, json=body) if isinstance(body, dict) else httpx.Response(status, text=body)

    provider, _ = _provider(handler)

    with pytest.raises(ProviderHTTPError) as raised:
        await provider.generate(_request())

    assert raised.value.code == status  # chat.py passes 429 / 503 on to the client
    assert expected_message in str(raised.value)


# ── A streamed answer ───────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_yields_text_then_a_final_chunk_with_usage():
    provider, sent = _provider(lambda request: _sse(
        _delta(role="assistant", content=""),
        _delta("Kuala "),
        _delta("Lumpur."),
        _delta(finish_reason="stop"),
        {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}},
        "[DONE]",
    ))

    chunks = await _collect(provider.generate_stream(_request(stream=True)))

    assert [c.text for c in chunks] == ["Kuala ", "Lumpur.", ""]
    assert chunks[-1].finish_reason == "stop"
    assert (chunks[-1].usage.prompt_tokens, chunks[-1].usage.completion_tokens) == (20, 5)
    # It asked for a stream, and for the token counts at the end of it.
    payload = json.loads(sent[0].content)
    assert payload["stream"] is True and payload["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_stream_skips_reasoning_and_keep_alive_lines():
    events = "".join([
        ": keep-alive\n\n",
        f"data: {json.dumps(_delta(reasoning_content='Thinking...'))}\n\n",
        f"data: {json.dumps(_delta('Kuala Lumpur.'))}\n\n",
        f"data: {json.dumps(_delta(finish_reason='stop'))}\n\n",
        "data: [DONE]\n\n",
    ])
    provider, _ = _provider(lambda request: httpx.Response(200, content=events.encode()))

    chunks = await _collect(provider.generate_stream(_request(stream=True)))

    assert "".join(c.text for c in chunks) == "Kuala Lumpur."


@pytest.mark.asyncio
async def test_stream_cut_short_sends_no_final_chunk():
    """No finish reason means the answer is incomplete, so chat.py won't cache it."""
    provider, _ = _provider(lambda request: _sse(_delta("Kuala ")))

    chunks = await _collect(provider.generate_stream(_request(stream=True)))

    assert [c.text for c in chunks] == ["Kuala "]
    assert all(c.finish_reason is None for c in chunks)


@pytest.mark.asyncio
async def test_stream_error_status_is_raised_before_any_text():
    provider, _ = _provider(lambda request: httpx.Response(503, json={"error": {"message": "Overloaded"}}))

    with pytest.raises(ProviderHTTPError) as raised:
        await anext(provider.generate_stream(_request(stream=True)))

    assert raised.value.code == 503 and "Overloaded" in str(raised.value)


@pytest.mark.asyncio
async def test_error_event_inside_a_stream_is_raised():
    provider, _ = _provider(lambda request: _sse(_delta("Kuala "), {"error": {"message": "GPU fell over"}}))

    with pytest.raises(ProviderHTTPError, match="GPU fell over"):
        await _collect(provider.generate_stream(_request(stream=True)))


# ── Routing by model ────────────────────────────────────────

@pytest.mark.parametrize(("model", "google"), [
    ("gemini-3.5-flash-lite", True),
    ("gemini-3.5-flash", True),
    ("gemma-3-27b-it", True),
    ("models/gemini-3.5-flash", True),
    ("Gemini-3.5-Flash", True),
    ("openai-gpt-oss-120b", False),
    ("mistral-small-3.2-24b-instruct-2506", False),
    ("qwen-qwen3-8-27b", False),
    ("gpt-5", False),
])
def test_which_models_are_googles(model, google):
    assert is_google_model(model) is google


def _named_provider(name: str) -> MagicMock:
    provider = MagicMock(spec=LLMProvider)
    provider.generate = AsyncMock(return_value=name)

    async def stream(request):
        yield StreamChunk(text=name)
        yield StreamChunk(finish_reason="stop")

    provider.generate_stream = MagicMock(side_effect=stream)
    return provider


@pytest.mark.asyncio
async def test_router_sends_each_model_to_its_provider():
    google, other = _named_provider("google"), _named_provider("other")
    router = RoutingProvider(google=google, other=other)

    assert await router.generate(_request(model="gemini-3.5-flash-lite")) == "google"
    assert await router.generate(_request(model=MODEL)) == "other"

    streamed = await _collect(router.generate_stream(_request(model=MODEL, stream=True)))
    assert streamed[0].text == "other"
    google.generate_stream.assert_not_called()


@pytest.mark.asyncio
async def test_router_stops_the_provider_when_the_caller_stops_reading():
    closed = asyncio.Event()

    async def endless(request):
        try:
            while True:
                yield StreamChunk(text="word ")
        finally:
            closed.set()

    other = MagicMock(spec=LLMProvider)
    other.generate_stream = MagicMock(side_effect=endless)
    stream = RoutingProvider(google=_named_provider("google"), other=other).generate_stream(_request(stream=True))

    await anext(stream)
    await stream.aclose()

    assert closed.is_set()


# ── Through the cache ───────────────────────────────────────

def _classifier() -> MagicMock:
    classifier = MagicMock(spec=IntentClassifier)
    classifier.classify_safe = AsyncMock(return_value=ClassifierResult(
        policy=TASK_POLICIES["factual"], tokens=10, is_fallback=False, intent="factual",
    ))
    return classifier


def _engine() -> CacheEngine:
    return CacheEngine(embedder=MockEmbedder(), store=InMemoryVectorStore(), lookup_log=InMemoryLookupLog())


@pytest.mark.asyncio
async def test_a_second_providers_answer_is_cached_and_served_again():
    openai_compatible, sent = _provider(lambda request: httpx.Response(200, json=_completion()))
    google = _named_provider("google")
    router = RoutingProvider(google=google, other=openai_compatible)
    engine = _engine()

    first_headers, second_headers = Response(), Response()
    first = await chat_completions(_request(), first_headers, engine, router, _classifier())
    second = await chat_completions(_request(), second_headers, engine, router, _classifier())

    assert first_headers.headers["X-Cache-Status"] == "MISS"
    assert second_headers.headers["X-Cache-Status"] == "HIT"
    assert first.choices[0].message.content == second.choices[0].message.content == "Kuala Lumpur."
    assert len(sent) == 1  # the second answer came from the cache
    google.generate.assert_not_called()

    # The same question to a Gemini model is a different cache entry.
    gemini_headers = Response()
    google.generate = AsyncMock(return_value=first)
    await chat_completions(
        _request(model="gemini-3.5-flash-lite"), gemini_headers, engine, router, _classifier()
    )
    assert gemini_headers.headers["X-Cache-Status"] == "MISS"


@pytest.mark.asyncio
@pytest.mark.parametrize(("upstream", "expected"), [(429, 429), (503, 503), (401, 502), (500, 502)])
async def test_provider_errors_reach_the_client_with_a_useful_status(upstream, expected):
    openai_compatible, _ = _provider(lambda request: httpx.Response(upstream, json={"error": {"message": "nope"}}))
    router = RoutingProvider(google=_named_provider("google"), other=openai_compatible)

    with pytest.raises(HTTPException) as raised:
        await chat_completions(_request(), Response(), _engine(), router, _classifier())

    assert raised.value.status_code == expected
    assert "nope" in raised.value.detail


# ── Startup ─────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("key_set", [True, False])
async def test_startup_adds_the_second_provider_only_with_its_key(key_set, monkeypatch):
    from semcache.api import dependencies
    from semcache.config import Settings
    from semcache.providers.gemini import GeminiProvider

    def redis_backed(**methods):
        instance = MagicMock()
        instance.initialize = AsyncMock()
        instance.close = AsyncMock()
        for name, value in methods.items():
            setattr(instance, name, AsyncMock(return_value=value))
        return MagicMock(return_value=instance)

    async def idle(engine):
        await asyncio.Event().wait()

    monkeypatch.setattr(dependencies, "RedisVectorStore", redis_backed())
    monkeypatch.setattr(dependencies, "RedisEmbeddingStore", redis_backed())
    monkeypatch.setattr(dependencies, "RedisLookupLog", redis_backed(labelled=[]))
    monkeypatch.setattr(dependencies, "gauge_refresh_loop", idle)
    extra = (
        {"openai_compatible_api_key": "secret-key", "openai_compatible_base_url": BASE_URL}
        if key_set else {}
    )
    monkeypatch.setattr(
        dependencies, "get_settings", lambda: Settings(_env_file=None, gemini_api_key="test-key", **extra)
    )

    async with dependencies.lifespan(MagicMock()):
        provider = dependencies.get_provider()
        if not key_set:
            assert isinstance(provider, GeminiProvider)  # exactly as before
        else:
            assert isinstance(provider, RoutingProvider)
            assert isinstance(provider.provider_for("gemini-3.5-flash-lite"), GeminiProvider)
            other = provider.provider_for(MODEL)
            assert isinstance(other, OpenAICompatibleProvider)
            assert str(other._client.base_url) == "https://models.example/v1/"
            assert other._client.headers["authorization"] == "Bearer secret-key"


def test_settings_read_the_documented_variable_names(monkeypatch):
    from semcache.config import Settings

    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "secret-key")
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", BASE_URL)

    settings = Settings(_env_file=None, gemini_api_key="test-key")

    assert settings.openai_compatible_api_key == "secret-key"
    assert settings.openai_compatible_base_url == BASE_URL
