"""
Tests for streaming responses ("stream": true) and LLM error handling.

The provider is faked with async generators, so these run offline and show
exactly which events the client receives on every path.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Response
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from google.genai import types

from semcache.api.chat import _provider_http_error, chat_completions
from semcache.api.dependencies import get_classifier, get_engine, get_provider
from semcache.cache.classifier import ClassifierResult, IntentClassifier
from semcache.cache.engine import CacheEngine, LookupResult
from semcache.cache.policy import DEFAULT_POLICY, TASK_POLICIES
from semcache.cache.store.base import CacheEntry
from semcache.main import app
from semcache.metrics import metrics
from semcache.providers.base import LLMProvider, StreamChunk
from semcache.providers.gemini import GeminiProvider
from semcache.schemas import ChatCompletionRequest, ChatMessage, UsageInfo

USAGE = UsageInfo(prompt_tokens=8, completion_tokens=5, total_tokens=13)


class Overloaded(Exception):
    """Stands in for google.genai's APIError, which carries the HTTP code."""

    code = 503


def _request(stream=True, include_usage=False, **kwargs) -> ChatCompletionRequest:
    kwargs.setdefault("messages", [ChatMessage(role="user", content="What is the capital of Malaysia?")])
    if include_usage:
        kwargs["stream_options"] = {"include_usage": True}
    return ChatCompletionRequest(model="gemini-3.5-flash-lite", stream=stream, **kwargs)


def _provider(chunks=None, *, fail_at=None, error=None) -> MagicMock:
    """A provider whose stream yields `chunks`, raising `error` before chunk `fail_at`."""
    chunks = chunks if chunks is not None else [
        StreamChunk(text="Kuala"),
        StreamChunk(text=" Lumpur."),
        StreamChunk(finish_reason="stop", usage=USAGE),
    ]
    provider = MagicMock(spec=LLMProvider)
    provider.closed = False

    async def stream(request):
        try:
            for i, chunk in enumerate(chunks):
                if fail_at == i:
                    raise error
                yield chunk
            if fail_at == len(chunks):
                raise error
        finally:
            provider.closed = True

    provider.generate_stream = MagicMock(side_effect=stream)
    return provider


def _engine(lookup: LookupResult | None = None) -> MagicMock:
    engine = MagicMock(spec=CacheEngine)
    engine.lookup = AsyncMock(return_value=lookup or LookupResult(
        hit=False, namespace="ns", embedding=[0.0], lookup_id="lookup-1"
    ))
    engine.store = AsyncMock()
    engine.default_policy = DEFAULT_POLICY
    return engine


def _classifier(intent="factual") -> MagicMock:
    classifier = MagicMock(spec=IntentClassifier)
    classifier.classify_safe = AsyncMock(return_value=ClassifierResult(
        policy=TASK_POLICIES[intent], tokens=20, is_fallback=False, intent=intent
    ))
    return classifier


async def _events(response: StreamingResponse) -> list:
    """Read a streaming response into its events: dicts, and "[DONE]"."""
    body = "".join([part if isinstance(part, str) else part.decode() async for part in response.body_iterator])
    events = []
    for block in body.strip().split("\n\n"):
        assert block.startswith("data: "), block
        payload = block.removeprefix("data: ")
        events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


def _text(events) -> str:
    return "".join(
        e["choices"][0]["delta"].get("content", "")
        for e in events if isinstance(e, dict) and e.get("choices")
    )


# ── Cache MISS: pass through, keep a copy, store when complete ──

@pytest.mark.asyncio
async def test_miss_streams_the_answer_then_caches_it():
    engine, provider = _engine(), _provider()

    response = await chat_completions(_request(), Response(), engine, provider, _classifier())

    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"
    assert response.headers["X-Cache-Status"] == "MISS"
    assert response.headers["X-Cache-Lookup-Id"] == "lookup-1"
    engine.store.assert_not_called()  # nothing is stored before the stream is read

    events = await _events(response)

    # role → content deltas → finish → [DONE], all sharing one id.
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert [e["choices"][0]["delta"].get("content") for e in events[1:3]] == ["Kuala", " Lumpur."]
    assert events[3]["choices"][0] == {"index": 0, "delta": {}, "finish_reason": "stop"}
    assert events[-1] == "[DONE]"
    assert len({e["id"] for e in events[:-1]}) == 1
    assert all(e["object"] == "chat.completion.chunk" for e in events[:-1])

    # The complete answer was cached, with the classifier's policy and intent.
    stored = engine.store.call_args.kwargs
    assert stored["response"] == "Kuala Lumpur."
    assert stored["policy"] == TASK_POLICIES["factual"]
    assert stored["intent"] == "factual"
    assert stored["response_metadata"] == {"finish_reason": "stop", "usage": USAGE.model_dump()}
    assert (metrics.cache_misses, metrics.llm_calls, metrics.classifier_calls_success) == (1, 1, 1)
    assert metrics.llm_tokens_prompt == 8


@pytest.mark.asyncio
async def test_usage_event_only_when_requested():
    without = await _events(await chat_completions(_request(), Response(), _engine(), _provider(), _classifier()))
    with_usage = await _events(await chat_completions(
        _request(include_usage=True), Response(), _engine(), _provider(), _classifier()
    ))

    assert not any("usage" in e for e in without if isinstance(e, dict))
    usage_event = with_usage[-2]
    assert usage_event["choices"] == []
    assert usage_event["usage"] == USAGE.model_dump()
    assert with_usage[-1] == "[DONE]"


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
async def test_unfinished_stream_is_delivered_but_not_cached(finish_reason):
    engine = _engine()
    provider = _provider([StreamChunk(text="Partial"), StreamChunk(finish_reason=finish_reason)])

    events = await _events(await chat_completions(_request(), Response(), engine, provider, _classifier()))

    assert _text(events) == "Partial"
    assert events[-2]["choices"][0]["finish_reason"] == finish_reason
    engine.store.assert_not_called()


@pytest.mark.asyncio
async def test_creative_answers_stream_but_are_not_cached(memory_store, mock_embedder):
    """With a real engine: the NO_CACHE policy applies to streamed answers too."""
    engine = CacheEngine(embedder=mock_embedder, store=memory_store)

    events = await _events(await chat_completions(
        _request(), Response(), engine, _provider(), _classifier("creative")
    ))

    assert _text(events) == "Kuala Lumpur."
    assert memory_store._entries == []


# ── Cache HIT: the stored answer, at once ───────────────────

@pytest.mark.asyncio
async def test_hit_streams_the_cached_answer_without_calling_the_llm():
    entry = CacheEntry(
        prompt="What is the capital of Malaysia?", response="Kuala Lumpur.",
        model="gemini-3.5-flash-lite", namespace="ns",
        response_metadata={"finish_reason": "stop", "usage": USAGE.model_dump()},
    )
    engine = _engine(LookupResult(hit=True, entry=entry, similarity=0.9731, lookup_id="lookup-2"))
    provider, classifier = _provider(), _classifier()

    response = await chat_completions(_request(include_usage=True), Response(), engine, provider, classifier)
    events = await _events(response)

    assert response.headers["X-Cache-Status"] == "HIT"
    assert response.headers["X-Cache-Similarity"] == "0.9731"
    assert response.headers["X-Cache-Lookup-Id"] == "lookup-2"
    assert _text(events) == "Kuala Lumpur."
    assert events[0]["id"].startswith("chatcmpl-cached-")
    assert events[-3]["choices"][0]["finish_reason"] == "stop"
    assert events[-2]["usage"] == USAGE.model_dump()  # the original call's usage
    assert events[-1] == "[DONE]"
    provider.generate_stream.assert_not_called()
    classifier.classify_safe.assert_not_called()
    assert (metrics.cache_hits, metrics.tokens_saved) == (1, 13)


@pytest.mark.asyncio
async def test_stream_then_stream_again_is_a_hit(memory_store, mock_embedder):
    """End to end with a real engine: a streamed miss is served as a streamed hit."""
    engine = CacheEngine(embedder=mock_embedder, store=memory_store)
    provider = _provider()

    first = await chat_completions(_request(), Response(), engine, provider, _classifier())
    miss_events = await _events(first)
    second = await chat_completions(_request(), Response(), engine, provider, _classifier())
    hit_events = await _events(second)

    assert first.headers["X-Cache-Status"] == "MISS"
    assert second.headers["X-Cache-Status"] == "HIT"
    assert _text(hit_events) == _text(miss_events) == "Kuala Lumpur."
    assert provider.generate_stream.call_count == 1

    # And a non-streaming client gets the same cached answer.
    plain = await chat_completions(_request(stream=False), Response(), engine, provider, _classifier())
    assert plain.choices[0].message.content == "Kuala Lumpur."
    assert provider.generate_stream.call_count == 1


# ── BYPASS: pass through, never cached ──────────────────────

@pytest.mark.asyncio
async def test_uncacheable_request_streams_without_the_cache():
    engine, classifier = _engine(), _classifier()
    multi_turn = _request(messages=[
        ChatMessage(role="user", content="Hi"),
        ChatMessage(role="assistant", content="Hello!"),
        ChatMessage(role="user", content="And the capital?"),
    ])

    response = await chat_completions(multi_turn, Response(), engine, _provider(), classifier)
    events = await _events(response)

    assert response.headers["X-Cache-Status"] == "BYPASS"
    assert response.headers["X-Cache-Bypass-Reason"] == "uncacheable"
    assert _text(events) == "Kuala Lumpur."
    assert events[-1] == "[DONE]"
    engine.lookup.assert_not_called()
    engine.store.assert_not_called()
    classifier.classify_safe.assert_not_called()
    assert (metrics.cache_bypasses, metrics.llm_calls) == (1, 1)


@pytest.mark.asyncio
async def test_cache_outage_still_streams_the_answer():
    engine = _engine()
    engine.lookup = AsyncMock(side_effect=ConnectionError("Redis down"))

    response = await chat_completions(_request(), Response(), engine, _provider(), _classifier())

    assert response.headers["X-Cache-Bypass-Reason"] == "cache-error"
    assert _text(await _events(response)) == "Kuala Lumpur."
    engine.store.assert_not_called()
    assert metrics.cache_lookup_errors == 1


# ── Failures ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_failure_before_the_first_chunk_is_a_real_http_error():
    """Nothing sent yet, so the client gets a proper status (503: retry), not a broken stream."""
    engine, classifier = _engine(), _classifier()
    provider = _provider(fail_at=0, error=Overloaded("high demand"))

    with pytest.raises(HTTPException) as excinfo:
        await chat_completions(_request(), Response(), engine, provider, classifier)

    assert excinfo.value.status_code == 503
    assert "high demand" in excinfo.value.detail
    engine.store.assert_not_called()


@pytest.mark.asyncio
async def test_failure_mid_stream_ends_with_an_error_event_and_caches_nothing():
    engine = _engine()
    provider = _provider(fail_at=1, error=ConnectionError("connection reset"))

    response = await chat_completions(_request(), Response(), engine, provider, _classifier())
    events = await _events(response)

    assert _text(events) == "Kuala"  # what arrived before the failure
    assert events[-1]["error"]["type"] == "upstream_error"
    assert "connection reset" in events[-1]["error"]["message"]
    assert "[DONE]" not in events  # the stream did not complete
    engine.store.assert_not_called()


@pytest.mark.asyncio
async def test_stream_that_ends_without_a_finish_reason_is_not_cached():
    engine = _engine()
    provider = _provider([StreamChunk(text="Kuala")])  # cut short: no final chunk

    events = await _events(await chat_completions(_request(), Response(), engine, provider, _classifier()))

    assert "ended unexpectedly" in events[-1]["error"]["message"]
    engine.store.assert_not_called()


@pytest.mark.asyncio
async def test_client_disconnect_caches_nothing_and_stops_the_provider():
    engine = _engine()
    provider = _provider()
    classifier = MagicMock(spec=IntentClassifier)
    classifier_started = asyncio.Event()

    async def slow_classify(prompt):
        classifier_started.set()
        await asyncio.sleep(60)

    classifier.classify_safe = AsyncMock(side_effect=slow_classify)

    response = await chat_completions(_request(), Response(), engine, provider, classifier)
    body = response.body_iterator
    await anext(body)  # role
    await anext(body)  # "Kuala"
    await classifier_started.wait()
    await body.aclose()  # what the server does when the client goes away
    await asyncio.sleep(0)

    engine.store.assert_not_called()
    assert provider.closed  # we stopped reading from the LLM
    assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]


# ── Non-streaming requests: LLM failures are 502/503, not 500 ──

class TestProviderErrors:
    def test_status_mapping(self):
        class RateLimited(Exception):
            code = 429

        assert _provider_http_error(Overloaded("busy")).status_code == 503
        assert _provider_http_error(RateLimited("slow down")).status_code == 429
        assert _provider_http_error(RuntimeError("boom")).status_code == 502
        original = HTTPException(status_code=418)
        assert _provider_http_error(original) is original

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cacheable", [True, False])
    async def test_failed_generation_is_not_a_500(self, cacheable):
        """On the miss path and the bypass path alike."""
        engine = _engine()
        provider = MagicMock(spec=LLMProvider)
        provider.generate = AsyncMock(side_effect=Overloaded("high demand"))
        request = _request(stream=False) if cacheable else _request(stream=False, tools=[{"type": "function"}])

        with pytest.raises(HTTPException) as excinfo:
            await chat_completions(request, Response(), engine, provider, _classifier())

        assert excinfo.value.status_code == 503
        engine.store.assert_not_called()


# ── GeminiProvider.generate_stream ──────────────────────────

def _gemini_chunk(text=None, finish_reason=None, usage=None, thought=False) -> types.GenerateContentResponse:
    parts = [types.Part(text=text, thought=thought)] if text is not None else []
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts), finish_reason=finish_reason)],
        usage_metadata=usage,
    )


async def _gemini_stream(responses) -> list[StreamChunk]:
    provider = GeminiProvider(api_key="fake-key")

    async def fake_stream():
        for response in responses:
            yield response

    provider._client.aio.models.generate_content_stream = AsyncMock(return_value=fake_stream())
    return [chunk async for chunk in provider.generate_stream(_request())]


@pytest.mark.asyncio
async def test_gemini_stream_yields_text_then_a_final_chunk():
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=8, candidates_token_count=5, total_token_count=13
    )
    chunks = await _gemini_stream([
        _gemini_chunk("Let me think", thought=True),  # thinking is never shown
        _gemini_chunk("Kuala"),
        _gemini_chunk(" Lumpur.", finish_reason="STOP", usage=usage),
    ])

    assert [c.text for c in chunks] == ["Kuala", " Lumpur.", ""]
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage == USAGE


@pytest.mark.asyncio
async def test_gemini_stream_maps_finish_reasons_and_blocked_prompts():
    truncated = await _gemini_stream([_gemini_chunk("Kuala", finish_reason="MAX_TOKENS")])
    blocked = await _gemini_stream([types.GenerateContentResponse(
        candidates=[],
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason="SAFETY"),
    )])
    cut_short = await _gemini_stream([_gemini_chunk("Kuala")])

    assert truncated[-1].finish_reason == "length"
    assert [c.finish_reason for c in blocked] == ["content_filter"]
    # No finish reason from Gemini → no final chunk, so the caller knows it was cut short.
    assert [c.finish_reason for c in cut_short] == [None]


# ── Over real HTTP ──────────────────────────────────────────

def test_streaming_over_http():
    """Through FastAPI: SSE content type, cache headers, and events on the wire."""
    # Zero-argument lambdas: FastAPI would read the helpers' parameters as request parameters.
    app.dependency_overrides[get_engine] = lambda: _engine()
    app.dependency_overrides[get_provider] = lambda: _provider()
    app.dependency_overrides[get_classifier] = lambda: _classifier()
    try:
        with TestClient(app).stream("POST", "/v1/chat/completions", json={
            "model": "gemini-3.5-flash-lite", "stream": True,
            "messages": [{"role": "user", "content": "What is the capital of Malaysia?"}],
        }) as resp:
            lines = [line for line in resp.iter_lines() if line]
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            assert resp.headers["x-cache-status"] == "MISS"
            assert resp.headers["cache-control"] == "no-cache"
    finally:
        app.dependency_overrides.clear()

    assert lines[-1] == "data: [DONE]"
    deltas = [json.loads(line.removeprefix("data: "))["choices"][0]["delta"] for line in lines[:-1]]
    assert "".join(d.get("content", "") for d in deltas) == "Kuala Lumpur."


def test_streaming_http_error_before_first_chunk_has_a_real_status():
    app.dependency_overrides[get_engine] = lambda: _engine()
    app.dependency_overrides[get_provider] = lambda: _provider(fail_at=0, error=Overloaded("high demand"))
    app.dependency_overrides[get_classifier] = lambda: _classifier()
    try:
        resp = TestClient(app).post("/v1/chat/completions", json={
            "model": "gemini-3.5-flash-lite", "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        })
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 503
    assert "high demand" in resp.json()["detail"]
