"""
The user never waits for the classifier.

On a cache miss the answer is returned as soon as the LLM produces it; the
classifier's result is awaited, and the answer stored, after the response
is sent (FastAPI background tasks). These tests use a classifier that
doesn't answer until released, and check the response is complete before it.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import BackgroundTasks, HTTPException, Response

from semcache.api.chat import chat_completions
from semcache.cache.classifier import ClassifierResult, IntentClassifier
from semcache.cache.engine import CacheEngine, LookupResult
from semcache.cache.policy import DEFAULT_POLICY, TASK_POLICIES
from semcache.providers.base import LLMProvider, StreamChunk
from semcache.schemas import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    UsageInfo,
)


def _request(stream=False) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gemini-3.5-flash-lite", stream=stream,
        messages=[ChatMessage(role="user", content="How do I sort a list?")],
    )


def _engine() -> MagicMock:
    engine = MagicMock(spec=CacheEngine)
    engine.lookup = AsyncMock(return_value=LookupResult(hit=False, namespace="ns", embedding=[0.0]))
    engine.store = AsyncMock()
    engine.default_policy = DEFAULT_POLICY
    return engine


def _provider() -> MagicMock:
    provider = MagicMock(spec=LLMProvider)
    provider.generate = AsyncMock(return_value=ChatCompletionResponse(
        id="id", created=0, model="gemini-3.5-flash-lite",
        choices=[ChatCompletionChoice(message=ChatCompletionChoiceMessage(content="Use sorted()."))],
        usage=UsageInfo(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    ))

    async def stream(request):
        yield StreamChunk(text="Use sorted().")
        yield StreamChunk(finish_reason="stop")

    provider.generate_stream = MagicMock(side_effect=stream)
    return provider


class SlowClassifier:
    """A classifier that answers only when `release()` is called."""

    def __init__(self) -> None:
        self.mock = MagicMock(spec=IntentClassifier)
        self._released = asyncio.Event()
        self.mock.classify_safe = AsyncMock(side_effect=self._classify)

    async def _classify(self, prompt):
        await self._released.wait()
        return ClassifierResult(policy=TASK_POLICIES["how_to"], tokens=10, is_fallback=False, intent="how_to")

    def release(self) -> None:
        self._released.set()


@pytest.mark.asyncio
async def test_response_is_returned_before_the_classifier_finishes():
    engine, background = _engine(), BackgroundTasks()
    classifier = SlowClassifier()

    response = await asyncio.wait_for(
        chat_completions(_request(), Response(), engine, _provider(), classifier.mock, None, background),
        timeout=2,  # the classifier never answers on its own: waiting for it would time out
    )

    assert response.choices[0].message.content == "Use sorted()."
    engine.store.assert_not_called()  # storing waits for the classifier, after the response

    classifier.release()
    await background()  # what FastAPI does once the response is sent

    stored = engine.store.call_args.kwargs
    assert stored["response"] == "Use sorted()."
    assert stored["policy"] == TASK_POLICIES["how_to"]
    assert stored["intent"] == "how_to"


@pytest.mark.asyncio
async def test_streamed_answer_completes_before_the_classifier_finishes():
    engine, background = _engine(), BackgroundTasks()
    classifier = SlowClassifier()

    response = await chat_completions(_request(stream=True), Response(), engine, _provider(), classifier.mock, None, background)
    events = await asyncio.wait_for(
        _read_all(response.body_iterator), timeout=2  # the whole stream, [DONE] included
    )

    assert events[-1] == "data: [DONE]"
    assert "Use sorted()." in "".join(events)
    engine.store.assert_not_called()

    classifier.release()
    await background()

    assert engine.store.call_args.kwargs["intent"] == "how_to"


async def _read_all(body) -> list[str]:
    return [block for part in [p async for p in body] for block in part.strip().split("\n\n") if block]


@pytest.mark.asyncio
async def test_a_failed_answer_cancels_the_classifier():
    engine, classifier = _engine(), SlowClassifier()
    provider = _provider()
    provider.generate = AsyncMock(side_effect=RuntimeError("LLM down"))

    with pytest.raises(HTTPException) as excinfo:
        await chat_completions(_request(), Response(), engine, provider, classifier.mock, None, BackgroundTasks())

    assert excinfo.value.status_code == 502
    await asyncio.sleep(0)
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert pending == []  # the classifier task was cancelled, not left running
    engine.store.assert_not_called()


def test_over_http_the_answer_is_stored_after_the_response():
    """End to end: FastAPI runs the store step after sending the response."""
    from fastapi.testclient import TestClient

    from semcache.api.dependencies import get_classifier, get_engine, get_provider
    from semcache.main import app

    engine = _engine()
    classifier = MagicMock(spec=IntentClassifier)
    classifier.classify_safe = AsyncMock(return_value=ClassifierResult(
        policy=TASK_POLICIES["how_to"], tokens=10, is_fallback=False, intent="how_to"
    ))
    app.dependency_overrides[get_engine] = lambda: engine
    app.dependency_overrides[get_provider] = lambda: _provider()
    app.dependency_overrides[get_classifier] = lambda: classifier
    try:
        resp = TestClient(app).post("/v1/chat/completions", json=json.loads(_request().model_dump_json()))
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.headers["x-cache-status"] == "MISS"
    assert engine.store.call_args.kwargs["intent"] == "how_to"
