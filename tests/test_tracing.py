"""
Langfuse tracing (semcache/tracing.py).

These run the real Langfuse SDK, with its spans captured in memory instead
of being sent anywhere, so they check what Langfuse would actually receive.
"""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import BackgroundTasks, HTTPException, Response
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from semcache import tracing
from semcache.api.chat import chat_completions
from semcache.api.tuning import FeedbackRequest, feedback
from semcache.cache.classifier import ClassifierResult, IntentClassifier
from semcache.cache.engine import CacheEngine
from semcache.cache.lookup_log import LookupEvent, _from_hash, _to_hash
from semcache.cache.policy import TASK_POLICIES
from semcache.providers.base import LLMProvider, StreamChunk
from semcache.providers.traced import TracedProvider
from semcache.schemas import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    UsageInfo,
)
from tests.conftest import InMemoryLookupLog, InMemoryVectorStore, MockEmbedder

MODEL = "gemini-3.5-flash-lite"
QUESTION = "What is the capital of Malaysia?"
ANSWER = "Kuala Lumpur."


# ── Helpers ─────────────────────────────────────────────────

@pytest.fixture
def exporter():
    """Tracing on, with spans kept in memory. Off again afterwards."""
    exporter = InMemorySpanExporter()
    on = tracing.configure(
        # A new key per test: the SDK keeps one client per public key.
        f"pk-test-{uuid.uuid4().hex}", "sk-test",
        base_url="http://127.0.0.1:9", span_exporter=exporter,
    )
    assert on
    yield exporter
    tracing.shutdown()


@pytest.fixture(autouse=True)
def tracing_off_afterwards():
    yield
    tracing.shutdown()


def _request(question=QUESTION, stream=False, messages=None) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=MODEL, stream=stream,
        messages=messages or [ChatMessage(role="user", content=question)],
    )


def _engine() -> CacheEngine:
    return CacheEngine(
        embedder=MockEmbedder(), store=InMemoryVectorStore(), lookup_log=InMemoryLookupLog()
    )


def _provider(error: Exception | None = None, stream_chunks=None) -> LLMProvider:
    inner = MagicMock(spec=LLMProvider)
    inner.generate = AsyncMock(
        side_effect=error,
        return_value=ChatCompletionResponse(
            id="id", created=0, model=MODEL,
            choices=[ChatCompletionChoice(message=ChatCompletionChoiceMessage(content=ANSWER))],
            usage=UsageInfo(prompt_tokens=7, completion_tokens=3, total_tokens=10),
        ),
    )

    async def stream(request):
        for chunk in stream_chunks or [
            StreamChunk(text="Kuala "),
            StreamChunk(text="Lumpur."),
            StreamChunk(finish_reason="stop", usage=UsageInfo(prompt_tokens=7, completion_tokens=3, total_tokens=10)),
        ]:
            yield chunk

    inner.generate_stream = MagicMock(side_effect=stream)
    return TracedProvider(inner)


def _classifier(intent="factual", tokens=12) -> MagicMock:
    classifier = MagicMock(spec=IntentClassifier)
    classifier.model = "gemini-3.5-flash-lite"
    classifier.classify_safe = AsyncMock(return_value=ClassifierResult(
        policy=TASK_POLICIES[intent], tokens=tokens, is_fallback=False, intent=intent,
    ))
    return classifier


async def _ask(engine, provider=None, classifier=None, **request_kwargs):
    return await chat_completions(
        _request(**request_kwargs), Response(), engine, provider or _provider(), classifier or _classifier(),
    )


class Spans:
    """The spans Langfuse received, by step name."""

    def __init__(self, exporter: InMemorySpanExporter) -> None:
        tracing.flush()
        self.all = exporter.get_finished_spans()
        exporter.clear()

    def names(self) -> list[str]:
        return sorted(s.name for s in self.all)

    def get(self, name: str):
        matches = [s for s in self.all if s.name == name]
        assert len(matches) == 1, f"expected one {name!r} span, got {len(matches)} in {self.names()}"
        return matches[0]

    def attr(self, name: str, key: str):
        return self.get(name).attributes.get(f"langfuse.observation.{key}")

    def json(self, name: str, key: str):
        return json.loads(self.attr(name, key))

    def parent(self, name: str) -> str | None:
        span = self.get(name)
        parents = [s.name for s in self.all if span.parent and s.context.span_id == span.parent.span_id]
        return parents[0] if parents else None

    def trace_ids(self) -> set[str]:
        return {format(s.context.trace_id, "032x") for s in self.all}


# ── Off by default ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_off_without_keys():
    assert tracing.configure(None, None) is False
    assert tracing.configure("pk-only", None) is False
    assert not tracing.enabled()

    # Every call is a harmless no-op.
    root = tracing.start_trace("chat-completion", input="q")
    with root.step("cache-lookup") as step:
        tracing.step("embedding", as_type="embedding").end()
        step.update(output={"hit": False})
    root.end(metadata={"cache_status": "miss"})
    tracing.score("abc", "good_match", True)
    assert root.trace_id is None and tracing.current_trace_id() is None

    # And a request is served exactly as before.
    response = await _ask(_engine())
    assert response.choices[0].message.content == ANSWER


def test_unknown_step_detail_is_a_programming_error():
    with pytest.raises(TypeError):
        tracing.step("x", colour="blue")


# ── What a trace contains ───────────────────────────────────

@pytest.mark.asyncio
async def test_miss_records_every_step_in_one_trace(exporter):
    await _ask(_engine())
    spans = Spans(exporter)

    assert spans.names() == sorted([
        "chat-completion", "cache-lookup", "vector-search", "classify-intent", "llm-generation", "store-answer",
    ])
    assert len(spans.trace_ids()) == 1

    # The outline.
    assert spans.parent("chat-completion") is None
    assert spans.parent("cache-lookup") == "chat-completion"
    assert spans.parent("vector-search") == "cache-lookup"
    for step in ("classify-intent", "llm-generation", "store-answer"):
        assert spans.parent(step) == "chat-completion"

    # The request: question in, answer out, and how the cache handled it.
    assert spans.json("chat-completion", "input") == [{"role": "user", "content": QUESTION}]
    assert spans.attr("chat-completion", "output") == ANSWER
    assert spans.attr("chat-completion", "metadata.cache_status") == "miss"

    # The LLM call, with its tokens.
    assert spans.attr("llm-generation", "type") == "generation"
    assert spans.attr("llm-generation", "model.name") == MODEL
    assert spans.attr("llm-generation", "output") == ANSWER
    assert spans.json("llm-generation", "usage_details") == {"input": 7, "output": 3}

    # The classifier's decision, and what the stored answer got from it.
    assert spans.json("classify-intent", "output") == {"intent": "factual", "used_default_policy": False}
    assert spans.json("classify-intent", "usage_details") == {"total": 12}
    stored = spans.json("store-answer", "output")
    assert stored["stored"] is True and stored["intent"] == "factual"
    assert stored["required_similarity"] == TASK_POLICIES["factual"].similarity_threshold

    assert spans.json("cache-lookup", "output")["hit"] is False


@pytest.mark.asyncio
async def test_hit_shows_the_candidate_and_makes_no_llm_call(exporter):
    engine = _engine()
    await _ask(engine)
    Spans(exporter)  # discard the miss

    await _ask(engine)
    spans = Spans(exporter)

    assert spans.names() == ["cache-lookup", "chat-completion", "vector-search"]
    assert spans.attr("chat-completion", "metadata.cache_status") == "hit"
    assert spans.attr("chat-completion", "output") == ANSWER
    assert spans.attr("chat-completion", "metadata.cached_question") == QUESTION

    # The candidates explain the decision: what matched, how closely, what it needed.
    [candidate] = spans.json("vector-search", "output")
    assert candidate["cached_question"] == QUESTION
    assert candidate["similarity"] == pytest.approx(1.0)
    assert candidate["required"] == TASK_POLICIES["factual"].similarity_threshold


@pytest.mark.asyncio
async def test_only_the_request_is_a_root(exporter):
    """
    Storing happens after the response. Its step must still belong to the
    request's trace as a child, not show up in Langfuse as a second root.
    """
    background = BackgroundTasks()
    await chat_completions(_request(), Response(), _engine(), _provider(), _classifier(), None, background)
    await background()  # what FastAPI runs after sending the response
    spans = Spans(exporter)

    roots = [s.name for s in spans.all if s.attributes.get("langfuse.internal.is_app_root")]
    assert roots == ["chat-completion"]
    assert spans.json("store-answer", "output")["stored"] is True
    # The request's span ended when the answer was ready, before the store did.
    assert spans.get("chat-completion").end_time <= spans.get("store-answer").end_time


@pytest.mark.asyncio
async def test_uncacheable_request_is_traced_as_a_bypass(exporter):
    conversation = [
        ChatMessage(role="user", content=QUESTION),
        ChatMessage(role="assistant", content=ANSWER),
        ChatMessage(role="user", content="And its population?"),
    ]
    await _ask(_engine(), messages=conversation)
    spans = Spans(exporter)

    assert spans.names() == ["chat-completion", "llm-generation"]
    assert spans.attr("chat-completion", "metadata.cache_status") == "bypass"
    assert spans.attr("chat-completion", "metadata.bypass_reason") == "uncacheable"
    assert len(spans.json("llm-generation", "input")) == 3


# ── Streaming ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_streamed_miss_is_traced_like_any_other(exporter):
    response = await _ask(_engine(), stream=True)
    async for _ in response.body_iterator:
        pass
    spans = Spans(exporter)

    assert "llm-generation" in spans.names() and "store-answer" in spans.names()
    assert spans.attr("chat-completion", "output") == ANSWER
    assert spans.attr("chat-completion", "metadata.cache_status") == "miss"
    assert spans.attr("llm-generation", "output") == ANSWER
    assert spans.json("llm-generation", "usage_details") == {"input": 7, "output": 3}
    assert spans.attr("llm-generation", "completion_start_time")  # time to first word
    assert spans.attr("chat-completion", "level") is None  # a normal ending


@pytest.mark.asyncio
async def test_client_disconnect_still_closes_the_trace(exporter):
    response = await _ask(_engine(), stream=True)
    events = response.body_iterator
    await anext(events)
    await anext(events)  # the first words arrived...
    await events.aclose()  # ...then the client went away
    spans = Spans(exporter)

    assert spans.attr("chat-completion", "level") == "WARNING"
    assert spans.attr("chat-completion", "metadata.cache_status") == "error"
    assert spans.attr("llm-generation", "level") == "WARNING"
    assert "store-answer" not in spans.names()  # a partial answer is never cached


def test_streaming_over_http_is_traced(exporter):
    """Through FastAPI, where the stream is sent by a different task than the handler's."""
    from fastapi.testclient import TestClient

    from semcache.api.dependencies import get_classifier, get_engine, get_provider
    from semcache.main import app

    engine, provider, classifier = _engine(), _provider(), _classifier()
    app.dependency_overrides[get_engine] = lambda: engine
    app.dependency_overrides[get_provider] = lambda: provider
    app.dependency_overrides[get_classifier] = lambda: classifier
    try:
        body = {"model": MODEL, "stream": True, "messages": [{"role": "user", "content": QUESTION}]}
        resp = TestClient(app).post("/v1/chat/completions", json=body)
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200 and "Lumpur" in resp.text
    spans = Spans(exporter)

    assert len(spans.trace_ids()) == 1
    assert spans.attr("chat-completion", "output") == ANSWER
    assert spans.json("store-answer", "output")["stored"] is True
    assert [s.name for s in spans.all if s.attributes.get("langfuse.internal.is_app_root")] == ["chat-completion"]


# ── Failures ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_llm_failure_is_recorded_as_an_error(exporter):
    with pytest.raises(HTTPException):
        await _ask(_engine(), provider=_provider(error=RuntimeError("model overloaded")))
    spans = Spans(exporter)

    assert spans.attr("llm-generation", "level") == "ERROR"
    assert "model overloaded" in spans.attr("llm-generation", "status_message")
    assert spans.attr("chat-completion", "level") == "ERROR"
    assert spans.attr("chat-completion", "metadata.cache_status") == "error"


@pytest.mark.asyncio
async def test_a_broken_langfuse_never_breaks_a_request(exporter, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("langfuse is down")

    monkeypatch.setattr(tracing._client, "start_observation", broken)
    monkeypatch.setattr(tracing._client, "create_score", broken)

    response = await _ask(_engine())
    assert response.choices[0].message.content == ANSWER
    tracing.score("0" * 32, "good_match", True)  # doesn't raise either


# ── Feedback ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_feedback_is_scored_on_the_requests_trace(exporter, monkeypatch):
    engine = _engine()
    await _ask(engine)
    await _ask(engine)  # a hit, which can be judged
    hit = (await engine.recent_lookups())[0]
    assert hit.outcome == "hit"

    # The lookup remembers which trace it belongs to.
    spans = Spans(exporter)
    assert hit.trace_id in spans.trace_ids()

    create_score = MagicMock()
    monkeypatch.setattr(tracing._client, "create_score", create_score)
    await feedback(FeedbackRequest(lookup_id=hit.id, good_match=False), engine)

    create_score.assert_called_once()
    sent = create_score.call_args.kwargs
    assert sent["trace_id"] == hit.trace_id
    assert (sent["name"], sent["value"], sent["data_type"]) == ("good_match", 0, "BOOLEAN")


@pytest.mark.asyncio
async def test_lookups_have_no_trace_id_when_tracing_is_off():
    engine = _engine()
    await _ask(engine)
    assert (await engine.recent_lookups())[0].trace_id is None


def test_trace_id_survives_the_redis_round_trip():
    event = LookupEvent(prompt="q", model="m", outcome="hit", similarity=0.97, trace_id="ab" * 16)
    stored = {k.encode(): v.encode() for k, v in _to_hash(event).items()}
    assert _from_hash(event.id, stored).trace_id == "ab" * 16

    # Events written before tracing existed have no such field.
    del stored[b"trace_id"]
    assert _from_hash(event.id, stored).trace_id is None


# ── The embedding step ──────────────────────────────────────

@pytest.mark.asyncio
async def test_embedding_api_call_is_a_step_and_a_remembered_one_is_not(exporter):
    from types import SimpleNamespace

    from semcache.embeddings.gemini import GeminiEmbedder
    from semcache.embeddings.memory import CachedEmbedder, EmbeddingStore

    class Memory(EmbeddingStore):
        def __init__(self):
            self.vectors = {}

        async def get(self, key):
            return self.vectors.get(key)

        async def set(self, key, vector, ttl_seconds):
            self.vectors[key] = vector

    gemini = GeminiEmbedder(api_key="test-key", model="gemini-embedding-001", dims=4)
    gemini._client = MagicMock()
    gemini._client.aio.models.embed_content = AsyncMock(
        return_value=SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 0.0, 0.0, 0.0])])
    )
    embedder = CachedEmbedder(gemini, Memory(), identity="gemini-embedding-001:4", ttl_seconds=60)
    engine = CacheEngine(embedder=embedder, store=InMemoryVectorStore(), lookup_log=InMemoryLookupLog())

    await _ask(engine)  # the embedding API is called
    first = Spans(exporter)
    assert first.parent("embedding") == "cache-lookup"
    assert first.attr("embedding", "type") == "embedding"
    assert first.attr("embedding", "model.name") == "gemini-embedding-001"

    await _ask(engine)  # the same question: its embedding is remembered
    assert "embedding" not in Spans(exporter).names()


# ── Startup ─────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("keys_set", [True, False])
async def test_startup_turns_tracing_on_only_with_keys(keys_set, monkeypatch):
    import asyncio

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
    keys = (
        {"langfuse_public_key": f"pk-test-{uuid.uuid4().hex}", "langfuse_secret_key": "sk-test",
         "langfuse_base_url": "http://127.0.0.1:9"}
        if keys_set else {}
    )
    monkeypatch.setattr(
        dependencies, "get_settings", lambda: Settings(_env_file=None, gemini_api_key="test-key", **keys)
    )

    async with dependencies.lifespan(MagicMock()):
        assert tracing.enabled() is keys_set
        provider = dependencies.get_provider()
        assert isinstance(provider, TracedProvider if keys_set else GeminiProvider)
    assert not tracing.enabled()  # shut down with the server
