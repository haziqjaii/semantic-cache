"""
The model list (GET /v1/models) and prices for models added by configuration.
"""

import asyncio
import copy
import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import Response
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError

from semcache import config, tracing
from semcache.api.chat import chat_completions
from semcache.api.dependencies import get_provider
from semcache.cache.classifier import ClassifierResult, IntentClassifier
from semcache.cache.engine import CacheEngine
from semcache.cache.policy import TASK_POLICIES
from semcache.config import (
    ModelPrice,
    Settings,
    add_model_prices,
    estimate_cost_myr,
    estimate_cost_usd,
)
from semcache.main import app
from semcache.metrics import metrics
from semcache.providers.base import LLMProvider
from semcache.providers.gemini import GeminiProvider
from semcache.providers.openai_compatible import OpenAICompatibleProvider
from semcache.providers.router import RoutingProvider
from semcache.providers.traced import TracedProvider
from semcache.schemas import ChatCompletionRequest, ChatMessage
from tests.conftest import InMemoryLookupLog, InMemoryVectorStore, MockEmbedder

MODEL = "openai-gpt-oss-120b"
PRICE = {MODEL: {"input": 0.15, "output": 0.60}}


@pytest.fixture(autouse=True)
def clean_state():
    """Each test starts with the built-in prices and zeroed metrics."""
    original = copy.deepcopy(config.PRICING_TABLE["models"])
    metrics.reset()
    yield
    config.PRICING_TABLE["models"].clear()
    config.PRICING_TABLE["models"].update(original)
    metrics.reset()
    tracing.shutdown()


def _openai_compatible(handler, **kwargs) -> tuple[OpenAICompatibleProvider, list[httpx.Request]]:
    sent: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return handler(request)

    client = httpx.AsyncClient(base_url="https://models.example/v1/", transport=httpx.MockTransport(record))
    return OpenAICompatibleProvider(api_key="secret-key", client=client, **kwargs), sent


def _model_list(*ids) -> httpx.Response:
    return httpx.Response(200, json={"object": "list", "data": [{"id": i, "object": "model"} for i in ids]})


# ── Prices from configuration ───────────────────────────────

def test_extra_prices_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("EXTRA_MODEL_PRICES", json.dumps(PRICE))

    settings = Settings(_env_file=None, gemini_api_key="test-key")

    assert settings.extra_model_prices == {MODEL: ModelPrice(input=0.15, output=0.60)}


def test_extra_prices_are_read_from_a_dotenv_file(tmp_path):
    """The line as it's documented in .env.example, unquoted, with spaces in the JSON."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        'GEMINI_API_KEY=test-key\n'
        'EXTRA_MODEL_PRICES={"openai-gpt-oss-120b": {"input": 0.15, "output": 0.60}, "Mistral Small 4": {"input": 0.1}}\n'
        "OPENAI_COMPATIBLE_MODELS=openai-gpt-oss-120b, Mistral Small 4\n"
    )

    settings = Settings(_env_file=env_file)

    assert settings.extra_model_prices[MODEL] == ModelPrice(input=0.15, output=0.60)
    assert settings.extra_model_prices["Mistral Small 4"] == ModelPrice(input=0.1, output=0.0)
    assert settings.openai_compatible_models == "openai-gpt-oss-120b, Mistral Small 4"


@pytest.mark.parametrize("bad", ['{"m": {"input": -1}}', '{"m": {"output": 1}}', "not json"])
def test_bad_prices_stop_startup_with_an_error(monkeypatch, bad):
    monkeypatch.setenv("EXTRA_MODEL_PRICES", bad)

    with pytest.raises((ValidationError, ValueError)):
        Settings(_env_file=None, gemini_api_key="test-key")


def test_no_extra_prices_by_default():
    assert Settings(_env_file=None, gemini_api_key="test-key").extra_model_prices == {}


def test_added_prices_are_used_for_cost():
    assert estimate_cost_usd(MODEL, 1000, 500) is None  # unknown until given a price

    add_model_prices({MODEL: ModelPrice(input=0.15, output=0.60)})

    cost = estimate_cost_usd(MODEL, 1_000_000, 500_000)
    assert cost == {"input": pytest.approx(0.15), "output": pytest.approx(0.30), "total": pytest.approx(0.45)}
    assert estimate_cost_myr(MODEL, 1_000_000, 500_000) == pytest.approx(0.45 * config.PRICING_TABLE["usd_to_myr"])


def test_built_in_prices_are_unchanged():
    price = config.PRICING_TABLE["models"]["gemini-3.5-flash"]
    expected_usd = (2000 * price["input"] + 1000 * price["output"]) / 1_000_000

    assert estimate_cost_usd("gemini-3.5-flash", 2000, 1000)["total"] == pytest.approx(expected_usd)
    assert estimate_cost_myr("gemini-3.5-flash", 2000, 1000) == pytest.approx(
        expected_usd * config.PRICING_TABLE["usd_to_myr"]
    )


# ── A provider's model list ─────────────────────────────────

@pytest.mark.asyncio
async def test_lists_the_providers_chat_models_only():
    provider, sent = _openai_compatible(lambda request: _model_list(
        "openai-gpt-oss-120b", "Mistral Small 3.2 24B Instruct 2506", "qwen-qwen3-8-27b",
        "e5-mistral-7b", "qwen-qwen3-embedding-8b", "bge-m3", "whisper-large-v3",
    ))

    models = await provider.list_models()

    assert models == ["openai-gpt-oss-120b", "Mistral Small 3.2 24B Instruct 2506", "qwen-qwen3-8-27b"]
    assert str(sent[0].url) == "https://models.example/v1/models"


@pytest.mark.asyncio
async def test_model_list_is_reused_for_a_while(monkeypatch):
    provider, sent = _openai_compatible(lambda request: _model_list("a", "b"))

    assert await provider.list_models() == ["a", "b"]
    assert await provider.list_models() == ["a", "b"]
    assert len(sent) == 1  # the second answer came from memory

    monkeypatch.setattr("semcache.providers.openai_compatible.MODEL_LIST_TTL_SECONDS", 0.0)
    await provider.list_models()
    assert len(sent) == 2  # asked again once it was old


@pytest.mark.asyncio
async def test_unreachable_provider_gives_the_last_list_not_an_error(monkeypatch):
    responses = iter([_model_list("a"), httpx.Response(503, text="down")])
    provider, _ = _openai_compatible(lambda request: next(responses))
    monkeypatch.setattr("semcache.providers.openai_compatible.MODEL_LIST_TTL_SECONDS", 0.0)

    assert await provider.list_models() == ["a"]
    assert await provider.list_models() == ["a"]  # the 503 is survived


@pytest.mark.asyncio
async def test_provider_that_never_answered_gives_an_empty_list():
    def refuse(request):
        raise httpx.ConnectError("no route")

    provider, _ = _openai_compatible(refuse)

    assert await provider.list_models() == []


@pytest.mark.asyncio
async def test_configured_model_list_is_used_without_asking_the_provider():
    provider, sent = _openai_compatible(lambda request: _model_list("x"), models=["one", "two"])

    assert await provider.list_models() == ["one", "two"]
    assert sent == []


@pytest.mark.asyncio
async def test_gemini_lists_its_priced_chat_models():
    add_model_prices({"gemini-9-pro": ModelPrice(input=1, output=2), MODEL: ModelPrice(input=1, output=2)})

    models = await GeminiProvider(api_key="test-key").list_models()

    assert "gemini-3.5-flash-lite" in models and "gemini-3.5-flash" in models
    assert "gemini-9-pro" in models  # a newly priced Gemini model appears
    assert MODEL not in models  # another provider's model doesn't
    assert not any("embedding" in m for m in models)


@pytest.mark.asyncio
async def test_router_lists_both_providers_models():
    other, _ = _openai_compatible(lambda request: _model_list(MODEL, "qwen-qwen3-8-27b"))
    router = RoutingProvider(google=GeminiProvider(api_key="test-key"), other=other)

    models = await TracedProvider(router).list_models()  # tracing passes the list through

    assert models[-2:] == [MODEL, "qwen-qwen3-8-27b"]
    assert "gemini-3.5-flash-lite" in models


def test_a_provider_without_a_list_offers_none():
    class Minimal(LLMProvider):
        async def generate(self, request): ...

        def generate_stream(self, request): ...

    assert asyncio.run(Minimal().list_models()) == []


# ── GET /v1/models ──────────────────────────────────────────

def test_models_endpoint_lists_what_the_proxy_offers():
    add_model_prices({MODEL: ModelPrice(input=0.15, output=0.60)})
    other, _ = _openai_compatible(lambda request: _model_list(MODEL, "qwen-qwen3-8-27b"))
    router = RoutingProvider(google=GeminiProvider(api_key="test-key"), other=other)
    app.dependency_overrides[get_provider] = lambda: router
    try:
        resp = TestClient(app).get("/v1/models")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    by_id = {m["id"]: m for m in body["data"]}
    assert by_id["gemini-3.5-flash-lite"] == {
        "id": "gemini-3.5-flash-lite", "object": "model", "owned_by": "google", "priced": True,
    }
    assert by_id[MODEL]["owned_by"] == "openai-compatible" and by_id[MODEL]["priced"] is True
    assert by_id["qwen-qwen3-8-27b"]["priced"] is False


# ── Cost on traces, and savings in RM ───────────────────────

def _request(model=MODEL) -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model, messages=[ChatMessage(role="user", content="What is the capital of Malaysia?")])


def _answering_provider() -> OpenAICompatibleProvider:
    body = {
        "choices": [{"message": {"role": "assistant", "content": "Kuala Lumpur."}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
    }
    return _openai_compatible(lambda request: httpx.Response(200, json=body))[0]


def _classifier() -> MagicMock:
    classifier = MagicMock(spec=IntentClassifier)
    classifier.model = "gemini-3.5-flash-lite"
    classifier.classify_safe = AsyncMock(return_value=ClassifierResult(
        policy=TASK_POLICIES["factual"], tokens=10, is_fallback=False, intent="factual",
    ))
    return classifier


def _engine() -> CacheEngine:
    return CacheEngine(embedder=MockEmbedder(), store=InMemoryVectorStore(), lookup_log=InMemoryLookupLog())


@pytest.mark.asyncio
@pytest.mark.parametrize("priced", [True, False])
async def test_trace_shows_the_cost_only_for_priced_models(priced):
    if priced:
        add_model_prices({MODEL: ModelPrice(input=0.15, output=0.60)})
    exporter = InMemorySpanExporter()
    assert tracing.configure(
        f"pk-test-{uuid.uuid4().hex}", "sk-test", base_url="http://127.0.0.1:9", span_exporter=exporter
    )

    await chat_completions(_request(), Response(), _engine(), TracedProvider(_answering_provider()), _classifier())
    tracing.flush()

    [generation] = [s for s in exporter.get_finished_spans() if s.name == "llm-generation"]
    cost = generation.attributes.get("langfuse.observation.cost_details")
    if priced:
        # 1000 input tokens at $0.15 per million, 500 output at $0.60 per million.
        assert json.loads(cost) == {
            "input": pytest.approx(0.00015), "output": pytest.approx(0.0003), "total": pytest.approx(0.00045),
        }
    else:
        assert cost is None  # no price known: no cost claimed


@pytest.mark.asyncio
@pytest.mark.parametrize("priced", [True, False])
async def test_a_hit_counts_money_saved_only_for_priced_models(priced):
    if priced:
        add_model_prices({MODEL: ModelPrice(input=0.15, output=0.60)})
    engine, provider = _engine(), _answering_provider()

    await chat_completions(_request(), Response(), engine, provider, _classifier())  # miss
    await chat_completions(_request(), Response(), engine, provider, _classifier())  # hit

    assert metrics.cache_hits == 1
    assert metrics.tokens_saved == 1500  # tokens are counted either way
    if priced:
        assert metrics.cost_saved_myr == pytest.approx(0.00045 * config.PRICING_TABLE["usd_to_myr"])
        assert MODEL not in metrics.unpriced_models
    else:
        assert metrics.cost_saved_myr == 0
        assert MODEL in metrics.unpriced_models


# ── Startup ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_startup_applies_extra_prices_and_the_configured_model_list(monkeypatch):
    from semcache.api import dependencies

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
    monkeypatch.setattr(dependencies, "get_settings", lambda: Settings(
        _env_file=None, gemini_api_key="test-key",
        openai_compatible_api_key="secret-key", openai_compatible_base_url="https://models.example/v1",
        openai_compatible_models=" openai-gpt-oss-120b , qwen-qwen3-8-27b,",
        extra_model_prices={MODEL: {"input": 0.15, "output": 0.60}},
    ))

    async with dependencies.lifespan(MagicMock()):
        assert estimate_cost_usd(MODEL, 1_000_000)["total"] == pytest.approx(0.15)
        models = await dependencies.get_provider().list_models()
        assert models[-2:] == ["openai-gpt-oss-120b", "qwen-qwen3-8-27b"]
