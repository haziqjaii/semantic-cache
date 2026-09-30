"""
HTTP tests for the cache admin endpoints (/v1/cache/...).

The engine runs on the in-memory store; settings are injected so the tests
don't depend on .env.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from semcache.api.dependencies import get_classifier, get_engine, get_provider
from semcache.cache.classifier import ClassifierResult, IntentClassifier
from semcache.cache.engine import CacheEngine
from semcache.cache.keys import hash_system_prompt
from semcache.cache.policy import DEFAULT_POLICY
from semcache.config import Settings, get_settings
from semcache.main import app
from semcache.providers.base import LLMProvider
from semcache.schemas import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionResponse,
)


@pytest.fixture
def engine(mock_embedder, memory_store) -> CacheEngine:
    return CacheEngine(embedder=mock_embedder, store=memory_store)


def _settings(admin_token: str | None = None) -> Settings:
    return Settings(_env_file=None, gemini_api_key="test-key", admin_token=admin_token)


@pytest.fixture
def client(engine):
    provider = MagicMock(spec=LLMProvider)
    provider.generate = AsyncMock(side_effect=lambda request: ChatCompletionResponse(
        id="id", created=0, model=request.model,
        choices=[ChatCompletionChoice(message=ChatCompletionChoiceMessage(content="An answer"))],
    ))
    classifier = MagicMock(spec=IntentClassifier)
    classifier.classify_safe = AsyncMock(
        return_value=ClassifierResult(policy=DEFAULT_POLICY, tokens=0, is_fallback=False)
    )

    app.dependency_overrides[get_engine] = lambda: engine
    app.dependency_overrides[get_provider] = lambda: provider
    app.dependency_overrides[get_classifier] = lambda: classifier
    app.dependency_overrides[get_settings] = lambda: _settings()
    yield TestClient(app)
    app.dependency_overrides.clear()


def _ask(client, content, *, model="m1", system=None, tags=None):
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": content})
    headers = {"X-Cache-Tags": tags} if tags else {}
    resp = client.post(
        "/v1/chat/completions", json={"model": model, "messages": messages}, headers=headers
    )
    assert resp.status_code == 200, resp.text
    return resp


# ── Tags arrive through the chat endpoint ───────────────────

def test_chat_tags_header_is_stored(client, memory_store):
    _ask(client, "What is Python?", tags="Docs, python:intro")

    (_, entry), = memory_store._entries
    assert entry.tags == ["docs", "python:intro"]


def test_invalid_tags_header_is_a_400(client, memory_store):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "m1", "messages": [{"role": "user", "content": "Hi"}]},
        headers={"X-Cache-Tags": "bad tag!"},
    )
    assert resp.status_code == 400
    assert "Invalid cache tag" in resp.json()["detail"]
    assert memory_store._entries == []


# ── Invalidation ────────────────────────────────────────────

def test_invalidate_by_system_prompt_text(client, memory_store):
    _ask(client, "q1", model="m1", system="You are a support agent.")
    _ask(client, "q2", model="m2", system="You are a support agent.")
    _ask(client, "q3", model="m1", system="You are a code reviewer.")

    resp = client.post("/v1/cache/invalidate", json={"system_prompt": "You are a support agent."})

    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] == 2
    assert body["filter"] == {"system_prompt_hash": hash_system_prompt("You are a support agent.")}
    assert len(memory_store._entries) == 1


def test_invalidate_requests_without_a_system_prompt(client, memory_store):
    """system_prompt "" selects entries whose requests had no system prompt."""
    _ask(client, "q1")
    _ask(client, "q2", system="You are a support agent.")

    assert client.post("/v1/cache/invalidate", json={"system_prompt": ""}).json()["deleted"] == 1


def test_invalidate_by_model_and_tag_combined(client):
    _ask(client, "q1", model="gemini-3.5-flash", tags="pricing")
    _ask(client, "q2", model="gemini-3.5-flash", tags="support")
    _ask(client, "q3", model="gemini-3.5-flash-lite", tags="pricing")

    body = client.post(
        "/v1/cache/invalidate", json={"model": "gemini-3.5-flash", "tag": "Pricing"}
    ).json()

    assert body["deleted"] == 1
    assert body["filter"] == {"model": "gemini-3.5-flash", "tag": "pricing"}


def test_dry_run_counts_without_deleting(client, memory_store):
    _ask(client, "q1", tags="billing:v1")
    _ask(client, "q2", tags="billing:v2")

    body = client.post(
        "/v1/cache/invalidate", json={"tag_prefix": "billing:", "dry_run": True}
    ).json()

    assert body == {"dry_run": True, "matched": 2, "filter": {"tag_prefix": "billing:"}}
    assert len(memory_store._entries) == 2


def test_all_clears_everything(client, memory_store):
    _ask(client, "q1")
    _ask(client, "q2")

    body = client.post("/v1/cache/invalidate", json={"all": True}).json()

    assert body == {"dry_run": False, "deleted": 2, "filter": {"all": True}}
    assert memory_store._entries == []


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "at least one filter"),
        ({"dry_run": True}, "at least one filter"),
        ({"all": True, "model": "m1"}, "cannot be combined"),
        ({"model": ""}, "must not be empty"),
        ({"system_prompt": "x", "system_prompt_hash": "0123456789abcdef"}, "not both"),
        ({"system_prompt_hash": "xyz"}, "16 hex"),
        ({"tag": "bad tag"}, "Invalid cache tag"),
        ({"tag_prefix": "b"}, "at least 2"),
    ],
)
def test_invalid_requests_are_rejected_and_delete_nothing(client, memory_store, payload, message):
    _ask(client, "q1")

    resp = client.post("/v1/cache/invalidate", json=payload)

    assert resp.status_code == 422
    assert message in resp.text
    assert len(memory_store._entries) == 1


# ── Admin token ─────────────────────────────────────────────

def test_admin_token_is_enforced_when_configured(client, memory_store):
    _ask(client, "q1")
    app.dependency_overrides[get_settings] = lambda: _settings(admin_token="s3cret")

    missing = client.post("/v1/cache/invalidate", json={"all": True})
    wrong = client.post(
        "/v1/cache/invalidate", json={"all": True}, headers={"Authorization": "Bearer nope"}
    )
    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert len(memory_store._entries) == 1

    ok = client.post(
        "/v1/cache/invalidate", json={"all": True}, headers={"Authorization": "Bearer s3cret"}
    )
    assert ok.status_code == 200
    assert memory_store._entries == []


def test_stats_endpoint(client):
    _ask(client, "q1")
    assert client.get("/v1/cache/stats").json() == {"total_entries": 1}


# ── Listing cached entries ──────────────────────────────────

def test_entries_lists_newest_first_with_details(client):
    _ask(client, "first question", tags="docs")
    _ask(client, "second question", model="m2")

    entries = client.get("/v1/cache/entries").json()["entries"]

    assert [e["prompt"] for e in entries] == ["second question", "first question"]
    first = entries[1]
    assert first["tags"] == ["docs"]
    assert first["model"] == "m1"
    assert first["hit_count"] == 0
    assert first["required_similarity"] == DEFAULT_POLICY.similarity_threshold
    assert first["expires_at"] > first["created_at"]


def test_entries_respects_limit_and_bounds(client):
    for i in range(3):
        _ask(client, f"question {i}")

    assert len(client.get("/v1/cache/entries?limit=2").json()["entries"]) == 2
    assert client.get("/v1/cache/entries?limit=0").status_code == 422
    assert client.get("/v1/cache/entries?limit=201").status_code == 422


def test_entries_requires_admin_token_when_configured(client):
    app.dependency_overrides[get_settings] = lambda: _settings(admin_token="s3cret")

    assert client.get("/v1/cache/entries").status_code == 401
    assert client.get(
        "/v1/cache/entries", headers={"Authorization": "Bearer s3cret"}
    ).status_code == 200
