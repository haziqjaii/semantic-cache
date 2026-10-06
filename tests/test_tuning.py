"""
Tests for the lookup log, near-miss analyzer, threshold tuner, and
thresholds learned from feedback.

Similarities are controlled exactly with 2-D vectors: a stored entry at
[s, sqrt(1 - s²)] has cosine similarity s with the query [1, 0].
"""

import math
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from semcache.api.dependencies import get_engine
from semcache.cache.engine import CacheEngine
from semcache.cache.lookup_log import HIT, MISS, NEAR_MISS, LookupEvent
from semcache.cache.store.base import CacheEntry
from semcache.cache.tuning import (
    MIN_LABELS,
    TARGET_PRECISION,
    learn_thresholds,
    tune,
)
from semcache.config import Settings, get_settings
from semcache.main import app
from tests.conftest import InMemoryVectorStore, MockEmbedder

QUERY = [1.0, 0.0]


_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def _name(similarity: float) -> str:
    """
    A cached question's text: "cached at nine-seven" for similarity 0.97.

    Spelled out, because a cached answer is only reused when its question
    has the same numbers as the one asked (cache/text.py), and the
    questions asked in these tests have none.
    """
    return "cached at " + "-".join(_DIGITS[int(d)] for d in f"{similarity:.2f}"[2:])


def _vector(similarity: float) -> list[float]:
    return [similarity, math.sqrt(1 - similarity**2)]


def _event(similarity, good_match=None, intent="factual", outcome=NEAR_MISS) -> LookupEvent:
    return LookupEvent(
        prompt="q", model="m", outcome=outcome, similarity=similarity,
        required_similarity=0.95, intent=intent, candidate_prompt="c", good_match=good_match,
    )


# ── The tuner ───────────────────────────────────────────────

class TestTune:
    def test_hit_rate_and_wrong_rate_per_threshold(self):
        events = [
            _event(0.99, good_match=True),
            _event(0.96, good_match=True),
            _event(0.93, good_match=False),
            _event(0.91),  # unlabelled
            LookupEvent(prompt="q", model="m", outcome=MISS),  # no candidate
        ]
        rows = {row.threshold: row for row in tune(events)}

        assert rows[0.90].would_hit == 4
        assert rows[0.90].hit_rate == 4 / 5
        assert rows[0.90].labelled == 3
        assert rows[0.90].wrong == 1
        assert rows[0.90].wrong_rate == pytest.approx(1 / 3)

        assert rows[0.95].would_hit == 2
        assert rows[0.95].wrong_rate == 0.0

        assert rows[1.00].would_hit == 0
        assert rows[1.00].wrong_rate is None  # no labels to judge
        assert [r.threshold for r in tune(events)] == [
            0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0
        ]

    def test_filter_by_intent(self):
        events = [_event(0.97, intent="factual"), _event(0.92, intent="how_to")]
        rows = tune(events, intent="how_to")
        assert rows[0].would_hit == 1
        assert rows[0].hit_rate == 1.0

    def test_no_events(self):
        assert all(r.hit_rate == 0.0 and r.wrong_rate is None for r in tune([]))


# ── Learning thresholds from feedback ───────────────────────

class TestLearnThresholds:
    def test_nothing_learned_below_min_labels(self):
        events = [_event(0.92, good_match=True) for _ in range(MIN_LABELS - 1)]
        assert learn_thresholds(events) == {}

    def test_picks_lowest_threshold_meeting_target_precision(self):
        # All matches ≥ 0.93 were good; the ones at 0.91 were wrong.
        events = (
            [_event(0.91, good_match=False) for _ in range(3)]
            + [_event(0.93, good_match=True) for _ in range(5)]
            + [_event(0.97, good_match=True) for _ in range(5)]
        )
        learned = learn_thresholds(events)["factual"]
        assert learned.threshold == 0.92  # the first threshold that excludes the 0.91s
        assert learned.labels == 13
        assert learned.precision == 1.0 >= TARGET_PRECISION

    def test_can_loosen_below_the_default(self):
        """If near misses at 0.91 are consistently right, factual drops from 0.95."""
        events = [_event(0.91, good_match=True) for _ in range(MIN_LABELS)]
        assert learn_thresholds(events)["factual"].threshold == 0.91

    def test_never_below_the_lowest_judged_similarity(self):
        """All labels at 0.93 say nothing about 0.90–0.92, so don't go there."""
        events = [_event(0.93, good_match=True) for _ in range(MIN_LABELS)]
        assert learn_thresholds(events)["factual"].threshold == 0.93

    def test_threshold_always_includes_the_judged_pairs(self):
        """0.9199 is below 0.92, so only 0.91 would have matched it."""
        events = [_event(0.9199, good_match=True) for _ in range(MIN_LABELS)]
        assert learn_thresholds(events)["factual"].threshold == 0.91

    def test_strictest_when_nothing_meets_the_target(self):
        events = [_event(0.99, good_match=False) for _ in range(MIN_LABELS)]
        learned = learn_thresholds(events)["factual"]
        assert learned.threshold == 0.99
        assert learned.precision is None

    def test_intents_are_learned_separately_and_unlabelled_ignored(self):
        events = (
            [_event(0.95, good_match=True, intent="factual") for _ in range(MIN_LABELS)]
            + [_event(0.95, good_match=True, intent="how_to") for _ in range(3)]
            + [_event(0.95, intent="how_to") for _ in range(20)]  # unlabelled
        )
        assert set(learn_thresholds(events)) == {"factual"}


# ── Engine: logging lookups and using learned thresholds ────

async def _engine_with(entries, lookup_log):
    """Engine whose query embeds to QUERY; entries are (similarity, required, intent)."""
    store = InMemoryVectorStore()
    embedder = MockEmbedder(dims=2)
    engine = CacheEngine(embedder=embedder, store=store, lookup_log=lookup_log)
    namespace = (await engine.lookup(prompt="setup", model="m")).namespace
    lookup_log.events.clear()
    for similarity, required, intent in entries:
        entry = CacheEntry(
            prompt=_name(similarity), response="answer", model="m", namespace=namespace,
            required_similarity=required, intent=intent,
        )
        await store.store(_vector(similarity), entry)
    embedder.embed = AsyncMock(return_value=QUERY)
    return engine


class TestEngineLogging:
    @pytest.mark.asyncio
    async def test_hit_is_logged_with_its_candidate(self, lookup_log):
        engine = await _engine_with([(0.97, 0.95, "factual")], lookup_log)

        result = await engine.lookup(prompt="What is Python?", model="m")

        assert result.hit
        event = lookup_log.events[result.lookup_id]
        assert event.outcome == HIT
        assert event.similarity == pytest.approx(0.97)
        assert event.required_similarity == 0.95
        assert event.intent == "factual"
        assert event.candidate_prompt == _name(0.97)

    @pytest.mark.asyncio
    async def test_near_miss_logs_the_closest_candidate(self, lookup_log):
        engine = await _engine_with([(0.93, 0.95, "factual"), (0.91, 0.95, "factual")], lookup_log)

        result = await engine.lookup(prompt="q", model="m")

        assert not result.hit
        event = lookup_log.events[result.lookup_id]
        assert event.outcome == NEAR_MISS
        assert event.similarity == pytest.approx(0.93)

    @pytest.mark.asyncio
    async def test_miss_without_candidates(self, lookup_log):
        engine = await _engine_with([(0.50, 0.95, "factual")], lookup_log)  # below the 0.90 floor

        result = await engine.lookup(prompt="q", model="m")

        event = lookup_log.events[result.lookup_id]
        assert event.outcome == MISS
        assert not event.has_candidate

    @pytest.mark.asyncio
    async def test_log_failure_never_breaks_the_lookup(self, lookup_log):
        engine = await _engine_with([(0.97, 0.95, "factual")], lookup_log)
        lookup_log.record = AsyncMock(side_effect=ConnectionError("Redis down"))

        result = await engine.lookup(prompt="q", model="m")

        assert result.hit
        assert result.lookup_id is None


class TestLearnedThresholdsInLookups:
    @pytest.mark.asyncio
    async def test_feedback_loosens_a_threshold_and_near_misses_start_hitting(self, lookup_log):
        engine = await _engine_with([(0.925, 0.95, "factual")], lookup_log)

        # At 0.925 vs the default 0.95, this is a near miss…
        first = await engine.lookup(prompt="q", model="m")
        assert not first.hit

        # …people say near misses like it are the same question…
        for _ in range(MIN_LABELS):
            near_miss = await engine.lookup(prompt="q", model="m")
            await engine.label_lookup(near_miss.lookup_id, good_match=True)

        # …so factual's threshold is learned lower, and it now hits.
        assert engine.learned_thresholds == {"factual": 0.92}
        assert (await engine.lookup(prompt="q", model="m")).hit

    @pytest.mark.asyncio
    async def test_bad_feedback_tightens_a_threshold(self, lookup_log):
        engine = await _engine_with([(0.96, 0.95, "factual")], lookup_log)

        for _ in range(MIN_LABELS):
            hit = await engine.lookup(prompt="q", model="m")
            assert hit.hit
            await engine.label_lookup(hit.lookup_id, good_match=False)

        # Wrong answers at 0.96 push factual to the strictest threshold.
        assert engine.learned_thresholds == {"factual": 0.99}
        assert not (await engine.lookup(prompt="q", model="m")).hit

    @pytest.mark.asyncio
    async def test_other_intents_keep_their_own_threshold(self, lookup_log):
        engine = await _engine_with([(0.92, 0.95, "factual")], lookup_log)
        engine._learned_thresholds = {"how_to": 0.90}

        assert not (await engine.lookup(prompt="q", model="m")).hit

    @pytest.mark.asyncio
    async def test_label_errors(self, lookup_log):
        engine = await _engine_with([], lookup_log)
        miss = await engine.lookup(prompt="q", model="m")

        with pytest.raises(LookupError):
            await engine.label_lookup("no-such-id", good_match=True)
        with pytest.raises(ValueError, match="nothing to judge"):
            await engine.label_lookup(miss.lookup_id, good_match=True)


# ── HTTP endpoints ──────────────────────────────────────────

@pytest.fixture
def api(lookup_log):
    """TestClient plus a helper to seed the engine with controlled entries."""
    state = {}

    async def seed(entries):
        state["engine"] = await _engine_with(entries, lookup_log)
        return state["engine"]

    app.dependency_overrides[get_engine] = lambda: state["engine"]
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, gemini_api_key="k")
    yield TestClient(app), seed
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_feedback_endpoint(api, lookup_log):
    client, seed = api
    engine = await seed([(0.97, 0.95, "factual")])
    hit = await engine.lookup(prompt="q", model="m")

    resp = client.post("/v1/cache/feedback", json={"lookup_id": hit.lookup_id, "good_match": False})

    assert resp.status_code == 200
    assert resp.json()["lookup"]["good_match"] is False
    assert lookup_log.events[hit.lookup_id].good_match is False
    assert client.post("/v1/cache/feedback", json={"lookup_id": "nope", "good_match": True}).status_code == 404

    miss = await (await seed([])).lookup(prompt="q", model="m")
    assert client.post("/v1/cache/feedback", json={"lookup_id": miss.lookup_id, "good_match": True}).status_code == 400


@pytest.mark.asyncio
async def test_near_misses_endpoint(api):
    client, seed = api
    engine = await seed([(0.93, 0.95, "factual")])
    await engine.lookup(prompt="What is Python?", model="m")
    await engine.lookup(prompt="Explain Python", model="m")

    body = client.get("/v1/cache/near-misses?limit=1").json()

    assert body["total"] == 2
    assert body["labelled"] == 0
    (latest,) = body["near_misses"]
    assert latest["prompt"] == "Explain Python"  # newest first
    assert latest["candidate_prompt"] == _name(0.93)
    assert latest["similarity"] == pytest.approx(0.93)
    assert latest["required_similarity"] == 0.95


@pytest.mark.asyncio
async def test_lookup_endpoint_says_which_cached_question_served_a_hit(api):
    client, seed = api
    engine = await seed([(0.97, 0.95, "factual")])
    hit = await engine.lookup(prompt="What's Python?", model="m")

    body = client.get(f"/v1/cache/lookups/{hit.lookup_id}").json()["lookup"]

    assert body["prompt"] == "What's Python?"
    assert body["outcome"] == "hit"
    assert body["candidate_prompt"] == _name(0.97)
    assert client.get("/v1/cache/lookups/nope").status_code == 404


@pytest.mark.asyncio
async def test_near_misses_and_feedback_need_the_admin_token(api):
    client, seed = api
    await seed([])
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, gemini_api_key="k", admin_token="s3cret"
    )

    assert client.get("/v1/cache/near-misses").status_code == 401
    assert client.get("/v1/cache/lookups/x").status_code == 401  # it shows prompts
    assert client.post("/v1/cache/feedback", json={"lookup_id": "x", "good_match": True}).status_code == 401
    # The tuner and thresholds are aggregate numbers: no prompts, no side effects.
    assert client.get("/v1/cache/tuner").status_code == 200
    assert client.get("/v1/cache/thresholds").status_code == 200


@pytest.mark.asyncio
async def test_tuner_endpoint(api):
    client, seed = api
    engine = await seed([(0.93, 0.95, "factual")])
    near_miss = await engine.lookup(prompt="q", model="m")
    await engine.label_lookup(near_miss.lookup_id, good_match=True)

    body = client.get("/v1/cache/tuner").json()
    rows = {row["threshold"]: row for row in body["rows"]}

    assert body["lookups_analysed"] == 1
    assert body["labelled"] == 1
    assert rows[0.93]["would_hit"] == 1
    assert rows[0.93]["wrong_rate"] == 0.0
    assert rows[0.94]["would_hit"] == 0
    assert client.get("/v1/cache/tuner?intent=factual").json()["lookups_analysed"] == 1
    assert client.get("/v1/cache/tuner?intent=creative").status_code == 422


@pytest.mark.asyncio
async def test_thresholds_endpoint(api):
    client, seed = api
    engine = await seed([(0.925, 0.95, "factual")])
    for _ in range(MIN_LABELS):
        near_miss = await engine.lookup(prompt="q", model="m")
        await engine.label_lookup(near_miss.lookup_id, good_match=True)

    intents = {i["intent"]: i for i in client.get("/v1/cache/thresholds").json()["intents"]}

    assert set(intents) == {"factual", "how_to", "time_sensitive", "classification"}
    assert intents["factual"] == {
        "intent": "factual", "default": 0.95, "learned": 0.92, "in_use": 0.92,
        "labels": MIN_LABELS, "precision": 1.0,
    }
    assert intents["how_to"]["learned"] is None
    assert intents["how_to"]["in_use"] == 0.93


def test_chat_returns_lookup_id_header():
    """The lookup id travels to clients in X-Cache-Lookup-Id."""
    from fastapi import Response

    from semcache.api.chat import _set_cache_headers

    response = Response()
    _set_cache_headers(response, "HIT", similarity=0.97, lookup_id="abc123")
    assert response.headers["X-Cache-Lookup-Id"] == "abc123"


@pytest.mark.asyncio
async def test_classifier_intent_is_stored_on_the_entry(lookup_log):
    store = InMemoryVectorStore()
    engine = CacheEngine(embedder=MockEmbedder(), store=store, lookup_log=lookup_log)
    result = await engine.lookup(prompt="How do I sort?", model="m")

    await engine.store(result, prompt="How do I sort?", response="Use sorted()", model="m", intent="how_to")

    (_, entry), = store._entries
    assert entry.intent == "how_to"



@pytest.mark.asyncio
async def test_thresholds_endpoint_counts_labels_before_learning(api):
    """Progress toward learning is visible: 1 label shows as 1, not 0."""
    client, seed = api
    engine = await seed([(0.97, 0.95, "factual")])
    hit = await engine.lookup(prompt="q", model="m")
    await engine.label_lookup(hit.lookup_id, good_match=True)

    intents = {i["intent"]: i for i in client.get("/v1/cache/thresholds").json()["intents"]}

    assert intents["factual"]["labels"] == 1
    assert intents["factual"]["learned"] is None
