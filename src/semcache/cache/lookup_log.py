"""
Lookup log — a record of recent cache lookups, and the feedback on them.

Every cacheable request writes one LookupEvent: the question, the closest
cached question it was compared with, their similarity, and the outcome
(hit, near miss, or miss). Three features read from it:

  - Near-miss analyzer: lookups that almost matched a cached answer.
  - Threshold tuner:    "what if the threshold were 0.92?" using real lookups.
  - Learned thresholds: per-intent thresholds chosen from human feedback.

FEEDBACK
    A person labels a lookup that had a candidate:
      - a hit:       "was the cached answer right for this question?"
      - a near miss: "would the cached answer have been right?"
    Both are the same question — does the candidate's answer fit this
    question? — so one boolean, `good_match`, covers both.

STORAGE (Redis)
    semcache:lookup:{id}       hash, one per event (expires after 7 days)
    semcache:lookups           sorted set of event ids by time (last 5,000)
    semcache:lookups:labelled  set of labelled ids; labelled events never
                               expire, because they are the training data
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime

from redis.asyncio import Redis

# Outcomes of a lookup.
HIT = "hit"
NEAR_MISS = "near_miss"  # a candidate was found, but below its threshold
MISS = "miss"  # no candidate above the search floor


@dataclass
class LookupEvent:
    prompt: str
    model: str
    outcome: str
    # The closest candidate considered (the one served, on a hit).
    # All None on a MISS: nothing similar enough was cached.
    similarity: float | None = None
    required_similarity: float | None = None
    intent: str | None = None
    candidate_prompt: str | None = None
    # Feedback: does the candidate's answer fit this question? None = unlabelled.
    good_match: bool | None = None
    # The Langfuse trace of the request, when tracing is on (see tracing.py),
    # so feedback can be attached to that trace.
    trace_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def has_candidate(self) -> bool:
        return self.similarity is not None


class LookupLog(ABC):
    """Where lookup events and their labels are kept."""

    @abstractmethod
    async def record(self, event: LookupEvent) -> None:
        """Save one event."""

    @abstractmethod
    async def get(self, event_id: str) -> LookupEvent | None:
        """One event by id, or None if unknown or expired."""

    @abstractmethod
    async def label(self, event_id: str, good_match: bool) -> bool:
        """Attach feedback to an event. Returns False if it doesn't exist."""

    @abstractmethod
    async def recent(self) -> list[LookupEvent]:
        """The retained recent events, newest first (for the tuner and analyzer)."""

    @abstractmethod
    async def labelled(self) -> list[LookupEvent]:
        """Every labelled event, however old (the training data for learning)."""


# ── Redis implementation ────────────────────────────────────

_EVENT_PREFIX = "semcache:lookup:"
_RECENT_KEY = "semcache:lookups"
_LABELLED_KEY = "semcache:lookups:labelled"

MAX_RECENT_EVENTS = 5000
EVENT_TTL_SECONDS = 7 * 24 * 3600


def _to_hash(event: LookupEvent) -> dict[str, str]:
    def opt(value) -> str:
        return "" if value is None else str(value)

    return {
        "prompt": event.prompt,
        "model": event.model,
        "outcome": event.outcome,
        "similarity": opt(event.similarity),
        "required_similarity": opt(event.required_similarity),
        "intent": opt(event.intent),
        "candidate_prompt": opt(event.candidate_prompt),
        "good_match": "" if event.good_match is None else ("1" if event.good_match else "0"),
        "trace_id": opt(event.trace_id),
        "created_at": event.created_at.isoformat(),
    }


def _from_hash(event_id: str, data: dict[bytes, bytes]) -> LookupEvent:
    d = {k.decode(): v.decode() for k, v in data.items()}
    return LookupEvent(
        id=event_id,
        prompt=d["prompt"],
        model=d["model"],
        outcome=d["outcome"],
        similarity=float(d["similarity"]) if d.get("similarity") else None,
        required_similarity=float(d["required_similarity"]) if d.get("required_similarity") else None,
        intent=d.get("intent") or None,
        candidate_prompt=d.get("candidate_prompt") or None,
        good_match=None if not d.get("good_match") else d["good_match"] == "1",
        trace_id=d.get("trace_id") or None,
        created_at=datetime.fromisoformat(d["created_at"]),
    )


class RedisLookupLog(LookupLog):
    """Lookup log in Redis (layout in the module docstring)."""

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._redis: Redis | None = None

    async def initialize(self) -> None:
        self._redis = Redis.from_url(self._redis_url)

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()

    @property
    def _client(self) -> Redis:
        if self._redis is None:
            raise RuntimeError("Lookup log not initialized. Call initialize() first.")
        return self._redis

    async def record(self, event: LookupEvent) -> None:
        key = f"{_EVENT_PREFIX}{event.id}"
        pipe = self._client.pipeline()
        pipe.hset(key, mapping=_to_hash(event))
        pipe.expire(key, EVENT_TTL_SECONDS)
        pipe.zadd(_RECENT_KEY, {event.id: event.created_at.timestamp()})
        pipe.zcard(_RECENT_KEY)
        *_, size = await pipe.execute()
        # Trim in batches (not on every write) to stay at ~MAX_RECENT_EVENTS.
        if size > MAX_RECENT_EVENTS + 500:
            await self._trim(size - MAX_RECENT_EVENTS)

    async def _trim(self, count: int) -> None:
        oldest = [i.decode() for i in await self._client.zrange(_RECENT_KEY, 0, count - 1)]
        if not oldest:
            return
        labelled = await self._client.smismember(_LABELLED_KEY, oldest)
        pipe = self._client.pipeline()
        pipe.zrem(_RECENT_KEY, *oldest)
        # Keep labelled events: learning still needs them.
        unlabelled = [f"{_EVENT_PREFIX}{i}" for i, keep in zip(oldest, labelled) if not keep]
        if unlabelled:
            pipe.delete(*unlabelled)
        await pipe.execute()

    async def get(self, event_id: str) -> LookupEvent | None:
        data = await self._client.hgetall(f"{_EVENT_PREFIX}{event_id}")
        return _from_hash(event_id, data) if data else None

    async def label(self, event_id: str, good_match: bool) -> bool:
        key = f"{_EVENT_PREFIX}{event_id}"
        # HSET on a missing key would create a half-empty event, so check first.
        if not await self._client.exists(key):
            return False
        pipe = self._client.pipeline()
        pipe.hset(key, "good_match", "1" if good_match else "0")
        pipe.persist(key)  # training data doesn't expire
        pipe.sadd(_LABELLED_KEY, event_id)
        await pipe.execute()
        return True

    async def _load(self, ids: list[str]) -> list[LookupEvent]:
        pipe = self._client.pipeline()
        for event_id in ids:
            pipe.hgetall(f"{_EVENT_PREFIX}{event_id}")
        rows = await pipe.execute()
        # Unlabelled events can expire while still listed; skip those.
        return [_from_hash(i, data) for i, data in zip(ids, rows) if data]

    async def recent(self) -> list[LookupEvent]:
        ids = [i.decode() for i in await self._client.zrevrange(_RECENT_KEY, 0, MAX_RECENT_EVENTS - 1)]
        return await self._load(ids)

    async def labelled(self) -> list[LookupEvent]:
        ids = [i.decode() for i in await self._client.smembers(_LABELLED_KEY)]
        return await self._load(ids)
