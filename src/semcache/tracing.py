"""
Optional request tracing with Langfuse.

Prometheus and Grafana show totals: hit rate, latency percentiles, cost
saved. Langfuse shows ONE request at a time: the question, each step the
cache took, how long each step ran, and what it decided. That's what you
open to answer "why did THIS request get a wrong answer from the cache?"

A traced request looks like this:

    chat-completion            the whole request (question in, answer out)
    ├─ cache-lookup            hit or miss, and the lookup id
    │  ├─ embedding            only when the embedding API was really called
    │  └─ vector-search        every candidate, its similarity, what it needed
    ├─ classify-intent         on a miss: the question type the classifier chose
    ├─ llm-generation          on a miss or bypass: the LLM call and its tokens
    └─ store-answer            on a miss: the TTL and threshold the answer got

OFF BY DEFAULT
    Tracing turns on only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY
    are set. Without them every function here does nothing, and the
    langfuse package is never even imported.

    When it's on, questions and answers are sent to the Langfuse server at
    LANGFUSE_BASE_URL (Langfuse Cloud unless you self-host).

NEVER IN THE WAY
    Tracing is for looking at requests, not for serving them. Every call
    into Langfuse is wrapped so a tracing failure is logged and ignored;
    it can't fail or slow a request. Spans are sent in the background.

HOW CODE USES IT
    chat.py starts the trace and owns its outline. Deeper code (the engine,
    the embedder, the LLM provider) adds steps with `tracing.step(...)`,
    which attaches to the request being served and does nothing otherwise:

        with tracing.step("vector-search", as_type="retriever") as step:
            candidates = await store.search(...)
            step.update(output=...)
"""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from typing import Any, Self

logger = logging.getLogger(__name__)

# The Langfuse client, or None while tracing is off.
_client: Any = None

# Details a step accepts, passed straight to Langfuse.
_DETAILS = frozenset({
    "input", "output", "metadata", "model", "model_parameters", "usage_details",
    "level", "status_message", "completion_start_time",
})


def _clean(details: dict[str, Any]) -> dict[str, Any]:
    """Keep the details Langfuse understands, dropping unset ones."""
    unknown = details.keys() - _DETAILS
    if unknown:
        raise TypeError(f"Unknown step details: {sorted(unknown)}")
    cleaned = {k: v for k, v in details.items() if v is not None}
    if isinstance(cleaned.get("metadata"), dict):
        cleaned["metadata"] = {k: v for k, v in cleaned["metadata"].items() if v is not None}
    if not isinstance(cleaned.get("model", ""), str):
        del cleaned["model"]  # e.g. a mock in tests
    return cleaned


class Step:
    """
    One step of a traced request (a Langfuse "observation").

    Safe to use whether tracing is on or off: with tracing off it wraps
    nothing, and every method returns at once. No method ever raises
    because of Langfuse.

    As a context manager, the step is the parent of any `tracing.step()`
    started inside the block, and ends when the block does:

        with tracing.step("cache-lookup") as lookup:
            ...                      # steps started here nest under it
            lookup.update(output=...)
    """

    def __init__(self, observation: Any = None) -> None:
        self._observation = observation
        self._ended = observation is None
        self._token: Token | None = None

    @property
    def trace_id(self) -> str | None:
        """The Langfuse trace this step belongs to (None with tracing off)."""
        return getattr(self._observation, "trace_id", None)

    @property
    def open(self) -> bool:
        """True until the step has ended. Always False with tracing off."""
        return not self._ended

    def step(self, name: str, *, as_type: str = "span", **details: Any) -> Step:
        """Start a step inside this one."""
        details = _clean(details)
        if self._ended:
            # Langfuse would file a step under an already-ended parent as a
            # second root of the trace, so don't start one.
            return Step()
        try:
            return Step(self._observation.start_observation(name=name, as_type=as_type, **details))
        except Exception:
            logger.warning("Langfuse: could not start step %r", name, exc_info=True)
            return Step()

    def update(self, **details: Any) -> None:
        """Add details (output, metadata, token usage, ...) to the step."""
        details = _clean(details)
        if self._ended or not details:
            return
        try:
            self._observation.update(**details)
        except Exception:
            logger.warning("Langfuse: could not update a step", exc_info=True)

    def end(self, **details: Any) -> None:
        """End the step, with any last details. Ending twice is harmless."""
        self.update(**details)
        if self._ended:
            return
        self._ended = True
        try:
            self._observation.end()
        except Exception:
            logger.warning("Langfuse: could not end a step", exc_info=True)

    def fail(self, error: BaseException | str) -> None:
        """End the step as failed."""
        self.end(level="ERROR", status_message=str(error) or type(error).__name__)

    def __enter__(self) -> Self:
        self._token = _current.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _current.reset(self._token)
            self._token = None
        if exc is not None:
            self.fail(exc)
        else:
            self.end()


# The step that `tracing.step()` attaches to: set per request by
# start_trace(), and by `with step:` for nesting. Each request runs in its
# own asyncio task, so requests never see each other's steps.
_current: ContextVar[Step | None] = ContextVar("semcache_current_step", default=None)


# ── Turning tracing on and off ──────────────────────────────

def configure(
    public_key: str | None,
    secret_key: str | None,
    base_url: str | None = None,
    **client_options: Any,
) -> bool:
    """
    Turn tracing on if both Langfuse keys are given. Returns whether it's on.

    Called once at startup. `client_options` go to the Langfuse client
    (tests pass an in-memory span exporter this way).
    """
    global _client
    shutdown()
    if not (public_key and secret_key):
        return False
    try:
        from langfuse import Langfuse

        if base_url:
            client_options["base_url"] = base_url
        _client = Langfuse(public_key=public_key, secret_key=secret_key, **client_options)
    except Exception:
        logger.exception("Langfuse tracing could not start; continuing without it")
        _client = None
        return False
    logger.info("Langfuse tracing is on (%s)", base_url or "Langfuse Cloud")
    return True


def enabled() -> bool:
    return _client is not None


def flush() -> None:
    """Send everything recorded so far (tests; shutdown does this too)."""
    if _client is not None:
        try:
            _client.flush()
        except Exception:
            logger.warning("Langfuse: flush failed", exc_info=True)


def shutdown() -> None:
    """Send what's left and turn tracing off. Called when the server stops."""
    global _client
    client, _client = _client, None
    if client is not None:
        try:
            client.shutdown()
        except Exception:
            logger.warning("Langfuse: shutdown failed", exc_info=True)


# ── Recording ───────────────────────────────────────────────

def start_trace(name: str, **details: Any) -> Step:
    """
    Start the trace for a request, and make it the request's current step.

    The caller ends it (`.end()` / `.fail()`) when the request is done.
    """
    details = _clean(details)
    root = Step()
    if _client is not None:
        try:
            root = Step(_client.start_observation(name=name, as_type="span", **details))
        except Exception:
            logger.warning("Langfuse: could not start trace %r", name, exc_info=True)
    _current.set(root)
    return root


def step(name: str, *, as_type: str = "span", **details: Any) -> Step:
    """
    Start a step under the request being served.

    Does nothing (and returns a step that does nothing) when tracing is
    off, or when called outside a traced request.
    """
    parent = _current.get()
    if parent is None:
        _clean(details)
        return Step()
    return parent.step(name, as_type=as_type, **details)


def current_trace_id() -> str | None:
    """The trace id of the request being served, if it's being traced."""
    parent = _current.get()
    return parent.trace_id if parent is not None and parent.open else None


def score(trace_id: str | None, name: str, value: bool, comment: str | None = None) -> None:
    """Attach a yes/no score to a finished trace (used for cache feedback)."""
    if _client is None or not trace_id:
        return
    try:
        _client.create_score(
            trace_id=trace_id, name=name, value=1 if value else 0,
            data_type="BOOLEAN", comment=comment,
        )
    except Exception:
        logger.warning("Langfuse: could not record score %r", name, exc_info=True)
