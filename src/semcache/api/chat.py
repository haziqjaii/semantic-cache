"""
The main /v1/chat/completions endpoint.

This is the user-facing API. Every request flows through here:
  1. Check if cacheable (Option A: single-turn only)
  2. Look up in cache (adaptive threshold via per-entry required_similarity)
  3. On HIT → return cached response instantly
  4. On MISS → generate from LLM, and classify intent at the same time
  5. Respond with the answer, THEN store it with the classifier's policy
     (TTL + required_similarity)

The user never waits for the classifier: the answer is returned as soon as
the LLM produces it, and waiting for the classifier and storing the answer
happen after the response is sent (FastAPI background tasks). So a slow
classifier model can't slow a request down. If the classifier fails for
any reason, we fall back to the engine's default policy — the user never
notices.

The cache is an optimization, never a dependency: if Redis or the embedding
API fails during lookup or store, the user still gets the LLM's answer.

Every response carries an X-Cache-Status header (HIT, MISS, or BYPASS), so
clients can see cache behavior without changing how they parse the body.

STREAMING ("stream": true)
    The same four outcomes, delivered as Server-Sent Events (see streaming.py):
      - HIT:    the cached answer is sent at once, in streaming format.
      - MISS:   the LLM's answer is passed through as it's generated, while
                we keep a copy. It's cached only if it finished normally;
                if the client disconnects or the answer is cut off, nothing
                is cached.
      - BYPASS: passed through, never cached.
    An LLM failure before the first word is a normal HTTP error (502/503).
    After that the status is already sent, so the stream ends with an
    error event instead.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Response
from fastapi.responses import StreamingResponse

from semcache.api.dependencies import get_classifier, get_engine, get_provider
from semcache.api.streaming import DONE, SSE_HEADERS, SSE_MEDIA_TYPE, ChunkWriter
from semcache.cache.classifier import ClassifierResult, IntentClassifier
from semcache.cache.engine import CacheEngine, LookupResult
from semcache.cache.keys import parse_cache_tags
from semcache.cache.policy import CachePolicy
from semcache.metrics import REQUEST_DURATION, metrics
from semcache.providers.base import LLMProvider, StreamChunk
from semcache.schemas import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    UsageInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Headers ─────────────────────────────────────────────────

def _cache_headers(
    status: str,
    *,
    similarity: float | None = None,
    bypass_reason: str | None = None,
    lookup_id: str | None = None,
) -> dict[str, str]:
    """Cache behavior as response headers, as a drop-in proxy should report it."""
    headers = {"X-Cache-Status": status}
    if similarity is not None:
        headers["X-Cache-Similarity"] = f"{similarity:.4f}"
    if bypass_reason is not None:
        headers["X-Cache-Bypass-Reason"] = bypass_reason
    if lookup_id is not None:
        # Clients send this back to POST /v1/cache/feedback.
        headers["X-Cache-Lookup-Id"] = lookup_id
    return headers


def _set_cache_headers(http_response: Response, status: str, **details) -> None:
    http_response.headers.update(_cache_headers(status, **details))


# ── Metrics ─────────────────────────────────────────────────

def _record_llm_usage(usage: UsageInfo | None) -> None:
    """Count an LLM call and its tokens, on every path that calls the provider."""
    metrics.llm_calls += 1
    if usage:
        metrics.llm_tokens_prompt += usage.prompt_tokens
        metrics.llm_tokens_completion += usage.completion_tokens


def _record_cache_saving(model: str, usage: UsageInfo | None) -> None:
    """Credit a cache hit with the tokens and cost of the LLM call it replaced."""
    if usage:
        metrics.tokens_saved += usage.prompt_tokens + usage.completion_tokens
    saved = metrics.price(model, usage.prompt_tokens, usage.completion_tokens) if usage else None
    if saved is None:
        # Entry cached without token counts, or its model has no pricing.
        metrics.cache_hits_unpriced += 1
    else:
        metrics.cost_saved_myr += saved


# ── LLM errors ──────────────────────────────────────────────

def _provider_http_error(exc: BaseException) -> HTTPException:
    """
    Turn a failed LLM call into an HTTP error the client can act on.

    Overload and rate-limit errors keep their status (503 / 429), so
    clients know to retry; anything else is a 502 (bad gateway). Without
    this, every provider failure would surface as a generic 500.
    """
    if isinstance(exc, HTTPException):
        return exc
    logger.error("LLM provider call failed", exc_info=exc)
    upstream_status = getattr(exc, "code", None)  # google-genai API errors carry the HTTP code
    status = upstream_status if upstream_status in (429, 503) else 502
    return HTTPException(status_code=status, detail=f"LLM provider error: {exc}")


# ── Shared steps of a cache miss ────────────────────────────

def _resolve_policy(
    classify_result: object, engine: CacheEngine
) -> tuple[CachePolicy, str | None]:
    """
    The cache policy and intent for a new entry, from the classifier's result.

    If the classifier failed (despite classify_safe's try/except), or
    returned an unexpected type, fall back to the engine's default.
    """
    if isinstance(classify_result, ClassifierResult):
        metrics.classifier_tokens_total += classify_result.tokens
        if classify_result.is_fallback:
            metrics.classifier_calls_fallback += 1
        else:
            metrics.classifier_calls_success += 1
        return classify_result.policy, classify_result.intent

    if isinstance(classify_result, BaseException):
        logger.warning("Classifier raised: %s", classify_result)
    else:
        logger.warning("Classifier returned unexpected type: %s", type(classify_result))
    metrics.classifier_calls_fallback += 1
    return engine.default_policy, None


async def _store_answer(
    engine: CacheEngine,
    lookup_result: LookupResult,
    *,
    prompt: str,
    model: str,
    text: str,
    finish_reason: str,
    usage: UsageInfo | None,
    policy: CachePolicy,
    intent: str | None,
    tags: list[str],
) -> None:
    """Cache a freshly generated answer, if it's fit to be served again."""
    if finish_reason != "stop" or not text.strip():
        # Truncated, filtered, or empty generations must not be served to
        # future users. Only answers that finished normally are cached.
        logger.warning(
            "Not caching response (finish_reason=%s, empty=%s) for prompt: %s",
            finish_reason,
            not text.strip(),
            prompt[:50],
        )
        return
    if lookup_result.embedding is None:
        return

    # Keep what a hit needs to replay the answer faithfully and to
    # credit the exact cost it saves.
    response_metadata: dict = {"finish_reason": finish_reason}
    if usage:
        response_metadata["usage"] = usage.model_dump()
    try:
        await engine.store(
            lookup_result=lookup_result,
            prompt=prompt,
            response=text,
            model=model,
            response_metadata=response_metadata,
            policy=policy,
            tags=tags,
            intent=intent,
        )
    except Exception:
        # The user already has their answer; losing one cache write is fine.
        logger.exception("Cache store failed, returning uncached response")
        metrics.cache_store_errors += 1


async def _generate_uncached(
    request: ChatCompletionRequest, provider: LLMProvider
) -> ChatCompletionResponse:
    """Generate straight from the LLM, with no cache read or write."""
    # Nothing will be stored, so a classifier call would be wasted.
    metrics.classifier_calls_skipped += 1
    try:
        response = await provider.generate(request)
    except Exception as exc:
        raise _provider_http_error(exc) from exc
    _record_llm_usage(response.usage)
    return response


# ── Streaming ───────────────────────────────────────────────

async def _stream_cached(
    writer: ChunkWriter,
    text: str,
    finish_reason: str,
    usage: UsageInfo | None,
    include_usage: bool,
    on_done: Callable[[], None],
) -> AsyncIterator[str]:
    """A cache hit in streaming format: the whole answer at once."""
    yield writer.role()
    yield writer.content(text)
    yield writer.finish(finish_reason)
    if include_usage and usage:
        yield writer.usage(usage)
    yield DONE
    on_done()


async def _stream_generation(
    writer: ChunkWriter,
    first: StreamChunk | None,
    chunks: AsyncIterator[StreamChunk],
    include_usage: bool,
    on_done: Callable[[], None],
    classify_task: asyncio.Task | None = None,
    on_complete: Callable[[str, str, UsageInfo | None], object] | None = None,
) -> AsyncIterator[str]:
    """
    Pass the LLM's answer to the client as it arrives, keeping a copy.

    `first` is the chunk already read from `chunks` (the caller reads it
    before responding, so early failures can be proper HTTP errors).

    When the answer is complete, `on_complete(text, finish_reason, usage)`
    runs (it arranges for the answer to be stored once the stream ends). If
    the client disconnects, or the stream breaks or is cut short, it never
    runs, so partial answers are never cached.
    """
    parts: list[str] = []
    finish_reason: str | None = None
    usage: UsageInfo | None = None
    completed = False
    try:
        yield writer.role()
        chunk = first
        while chunk is not None:
            if chunk.text:
                parts.append(chunk.text)
                yield writer.content(chunk.text)
            usage = chunk.usage or usage
            finish_reason = chunk.finish_reason or finish_reason
            chunk = await anext(chunks, None)

        if finish_reason is None:
            raise RuntimeError("The model's response ended unexpectedly.")

        completed = True
        if on_complete is not None:
            await on_complete("".join(parts), finish_reason, usage)

        yield writer.finish(finish_reason)
        if include_usage and usage:
            yield writer.usage(usage)
        yield DONE
        on_done()
    except Exception as exc:  # noqa: BLE001 - logged by _provider_http_error
        # The 200 status is already sent, so report the failure in-band.
        yield writer.error(_provider_http_error(exc).detail)
    finally:
        # Runs on normal completion, on errors, and when the client
        # disconnects (the generator is cancelled mid-stream).
        _record_llm_usage(usage)
        if classify_task is not None and not completed:
            classify_task.cancel()
        close = getattr(chunks, "aclose", None)
        if close is not None:
            with suppress(Exception):
                await close()  # stop reading from the provider


async def _start_stream(
    provider: LLMProvider, request: ChatCompletionRequest
) -> tuple[AsyncIterator[StreamChunk], StreamChunk | None]:
    """
    Open the provider stream and read its first chunk.

    Reading one chunk before responding means a provider that fails
    straight away (overloaded, bad model name) still produces a proper
    HTTP error instead of a broken stream.
    """
    chunks = provider.generate_stream(request)
    try:
        return chunks, await anext(chunks, None)
    except Exception as exc:
        raise _provider_http_error(exc) from exc


def _sse_response(events: AsyncIterator[str], cache_headers: dict[str, str]) -> StreamingResponse:
    return StreamingResponse(
        events, media_type=SSE_MEDIA_TYPE, headers={**SSE_HEADERS, **cache_headers}
    )


# ── The endpoint ────────────────────────────────────────────

@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    http_response: Response,
    engine: CacheEngine = Depends(get_engine),  # noqa: B008
    provider: LLMProvider = Depends(get_provider),  # noqa: B008
    classifier: IntentClassifier = Depends(get_classifier),  # noqa: B008
    x_cache_tags: Annotated[str | None, Header()] = None,
    # FastAPI always supplies this. Direct calls (tests, scripts) may leave it
    # out, and then the post-response work runs before returning instead.
    background_tasks: BackgroundTasks = None,  # type: ignore[assignment]
) -> ChatCompletionResponse | StreamingResponse:
    """
    OpenAI-compatible chat completions endpoint with semantic caching.

    Optional X-Cache-Tags header ("support, billing:v2") labels the entry
    this request creates, so it can be invalidated as a group later.

    With "stream": true the answer is sent as Server-Sent Events; add
    "stream_options": {"include_usage": true} to get token counts too.
    """
    start_time = time.time()
    # The latency label. "error" (never observed) until an outcome is known;
    # streamed responses observe theirs when the stream ends instead.
    cache_status = "error"
    streaming = bool(request.stream)
    include_usage = bool((request.stream_options or {}).get("include_usage"))

    def observe(status: str) -> None:
        REQUEST_DURATION.labels(cache_status=status, model=request.model).observe(
            time.time() - start_time
        )

    async def after_response(job: Callable[[], Awaitable[None]]) -> None:
        """Run `job` once the response has been sent, so the client doesn't wait for it."""
        if background_tasks is not None:
            background_tasks.add_task(job)
        else:
            await job()

    def classify_and_store(
        classify_task: asyncio.Task, text: str, finish_reason: str, usage: UsageInfo | None
    ) -> Callable[[], Awaitable[None]]:
        """The post-response job for a miss: wait for the classifier, then cache the answer."""

        async def job() -> None:
            classify_result = (await asyncio.gather(classify_task, return_exceptions=True))[0]
            policy, intent = _resolve_policy(classify_result, engine)
            await _store_answer(
                engine, lookup_result, prompt=user_prompt, model=request.model, text=text,
                finish_reason=finish_reason, usage=usage, policy=policy, intent=intent, tags=tags,
            )

        return job

    async def stream_uncached(status: str, bypass_reason: str) -> StreamingResponse:
        """Stream straight from the LLM, with no cache read or write."""
        metrics.classifier_calls_skipped += 1
        chunks, first = await _start_stream(provider, request)
        events = _stream_generation(
            ChunkWriter(request.model), first, chunks, include_usage,
            on_done=lambda: observe(status),
        )
        return _sse_response(events, _cache_headers("BYPASS", bypass_reason=bypass_reason))

    try:
        try:
            tags = parse_cache_tags(x_cache_tags)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # 1. Option A: Should we cache this at all?
        if not request.is_cacheable:
            metrics.cache_bypasses += 1
            if streaming:
                return await stream_uncached("bypass", "uncacheable")
            response = await _generate_uncached(request, provider)
            response.x_cache_status = "BYPASS (Uncacheable)"
            _set_cache_headers(http_response, "BYPASS", bypass_reason="uncacheable")
            cache_status = "bypass"
            return response

        # Extract the user prompt and system prompt for the cache engine
        system_prompt = request.system_prompt
        user_prompt = next(
            (m.content for m in request.messages if m.role == "user"), ""
        )

        if not user_prompt:
            raise HTTPException(status_code=400, detail="No user message provided.")

        # 2. Check the Cache (uses FLOOR_THRESHOLD + per-entry required_similarity)
        try:
            lookup_result = await engine.lookup(
                prompt=user_prompt,
                model=request.model,
                system_prompt=system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        except Exception:
            # Embedding API or Redis is down. Serve the request without the cache.
            logger.exception("Cache lookup failed, serving uncached response")
            metrics.cache_lookup_errors += 1
            if streaming:
                return await stream_uncached("cache_error", "cache-error")
            response = await _generate_uncached(request, provider)
            response.x_cache_status = "BYPASS (Cache error)"
            _set_cache_headers(http_response, "BYPASS", bypass_reason="cache-error")
            cache_status = "cache_error"
            return response

        # 3. Cache HIT — replay the stored answer, including its original
        # token usage and finish reason.
        if lookup_result.hit and lookup_result.entry:
            entry = lookup_result.entry
            metadata = entry.response_metadata or {}
            usage = UsageInfo(**metadata["usage"]) if metadata.get("usage") else None
            finish_reason = metadata.get("finish_reason", "stop")
            hit_headers = _cache_headers(
                "HIT", similarity=lookup_result.similarity, lookup_id=lookup_result.lookup_id
            )

            metrics.cache_hits += 1
            metrics.classifier_calls_skipped += 1  # No classify needed on hit
            _record_cache_saving(entry.model, usage)

            if streaming:
                events = _stream_cached(
                    ChunkWriter(request.model, cached=True), entry.response, finish_reason,
                    usage, include_usage, on_done=lambda: observe("hit"),
                )
                return _sse_response(events, hit_headers)

            http_response.headers.update(hit_headers)
            cache_status = "hit"
            return ChatCompletionResponse(
                id=f"chatcmpl-cached-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        message=ChatCompletionChoiceMessage(content=entry.response),
                        finish_reason=finish_reason,
                    )
                ],
                usage=usage,
                x_cache_status=f"HIT (similarity: {lookup_result.similarity:.4f})",
            )

        # 4. Cache MISS → generate from the LLM, classifying the question
        # at the same time. Only the answer is awaited here: the classifier's
        # result is needed only to store the answer, after the response.
        metrics.cache_misses += 1
        miss_headers = _cache_headers("MISS", lookup_id=lookup_result.lookup_id)
        classify_task = asyncio.create_task(classifier.classify_safe(user_prompt))

        if streaming:
            try:
                chunks, first = await _start_stream(provider, request)
            except BaseException:
                classify_task.cancel()
                raise

            async def store_when_complete(text: str, finish_reason: str, usage: UsageInfo | None) -> None:
                await after_response(classify_and_store(classify_task, text, finish_reason, usage))

            events = _stream_generation(
                ChunkWriter(request.model), first, chunks, include_usage,
                on_done=lambda: observe("miss"),
                classify_task=classify_task, on_complete=store_when_complete,
            )
            return _sse_response(events, miss_headers)

        try:
            response = await provider.generate(request)
        except BaseException as exc:
            classify_task.cancel()  # no answer, so nothing to classify it for
            if isinstance(exc, Exception):
                raise _provider_http_error(exc) from exc
            raise  # e.g. the request was cancelled
        _record_llm_usage(response.usage)

        # 5. Store the new response with the classifier's policy, once the
        # response is on its way.
        choice = response.choices[0]
        await after_response(
            classify_and_store(classify_task, choice.message.content, choice.finish_reason, response.usage)
        )

        http_response.headers.update(miss_headers)
        cache_status = "miss"
        return response

    finally:
        if cache_status != "error":
            observe(cache_status)
