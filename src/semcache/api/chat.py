"""
The main /v1/chat/completions endpoint.

This is the user-facing API. Every request flows through here:
  1. Check if cacheable (Option A: single-turn only)
  2. Look up in cache (adaptive threshold via per-entry required_similarity)
  3. On HIT → return cached response instantly
  4. On MISS → generate from LLM + classify intent CONCURRENTLY
  5. Store response with classifier's policy (TTL + required_similarity)

The classifier runs concurrently with the main LLM via asyncio.gather,
so it adds ZERO user-facing latency. If the classifier fails for any
reason, we fall back to the engine's default policy — the user never notices.

The cache is an optimization, never a dependency: if Redis or the embedding
API fails during lookup or store, the user still gets the LLM's answer.

Every response carries an X-Cache-Status header (HIT, MISS, or BYPASS), so
clients can see cache behavior without changing how they parse the body.
"""

import asyncio
import logging
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Response

from semcache.api.dependencies import get_classifier, get_engine, get_provider
from semcache.cache.classifier import ClassifierResult, IntentClassifier
from semcache.cache.engine import CacheEngine
from semcache.cache.keys import parse_cache_tags
from semcache.cache.policy import CachePolicy
from semcache.metrics import REQUEST_DURATION, metrics
from semcache.providers.base import LLMProvider
from semcache.schemas import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    UsageInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _set_cache_headers(
    http_response: Response,
    status: str,
    *,
    similarity: float | None = None,
    bypass_reason: str | None = None,
    lookup_id: str | None = None,
) -> None:
    """Report cache behavior in headers, as a drop-in proxy should."""
    http_response.headers["X-Cache-Status"] = status
    if similarity is not None:
        http_response.headers["X-Cache-Similarity"] = f"{similarity:.4f}"
    if bypass_reason is not None:
        http_response.headers["X-Cache-Bypass-Reason"] = bypass_reason
    if lookup_id is not None:
        # Clients send this back to POST /v1/cache/feedback.
        http_response.headers["X-Cache-Lookup-Id"] = lookup_id


def _record_llm_usage(response: ChatCompletionResponse) -> None:
    """Count an LLM call and its tokens, on every path that calls the provider."""
    metrics.llm_calls += 1
    if response.usage:
        metrics.llm_tokens_prompt += response.usage.prompt_tokens
        metrics.llm_tokens_completion += response.usage.completion_tokens


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


async def _generate_uncached(
    request: ChatCompletionRequest, provider: LLMProvider
) -> ChatCompletionResponse:
    """Generate straight from the LLM, with no cache read or write."""
    # Nothing will be stored, so a classifier call would be wasted.
    metrics.classifier_calls_skipped += 1
    response = await provider.generate(request)
    _record_llm_usage(response)
    return response


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    http_response: Response,
    engine: CacheEngine = Depends(get_engine),  # noqa: B008
    provider: LLMProvider = Depends(get_provider),  # noqa: B008
    classifier: IntentClassifier = Depends(get_classifier),  # noqa: B008
    x_cache_tags: Annotated[str | None, Header()] = None,
) -> ChatCompletionResponse:
    """
    OpenAI-compatible chat completions endpoint with semantic caching.

    Optional X-Cache-Tags header ("support, billing:v2") labels the entry
    this request creates, so it can be invalidated as a group later.
    """
    start_time = time.time()
    cache_status = "error" # Default fallback label

    try:
        try:
            tags = parse_cache_tags(x_cache_tags)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Streaming is complex (requires Server-Sent Events).
        # We will tackle that in a future phase if needed, or return a 400 for now.
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail="Streaming is not currently supported by this proxy.",
            )

        # 1. Option A: Should we cache this at all?
        if not request.is_cacheable:
            metrics.cache_bypasses += 1
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

            metrics.cache_hits += 1
            metrics.classifier_calls_skipped += 1  # No classify needed on hit
            _record_cache_saving(entry.model, usage)
            _set_cache_headers(
                http_response, "HIT",
                similarity=lookup_result.similarity, lookup_id=lookup_result.lookup_id,
            )
            cache_status = "hit"
            return ChatCompletionResponse(
                id=f"chatcmpl-cached-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        message=ChatCompletionChoiceMessage(content=entry.response),
                        finish_reason=metadata.get("finish_reason", "stop"),
                    )
                ],
                usage=usage,
                x_cache_status=f"HIT (similarity: {lookup_result.similarity:.4f})",
            )

        # 4. Cache MISS → Generate from LLM + Classify intent CONCURRENTLY
        metrics.cache_misses += 1

        # Run both tasks at the same time. The main generation takes 1-5s.
        # The classifier takes ~200ms. By running them together, the classifier
        # finishes well before the generation, adding zero wall-clock latency.
        #
        # return_exceptions=True ensures that if the classifier fails, it returns
        # the exception object instead of raising — so the generation still completes.
        gen_task = provider.generate(request)
        classify_task = classifier.classify_safe(user_prompt)

        results = await asyncio.gather(gen_task, classify_task, return_exceptions=True)

        # Unpack results — check types for safety
        gen_result = results[0]
        classify_result = results[1]

        # If the generation itself failed, that's a real error — re-raise it.
        if isinstance(gen_result, BaseException):
            raise gen_result

        response: ChatCompletionResponse = gen_result
        _record_llm_usage(response)

        # If the classifier failed (despite classify_safe's try/except),
        # or returned an unexpected type, fall back to the engine's default.
        intent: str | None = None
        if isinstance(classify_result, BaseException):
            logger.warning("Classifier raised in gather: %s", classify_result)
            resolved_policy: CachePolicy = engine.default_policy
            metrics.classifier_calls_fallback += 1
        elif isinstance(classify_result, ClassifierResult):
            resolved_policy = classify_result.policy
            intent = classify_result.intent
            metrics.classifier_tokens_total += classify_result.tokens
            if classify_result.is_fallback:
                metrics.classifier_calls_fallback += 1
            else:
                metrics.classifier_calls_success += 1
        else:
            logger.warning("Classifier returned unexpected type: %s", type(classify_result))
            resolved_policy = engine.default_policy
            metrics.classifier_calls_fallback += 1

        # 5. Store the new response with the classifier's policy
        choice = response.choices[0]
        generated_text = choice.message.content

        if choice.finish_reason != "stop" or not generated_text.strip():
            # Truncated, filtered, or empty generations must not be served to
            # future users. Only answers that finished normally are cached.
            logger.warning(
                "Not caching response (finish_reason=%s, empty=%s) for prompt: %s",
                choice.finish_reason,
                not generated_text.strip(),
                user_prompt[:50],
            )
        elif lookup_result.embedding is not None:
            # Keep what a hit needs to replay the answer faithfully and to
            # credit the exact cost it saves.
            response_metadata: dict = {"finish_reason": choice.finish_reason}
            if response.usage:
                response_metadata["usage"] = response.usage.model_dump()
            try:
                await engine.store(
                    lookup_result=lookup_result,
                    prompt=user_prompt,
                    response=generated_text,
                    model=request.model,
                    response_metadata=response_metadata,
                    policy=resolved_policy,
                    tags=tags,
                    intent=intent,
                )
            except Exception:
                # The user already has their answer; losing one cache write is fine.
                logger.exception("Cache store failed, returning uncached response")
                metrics.cache_store_errors += 1

        _set_cache_headers(http_response, "MISS", lookup_id=lookup_result.lookup_id)
        cache_status = "miss"
        return response

    finally:
        if cache_status != "error":
            duration = time.time() - start_time
            REQUEST_DURATION.labels(cache_status=cache_status, model=request.model).observe(duration)
