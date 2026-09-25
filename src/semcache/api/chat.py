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
reason, we fall back to DEFAULT_POLICY — the user never notices.
"""

import asyncio
import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException

from semcache.api.dependencies import get_classifier, get_engine, get_provider
from semcache.cache.classifier import IntentClassifier
from semcache.cache.engine import CacheEngine
from semcache.cache.policy import DEFAULT_POLICY, CachePolicy
from semcache.metrics import metrics
from semcache.providers.base import LLMProvider
from semcache.schemas import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    engine: CacheEngine = Depends(get_engine),  # noqa: B008
    provider: LLMProvider = Depends(get_provider),  # noqa: B008
    classifier: IntentClassifier = Depends(get_classifier),  # noqa: B008
) -> ChatCompletionResponse:
    """
    OpenAI-compatible chat completions endpoint with semantic caching.
    """
    # Streaming is complex (requires Server-Sent Events).
    # We will tackle that in a future phase if needed, or return a 400 for now.
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming is not currently supported by this proxy.",
        )

    # 1. Option A: Should we cache this at all?
    if not request.is_cacheable:
        # Don't waste a classifier call on uncacheable requests.
        metrics.cache_bypasses += 1
        metrics.classifier_calls_skipped += 1
        response = await provider.generate(request)
        response.x_cache_status = "BYPASS (Uncacheable)"
        return response

    # Extract the user prompt and system prompt for the cache engine
    system_prompt = next(
        (m.content for m in request.messages if m.role == "system"), None
    )
    user_prompt = next(
        (m.content for m in request.messages if m.role == "user"), ""
    )

    if not user_prompt:
        raise HTTPException(status_code=400, detail="No user message provided.")

    # 2. Check the Cache (uses FLOOR_THRESHOLD + per-entry required_similarity)
    lookup_result = await engine.lookup(
        prompt=user_prompt,
        model=request.model,
        system_prompt=system_prompt,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    # 3. Cache HIT
    if lookup_result.hit and lookup_result.entry:
        metrics.cache_hits += 1
        metrics.classifier_calls_skipped += 1  # No classify needed on hit
        return ChatCompletionResponse(
            id=f"chatcmpl-cached-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionChoiceMessage(
                        content=lookup_result.entry.response
                    ),
                    finish_reason="stop",
                )
            ],
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

    # If the classifier failed (despite classify_safe's try/except),
    # or returned an unexpected type, fall back to DEFAULT_POLICY.
    if isinstance(classify_result, BaseException):
        logger.warning("Classifier raised in gather: %s", classify_result)
        resolved_policy: CachePolicy = DEFAULT_POLICY
        metrics.classifier_calls_fallback += 1
    elif isinstance(classify_result, CachePolicy):
        resolved_policy = classify_result
        metrics.classifier_calls_success += 1
    else:
        logger.warning("Classifier returned unexpected type: %s", type(classify_result))
        resolved_policy = DEFAULT_POLICY
        metrics.classifier_calls_fallback += 1

    # 5. Store the new response with the classifier's policy
    generated_text = response.choices[0].message.content

    if lookup_result.embedding is not None:
        await engine.store(
            lookup_result=lookup_result,
            prompt=user_prompt,
            response=generated_text,
            model=request.model,
            policy=resolved_policy,
        )

    # Track LLM usage
    metrics.llm_calls += 1
    if response.usage:
        metrics.llm_tokens_prompt += response.usage.prompt_tokens
        metrics.llm_tokens_completion += response.usage.completion_tokens

    return response
