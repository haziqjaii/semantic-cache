"""
The main /v1/chat/completions endpoint.
"""

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException

from semcache.api.dependencies import get_engine, get_provider
from semcache.cache.engine import CacheEngine
from semcache.providers.base import LLMProvider
from semcache.schemas import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
)

router = APIRouter()


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    engine: CacheEngine = Depends(get_engine),  # noqa: B008
    provider: LLMProvider = Depends(get_provider),  # noqa: B008
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
        # Pass directly to provider, skip cache engine entirely
        response = await provider.generate(request)
        response.x_cache_status = "BYPASS (Uncacheable)"
        return response

    # Extract the user prompt and system prompt for the cache engine
    system_prompt = next((m.content for m in request.messages if m.role == "system"), None)
    user_prompt = next((m.content for m in request.messages if m.role == "user"), "")

    if not user_prompt:
        raise HTTPException(status_code=400, detail="No user message provided.")

    # 2. Check the Cache
    lookup_result = await engine.lookup(
        prompt=user_prompt,
        model=request.model,
        system_prompt=system_prompt,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    # 3. Cache HIT
    if lookup_result.hit and lookup_result.entry:
        return ChatCompletionResponse(
            id=f"chatcmpl-cached-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionChoiceMessage(content=lookup_result.entry.response),
                    finish_reason="stop",
                )
            ],
            # If we stored usage metadata in the cache entry, we could return it here.
            # For MVP, we leave it None.
            x_cache_status=f"HIT (similarity: {lookup_result.similarity:.4f})",
        )

    # 4. Cache MISS -> Generate from LLM
    response = await provider.generate(request)

    # 5. Store the new response asynchronously (or await it, it's fast enough)
    # We grab the text content from the response we just generated.
    generated_text = response.choices[0].message.content
    
    # We only store if it's not a NO_CACHE policy miss
    if lookup_result.embedding is not None:
        await engine.store(
            lookup_result=lookup_result,
            prompt=user_prompt,
            response=generated_text,
            model=request.model,
        )

    return response
