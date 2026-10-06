"""
Sends each request to the provider that serves its model.

The request's `model` name decides:

    gemini-..., gemma-...   →  Gemini
    anything else           →  the OpenAI-compatible provider
                               (mistral-..., qwen-..., gpt-..., ...)

Used only when an OpenAI-compatible provider is configured (see
dependencies.py). The cache itself doesn't care who answers: the model
name is part of every cache key, so answers from different models are
never mixed up.
"""

from collections.abc import AsyncIterator
from contextlib import suppress

from semcache.providers.base import LLMProvider, StreamChunk
from semcache.schemas import ChatCompletionRequest, ChatCompletionResponse

_GOOGLE_PREFIXES = ("gemini", "gemma", "models/gemini", "models/gemma")


def is_google_model(model: str) -> bool:
    return model.strip().lower().startswith(_GOOGLE_PREFIXES)


class RoutingProvider(LLMProvider):
    def __init__(self, google: LLMProvider, other: LLMProvider) -> None:
        self._google = google
        self._other = other

    def provider_for(self, model: str) -> LLMProvider:
        return self._google if is_google_model(model) else self._other

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return await self.provider_for(request.model).generate(request)

    async def generate_stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamChunk]:
        chunks = self.provider_for(request.model).generate_stream(request)
        try:
            async for chunk in chunks:
                yield chunk
        finally:
            # If the caller stops reading early, stop the provider's stream too.
            close = getattr(chunks, "aclose", None)
            if close is not None:
                with suppress(Exception):
                    await close()
