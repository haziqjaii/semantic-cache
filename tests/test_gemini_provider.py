"""
Tests for GeminiProvider's response translation.

The Gemini client is mocked, so these run offline. They cover the shapes a
real response can take when generation is blocked, empty, or multi-part.
"""

from unittest.mock import AsyncMock

import pytest
from google.genai import types

from semcache.providers.gemini import GeminiProvider
from semcache.schemas import ChatCompletionRequest, ChatMessage


def _provider_returning(response: types.GenerateContentResponse) -> GeminiProvider:
    provider = GeminiProvider(api_key="fake-key")
    provider._client.aio.models.generate_content = AsyncMock(return_value=response)
    return provider


async def _generate_text(response: types.GenerateContentResponse) -> str:
    request = ChatCompletionRequest(
        model="gemini-3.5-flash",
        messages=[ChatMessage(role="user", content="Hello")],
    )
    result = await _provider_returning(response).generate(request)
    return result.choices[0].message.content


@pytest.mark.asyncio
async def test_no_candidates_returns_empty_text():
    """A prompt blocked outright has no candidates at all."""
    assert await _generate_text(types.GenerateContentResponse(candidates=[])) == ""


@pytest.mark.asyncio
async def test_candidate_without_content_returns_empty_text():
    """A safety-blocked candidate has content=None — this used to crash."""
    response = types.GenerateContentResponse(
        candidates=[types.Candidate(content=None, finish_reason="SAFETY")]
    )
    assert await _generate_text(response) == ""


@pytest.mark.asyncio
async def test_joins_text_parts_and_skips_thoughts():
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(text="Let me think...", thought=True),
                        types.Part(text="Python is "),
                        types.Part(text="a language."),
                    ],
                )
            )
        ]
    )
    assert await _generate_text(response) == "Python is a language."
