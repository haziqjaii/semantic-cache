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


async def _generate(response: types.GenerateContentResponse, request: ChatCompletionRequest | None = None):
    request = request or ChatCompletionRequest(
        model="gemini-3.5-flash",
        messages=[ChatMessage(role="user", content="Hello")],
    )
    provider = _provider_returning(response)
    result = await provider.generate(request)
    return result, provider._client.aio.models.generate_content


def _finished(reason: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text="Hi")]),
                finish_reason=reason,
            )
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gemini_reason", "openai_reason"),
    [
        ("STOP", "stop"),
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
        ("RECITATION", "content_filter"),
        # No OpenAI equivalent: passed through, and never reported as "stop".
        ("OTHER", "other"),
    ],
)
async def test_finish_reason_is_mapped(gemini_reason, openai_reason):
    result, _ = await _generate(_finished(gemini_reason))
    assert result.choices[0].finish_reason == openai_reason


@pytest.mark.asyncio
async def test_blocked_prompt_reports_content_filter():
    response = types.GenerateContentResponse(
        candidates=[],
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason="SAFETY"),
    )
    result, _ = await _generate(response)
    assert result.choices[0].finish_reason == "content_filter"


@pytest.mark.asyncio
async def test_all_system_messages_become_the_system_instruction():
    request = ChatCompletionRequest(
        model="gemini-3.5-flash",
        messages=[
            ChatMessage(role="system", content="You are a teacher."),
            ChatMessage(role="system", content="Answer in French."),
            ChatMessage(role="user", content="What is Python?"),
        ],
    )

    _, generate_content = await _generate(_finished("STOP"), request)

    kwargs = generate_content.call_args.kwargs
    assert kwargs["config"].system_instruction == "You are a teacher.\n\nAnswer in French."
    assert [c.role for c in kwargs["contents"]] == ["user"]
