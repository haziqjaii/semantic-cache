"""
Gemini provider implementation using google-genai SDK.
"""

import time
import uuid
from collections.abc import AsyncIterator

from google import genai
from google.genai import types

from semcache.config import PRICING_TABLE
from semcache.providers.base import LLMProvider, StreamChunk
from semcache.schemas import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    UsageInfo,
)

# Gemini finish reasons → OpenAI's. Reasons with no OpenAI equivalent
# (e.g. OTHER, LANGUAGE) pass through lowercased — never as "stop", since
# chat.py only caches responses that finished normally.
_FINISH_REASONS: dict[types.FinishReason, str] = {
    types.FinishReason.STOP: "stop",
    types.FinishReason.FINISH_REASON_UNSPECIFIED: "stop",
    types.FinishReason.MAX_TOKENS: "length",
    types.FinishReason.SAFETY: "content_filter",
    types.FinishReason.RECITATION: "content_filter",
    types.FinishReason.BLOCKLIST: "content_filter",
    types.FinishReason.PROHIBITED_CONTENT: "content_filter",
    types.FinishReason.SPII: "content_filter",
    types.FinishReason.IMAGE_SAFETY: "content_filter",
    types.FinishReason.IMAGE_PROHIBITED_CONTENT: "content_filter",
    types.FinishReason.IMAGE_RECITATION: "content_filter",
}


def _map_finish_reason(reason: types.FinishReason) -> str:
    return _FINISH_REASONS.get(reason, reason.name.lower())


def _finish_reason(
    response: types.GenerateContentResponse, candidate: types.Candidate | None
) -> str:
    """The finish reason of a complete (non-streamed) response."""
    if candidate is None:
        # No candidates at all usually means the prompt itself was blocked.
        blocked = response.prompt_feedback and response.prompt_feedback.block_reason
        return "content_filter" if blocked else "stop"
    if candidate.finish_reason is None:
        return "stop"
    return _map_finish_reason(candidate.finish_reason)


def _text(candidate: types.Candidate | None) -> str:
    """
    The answer text in a candidate ("" if there is none).

    A blocked or empty response can have no candidates, or a candidate
    whose content is None. Join every text part, skipping thought
    summaries and non-text parts (e.g. function calls).
    """
    if candidate and candidate.content and candidate.content.parts:
        return "".join(
            part.text for part in candidate.content.parts
            if part.text and not part.thought
        )
    return ""


def _usage(response: types.GenerateContentResponse) -> UsageInfo | None:
    if not response.usage_metadata:
        return None
    prompt_tokens = response.usage_metadata.prompt_token_count or 0
    candidates_tokens = response.usage_metadata.candidates_token_count or 0
    # Thinking models report thought tokens separately from candidates, but both are billed as output.
    thoughts_tokens = getattr(response.usage_metadata, "thoughts_token_count", 0) or 0
    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=candidates_tokens + thoughts_tokens,
        total_tokens=response.usage_metadata.total_token_count or 0,
    )


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)

    async def list_models(self) -> list[str]:
        # The Gemini chat models we have prices for. Others still work when
        # requested by name; give one a price to list it here.
        return [
            model for model in PRICING_TABLE["models"]
            if model.startswith(("gemini", "gemma")) and "embedding" not in model
        ]

    def _translate(
        self, request: ChatCompletionRequest
    ) -> tuple[list[types.Content], types.GenerateContentConfig]:
        """Translate an OpenAI-format request into Gemini contents and config."""
        # Gemini handles system instructions in the config, not in the content
        # array. Other roles map "user" → "user" and "assistant" → "model".
        contents = [
            types.Content(
                role="user" if msg.role == "user" else "model",
                parts=[types.Part.from_text(text=msg.content)],
            )
            for msg in request.messages
            if msg.role in ("user", "assistant")
        ]
        config = types.GenerateContentConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            system_instruction=request.system_prompt,
        )
        return contents, config

    async def generate(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Translate OpenAI format to Gemini format, call Gemini,
        and translate the response back to OpenAI format.
        """
        contents, config = self._translate(request)

        response = await self._client.aio.models.generate_content(
            model=request.model,
            contents=contents,
            config=config,
        )

        candidate = response.candidates[0] if response.candidates else None

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionChoiceMessage(content=_text(candidate)),
                    finish_reason=_finish_reason(response, candidate),
                )
            ],
            usage=_usage(response),
            x_cache_status="MISS",  # Base generation is always a miss
        )

    async def generate_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream the answer as it's generated.

        Yields text as Gemini produces it, then one final chunk with the
        finish reason and token usage. If Gemini's stream ends without a
        finish reason, no final chunk is sent, which tells the caller the
        answer was cut short.
        """
        contents, config = self._translate(request)

        stream = await self._client.aio.models.generate_content_stream(
            model=request.model,
            contents=contents,
            config=config,
        )

        finish_reason: str | None = None
        usage: UsageInfo | None = None
        async for response in stream:
            candidate = response.candidates[0] if response.candidates else None
            # Usage arrives on the last chunk (and is cumulative if repeated).
            usage = _usage(response) or usage

            if candidate is None:
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    finish_reason = "content_filter"  # the prompt itself was blocked
                continue

            text = _text(candidate)
            if text:
                yield StreamChunk(text=text)
            if candidate.finish_reason is not None:
                finish_reason = _map_finish_reason(candidate.finish_reason)

        if finish_reason is not None:
            yield StreamChunk(finish_reason=finish_reason, usage=usage)
