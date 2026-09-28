"""
Gemini provider implementation using google-genai SDK.
"""

import time
import uuid

from google import genai
from google.genai import types

from semcache.providers.base import LLMProvider
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


def _finish_reason(
    response: types.GenerateContentResponse, candidate: types.Candidate | None
) -> str:
    if candidate is None:
        # No candidates at all usually means the prompt itself was blocked.
        blocked = response.prompt_feedback and response.prompt_feedback.block_reason
        return "content_filter" if blocked else "stop"
    reason = candidate.finish_reason
    if reason is None:
        return "stop"
    return _FINISH_REASONS.get(reason, reason.name.lower())


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)

    async def generate(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Translate OpenAI format to Gemini format, call Gemini,
        and translate the response back to OpenAI format.
        """
        # 1. Translate Messages
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

        # 2. Translate Config
        config = types.GenerateContentConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            system_instruction=request.system_prompt,
        )

        # 3. Call Gemini (Async)
        response = await self._client.aio.models.generate_content(
            model=request.model,
            contents=contents,
            config=config,
        )

        # 4. Translate Response back to OpenAI format
        usage = None
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count or 0
            candidates_tokens = response.usage_metadata.candidates_token_count or 0
            # Thinking models report thought tokens separately from candidates, but both are billed as output.
            thoughts_tokens = getattr(response.usage_metadata, "thoughts_token_count", 0) or 0

            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=candidates_tokens + thoughts_tokens,
                total_tokens=response.usage_metadata.total_token_count or 0,
            )

        # A blocked or empty response can have no candidates, or a candidate
        # whose content is None. Join every text part, skipping thought
        # summaries and non-text parts (e.g. function calls).
        text_content = ""
        candidate = response.candidates[0] if response.candidates else None
        if candidate and candidate.content and candidate.content.parts:
            text_content = "".join(
                part.text for part in candidate.content.parts
                if part.text and not part.thought
            )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionChoiceMessage(content=text_content),
                    finish_reason=_finish_reason(response, candidate),
                )
            ],
            usage=usage,
            x_cache_status="MISS",  # Base generation is always a miss
        )
