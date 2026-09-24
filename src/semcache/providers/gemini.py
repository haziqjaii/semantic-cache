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
        contents = []
        system_instruction = None

        for msg in request.messages:
            if msg.role == "system":
                # Gemini handles system instructions in the config, not in the content array
                system_instruction = msg.content
            elif msg.role in ("user", "assistant"):
                # Gemini roles are "user" and "model"
                gemini_role = "user" if msg.role == "user" else "model"
                contents.append(
                    types.Content(role=gemini_role, parts=[types.Part.from_text(text=msg.content)])
                )

        # 2. Translate Config
        config = types.GenerateContentConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            system_instruction=system_instruction,
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
            usage = UsageInfo(
                prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                completion_tokens=response.usage_metadata.candidates_token_count or 0,
                total_tokens=response.usage_metadata.total_token_count or 0,
            )

        text_content = ""
        if response.candidates and response.candidates[0].content.parts:
            text_content = response.candidates[0].content.parts[0].text

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionChoiceMessage(content=text_content),
                    finish_reason="stop",
                )
            ],
            usage=usage,
            x_cache_status="MISS",  # Base generation is always a miss
        )
