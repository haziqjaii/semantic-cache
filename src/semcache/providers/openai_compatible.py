"""
A provider for any service that speaks OpenAI's chat completions API.

Many model hosts copy OpenAI's API: the same URL shape
(`<base_url>/chat/completions`), the same JSON, the same streaming format.
One class therefore covers OpenAI itself and hosts that serve open models
(Mistral, Qwen, gpt-oss, ...) behind one address and one API key. Which
model answers is chosen per request, by the request's `model` name.

Our own API is OpenAI-shaped too (see schemas.py), so almost nothing needs
translating: the request's messages go out as they came in.

Thinking models may send their reasoning in a separate `reasoning_content`
field. Only `content`, the answer itself, is returned and cached. Their
reasoning tokens are still counted, because the host bills them as output.
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

import httpx

from semcache.providers.base import LLMProvider, StreamChunk
from semcache.schemas import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    UsageInfo,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"

# How long the provider's model list is reused before asking again.
MODEL_LIST_TTL_SECONDS = 300.0

# A provider's model list also names models that can't chat. Names
# containing one of these are left out of list_models().
_NOT_CHAT_MODELS = ("embed", "e5-", "bge-", "rerank", "whisper", "tts")


class ProviderHTTPError(Exception):
    """
    The provider answered with an error status.

    `code` is that HTTP status. chat.py reads it to pass "overloaded" (503)
    and "rate limited" (429) on to the client, so they know to retry.
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _error_message(body: str) -> str:
    """The provider's own explanation, from an error response body."""
    try:
        error = json.loads(body).get("error")
    except (ValueError, AttributeError):
        error = None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    if isinstance(error, str) and error:
        return error
    return body.strip()[:300] or "no details given"


def _usage(usage: dict | None) -> UsageInfo | None:
    if not usage:
        return None
    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=usage.get("total_tokens") or prompt_tokens + completion_tokens,
    )


class OpenAICompatibleProvider(LLMProvider):
    """
    Args:
        api_key: Sent as "Authorization: Bearer <key>".
        base_url: The API's address up to and including the version, e.g.
            "https://api.openai.com/v1". "/chat/completions" is added to it.
        timeout_seconds: How long to wait for the model (per read, when streaming).
        models: The models to offer in list_models(). Normally None: the
            list is then read from the provider (GET <base_url>/models).
        client: A ready-made httpx client (tests pass one with a fake transport).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 120.0,
        models: list[str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._fixed_models = models
        self._listed_models: list[str] = []
        self._listed_at: float | None = None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[str]:
        """
        The provider's chat models, asked for at most every few minutes.

        If the provider can't be reached, the last list it gave is returned
        (empty if there never was one): a missing list must not break anything.
        """
        if self._fixed_models is not None:
            return list(self._fixed_models)
        now = time.monotonic()
        if self._listed_at is not None and now - self._listed_at < MODEL_LIST_TTL_SECONDS:
            return list(self._listed_models)
        try:
            response = await self._client.get("models", timeout=10.0)
            response.raise_for_status()
            ids = [m["id"] for m in response.json().get("data") or [] if isinstance(m.get("id"), str)]
        except Exception as exc:  # noqa: BLE001 - any failure means "no list right now"
            logger.warning("Could not list the provider's models: %s", exc)
            return list(self._listed_models)
        self._listed_models = [
            model for model in ids if not any(word in model.lower() for word in _NOT_CHAT_MODELS)
        ]
        self._listed_at = now
        return list(self._listed_models)

    def _payload(self, request: ChatCompletionRequest, *, stream: bool) -> dict:
        payload: dict = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if stream:
            payload["stream"] = True
            # Ask for token counts at the end of the stream (they're how
            # savings are measured); without this a stream reports none.
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        response = await self._client.post("chat/completions", json=self._payload(request, stream=False))
        if response.status_code >= 400:
            raise ProviderHTTPError(response.status_code, _error_message(response.text))

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ProviderHTTPError(502, "The provider's response had no choices.")
        choice = choices[0]
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionChoiceMessage(content=(choice.get("message") or {}).get("content") or ""),
                    finish_reason=choice.get("finish_reason") or "stop",
                )
            ],
            usage=_usage(data.get("usage")),
            x_cache_status="MISS",  # Base generation is always a miss
        )

    async def generate_stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamChunk]:
        """
        Stream the answer as it's generated.

        Yields text as it arrives, then one final chunk with the finish
        reason and token usage. If the stream ends without a finish reason,
        no final chunk is sent, which tells the caller the answer was cut short.
        """
        finish_reason: str | None = None
        usage: UsageInfo | None = None
        async with self._client.stream(
            "POST", "chat/completions", json=self._payload(request, stream=True)
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode(errors="replace")
                raise ProviderHTTPError(response.status_code, _error_message(body))

            # Server-Sent Events: lines of `data: {json}`, ending with `data: [DONE]`.
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue  # blank separators, comments, keep-alives
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise ProviderHTTPError(502, _error_message(data))

                # Usage arrives in a last event whose `choices` is empty.
                usage = _usage(event.get("usage")) or usage
                for choice in event.get("choices") or []:
                    text = (choice.get("delta") or {}).get("content")
                    if text:
                        yield StreamChunk(text=text)
                    finish_reason = choice.get("finish_reason") or finish_reason

        if finish_reason is not None:
            yield StreamChunk(finish_reason=finish_reason, usage=usage)
