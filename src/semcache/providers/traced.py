"""
A provider wrapper that records each LLM call as a step in the request's trace.

Used only when Langfuse tracing is on (see tracing.py and dependencies.py).
It changes nothing about the call itself: the same request goes to the
wrapped provider, and the same response or stream comes back.
"""

from collections.abc import AsyncIterator
from contextlib import suppress
from datetime import UTC, datetime

from semcache import tracing
from semcache.providers.base import LLMProvider, StreamChunk
from semcache.schemas import ChatCompletionRequest, ChatCompletionResponse, UsageInfo

STEP_NAME = "llm-generation"


def _start(request: ChatCompletionRequest) -> tracing.Step:
    parameters = {"temperature": request.temperature, "max_tokens": request.max_tokens}
    return tracing.step(
        STEP_NAME,
        as_type="generation",
        model=request.model,
        input=[{"role": m.role, "content": m.content} for m in request.messages],
        model_parameters={k: v for k, v in parameters.items() if v is not None},
    )


def _usage_details(usage: UsageInfo | None) -> dict[str, int] | None:
    if usage is None:
        return None
    return {"input": usage.prompt_tokens, "output": usage.completion_tokens}


class TracedProvider(LLMProvider):
    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        with _start(request) as step:
            response = await self._inner.generate(request)
            choice = response.choices[0]
            step.update(
                output=choice.message.content,
                usage_details=_usage_details(response.usage),
                metadata={"finish_reason": choice.finish_reason},
            )
            return response

    async def generate_stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamChunk]:
        # Runs when the first chunk is requested, like the provider itself.
        step = _start(request)
        chunks = self._inner.generate_stream(request)
        parts: list[str] = []
        usage: UsageInfo | None = None
        finish_reason: str | None = None
        problem: dict = {}
        try:
            async for chunk in chunks:
                if chunk.text:
                    if not parts:
                        step.update(completion_start_time=datetime.now(UTC))  # time to first word
                    parts.append(chunk.text)
                usage = chunk.usage or usage
                finish_reason = chunk.finish_reason or finish_reason
                yield chunk
            if finish_reason is None:
                problem = {"level": "WARNING", "status_message": "The stream ended without a finish reason."}
        except Exception as exc:
            problem = {"level": "ERROR", "status_message": str(exc) or type(exc).__name__}
            raise
        except BaseException:
            # The caller stopped reading (client disconnect) or was cancelled.
            if finish_reason is None:
                problem = {"level": "WARNING", "status_message": "The stream was closed before the answer finished."}
            raise
        finally:
            step.end(
                output="".join(parts),
                usage_details=_usage_details(usage),
                metadata={"finish_reason": finish_reason} if finish_reason else None,
                **problem,
            )
            close = getattr(chunks, "aclose", None)
            if close is not None:
                with suppress(Exception):
                    await close()
