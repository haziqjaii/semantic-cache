"""
Abstract base class for LLM providers.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from semcache.schemas import ChatCompletionRequest, ChatCompletionResponse, UsageInfo


@dataclass
class StreamChunk:
    """
    One piece of a streamed response.

    A stream is any number of chunks carrying `text`, and ends with a chunk
    carrying `finish_reason` (in OpenAI's vocabulary: "stop", "length",
    "content_filter", ...). A stream that ends without one was cut short.
    """

    text: str = ""
    finish_reason: str | None = None
    usage: UsageInfo | None = None  # token counts, once the provider reports them


class LLMProvider(ABC):
    """
    Interface for generating text from an LLM.
    """

    @abstractmethod
    async def generate(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Generate a full (non-streamed) response.
        """

    @abstractmethod
    def generate_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """
        Generate a response as a stream of chunks (see StreamChunk).

        Nothing is sent to the provider until the first chunk is requested,
        so provider errors surface from the first `anext()`.
        """

    async def list_models(self) -> list[str]:
        """
        The chat models this provider offers, for GET /v1/models.

        Only a convenience for clients that show a list to pick from; any
        model name can still be requested. Empty if the provider can't say.
        """
        return []
