"""
Abstract base class for LLM providers.
"""

from abc import ABC, abstractmethod

from semcache.schemas import ChatCompletionRequest, ChatCompletionResponse


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
