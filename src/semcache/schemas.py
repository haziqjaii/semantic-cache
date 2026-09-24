"""
OpenAI-compatible request and response schemas.

We use these so that any app (like LangChain, LlamaIndex, or custom scripts)
can drop in our URL in place of `api.openai.com` and it will just work.
"""

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 0.7
    max_tokens: int | None = None
    stream: bool | None = False
    # If tools/functions are provided, we will skip caching.
    tools: list[Any] | None = None
    
    @property
    def is_cacheable(self) -> bool:
        """
        Option A: Single-turn Strict
        We only cache if it's a simple, single-turn conversation (no history)
        and no tools are involved.
        """
        if self.tools is not None and len(self.tools) > 0:
            return False
            
        # Count non-system messages
        user_or_assistant_msgs = [m for m in self.messages if m.role != "system"]
        
        # If there's more than one interaction, it has history. Skip cache.
        return len(user_or_assistant_msgs) <= 1


class ChatCompletionChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionChoiceMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo | None = None
    
    # Custom headers/metadata to indicate cache status
    # This won't break OpenAI clients, it just adds extra data
    x_cache_status: str = Field(default="MISS")
