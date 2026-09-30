"""
Server-Sent Events (SSE) in OpenAI's streaming format.

WHAT A STREAM LOOKS LIKE
    With "stream": true, the response is a series of events, each a line
    starting with "data: " followed by a blank line:

        data: {"object": "chat.completion.chunk", "choices": [{"delta": {"role": "assistant", "content": ""}}]}
        data: {... "choices": [{"delta": {"content": "Kuala"}}]}
        data: {... "choices": [{"delta": {"content": " Lumpur."}}]}
        data: {... "choices": [{"delta": {}, "finish_reason": "stop"}]}
        data: {... "choices": [], "usage": {...}}      ← only if the client asked
        data: [DONE]

    Clients append each delta's content to build the answer as it arrives.
    Every chunk of one response shares the same id, created and model.

ERRORS
    Once the first event is sent, the HTTP status (200) is already on the
    wire, so a later failure can't become a 502. Instead the stream ends
    with an error event and no [DONE]:

        data: {"error": {"message": "...", "type": "upstream_error"}}
"""

from __future__ import annotations

import json
import time
import uuid

from semcache.schemas import UsageInfo

SSE_MEDIA_TYPE = "text/event-stream"

# Tell proxies and browsers to pass events through as they're produced.
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

DONE = "data: [DONE]\n\n"


def _event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


class ChunkWriter:
    """Builds the events of one streamed response."""

    def __init__(self, model: str, *, cached: bool = False) -> None:
        self._base = {
            "id": f"chatcmpl-{'cached-' if cached else ''}{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
        }

    def _chunk(self, delta: dict, finish_reason: str | None = None) -> str:
        choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
        return _event({**self._base, "choices": [choice]})

    def role(self) -> str:
        """The opening event: the assistant is about to speak."""
        return self._chunk({"role": "assistant", "content": ""})

    def content(self, text: str) -> str:
        return self._chunk({"content": text})

    def finish(self, finish_reason: str) -> str:
        return self._chunk({}, finish_reason)

    def usage(self, usage: UsageInfo) -> str:
        """Token counts, sent only when the client set stream_options.include_usage."""
        return _event({**self._base, "choices": [], "usage": usage.model_dump()})

    def error(self, message: str) -> str:
        return _event({"error": {"message": message, "type": "upstream_error"}})
