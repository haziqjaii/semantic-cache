"""
Intent classifier — determines the cache policy for a prompt.

Uses gemini-3.5-flash-lite with structured outputs to classify prompts
into one of five intent categories:
  - FACTUAL:        "What is Python?" → 24h TTL, 0.95 threshold
  - HOW_TO:         "How do I sort a list?" → 1h TTL, 0.93 threshold
  - TIME_SENSITIVE: "What's the weather?" → 5min TTL, 0.97 threshold
  - CREATIVE:       "Write me a poem" → NO_CACHE, 0.99 threshold
  - CLASSIFICATION: "Is this spam?" → 24h TTL, 0.90 threshold

WHY AN LLM CLASSIFIER?
    Simple keyword matching ("if 'write' in prompt → creative") breaks
    on edge cases like "Write down the formula for water" (factual, not
    creative). An LLM understands intent, not just keywords.

WHY gemini-3.5-flash-lite?
    It's Google's cheapest, fastest model — built specifically for
    classification and extraction tasks. We don't need deep reasoning
    here, just a quick category label. It typically responds in ~200ms,
    and since we run it concurrently with the main LLM generation on
    cache misses, it adds zero user-facing latency.

FAILURE HANDLING:
    The classifier MUST NEVER break the user-facing response. If it
    times out, hits a rate limit, or returns garbage, we fall back to
    DEFAULT_POLICY and log the error. The user still gets their answer.
"""

from __future__ import annotations

import asyncio
import enum
import logging

from google import genai
from google.genai import types

from semcache.cache.policy import DEFAULT_POLICY, TASK_POLICIES, CachePolicy

logger = logging.getLogger(__name__)


class IntentCategory(enum.Enum):
    """The five intent categories we classify prompts into."""

    FACTUAL = "factual"
    HOW_TO = "how_to"
    TIME_SENSITIVE = "time_sensitive"
    CREATIVE = "creative"
    CLASSIFICATION = "classification"


# The system prompt that instructs the classifier LLM.
_CLASSIFIER_SYSTEM_PROMPT = """\
You are a prompt intent classifier. Given a user prompt, classify it into
exactly one of these categories:

- FACTUAL: Questions about stable facts, definitions, or concepts.
  Examples: "What is Python?", "Explain quantum computing", "Define osmosis"

- HOW_TO: Step-by-step instructions or tutorials.
  Examples: "How do I sort a list in Python?", "How to make pasta"

- TIME_SENSITIVE: Questions about current events, weather, prices, or anything
  that changes frequently.
  Examples: "What's the weather today?", "Current Bitcoin price"

- CREATIVE: Requests for unique, original content — poems, stories, code
  generation with creative latitude, brainstorming.
  Examples: "Write me a poem about rain", "Generate a unique startup idea"

- CLASSIFICATION: Binary or categorical classification tasks where the answer
  space is small and fixed.
  Examples: "Is this email spam?", "What sentiment is this review?"

Respond with ONLY the category name, nothing else.
"""


class IntentClassifier:
    """
    Classifies prompts into intent categories using gemini-3.5-flash-lite.

    Uses structured outputs (response_schema) to guarantee the LLM returns
    exactly one valid category — no parsing, no regex, no surprises.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        timeout_seconds: float = 5.0,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._timeout = timeout_seconds

    async def classify(self, prompt: str) -> CachePolicy:
        """
        Classify a prompt and return the corresponding CachePolicy.

        Calls gemini-3.5-flash-lite with structured output to get
        a category, then maps it to a predefined CachePolicy.

        Args:
            prompt: The user's message text.

        Returns:
            The CachePolicy for this prompt's intent category.

        Raises:
            Any exception from the Gemini SDK, asyncio timeout, etc.
            Callers should use classify_safe() instead.
        """
        response = await asyncio.wait_for(
            self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_CLASSIFIER_SYSTEM_PROMPT,
                    temperature=0.0,  # Deterministic — we want consistent classification
                    response_schema={
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": [c.value for c in IntentCategory],
                            }
                        },
                        "required": ["category"],
                    },
                    response_mime_type="application/json",
                ),
            ),
            timeout=self._timeout,
        )

        # Parse the structured output.
        import json

        result = json.loads(response.text)
        category_str = result.get("category", "factual")

        # Map to our predefined policies.
        policy = TASK_POLICIES.get(category_str, DEFAULT_POLICY)

        logger.info("Classified prompt as '%s' → TTL=%ds, threshold=%.2f",
                     category_str, policy.ttl_seconds, policy.similarity_threshold)

        return policy

    async def classify_safe(self, prompt: str) -> CachePolicy:
        """
        Safe wrapper around classify() that NEVER raises.

        If anything goes wrong (timeout, rate limit, malformed output,
        network error), we log it and return DEFAULT_POLICY. The user's
        request is never affected.

        This is what chat.py should always call.
        """
        try:
            return await self.classify(prompt)
        except Exception:
            logger.exception("Classifier failed, falling back to DEFAULT_POLICY")
            return DEFAULT_POLICY
