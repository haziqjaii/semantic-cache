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

GEMMA MODELS (e.g. CLASSIFIER_MODEL=gemma-4-26b-a4b-it):
    On the free tier, Gemma has its own, much larger quota (14,400 requests
    a day vs. 500 for flash-lite), so moving the classifier there leaves
    flash-lite's quota for answers. Gemma on the Gemini API rejects the
    options used for Gemini models (a separate system instruction, forced
    JSON output, temperature: all return "500 Internal error"), so for
    Gemma the instructions go inside the message and the plain-text reply
    ("FACTUAL") is read by _parse_category.

FAILURE HANDLING:
    The classifier MUST NEVER break the user-facing response. If it
    times out, hits a rate limit, or returns garbage, we fall back to
    DEFAULT_POLICY and log the error. The user still gets their answer.

    Brief Google errors ("overloaded" 503, "internal" 500, "too many
    requests" 429) are retried first, after 1 s, 2 s and 4 s. The user
    never waits for the classifier (it finishes after the response is
    sent), so retrying costs them nothing, and each fallback avoided is
    an answer cached with the right policy.
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import re
from dataclasses import dataclass

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


# The only shape a Gemini classifier may answer with: {"category": <one of five>}.
_CATEGORY_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": [c.value for c in IntentCategory],
        }
    },
    "required": ["category"],
}


def _is_gemma(model: str) -> bool:
    return model.startswith("gemma-")


def _parse_category(reply: str) -> str | None:
    """
    The category a reply names, or None if it names none (or several).

    Accepts the JSON a Gemini model is forced to give ({"category": "how_to"})
    and the plain text a Gemma model gives ("HOW_TO", "How-to", "time sensitive").
    """
    text = reply.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            text = str(value.get("category", ""))
    except ValueError:
        pass  # plain text

    words = re.sub(r"[^a-z]+", " ", text.lower())  # "HOW-TO" / "how_to" → "how to"
    named = [c.value for c in IntentCategory if re.search(rf"\b{c.value.replace('_', ' ')}\b", words)]
    return named[0] if len(named) == 1 else None


# Waits before each retry of a brief Google error (see FAILURE HANDLING).
RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0)


def _is_brief_error(exc: Exception) -> bool:
    """A Google error worth retrying: rate-limited (429) or a server-side failure (5xx)."""
    code = getattr(exc, "code", None)  # google.genai.errors.APIError carries the HTTP status
    return isinstance(code, int) and (code == 429 or code >= 500)


@dataclass
class ClassifierResult:
    policy: CachePolicy
    tokens: int
    is_fallback: bool
    # The category ("factual", "how_to", ...). Stored on the cache entry so
    # thresholds can be learned per intent. None on fallback.
    intent: str | None = None


class IntentClassifier:
    """
    Classifies prompts into intent categories using an LLM.

    Gemini models use structured outputs (response_schema) to guarantee
    exactly one valid category. Gemma models don't support that, so their
    plain-text reply is parsed instead (see the module docstring).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        timeout_seconds: float = 5.0,
        default_policy: CachePolicy | None = None,
        retry_delays: tuple[float, ...] = RETRY_DELAYS_SECONDS,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._timeout = timeout_seconds
        self._retry_delays = retry_delays
        # Used on failure or an unknown category. Pass the engine's
        # configured default so DEFAULT_SIMILARITY_THRESHOLD / DEFAULT_TTL_SECONDS
        # actually take effect.
        self._default_policy = default_policy or DEFAULT_POLICY

    async def classify(self, prompt: str) -> ClassifierResult:
        """
        Classify a prompt and return the corresponding CachePolicy.

        Calls gemini-3.5-flash-lite with structured output to get
        a category, then maps it to a predefined CachePolicy.

        Args:
            prompt: The user's message text.

        Returns:
            A ClassifierResult containing the policy and tokens used.

        Raises:
            Any exception from the Gemini SDK, asyncio timeout, etc.
            Callers should use classify_safe() instead.
        """
        if _is_gemma(self._model):
            # Gemma takes no system instruction, JSON mode or temperature:
            # the instructions travel in the message itself.
            request = self._client.aio.models.generate_content(
                model=self._model,
                contents=f"{_CLASSIFIER_SYSTEM_PROMPT}\nPrompt: {prompt}",
            )
        else:
            request = self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_CLASSIFIER_SYSTEM_PROMPT,
                    temperature=0.0,  # Deterministic — we want consistent classification
                    response_schema=_CATEGORY_SCHEMA,
                    response_mime_type="application/json",
                ),
            )
        response = await asyncio.wait_for(request, timeout=self._timeout)

        # Extract tokens used (safely handling None)
        tokens = (response.usage_metadata.total_token_count or 0) if response.usage_metadata else 0

        category_str = _parse_category(response.text or "")
        if category_str is None:
            # Counted as a fallback by classify_safe, rather than guessing a category.
            raise ValueError(f"Classifier reply names no single category: {response.text!r}")

        # Map to our predefined policies.
        policy = TASK_POLICIES[category_str]

        logger.info("Classified prompt as '%s' → TTL=%ds, threshold=%.2f (tokens: %d)",
                     category_str, policy.ttl_seconds, policy.similarity_threshold, tokens)

        return ClassifierResult(
            policy=policy,
            tokens=tokens,
            is_fallback=False,
            intent=category_str,
        )

    async def classify_safe(self, prompt: str) -> ClassifierResult:
        """
        Safe wrapper around classify() that NEVER raises.

        Brief Google errors (429, 5xx) are retried after each of the
        retry delays. If it still fails, or anything else goes wrong
        (timeout, malformed output, network error), we log it and return
        the default policy and 0 tokens. The user's request is never affected.

        This is what chat.py should always call.
        """
        for retry_delay in (*self._retry_delays, None):
            try:
                return await self.classify(prompt)
            except Exception as exc:
                if retry_delay is None or not _is_brief_error(exc):
                    logger.exception("Classifier failed, falling back to default policy")
                    return ClassifierResult(policy=self._default_policy, tokens=0, is_fallback=True)
                logger.warning("Classifier got %s; retrying in %.0f s", getattr(exc, "code", "?"), retry_delay)
                await asyncio.sleep(retry_delay)
        raise AssertionError("unreachable")  # the last attempt always returns
