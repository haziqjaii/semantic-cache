"""
Cache key (namespace) generation.

THE PROBLEM:
    Imagine two features in your app:
      1. A customer support chatbot with system prompt: "You are a helpful agent..."
      2. A code review bot with system prompt: "You are a senior engineer..."

    A user asks both: "What is Python?"
    The MEANING is identical, but the EXPECTED ANSWER is completely different.
    If they share a cache entry, the support bot might return a code review.

THE SOLUTION:
    We don't cache by prompt alone. We cache by prompt + context.
    The "context" is everything that affects the expected output:
      - model name (GPT-4 and Gemini give different answers)
      - system prompt (defines the AI's persona/behavior)
      - temperature (0.0 = deterministic, 1.0 = creative)
      - max_tokens (truncated responses shouldn't match full ones)

    We hash these into a "namespace" — a short hex string.
    Two requests only share a cache entry if they have the SAME namespace
    AND their prompt embeddings are similar enough.

    Think of it like folders in a filing cabinet:
      namespace "a3f2c1" = all requests for the support bot with temp=0.7
      namespace "b8d4e2" = all requests for the code review bot with temp=0.0
      Within each folder, we search by semantic similarity.

WHY SHA-256?
    - Deterministic: same inputs always produce the same hash
    - Fast: microseconds, negligible overhead
    - Collision-resistant: effectively impossible for two different inputs
      to produce the same hash
    - Fixed length: always 64 hex chars, regardless of input size
"""

import hashlib
import json
import re


def build_namespace(
    model: str,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Generate a deterministic namespace hash from request parameters.

    All parameters that affect the expected output must be included.
    Parameters that DON'T affect output (like `stream: true`) are excluded.

    Args:
        model: The LLM model name (e.g., "gemini-3.5-flash").
        system_prompt: The system instruction, if any.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.

    Returns:
        A 16-char hex string (first 64 bits of SHA-256).
        We truncate to 16 chars because:
          - 64 bits = 18 quintillion possible values
          - Even with 1 million namespaces, collision probability is ~0.000003%
          - Shorter keys = less Redis memory
    """
    # We build a dict and JSON-serialize it to get a canonical string.
    # Why JSON, not f-string? Because JSON serialization is deterministic
    # for the same input (same key order when using sort_keys=True),
    # while f-strings could have subtle formatting differences.
    key_parts = {
        "model": model,
        "system_prompt": system_prompt or "",
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    canonical = json.dumps(key_parts, sort_keys=True)
    hash_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # Return first 16 chars — enough uniqueness, saves Redis memory.
    return hash_digest[:16]


def hash_system_prompt(system_prompt: str | None) -> str:
    """
    A 16-char hash of the system prompt alone (None and "" hash the same).

    The namespace mixes the system prompt with model and parameters, so it
    can't answer "every entry for this system prompt". This hash is stored
    on each entry for exactly that kind of invalidation.
    """
    return hashlib.sha256((system_prompt or "").encode("utf-8")).hexdigest()[:16]


# Cache tags: short labels clients attach to entries (X-Cache-Tags header)
# so they can invalidate them as a group later.
MAX_CACHE_TAGS = 10
_TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,63}$")


def normalize_cache_tag(tag: str) -> str:
    """
    Lowercase and validate one cache tag.

    Tags are matched case-insensitively (as Redis TAG fields are), so we
    store them lowercased. Commas are not allowed: they separate tags.

    Raises:
        ValueError: If the tag is empty or has characters outside
            letters, digits, '_', '.', ':' and '-' (max 64 chars).
    """
    normalized = tag.strip().lower()
    if not _TAG_PATTERN.match(normalized):
        raise ValueError(
            f"Invalid cache tag {tag!r}: use 1-64 letters, digits, '_', '.', ':' or '-', "
            "starting with a letter or digit."
        )
    return normalized


def parse_cache_tags(header: str | None) -> list[str]:
    """Parse a comma-separated tag list ("support, Billing:v2") into sorted, unique tags."""
    if not header:
        return []
    tags = sorted({normalize_cache_tag(t) for t in header.split(",") if t.strip()})
    if len(tags) > MAX_CACHE_TAGS:
        raise ValueError(f"At most {MAX_CACHE_TAGS} cache tags are allowed, got {len(tags)}.")
    return tags
