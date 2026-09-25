"""
Cache policy — TTL tiers and similarity threshold rules.

This module defines the RULES, not the classification logic.
Think of it like a rulebook:
  - policy.py says "factual questions get 24h TTL"
  - ttl_classifier.py (Phase 3) reads a prompt and decides "this IS a factual question"

WHY SEPARATE THEM?
    The rules rarely change. The classifier will evolve (from simple keywords
    to ML-based). Separating them means upgrading the classifier never touches
    the policy definitions.

THRESHOLD PHILOSOPHY:
    Higher threshold (0.98) = fewer cache hits, but almost never wrong.
    Lower threshold (0.90) = more cache hits, but risk serving wrong answers.

    The sweet spot depends on the use case:
      - Classification tasks ("Is this spam?") → 0.90 is fine
        (the answer space is just "yes" or "no")
      - Creative writing ("Write me a poem about...") → 0.98 or skip cache
        (users expect unique outputs)
      - Factual questions ("What is Python?") → 0.95 is the sweet spot
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TTLTier(Enum):
    """
    Predefined TTL tiers for cached responses.

    Why tiers instead of arbitrary TTL values?
        Consistency and predictability. When debugging cache behavior,
        you want to know "this is a LONG entry" not "this entry expires
        in 87,432 seconds." Tiers make the system easier to reason about.
    """

    LONG = 86400       # 24 hours — stable facts, definitions
    MEDIUM = 3600      # 1 hour — semi-stable info, how-to guides
    SHORT = 300        # 5 minutes — time-sensitive or rapidly changing
    NO_CACHE = 0       # Don't cache — creative tasks, personal queries


@dataclass
class CachePolicy:
    """
    A complete cache policy for a request.

    This gets attached to every cache entry, controlling:
      - How long it lives (ttl_tier)
      - How similar a new query must be to reuse it (similarity_threshold)
    """

    ttl_tier: TTLTier = TTLTier.LONG
    similarity_threshold: float = 0.95

    @property
    def ttl_seconds(self) -> int:
        """Get the TTL in seconds from the tier."""
        return self.ttl_tier.value


# ── Default policies per task type ──────────────────────────
# Used by the IntentClassifier (Phase 3) to map prompt categories
# to cache behavior. The classifier returns a category string,
# and we look up the corresponding policy here.

DEFAULT_POLICY = CachePolicy(
    ttl_tier=TTLTier.LONG,
    similarity_threshold=0.95,
)

# Pre-defined policies for known task types (extensible).
TASK_POLICIES: dict[str, CachePolicy] = {
    "factual": CachePolicy(ttl_tier=TTLTier.LONG, similarity_threshold=0.95),
    "how_to": CachePolicy(ttl_tier=TTLTier.MEDIUM, similarity_threshold=0.93),
    "time_sensitive": CachePolicy(ttl_tier=TTLTier.SHORT, similarity_threshold=0.97),
    "creative": CachePolicy(ttl_tier=TTLTier.NO_CACHE, similarity_threshold=0.99),
    "classification": CachePolicy(ttl_tier=TTLTier.LONG, similarity_threshold=0.90),
}

# The most permissive threshold across all policies.
# Used by engine.lookup() when querying Redis — we always search at this
# floor so we never miss a candidate that some intent would have accepted.
# The per-entry required_similarity check happens in Python afterwards.
FLOOR_THRESHOLD: float = min(p.similarity_threshold for p in TASK_POLICIES.values())

