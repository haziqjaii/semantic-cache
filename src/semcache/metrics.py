"""
Prometheus metrics for the semantic cache.

We define all counters and histograms here so they're importable
from any module (api, engine, classifier) without circular imports.

Phase 4 will expose these via a /metrics endpoint. For now, we
just define them so the counters are tracked in-memory.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CacheMetrics:
    """
    Simple in-memory metrics tracker.

    We use a plain dataclass instead of Prometheus client for now
    so there's no extra dependency. Phase 4 will swap this for
    proper prometheus_client counters.
    """

    # Cache performance
    cache_hits: int = 0
    cache_misses: int = 0
    cache_bypasses: int = 0  # Uncacheable requests (multi-turn, tools)
    cache_near_misses: int = 0  # Found candidate but below required_similarity

    # Classifier tracking — essential for accurate cost reporting.
    # Without these, Phase 4's "cost saved" number would be inflated
    # because it wouldn't count the extra API call per miss.
    classifier_calls_success: int = 0
    classifier_calls_fallback: int = 0  # Classifier failed, used DEFAULT_POLICY
    classifier_calls_skipped: int = 0  # Request was uncacheable, no classify needed
    classifier_tokens_total: int = 0

    # LLM generation
    llm_calls: int = 0
    llm_tokens_prompt: int = 0
    llm_tokens_completion: int = 0


# Global singleton — importable from anywhere
metrics = CacheMetrics()
