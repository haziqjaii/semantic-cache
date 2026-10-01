"""
Prometheus metrics for the semantic cache.

We define all counters and histograms here so they're importable
from any module (api, engine, classifier) without circular imports.
"""

from __future__ import annotations

import logging
import time
from dataclasses import MISSING, dataclass, field, fields
from typing import ClassVar

from prometheus_client import Histogram
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

from semcache.config import estimate_cost_myr

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """
    Simple in-memory metrics tracker.

    We use a plain dataclass for counters so that we can easily reset them in tests,
    and we expose them to Prometheus via a custom Collector (CacheMetricsCollector).
    """

    # Cache performance
    cache_hits: int = 0
    cache_misses: int = 0
    cache_bypasses: int = 0  # Uncacheable requests (multi-turn, tools)
    cache_near_misses: int = 0  # Found candidate but below required_similarity
    cache_lookup_errors: int = 0  # Lookup failed; served uncached from the LLM
    cache_store_errors: int = 0  # Store failed after a miss; response still returned

    # Classifier tracking
    classifier_calls_success: int = 0
    classifier_calls_fallback: int = 0
    classifier_calls_skipped: int = 0
    classifier_tokens_total: int = 0

    # LLM generation — every call, including bypasses
    llm_calls: int = 0
    llm_tokens_prompt: int = 0
    llm_tokens_completion: int = 0

    # Embedding tracking: API calls made, and texts whose embedding was
    # remembered instead (see embeddings/memory.py).
    embedding_calls: int = 0
    embedding_tokens_total: int = 0
    embedding_cache_hits: int = 0

    # What cache hits saved. Each hit saves exactly the tokens its cached
    # answer originally used, priced in MYR by that answer's model.
    tokens_saved: int = 0
    cost_saved_myr: float = 0.0

    # Hits that couldn't be priced: their model is missing from
    # PRICING_TABLE, or the entry was cached without token counts.
    # Reported, never guessed.
    cache_hits_unpriced: int = 0
    unpriced_models: set[str] = field(default_factory=set)

    def price(self, model: str, input_tokens: int, output_tokens: int = 0) -> float | None:
        """
        Estimated MYR cost of a call, or None if the model has no pricing.

        Unpriced models are remembered (and logged once) so /v1/analytics
        can say the savings figure is incomplete.
        """
        cost = estimate_cost_myr(model, input_tokens, output_tokens)
        if cost is None and model not in self.unpriced_models:
            logger.warning("No pricing for model '%s'; its cost is excluded from analytics", model)
            self.unpriced_models.add(model)
        return cost

    def reset(self) -> None:
        """Reset every field to its default (mostly for tests)."""
        for f in fields(self):
            default = f.default_factory() if f.default_factory is not MISSING else f.default
            setattr(self, f.name, default)


# Global singleton — importable from anywhere
metrics = CacheMetrics()

# Global variable to store process start time for uptime tracking
process_start_time = time.time()

# ── Prometheus Histograms (Updated live, bypassing dataclass) ──

REQUEST_DURATION = Histogram(
    "semcache_request_duration_seconds",
    "Request latency in seconds",
    labelnames=["cache_status", "model"]
)

SIMILARITY_SCORE = Histogram(
    "semcache_similarity_score",
    "Similarity score of cache candidates",
    labelnames=["outcome"],
    buckets=(0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0)
)


class CacheMetricsCollector:
    """
    Custom Prometheus Collector that yields CounterMetricFamily from our CacheMetrics
    dataclass at scrape time. This avoids sync issues and keeps the dataclass the source of truth.
    """
    
    # We will update these from a background task
    _cache_entries = 0
    _evicted_keys = 0
    _expired_keys = 0
    _thresholds: ClassVar[dict[str, float]] = {}  # intent → similarity threshold in use

    def collect(self):
        # 1. Cache outcomes
        yield CounterMetricFamily("semcache_cache_hits_total", "Total cache hits", value=metrics.cache_hits)
        yield CounterMetricFamily("semcache_cache_misses_total", "Total cache misses", value=metrics.cache_misses)
        yield CounterMetricFamily("semcache_cache_bypasses_total", "Total cache bypasses", value=metrics.cache_bypasses)
        yield CounterMetricFamily("semcache_cache_near_misses_total", "Total cache near misses", value=metrics.cache_near_misses)

        e = CounterMetricFamily("semcache_cache_errors_total", "Total cache backend failures", labels=["stage"])
        e.add_metric(["lookup"], metrics.cache_lookup_errors)
        e.add_metric(["store"], metrics.cache_store_errors)
        yield e

        # 2. Classifier calls (using labels for status)
        c = CounterMetricFamily("semcache_classifier_calls_total", "Total classifier calls", labels=["status"])
        c.add_metric(["success"], metrics.classifier_calls_success)
        c.add_metric(["fallback"], metrics.classifier_calls_fallback)
        c.add_metric(["skipped"], metrics.classifier_calls_skipped)
        yield c

        yield CounterMetricFamily("semcache_classifier_tokens_total", "Total classifier tokens used", value=metrics.classifier_tokens_total)

        # 3. LLM Generation
        yield CounterMetricFamily("semcache_llm_calls_total", "Total LLM generation calls", value=metrics.llm_calls)
        yield CounterMetricFamily("semcache_llm_tokens_prompt_total", "Total LLM prompt tokens", value=metrics.llm_tokens_prompt)
        yield CounterMetricFamily("semcache_llm_tokens_completion_total", "Total LLM completion tokens", value=metrics.llm_tokens_completion)

        # 4. Embeddings
        yield CounterMetricFamily("semcache_embedding_calls_total", "Total embedding API calls", value=metrics.embedding_calls)
        yield CounterMetricFamily("semcache_embedding_tokens_total", "Total embedding tokens", value=metrics.embedding_tokens_total)
        yield CounterMetricFamily(
            "semcache_embedding_cache_hits_total",
            "Embeddings served from memory instead of the embedding API",
            value=metrics.embedding_cache_hits,
        )

        # 5. What cache hits saved
        yield CounterMetricFamily(
            "semcache_tokens_saved_total",
            "LLM tokens avoided by cache hits",
            value=metrics.tokens_saved,
        )
        yield CounterMetricFamily(
            "semcache_cost_saved_myr_total",
            "Estimated LLM cost avoided by cache hits (MYR)",
            value=metrics.cost_saved_myr,
        )

        # 6. Gauges (updated via background task)
        g_entries = GaugeMetricFamily("semcache_cache_entries", "Total number of entries in the cache")
        g_entries.add_metric([], self._cache_entries)
        yield g_entries

        g_evict = GaugeMetricFamily("semcache_evicted_keys_total", "Total keys evicted from cache by Redis")
        g_evict.add_metric([], self._evicted_keys)
        yield g_evict

        g_exp = GaugeMetricFamily("semcache_expired_keys_total", "Total keys expired in cache")
        g_exp.add_metric([], self._expired_keys)
        yield g_exp

        # Charted next to the hit rate, this shows what a threshold change did.
        g_threshold = GaugeMetricFamily(
            "semcache_similarity_threshold",
            "Similarity threshold in use per intent (default, or learned from feedback)",
            labels=["intent"],
        )
        for intent, threshold in sorted(self._thresholds.items()):
            g_threshold.add_metric([intent], threshold)
        yield g_threshold

        g_uptime = GaugeMetricFamily("semcache_uptime_seconds", "Process uptime in seconds")
        g_uptime.add_metric([], time.time() - process_start_time)
        yield g_uptime
