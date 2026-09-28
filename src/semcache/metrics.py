"""
Prometheus metrics for the semantic cache.

We define all counters and histograms here so they're importable
from any module (api, engine, classifier) without circular imports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generator

from prometheus_client import Histogram
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.registry import CollectorRegistry


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

    # Classifier tracking
    classifier_calls_success: int = 0
    classifier_calls_fallback: int = 0
    classifier_calls_skipped: int = 0
    classifier_tokens_total: int = 0

    # LLM generation
    llm_calls: int = 0
    llm_tokens_prompt: int = 0
    llm_tokens_completion: int = 0
    
    # Embedding tracking
    embedding_calls: int = 0
    embedding_tokens_total: int = 0

    def reset(self) -> None:
        """Reset all counters to 0 (mostly for tests)."""
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_bypasses = 0
        self.cache_near_misses = 0
        self.classifier_calls_success = 0
        self.classifier_calls_fallback = 0
        self.classifier_calls_skipped = 0
        self.classifier_tokens_total = 0
        self.llm_calls = 0
        self.llm_tokens_prompt = 0
        self.llm_tokens_completion = 0
        self.embedding_calls = 0
        self.embedding_tokens_total = 0


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

CACHE_ENTRIES = GaugeMetricFamily(
    "semcache_cache_entries",
    "Total number of entries in the cache"
)

EVICTED_KEYS = GaugeMetricFamily(
    "semcache_evicted_keys_total",
    "Total keys evicted from cache by Redis"
)

EXPIRED_KEYS = GaugeMetricFamily(
    "semcache_expired_keys_total",
    "Total keys expired in cache"
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

    def collect(self):
        # 1. Cache outcomes
        yield CounterMetricFamily("semcache_cache_hits_total", "Total cache hits", value=metrics.cache_hits)
        yield CounterMetricFamily("semcache_cache_misses_total", "Total cache misses", value=metrics.cache_misses)
        yield CounterMetricFamily("semcache_cache_bypasses_total", "Total cache bypasses", value=metrics.cache_bypasses)
        yield CounterMetricFamily("semcache_cache_near_misses_total", "Total cache near misses", value=metrics.cache_near_misses)

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

        # 5. Gauges (updated via background task)
        g_entries = GaugeMetricFamily("semcache_cache_entries", "Total number of entries in the cache")
        g_entries.add_metric([], self._cache_entries)
        yield g_entries

        g_evict = GaugeMetricFamily("semcache_evicted_keys_total", "Total keys evicted from cache by Redis")
        g_evict.add_metric([], self._evicted_keys)
        yield g_evict

        g_exp = GaugeMetricFamily("semcache_expired_keys_total", "Total keys expired in cache")
        g_exp.add_metric([], self._expired_keys)
        yield g_exp

        g_uptime = GaugeMetricFamily("semcache_uptime_seconds", "Process uptime in seconds")
        g_uptime.add_metric([], time.time() - process_start_time)
        yield g_uptime
