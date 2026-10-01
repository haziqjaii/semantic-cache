"""
Tests for the Prometheus gauge refresh background task.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semcache.api.dependencies import gauge_refresh_loop, refresh_cache_gauges
from semcache.cache.engine import CacheEngine
from semcache.metrics import CacheMetricsCollector


@pytest.fixture(autouse=True)
def reset_gauges():
    yield
    CacheMetricsCollector._cache_entries = 0
    CacheMetricsCollector._evicted_keys = 0
    CacheMetricsCollector._expired_keys = 0
    CacheMetricsCollector._thresholds = {}


@pytest.mark.asyncio
async def test_refresh_copies_engine_stats_into_gauges():
    engine = MagicMock(spec=CacheEngine)
    engine.stats = AsyncMock(
        return_value={"total_entries": 7, "evicted_keys": 2, "expired_keys": 5}
    )

    engine.thresholds_in_use = MagicMock(return_value={"factual": 0.92})

    await refresh_cache_gauges(engine)

    assert CacheMetricsCollector._thresholds == {"factual": 0.92}
    assert CacheMetricsCollector._cache_entries == 7
    assert CacheMetricsCollector._evicted_keys == 2
    assert CacheMetricsCollector._expired_keys == 5


@pytest.mark.asyncio
async def test_refresh_works_for_stores_without_backend_stats(mock_embedder, memory_store):
    """The in-memory store has no eviction counters; gauges fall back to 0."""
    engine = CacheEngine(embedder=mock_embedder, store=memory_store)

    await refresh_cache_gauges(engine)

    assert CacheMetricsCollector._cache_entries == 0
    assert CacheMetricsCollector._evicted_keys == 0


@pytest.mark.asyncio
async def test_loop_logs_failure_once_then_recovery(caplog):
    """A sustained outage logs one error, not one every 15 seconds."""
    engine = MagicMock(spec=CacheEngine)
    engine.stats = AsyncMock(side_effect=[
        ConnectionError("Redis down"),
        ConnectionError("Redis down"),
        ConnectionError("Redis down"),
        {"total_entries": 3},
    ])
    engine.thresholds_in_use = MagicMock(return_value={})
    # Let the loop run four iterations, then stop it like shutdown does.
    sleep = AsyncMock(side_effect=[None, None, None, asyncio.CancelledError()])

    with (
        patch("semcache.api.dependencies.asyncio.sleep", sleep),
        caplog.at_level(logging.INFO, logger="semcache.api.dependencies"),
        pytest.raises(asyncio.CancelledError),
    ):
        await gauge_refresh_loop(engine)

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    recoveries = [r for r in caplog.records if "recovered" in r.getMessage()]
    assert len(errors) == 1
    assert len(recoveries) == 1
    assert CacheMetricsCollector._cache_entries == 3
