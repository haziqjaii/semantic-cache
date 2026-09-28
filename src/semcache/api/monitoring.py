import time

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

from semcache.config import PRICING_TABLE
from semcache.metrics import CacheMetricsCollector, metrics, process_start_time

# Register custom collector once
try:
    REGISTRY.register(CacheMetricsCollector())
except ValueError:
    # Already registered in tests
    pass

router = APIRouter()

# Mounted at the root (not under /v1), where Prometheus scrapes by default.
metrics_router = APIRouter()


@metrics_router.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    """Prometheus exposition, served directly at /metrics (no redirect)."""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@router.get("/analytics")
async def get_analytics():
    """
    Business metrics dashboard computing derived metrics like cost and time saved.

    All costs are estimates in MYR, priced when each call happens (see
    PRICING_TABLE). The same figures are exported to Prometheus at /metrics.
    """
    total_requests = (
        metrics.cache_hits
        + metrics.cache_misses
        + metrics.cache_bypasses
        + metrics.cache_lookup_errors
    )
    hit_rate = metrics.cache_hits / total_requests if total_requests > 0 else 0
    cacheable_requests = metrics.cache_hits + metrics.cache_misses
    hit_rate_cacheable = metrics.cache_hits / cacheable_requests if cacheable_requests > 0 else 0

    # Each hit saved exactly what its cached answer originally cost; the
    # classifier and embeddings are what the cache itself costs to run.
    overhead_cost = metrics.classifier_cost_myr + metrics.embedding_cost_myr
    net_cost_saved = metrics.cost_saved_myr - overhead_cost

    # Classifier stats
    total_classifier_calls = metrics.classifier_calls_success + metrics.classifier_calls_fallback
    classifier_success_rate = metrics.classifier_calls_success / total_classifier_calls if total_classifier_calls > 0 else 0
    classifier_fallback_rate = metrics.classifier_calls_fallback / total_classifier_calls if total_classifier_calls > 0 else 0

    return JSONResponse(content={
        "meta": {
            "since_process_start": True,
            "uptime_seconds": time.time() - process_start_time,
            "pricing_as_of": PRICING_TABLE["as_of"],
            "currency": PRICING_TABLE["currency"],
            "usd_to_myr": PRICING_TABLE["usd_to_myr"],
            "usd_to_myr_as_of": PRICING_TABLE["usd_to_myr_as_of"],
        },
        "cache": {
            "total_requests": total_requests,
            "cache_hits": metrics.cache_hits,
            "cache_misses": metrics.cache_misses,
            "cache_bypasses": metrics.cache_bypasses,
            "cache_near_misses": metrics.cache_near_misses,
            "cache_lookup_errors": metrics.cache_lookup_errors,
            "cache_store_errors": metrics.cache_store_errors,
            "hit_rate": hit_rate,
            "hit_rate_cacheable": hit_rate_cacheable,
        },
        "cost": {
            "llm_tokens_prompt": metrics.llm_tokens_prompt,
            "llm_tokens_completion": metrics.llm_tokens_completion,
            "classifier_tokens": metrics.classifier_tokens_total,
            "embedding_tokens": metrics.embedding_tokens_total,
            "estimated_llm_cost_myr": metrics.llm_cost_myr,
            "estimated_classifier_cost_myr": metrics.classifier_cost_myr,
            "estimated_embedding_cost_myr": metrics.embedding_cost_myr,
            "estimated_cost_saved_myr": metrics.cost_saved_myr,
            "net_cost_saved_myr": net_cost_saved,
            # Estimates above exclude these: calls to models missing from
            # PRICING_TABLE, and hits on entries cached without token counts.
            "unpriced_llm_calls": metrics.llm_calls_unpriced,
            "unpriced_cache_hits": metrics.cache_hits_unpriced,
            "unpriced_models": sorted(metrics.unpriced_models),
        },
        "latency": {
            # Note: We can't compute this exactly from Histograms trivially in Python without parsing buckets.
            # Usually PromQL does this. But we provide the raw data in /metrics.
            "message": "See /metrics (semcache_request_duration_seconds) for accurate latency savings."
        },
        "classifier": {
            "total_calls": total_classifier_calls,
            "success_rate": classifier_success_rate,
            "fallback_rate": classifier_fallback_rate,
            "tokens_used": metrics.classifier_tokens_total,
        }
    })
