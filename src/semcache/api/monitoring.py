import time

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from prometheus_client import REGISTRY, make_asgi_app

from semcache.config import PRICING_TABLE, Settings, estimate_cost_usd, get_settings
from semcache.metrics import CacheMetricsCollector, metrics, process_start_time

# Register custom collector once
try:
    REGISTRY.register(CacheMetricsCollector())
except ValueError:
    # Already registered in tests
    pass

# Create ASGI app for /metrics
metrics_app = make_asgi_app(registry=REGISTRY)

router = APIRouter()


@router.get("/analytics")
async def get_analytics(settings: Settings = Depends(get_settings)):  # noqa: B008
    """
    Business metrics dashboard computing derived metrics like cost and time saved.
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

    # Saved cost: each hit replaces one single-turn LLM call, so value hits
    # at the average cost of those calls (each priced by its own model).
    # Multi-turn bypasses are excluded; they're typically much longer.
    cacheable_calls = metrics.llm_calls_cacheable
    avg_cost_per_call = metrics.llm_cost_usd_cacheable / cacheable_calls if cacheable_calls > 0 else 0
    saved_cost = avg_cost_per_call * metrics.cache_hits

    # Spent cost: classifier and embedding tokens are tracked as totals, so
    # (as before) they're priced as input tokens.
    unpriced_models = set(metrics.llm_unpriced_models)
    spent_classifier = estimate_cost_usd(settings.classifier_model, metrics.classifier_tokens_total)
    if spent_classifier is None:
        unpriced_models.add(settings.classifier_model)
    spent_embedding = estimate_cost_usd(settings.embedding_model, metrics.embedding_tokens_total)
    if spent_embedding is None:
        unpriced_models.add(settings.embedding_model)

    net_cost_saved = saved_cost - (spent_classifier or 0.0) - (spent_embedding or 0.0)

    # Classifier stats
    total_classifier_calls = metrics.classifier_calls_success + metrics.classifier_calls_fallback
    classifier_success_rate = metrics.classifier_calls_success / total_classifier_calls if total_classifier_calls > 0 else 0
    classifier_fallback_rate = metrics.classifier_calls_fallback / total_classifier_calls if total_classifier_calls > 0 else 0

    return JSONResponse(content={
        "meta": {
            "since_process_start": True,
            "uptime_seconds": time.time() - process_start_time,
            "pricing_as_of": PRICING_TABLE["as_of"]
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
            "estimated_llm_cost_usd": metrics.llm_cost_usd,
            "estimated_cost_saved_usd": saved_cost,
            "net_cost_saved_usd": net_cost_saved,
            # Estimates above exclude these (models missing from PRICING_TABLE).
            "unpriced_llm_calls": metrics.llm_calls_unpriced,
            "unpriced_models": sorted(unpriced_models),
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
