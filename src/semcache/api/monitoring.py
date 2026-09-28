import time
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from prometheus_client import REGISTRY, make_asgi_app

from semcache.config import PRICING_TABLE, get_settings
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
async def get_analytics():
    """
    Business metrics dashboard computing derived metrics like cost and time saved.
    """
    total_requests = (
        metrics.cache_hits
        + metrics.cache_misses
        + metrics.cache_bypasses
    )
    hit_rate = metrics.cache_hits / total_requests if total_requests > 0 else 0
    cacheable_requests = metrics.cache_hits + metrics.cache_misses
    hit_rate_cacheable = metrics.cache_hits / cacheable_requests if cacheable_requests > 0 else 0

    # Pricing calculations
    settings = get_settings()
    llm_price = PRICING_TABLE["models"].get(
        # We assume they use 3.5-flash since this isn't tracking model breakdown fully
        "gemini-3.5-flash", {"input": 1.50, "output": 9.00}
    )
    classifier_price = PRICING_TABLE["models"].get(
        settings.classifier_model, {"input": 0.075, "output": 0.30}
    )
    embedding_price = PRICING_TABLE["models"].get(
        settings.embedding_model, {"input": 0.15, "output": 0.0}
    )

    # Estimate average LLM tokens per call to figure out what was saved
    avg_prompt_tokens = metrics.llm_tokens_prompt / metrics.llm_calls if metrics.llm_calls > 0 else 0
    avg_completion_tokens = metrics.llm_tokens_completion / metrics.llm_calls if metrics.llm_calls > 0 else 0

    saved_prompt_tokens = avg_prompt_tokens * metrics.cache_hits
    saved_completion_tokens = avg_completion_tokens * metrics.cache_hits

    # Saved cost
    saved_cost = (
        (saved_prompt_tokens / 1_000_000) * llm_price["input"]
        + (saved_completion_tokens / 1_000_000) * llm_price["output"]
    )
    
    # Spent cost
    spent_classifier = (metrics.classifier_tokens_total / 1_000_000) * classifier_price["input"]
    spent_embedding = (metrics.embedding_tokens_total / 1_000_000) * embedding_price["input"]
    
    net_cost_saved = saved_cost - spent_classifier - spent_embedding

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
            "hit_rate": hit_rate,
            "hit_rate_cacheable": hit_rate_cacheable,
        },
        "cost": {
            "llm_tokens_prompt": metrics.llm_tokens_prompt,
            "llm_tokens_completion": metrics.llm_tokens_completion,
            "classifier_tokens": metrics.classifier_tokens_total,
            "embedding_tokens": metrics.embedding_tokens_total,
            "estimated_llm_cost_usd": (
                (metrics.llm_tokens_prompt / 1_000_000) * llm_price["input"]
                + (metrics.llm_tokens_completion / 1_000_000) * llm_price["output"]
            ),
            "estimated_cost_saved_usd": saved_cost,
            "net_cost_saved_usd": net_cost_saved,
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
