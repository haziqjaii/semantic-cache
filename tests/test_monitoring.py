import pytest
from fastapi.testclient import TestClient

from semcache.config import PRICING_TABLE, estimate_cost_myr
from semcache.main import app
from semcache.metrics import REQUEST_DURATION, metrics


@pytest.fixture
def client():
    return TestClient(app)


def test_estimate_cost_is_converted_to_ringgit():
    # gemini-3.5-flash: 1M input tokens = $1.50
    assert estimate_cost_myr("gemini-3.5-flash", 1_000_000) == pytest.approx(
        1.50 * PRICING_TABLE["usd_to_myr"]
    )
    assert estimate_cost_myr("not-a-real-model", 1_000_000) is None


def test_analytics_math(client):
    """Test that the analytics JSON math computes correctly."""
    # Set up some known values
    metrics.cache_hits = 10
    metrics.cache_misses = 5
    metrics.cache_bypasses = 5
    metrics.cache_near_misses = 2

    metrics.classifier_calls_success = 4
    metrics.classifier_calls_fallback = 1
    metrics.classifier_calls_skipped = 15
    metrics.classifier_tokens_total = 1000

    metrics.llm_calls = 10
    metrics.llm_tokens_prompt = 90000
    metrics.llm_tokens_completion = 45000

    metrics.embedding_calls = 15
    metrics.embedding_tokens_total = 3000

    # What hits saved, recorded as they happen.
    metrics.tokens_saved = 120000
    metrics.cost_saved_myr = 2.50

    response = client.get("/v1/analytics")

    assert response.status_code == 200
    data = response.json()

    # Check cache stats
    assert data["cache"]["total_requests"] == 20  # hits(10) + misses(5) + bypasses(5)
    assert data["cache"]["hit_rate"] == 0.5  # 10 / 20
    assert data["cache"]["hit_rate_cacheable"] == 10 / 15  # hits / (hits + misses)

    # Tokens: spent = LLM prompt + completion; saved = avoided by hits
    assert data["tokens"]["spent"] == 135000
    assert data["tokens"]["saved"] == 120000
    assert data["tokens"]["classifier"] == 1000
    assert data["tokens"]["embedding_estimated"] == 3000

    # Money: only the savings, in MYR
    assert data["meta"]["currency"] == "MYR"
    assert data["meta"]["usd_to_myr"] == PRICING_TABLE["usd_to_myr"]
    assert data["savings"] == {"saved_myr": 2.50, "unpriced_hits": 0, "unpriced_models": []}
    assert "cost" not in data

    # Check classifier stats
    assert data["classifier"]["total_calls"] == 5
    assert data["classifier"]["success_rate"] == 4 / 5
    assert data["classifier"]["fallback_rate"] == 1 / 5


def test_analytics_reports_hits_that_could_not_be_priced(client):
    """Unpriced hits are reported, not priced as some other model."""
    metrics.cache_hits_unpriced = 3
    metrics.unpriced_models = {"gemini-9-ultra"}

    data = client.get("/v1/analytics").json()

    assert data["savings"]["unpriced_hits"] == 3
    assert data["savings"]["unpriced_models"] == ["gemini-9-ultra"]


def test_prometheus_metrics_endpoint(client):
    """Test that the Prometheus text format endpoint works."""
    metrics.cache_hits = 42

    # Served directly: a scrape target shouldn't redirect (it used to 307 to /metrics/).
    response = client.get("/metrics", follow_redirects=False)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "semcache_cache_hits_total 42.0" in response.text


def test_prometheus_exports_savings(client):
    metrics.cost_saved_myr = 12.5
    metrics.tokens_saved = 4000

    text = client.get("/metrics").text

    assert "semcache_cost_saved_myr_total 12.5" in text
    assert "semcache_tokens_saved_total 4000.0" in text
    assert "semcache_llm_cost_myr_total" not in text
    assert "semcache_overhead_cost_myr_total" not in text


def test_prometheus_has_per_model_request_counts(client):
    """
    Per-model hit rate comes from the request histogram's _count series,
    which is labelled by cache_status and model.
    """
    REQUEST_DURATION.labels(cache_status="hit", model="per-model-test").observe(0.01)
    REQUEST_DURATION.labels(cache_status="miss", model="per-model-test").observe(1.0)

    text = client.get("/metrics").text

    assert 'semcache_request_duration_seconds_count{cache_status="hit",model="per-model-test"} 1.0' in text
    assert 'semcache_request_duration_seconds_count{cache_status="miss",model="per-model-test"} 1.0' in text


def test_cache_errors_in_analytics_and_prometheus(client):
    """Lookup errors count as requests; both error stages are exported."""
    metrics.cache_hits = 1
    metrics.cache_lookup_errors = 1
    metrics.cache_store_errors = 3

    data = client.get("/v1/analytics").json()
    assert data["cache"]["total_requests"] == 2
    assert data["cache"]["cache_lookup_errors"] == 1
    assert data["cache"]["cache_store_errors"] == 3

    text = client.get("/metrics").text
    assert 'semcache_cache_errors_total{stage="lookup"} 1.0' in text
    assert 'semcache_cache_errors_total{stage="store"} 3.0' in text


def test_metrics_reset_restores_every_default():
    metrics.cache_hits = 5
    metrics.cost_saved_myr = 1.23
    metrics.unpriced_models.add("some-model")

    metrics.reset()

    assert metrics.cache_hits == 0
    assert metrics.cost_saved_myr == 0.0
    assert metrics.unpriced_models == set()
