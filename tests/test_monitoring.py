import pytest
from fastapi.testclient import TestClient

from semcache.config import Settings, get_settings
from semcache.main import app
from semcache.metrics import metrics


def _test_settings(**overrides) -> Settings:
    """
    Explicit settings that ignore .env, so these tests run the same in CI
    (no .env, no API key) as on a dev machine with a customised .env.
    """
    values = {
        "gemini_api_key": "test-key",
        "classifier_model": "gemini-3.5-flash-lite",
        "embedding_model": "gemini-embedding-001",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def client():
    # A zero-arg lambda: FastAPI would treat **overrides as a request parameter.
    app.dependency_overrides[get_settings] = lambda: _test_settings()
    yield TestClient(app)
    app.dependency_overrides.clear()


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

    # gemini-3.5-flash: $1.50/M input, $9.00/M output.
    # 5 single-turn misses at 10k/5k tokens = $0.06 each → $0.30,
    # plus 2 long multi-turn bypasses at 20k/10k tokens = $0.12 each → $0.24.
    metrics.llm_calls = 7
    metrics.llm_tokens_prompt = 90000
    metrics.llm_tokens_completion = 45000
    metrics.llm_cost_usd = 0.54
    metrics.llm_calls_cacheable = 5
    metrics.llm_cost_usd_cacheable = 0.30

    metrics.embedding_calls = 15
    metrics.embedding_tokens_total = 3000

    response = client.get("/v1/analytics")

    assert response.status_code == 200
    data = response.json()

    # Check cache stats
    assert data["cache"]["total_requests"] == 20  # hits(10) + misses(5) + bypasses(5)
    assert data["cache"]["hit_rate"] == 0.5  # 10 / 20
    assert data["cache"]["hit_rate_cacheable"] == 10 / 15  # hits / (hits + misses)

    # Total spend includes bypasses.
    assert data["cost"]["estimated_llm_cost_usd"] == pytest.approx(0.54)

    # Each hit is valued at the average single-turn call: $0.30 / 5 = $0.06.
    # Saved (hits=10): $0.60. Bypasses don't inflate it.
    assert data["cost"]["estimated_cost_saved_usd"] == pytest.approx(0.60)

    # Spent cost:
    # Classifier (flash-lite, $0.075/M input): 1000 tokens = 0.000075
    # Embedding ($0.15/M input): 3000 tokens = 0.00045
    net_cost = 0.60 - 0.000075 - 0.00045
    assert data["cost"]["net_cost_saved_usd"] == pytest.approx(net_cost)
    assert data["cost"]["unpriced_llm_calls"] == 0
    assert data["cost"]["unpriced_models"] == []

    # Check classifier stats
    assert data["classifier"]["total_calls"] == 5
    assert data["classifier"]["success_rate"] == 4 / 5
    assert data["classifier"]["fallback_rate"] == 1 / 5


def test_analytics_reports_unpriced_models(client):
    """Models missing from PRICING_TABLE are reported, not priced as another model."""
    app.dependency_overrides[get_settings] = lambda: _test_settings(
        classifier_model="some-future-classifier"
    )
    metrics.classifier_tokens_total = 1000
    metrics.llm_calls_unpriced = 2
    metrics.llm_unpriced_models = {"gemini-9-ultra"}

    data = client.get("/v1/analytics").json()

    assert data["cost"]["unpriced_llm_calls"] == 2
    assert data["cost"]["unpriced_models"] == ["gemini-9-ultra", "some-future-classifier"]
    # The unpriced classifier contributes nothing rather than a guessed price.
    assert data["cost"]["net_cost_saved_usd"] == 0


def test_prometheus_metrics_endpoint(client):
    """Test that the Prometheus text format endpoint works."""
    metrics.cache_hits = 42

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "semcache_cache_hits_total 42.0" in response.text


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
    metrics.llm_cost_usd = 1.23
    metrics.llm_unpriced_models.add("some-model")

    metrics.reset()

    assert metrics.cache_hits == 0
    assert metrics.llm_cost_usd == 0.0
    assert metrics.llm_unpriced_models == set()
