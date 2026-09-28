import pytest
from fastapi.testclient import TestClient

from semcache.main import app
from semcache.metrics import metrics

@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset all in-memory metrics before each test."""
    metrics.reset()

def test_analytics_math():
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
    
    # 5 single-turn misses + 2 long multi-turn bypasses (20k/10k tokens each)
    metrics.llm_calls = 7
    metrics.llm_tokens_prompt = 90000
    metrics.llm_tokens_completion = 45000
    metrics.llm_calls_cacheable = 5
    metrics.llm_tokens_prompt_cacheable = 50000
    metrics.llm_tokens_completion_cacheable = 25000

    metrics.embedding_calls = 15
    metrics.embedding_tokens_total = 3000
    
    client = TestClient(app)
    response = client.get("/v1/analytics")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check cache stats
    assert data["cache"]["total_requests"] == 20  # hits(10) + misses(5) + bypasses(5)
    assert data["cache"]["hit_rate"] == 0.5  # 10 / 20
    assert data["cache"]["hit_rate_cacheable"] == 10 / 15  # hits / (hits + misses)
    
    # Check cost calculations
    # LLM price: $1.50/M input, $9.00/M output
    # Total spend includes bypasses: (90000 / 1M) * 1.50 + (45000 / 1M) * 9.00 = 0.135 + 0.405
    assert abs(data["cost"]["estimated_llm_cost_usd"] - 0.54) < 1e-6

    # Avg tokens use cacheable calls only: 10,000 prompt, 5,000 completion per call
    # Saved tokens (hits=10): 100,000 prompt, 50,000 completion
    # Saved LLM cost: (100000 / 1M) * 1.50 + (50000 / 1M) * 9.00 = 0.15 + 0.45 = 0.60
    assert abs(data["cost"]["estimated_cost_saved_usd"] - 0.60) < 1e-6
    
    # Spent cost:
    # Classifier price: $0.075/M input, $0.30/M output. We only have input tokens tracked right now.
    # 1000 tokens = 1000/1M * 0.075 = 0.000075
    # Embedding price: $0.15/M input
    # 3000 tokens = 3000/1M * 0.15 = 0.00045
    net_cost = 0.60 - 0.000075 - 0.00045
    assert abs(data["cost"]["net_cost_saved_usd"] - net_cost) < 1e-6
    
    # Check classifier stats
    assert data["classifier"]["total_calls"] == 5
    assert data["classifier"]["success_rate"] == 4 / 5
    assert data["classifier"]["fallback_rate"] == 1 / 5

def test_prometheus_metrics_endpoint():
    """Test that the Prometheus text format endpoint works."""
    metrics.cache_hits = 42
    
    client = TestClient(app)
    response = client.get("/metrics")
    
    assert response.status_code == 200
    assert "semcache_cache_hits_total 42.0" in response.text


def test_cache_errors_in_analytics_and_prometheus():
    """Lookup errors count as requests; both error stages are exported."""
    metrics.cache_hits = 1
    metrics.cache_lookup_errors = 1
    metrics.cache_store_errors = 3

    client = TestClient(app)

    data = client.get("/v1/analytics").json()
    assert data["cache"]["total_requests"] == 2
    assert data["cache"]["cache_lookup_errors"] == 1
    assert data["cache"]["cache_store_errors"] == 3

    text = client.get("/metrics").text
    assert 'semcache_cache_errors_total{stage="lookup"} 1.0' in text
    assert 'semcache_cache_errors_total{stage="store"} 3.0' in text
