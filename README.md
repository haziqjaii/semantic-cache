# Semantic Cache for LLMs

A semantic caching proxy for LLM APIs, designed to cut latency and API costs.

## Endpoints

* **`POST /v1/chat/completions`**: OpenAI-compatible endpoint. Drops into existing applications effortlessly. Every response has an `X-Cache-Status` header (`HIT`, `MISS`, or `BYPASS`), plus `X-Cache-Similarity` on hits and `X-Cache-Bypass-Reason` (`uncacheable` or `cache-error`) on bypasses. Cache hits replay the original answer's `usage` and `finish_reason`.
* **`GET /v1/analytics`**: JSON dashboard showing cache hit rates, cost savings, and classifier performance. (Note: metrics reset on process restart).
* **`GET /metrics`**: Prometheus-formatted metrics (counters, request duration histograms, similarity score histograms, live cache sizes, and costs).
* **`GET /v1/cache/stats`**: Live Redis store stats (entry count, plus Redis-wide evicted/expired key counters).

## Costs (in Malaysian ringgit)

All cost figures are estimates in **MYR**. Model prices live in `PRICING_TABLE` (`src/semcache/config.py`) in USD per million tokens, as providers publish them, and are converted with its `usd_to_myr` rate. Update the rate and its `usd_to_myr_as_of` date along with the prices.

* **Savings are exact per hit:** each hit is credited with what its cached answer originally cost (its stored token usage, priced by its model).
* **Overhead** is what running the cache costs: the intent classifier plus prompt embeddings. Embedding tokens are estimated at ~4 characters per token.
* **Net savings** = savings − overhead.
* Models missing from `PRICING_TABLE` are never priced as another model. `/v1/analytics` lists them under `unpriced_models`, with counts of the calls and hits it couldn't price.

## Monitoring queries (PromQL)

```promql
# Hit rate per model (last 5 min)
sum by (model) (rate(semcache_request_duration_seconds_count{cache_status="hit"}[5m]))
  / sum by (model) (rate(semcache_request_duration_seconds_count[5m]))

# Net savings per hour, in MYR
increase(semcache_cost_saved_myr_total[1h])
  - sum(increase(semcache_overhead_cost_myr_total[1h]))

# P95 latency, cached vs. uncached
histogram_quantile(0.95, sum by (le, cache_status) (rate(semcache_request_duration_seconds_bucket[5m])))
```

## Development

1. Start Redis Stack (Redis plus the RediSearch module needed for vector search):
   `docker compose up -d` (set `REDIS_PORT` to use a port other than 6379, and update `REDIS_URL` to match)
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.
3. Run the server with `uv run semcache`. The default configuration uses `reload=True`.

Run the unit tests with `uv run pytest -m "not integration"`. They need neither Redis nor an API key.
The integration tests (`uv run pytest -m integration`) need Redis on `localhost:6379`. **They flush that Redis database**, so they erase any cached entries in it. To keep your dev cache, run them against a separate Redis instead:

```sh
REDIS_PORT=6390 docker compose -p semcache-test up -d
SEMCACHE_TEST_REDIS_URL=redis://localhost:6390 uv run pytest -m integration
docker compose -p semcache-test down -v
```

> **Note on Prometheus Metrics in Production**:
> The metrics counters and Prometheus exposition in this project are currently designed for a single-process server.
> If you run `uvicorn` with multiple workers (e.g., `--workers 4`), the metrics will be per-process. To aggregate metrics across multiple workers, you must configure Prometheus Client's multiprocess mode (setting `PROMETHEUS_MULTIPROC_DIR`). This matters as soon as you deploy without `reload=True` and scale up the workers.
