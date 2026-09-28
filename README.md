# Semantic Cache for LLMs

A semantic caching proxy for LLM APIs, designed to cut latency and API costs.

## Endpoints

* **`POST /v1/chat/completions`**: OpenAI-compatible endpoint. Drops into existing applications effortlessly.
* **`GET /v1/analytics`**: JSON dashboard showing cache hit rates, cost savings, and classifier performance. (Note: metrics reset on process restart).
* **`GET /metrics`**: Prometheus-formatted metrics (counters, request duration histograms, similarity score histograms, live cache sizes).
* **`GET /v1/cache/stats`**: Live Redis store stats (entry count, plus Redis-wide evicted/expired key counters).

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
