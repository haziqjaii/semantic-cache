# Semantic Cache for LLMs

A semantic caching proxy for LLM APIs, designed to cut latency and API costs.

## Endpoints

* **`POST /v1/chat/completions`**: OpenAI-compatible endpoint. Drops into existing applications effortlessly.
* **`GET /v1/analytics`**: JSON dashboard showing cache hit rates, cost savings, and classifier performance. (Note: metrics reset on process restart).
* **`GET /metrics`**: Prometheus-formatted metrics (counters, request duration histograms, similarity score histograms, live cache sizes).
* **`GET /v1/cache/stats`**: Live Redis store stats.

## Development

Run the server with `uv run semcache`. The default configuration uses `reload=True`.

> **Note on Prometheus Metrics in Production**: 
> The metrics counters and Prometheus exposition in this project are currently designed for a single-process server. 
> If you run `uvicorn` with multiple workers (e.g., `--workers 4`), the metrics will be per-process. To aggregate metrics across multiple workers, you must configure Prometheus Client's multiprocess mode (setting `PROMETHEUS_MULTIPROC_DIR`). This is important if you deploy this using Docker Compose without `reload=True` and decide to scale up the workers.
