# Semantic Cache for LLMs

A semantic caching proxy for LLM APIs, designed to cut latency and API costs.

## Playground

Start the server (see [Development](#development)) and open **http://localhost:8000**. The playground lets you try the cache by hand:

* **Ask** questions and see each response's `HIT` / `MISS` / `BYPASS` status, similarity, latency and tokens. The example buttons include a reworded question, to show a semantic hit, and a poem, which is never cached.
* **Totals** since the server started: hit rate, tokens spent and saved, and money saved (RM).
* **Feedback:** on a `HIT`, say whether the cached answer was right.
* **Near misses:** questions that almost matched a cached one. Mark each as *Same question* or *Different*.
* **Threshold tuner:** hit rate vs. wrong-answer rate at each threshold, and the threshold each question type uses now.
* **Invalidate** entries by model, tag, tag prefix or system prompt, with a preview before deleting.
* **Cached entries**, newest first, with their tags, hit counts, required similarity and expiry. This list is admin-only, like invalidation: if the server sets `ADMIN_TOKEN`, enter it under *Invalidate* (it's saved in your browser only).

The page only calls the public API, so what it shows is exactly what any client would see.

## Endpoints

* **`POST /v1/chat/completions`**: OpenAI-compatible endpoint. Drops into existing applications effortlessly. Every response has an `X-Cache-Status` header (`HIT`, `MISS`, or `BYPASS`), plus `X-Cache-Similarity` on hits, `X-Cache-Bypass-Reason` (`uncacheable` or `cache-error`) on bypasses, and `X-Cache-Lookup-Id` (for feedback) on hits and misses. Cache hits replay the original answer's `usage` and `finish_reason`.
* **`GET /v1/analytics`**: JSON dashboard showing hit rates, tokens spent and saved, money saved (RM), and classifier performance. (Note: metrics reset on process restart).
* **`GET /metrics`**: Prometheus-formatted metrics (counters, request duration histograms, similarity score histograms, live cache sizes, tokens and money saved).
* **`GET /v1/cache/stats`**: Live Redis store stats (entry count, plus Redis-wide evicted/expired key counters).
* **`POST /v1/cache/invalidate`**: Delete cached entries by model, system prompt, or tag (see below).
* **`GET /v1/cache/entries?limit=50`**: The most recently cached entries, newest first (admin-only, like invalidation).
* **`POST /v1/cache/feedback`**, **`GET /v1/cache/near-misses`**, **`GET /v1/cache/tuner`**, **`GET /v1/cache/thresholds`**: threshold tuning (see below). Feedback and near misses are admin-only.
* **`GET /playground`** (and `/`, which redirects there): the web playground.

## Cache invalidation

Changing a system prompt, model, temperature or `max_tokens` already puts requests in a new cache namespace, so outdated entries can't be *served*. They just wait for their TTL to expire. Invalidation is for when the **answers** go stale while the requests stay the same: a model upgraded behind the same name, facts that changed (prices, policies), or a feature you want to reset. It also frees memory.

Label entries by sending an `X-Cache-Tags` header with chat requests (comma-separated; letters, digits, `_ . : -`; up to 10; case-insensitive). Tags group entries for invalidation; they don't split the cache.

```sh
curl localhost:8000/v1/chat/completions -H "X-Cache-Tags: pricing, shop:v2" -d '{...}'
```

Then invalidate with any combination of filters (combined with AND):

| Body | Clears |
|---|---|
| `{"model": "gemini-3.5-flash"}` | Everything that model generated |
| `{"system_prompt": "You are a support agent."}` | Everything for that exact system prompt, across all models and parameters (`""` = requests without one) |
| `{"system_prompt_hash": "7ec53d266067f78f"}` | The same, by the prompt's 16-char hash |
| `{"tag": "pricing"}` | Entries with that tag |
| `{"tag_prefix": "shop:"}` | Entries with a tag starting with `shop:` |
| `{"all": true}` | The whole cache (can't be combined with filters) |

Add `"dry_run": true` to see how many entries match without deleting. An empty body is rejected, so a missing filter can never wipe the cache.

Set `ADMIN_TOKEN` to require `Authorization: Bearer <token>` on this endpoint. It's open when unset (fine on localhost), and the server logs a warning at startup.

Entries cached before this feature have no system prompt hash or tags, so they can only be cleared by `model` or `all` (or they expire within 24 hours). When the app starts against an older index, it rebuilds the index definition automatically and keeps every cached entry.

## Tuning thresholds: near misses, feedback, and learning

How similar must two questions be to share an answer? Too loose and people get wrong answers; too strict and the cache rarely hits. The cache learns the answer from feedback:

1. **Every cacheable request is logged** (in Redis, the most recent 5,000) with the closest cached question and their similarity: a **hit**, a **near miss** (similar, but below the threshold), or a **miss** (nothing similar). Responses carry the log id in `X-Cache-Lookup-Id`.
2. **People label lookups** with `POST /v1/cache/feedback {"lookup_id": "...", "good_match": true|false}`: for a hit, *was the cached answer right?*; for a near miss, *would it have been right?* Labelled lookups are kept permanently as training data.
3. **The near-miss analyzer** (`GET /v1/cache/near-misses`) lists lookups that almost matched, to label and to spot phrasings the cache is missing.
4. **The threshold tuner** (`GET /v1/cache/tuner?intent=factual`) shows, for each threshold from 0.90 to 1.00, how many recent lookups would have hit and how many of those were labelled wrong.
5. **Learned thresholds** (`GET /v1/cache/thresholds`): once a question type (factual, how_to, …) has **10 labels**, it uses the **lowest** threshold where at least **95%** of labelled matches were right, with at least 5 labels at or above it. It never goes below the lowest similarity anyone has judged, and stays within 0.90–0.99. Lookups use it immediately, including for entries already in the cache.

## Savings (in Malaysian ringgit)

The one money figure is **what cache hits saved**, in **MYR**. Everything else is counted in tokens: `/v1/analytics` reports `tokens.spent` (LLM tokens actually used) and `tokens.saved` (LLM tokens avoided by hits).

* **Savings are exact per hit:** each hit is credited with the tokens its cached answer originally used, priced by that answer's model.
* Model prices live in `PRICING_TABLE` (`src/semcache/config.py`) in USD per million tokens, as providers publish them, and are converted with its `usd_to_myr` rate. Update the rate and its `usd_to_myr_as_of` date along with the prices.
* Hits on models missing from `PRICING_TABLE` are never priced as another model. `/v1/analytics` lists them under `savings.unpriced_models`.

## Monitoring queries (PromQL)

```promql
# Hit rate per model (last 5 min)
sum by (model) (rate(semcache_request_duration_seconds_count{cache_status="hit"}[5m]))
  / sum by (model) (rate(semcache_request_duration_seconds_count[5m]))

# Money saved per hour, in MYR
increase(semcache_cost_saved_myr_total[1h])

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
