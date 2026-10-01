"""
Builds the Grafana dashboard (dashboards/semcache.json).

Grafana dashboards are long JSON files. Defining the panels here keeps
them readable and consistent; run this after changing a panel:

    uv run python monitoring/grafana/build_dashboard.py

A test checks that the committed JSON matches what this script produces.
"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path(__file__).parent / "dashboards" / "semcache.json"
DATASOURCE = {"type": "prometheus", "uid": "prometheus"}  # see provisioning/datasources

# ── Queries (PromQL) ────────────────────────────────────────
HITS = "sum(semcache_cache_hits_total)"
MISSES = "sum(semcache_cache_misses_total)"
# clamp_min avoids 0 / 0 before any traffic (which would show as a blank tile): it reads 0% instead.
HIT_RATE = f"{HITS} / clamp_min({HITS} + {MISSES}, 1)"
RECENT_HITS = "sum(rate(semcache_cache_hits_total[$__rate_interval]))"
RECENT_MISSES = "sum(rate(semcache_cache_misses_total[$__rate_interval]))"
RECENT_HIT_RATE = f"{RECENT_HITS} / ({RECENT_HITS} + {RECENT_MISSES})"
TOKENS_SPENT = "sum(semcache_llm_tokens_prompt_total) + sum(semcache_llm_tokens_completion_total)"
TOKENS_SAVED = "sum(semcache_tokens_saved_total)"
# Counters that exist from startup (the duration histogram has no series
# until the first request, which would show "No data").
REQUESTS = (
    f"{HITS} + {MISSES} + sum(semcache_cache_bypasses_total)"
    ' + sum(semcache_cache_errors_total{stage="lookup"})'
)
MODEL_HIT_RATE = (
    'sum by (model) (rate(semcache_request_duration_seconds_count{cache_status="hit"}[$__rate_interval]))'
    ' / sum by (model) (rate(semcache_request_duration_seconds_count{cache_status=~"hit|miss"}[$__rate_interval]))'
)


def latency(quantile: float) -> str:
    return (
        f"histogram_quantile({quantile}, sum by (le, cache_status) "
        "(rate(semcache_request_duration_seconds_bucket[$__rate_interval])))"
    )


# ── Panel builders ──────────────────────────────────────────

def _targets(queries: list[tuple[str, str]], **extra) -> list[dict]:
    return [
        {"refId": chr(ord("A") + i), "expr": expr, "legendFormat": legend, **extra}
        for i, (expr, legend) in enumerate(queries)
    ]


def stat(title, expr, *, unit="none", decimals=None, color="blue", description="") -> dict:
    """A single big number."""
    defaults = {"unit": unit, "color": {"mode": "fixed", "fixedColor": color}}
    if decimals is not None:
        defaults["decimals"] = decimals
    return {
        "type": "stat",
        "title": title,
        "description": description,
        "datasource": DATASOURCE,
        "targets": _targets([(expr, "")], instant=True),
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "colorMode": "value",
            "graphMode": "none",
            "textMode": "value",
        },
    }


def timeseries(
    title, queries, *, unit="none", description="", stacked=False, filled=True, min=0, max=None, overrides=None
) -> dict:
    """Lines over time (`filled=False` for plain lines, e.g. levels rather than amounts)."""
    defaults = {
        "unit": unit,
        "custom": {
            "lineWidth": 2,
            "fillOpacity": 25 if stacked else 8 if filled else 0,
            "showPoints": "never",
            "spanNulls": True,
            "stacking": {"mode": "normal" if stacked else "none"},
        },
    }
    if min is not None:
        defaults["min"] = min
    if max is not None:
        defaults["max"] = max
    return {
        "type": "timeseries",
        "title": title,
        "description": description,
        "datasource": DATASOURCE,
        "targets": _targets(queries),
        "fieldConfig": {"defaults": defaults, "overrides": overrides or []},
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }


def histogram_bars(title, expr, *, color, description="") -> dict:
    """One bar per histogram bucket (Grafana turns cumulative buckets into per-bucket counts)."""
    return {
        "type": "bargauge",
        "title": title,
        "description": description,
        "datasource": DATASOURCE,
        "targets": _targets([(expr, "{{le}}")], format="heatmap", instant=True),
        "fieldConfig": {
            "defaults": {"unit": "none", "min": 0, "color": {"mode": "fixed", "fixedColor": color}},
            "overrides": [],
        },
        "options": {
            "displayMode": "basic",
            "orientation": "vertical",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "showUnfilled": False,
        },
    }


def right_axis(series: str, unit: str) -> dict:
    """Put one series on the right-hand axis, dashed, with its own unit."""
    return {
        "matcher": {"id": "byName", "options": series},
        "properties": [
            {"id": "custom.axisPlacement", "value": "right"},
            {"id": "unit", "value": unit},
            {"id": "min", "value": 0},
            {"id": "max", "value": 1},
            {"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [8, 6]}},
        ],
    }


# ── The dashboard ───────────────────────────────────────────

def layout(rows: list[tuple[int, list[tuple[int, dict]]]]) -> list[dict]:
    """Place panels row by row on Grafana's 24-column grid: (height, [(width, panel), ...])."""
    panels, y, panel_id = [], 0, 1
    for height, row in rows:
        x = 0
        for width, panel in row:
            panels.append({**panel, "id": panel_id, "gridPos": {"h": height, "w": width, "x": x, "y": y}})
            x += width
            panel_id += 1
        y += height
    return panels


def build() -> dict:
    rows = [
        # The headline numbers.
        (4, [
            (4, stat("Hit rate", HIT_RATE, unit="percentunit", decimals=1, color="green",
                     description="Cache hits as a share of cacheable requests (hits + misses), since the app started.")),
            (4, stat("Saved (RM)", "sum(semcache_cost_saved_myr_total)", decimals=4, color="green",
                     description="What the cache hits would have cost at the LLM, in Malaysian ringgit.")),
            (4, stat("Tokens saved", TOKENS_SAVED, unit="short", color="green",
                     description="LLM tokens that cache hits avoided.")),
            (4, stat("Tokens spent", TOKENS_SPENT, unit="short",
                     description="LLM tokens actually used (prompt + completion).")),
            (4, stat("Requests", REQUESTS, unit="short",
                     description="Chat requests served (hits, misses and bypasses).")),
            (4, stat("Cached entries", "sum(semcache_cache_entries)", unit="short",
                     description="Answers in the cache right now.")),
        ]),
        (8, [
            (12, timeseries(
                "Hit rate over time",
                [(HIT_RATE, "since start"), (RECENT_HIT_RATE, "recent")],
                unit="percentunit", max=1,
                description="'Since start' climbs as the cache fills (convergence). 'Recent' is the rate right now.",
            )),
            (12, timeseries(
                "Latency: cached vs. uncached",
                [(latency(0.5), "{{cache_status}} p50"), (latency(0.95), "{{cache_status}} p95"),
                 (latency(0.99), "{{cache_status}} p99")],
                unit="s",
                description="Response time percentiles by outcome. Hits skip the LLM call.",
            )),
        ]),
        (8, [
            (12, timeseries(
                "Requests by outcome",
                [("sum by (cache_status) (rate(semcache_request_duration_seconds_count[$__rate_interval]))",
                  "{{cache_status}}")],
                unit="reqps", stacked=True,
                description="Requests per second, split into hits, misses and bypasses.",
            )),
            (6, timeseries(
                "Savings (RM, cumulative)",
                [("sum(semcache_cost_saved_myr_total)", "saved")],
                description="Money saved by cache hits so far.",
            )),
            (6, timeseries(
                "Tokens: saved vs. spent",
                [(TOKENS_SAVED, "saved"), (TOKENS_SPENT, "spent")],
                unit="short",
                description="Cumulative LLM tokens avoided by hits, against tokens actually used.",
            )),
        ]),
        (8, [
            (12, timeseries(
                "Similarity thresholds and hit rate",
                [("semcache_similarity_threshold", "{{intent}}"), (RECENT_HIT_RATE, "hit rate")],
                min=0.88, max=1.0, filled=False, overrides=[right_axis("hit rate", "percentunit")],
                description="The threshold each question type uses (left axis) against the hit rate "
                            "(right axis, dashed). A threshold learned from feedback shows as a step.",
            )),
            (6, histogram_bars(
                "Similarity of hits",
                'sum by (le) (semcache_similarity_score_bucket{outcome="hit"})', color="green",
                description="How similar matched questions were to the cached question they hit.",
            )),
            (6, histogram_bars(
                "Similarity of near misses",
                'sum by (le) (semcache_similarity_score_bucket{outcome="near_miss"})', color="orange",
                description="Candidates that fell below their threshold. Many just under it suggests the threshold is too strict.",
            )),
        ]),
        (7, [
            (8, timeseries(
                "Hit rate per model",
                [(MODEL_HIT_RATE, "{{model}}")],
                unit="percentunit", max=1,
            )),
            (8, timeseries(
                "Cache size",
                [("sum(semcache_cache_entries)", "entries"), ("sum(semcache_expired_keys_total)", "expired (Redis-wide)"),
                 ("sum(semcache_evicted_keys_total)", "evicted (Redis-wide)")],
                unit="short",
                description="Entries in the cache, and keys Redis has expired or evicted.",
            )),
            (8, timeseries(
                "Near misses, classifier fallbacks, cache errors",
                [("sum(semcache_cache_near_misses_total)", "near misses"),
                 ('sum(semcache_classifier_calls_total{status="fallback"})', "classifier fallbacks"),
                 ("sum(semcache_cache_errors_total)", "cache errors")],
                unit="short",
                description="Things worth a look: almost-matches, classifier failures, and Redis/embedding failures.",
            )),
        ]),
    ]
    return {
        "uid": "semcache",
        "title": "Semantic Cache",
        "tags": ["semcache"],
        "schemaVersion": 39,
        "version": 1,
        "editable": True,
        "graphTooltip": 1,  # shared crosshair across panels
        "refresh": "5s",
        "time": {"from": "now-15m", "to": "now"},
        "timepicker": {"refresh_intervals": ["5s", "10s", "30s", "1m", "5m"]},
        "templating": {"list": []},
        "annotations": {"list": []},
        "panels": layout(rows),
    }


def render() -> str:
    return json.dumps(build(), indent=2) + "\n"


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render(), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(build()['panels'])} panels)")
