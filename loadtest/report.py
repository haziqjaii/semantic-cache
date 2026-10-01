"""
The load test's report — raw results in, headline numbers out.

THE NUMBERS
    hit rate          share of cacheable requests (hits + misses) served
                      from the cache
    convergence       the hit rate in each window of requests: it starts
                      near 0 and climbs as the cache fills
    latency           p50 / p95 / p99 for hits and for misses. p95 = 95%
                      of requests were at least this fast
    tokens saved      LLM tokens the hits avoided. "Cost without the
                      cache" is spent + saved, so
                      % saved = saved / (spent + saved)
    wrong answers     hits that returned an answer generated for a
                      DIFFERENT question (see workload.py: families).
                      This is the price of a threshold that's too loose.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

HIT, MISS, BYPASS, ERROR = "HIT", "MISS", "BYPASS", "ERROR"


@dataclass
class Result:
    """What happened to one request."""

    index: int
    family: str
    kind: str
    status: str  # HIT, MISS, BYPASS or ERROR
    latency: float  # seconds, for the attempt that succeeded
    similarity: float | None = None
    answer: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    error: str | None = None


def percentile(values: list[float], p: float) -> float | None:
    """The value below which `p` percent of `values` fall (nearest-rank method)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def _latency(results: list[Result]) -> dict | None:
    times = [r.latency for r in results]
    if not times:
        return None
    return {
        "count": len(times),
        "p50": percentile(times, 50),
        "p95": percentile(times, 95),
        "p99": percentile(times, 99),
        "mean": sum(times) / len(times),
    }


def _hit_rate(results: list[Result]) -> float | None:
    hits = sum(1 for r in results if r.status == HIT)
    cacheable = hits + sum(1 for r in results if r.status == MISS)
    return hits / cacheable if cacheable else None


def find_wrong_hits(results: list[Result]) -> tuple[list[Result], int]:
    """
    Hits that served another family's answer, and how many hits couldn't be checked.

    Every miss produced an answer for a known family. A hit replays one of
    those answers word for word, so its text tells us which family it
    really came from.
    """
    origin: dict[str, str] = {}
    for result in results:
        if result.status == MISS and result.answer:
            origin.setdefault(result.answer, result.family)

    wrong, unverified = [], 0
    for result in results:
        if result.status != HIT:
            continue
        source = origin.get(result.answer)
        if source is None:
            unverified += 1  # e.g. answered from an entry cached before this run
        elif source != result.family:
            wrong.append(result)
    return wrong, unverified


def summarize(results: list[Result], window: int = 100) -> dict:
    """The report, as plain data (see the module docstring for what each part means)."""
    ordered = sorted(results, key=lambda r: r.index)
    hits = [r for r in ordered if r.status == HIT]
    misses = [r for r in ordered if r.status == MISS]
    wrong, unverified = find_wrong_hits(ordered)

    spent = sum(r.prompt_tokens + r.completion_tokens for r in ordered if r.status in (MISS, BYPASS))
    saved = sum(r.prompt_tokens + r.completion_tokens for r in hits)  # hits replay the original usage

    hit_latency, miss_latency = _latency(hits), _latency(misses)
    speedup = None
    if hit_latency and miss_latency and hit_latency["p95"]:
        speedup = {
            "p50": miss_latency["p50"] / hit_latency["p50"] if hit_latency["p50"] else None,
            "p95": miss_latency["p95"] / hit_latency["p95"],
            "p95_reduction": 1 - hit_latency["p95"] / miss_latency["p95"] if miss_latency["p95"] else None,
        }

    by_kind = {}
    for kind in sorted({r.kind for r in ordered}):
        of_kind = [r for r in ordered if r.kind == kind]
        by_kind[kind] = {"requests": len(of_kind), "hit_rate": _hit_rate(of_kind)}

    return {
        "requests": len(ordered),
        "outcomes": {status: sum(1 for r in ordered if r.status == status) for status in (HIT, MISS, BYPASS, ERROR)},
        "retries": sum(r.retries for r in ordered),
        "hit_rate": _hit_rate(ordered),
        "convergence": [
            {"through_request": min(start + window, len(ordered)), "hit_rate": _hit_rate(ordered[start:start + window])}
            for start in range(0, len(ordered), window)
        ],
        "by_kind": by_kind,
        "latency_seconds": {"hit": hit_latency, "miss": miss_latency, "speedup": speedup},
        "tokens": {
            "spent": spent,
            "saved": saved,
            "without_cache": spent + saved,
            "saved_share": saved / (spent + saved) if spent + saved else None,
        },
        "wrong_answers": {
            "count": len(wrong),
            "share_of_hits": len(wrong) / len(hits) if hits else None,
            "unverified_hits": unverified,
            "examples": [
                {"asked": r.family, "similarity": r.similarity, "answer": r.answer[:120]} for r in wrong[:10]
            ],
        },
    }


# ── Markdown ────────────────────────────────────────────────

def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} s"


def render_markdown(summary: dict, *, title: str, settings: dict, saved_myr: float | None = None) -> str:
    """The report as Markdown, ready to paste into the README."""
    out = [f"## {title}", ""]
    out.append(", ".join(f"{key}: `{value}`" for key, value in settings.items()))
    out.append("")

    outcomes, tokens, latency = summary["outcomes"], summary["tokens"], summary["latency_seconds"]
    wrong = summary["wrong_answers"]

    requests = (
        f"{summary['requests']} ({outcomes[HIT]} hits, {outcomes[MISS]} misses, "
        f"{outcomes[BYPASS]} bypassed, {outcomes[ERROR]} errors)"
    )
    tokens_saved = (
        f"**{_pct(tokens['saved_share'])}** "
        f"({tokens['saved']:,} of {tokens['without_cache']:,} without the cache)"
    )
    wrong_answers = (
        f"{wrong['count']} of {outcomes[HIT]} hits ({_pct(wrong['share_of_hits'])}) "
        "served another question's answer"
    )

    rows = [
        ("Requests", requests),
        ("**Hit rate**", f"**{_pct(summary['hit_rate'])}**"),
        ("**LLM tokens saved**", tokens_saved),
    ]
    if saved_myr is not None:
        rows.append(("Money saved", f"RM {saved_myr:.4f}"))
    if latency["hit"] and latency["miss"]:
        hit, miss = latency["hit"], latency["miss"]
        p95_lower = f"(**{_pct(latency['speedup']['p95_reduction'])} lower**)"
        rows += [
            ("Latency p50", f"{_seconds(hit['p50'])} cached vs {_seconds(miss['p50'])} uncached"),
            ("Latency p95", f"{_seconds(hit['p95'])} cached vs {_seconds(miss['p95'])} uncached {p95_lower}"),
            ("Latency p99", f"{_seconds(hit['p99'])} cached vs {_seconds(miss['p99'])} uncached"),
        ]
    rows.append(("Wrong answers", wrong_answers))

    out += ["| Result | |", "|---|---|"]
    out += [f"| {label} | {value} |" for label, value in rows]
    out.append("")

    out += ["**Hit rate as the cache fills** (per window of requests):", "", "| Through request | Hit rate |", "|---|---|"]
    out += [f"| {w['through_request']} | {_pct(w['hit_rate'])} |" for w in summary["convergence"]]
    out.append("")

    out += ["**By kind of question:**", "", "| Kind | Requests | Hit rate |", "|---|---|---|"]
    out += [f"| {kind} | {row['requests']} | {_pct(row['hit_rate'])} |" for kind, row in summary["by_kind"].items()]
    out.append("")
    return "\n".join(out)


def results_to_json(results: list[Result]) -> list[dict]:
    return [asdict(r) for r in sorted(results, key=lambda r: r.index)]
