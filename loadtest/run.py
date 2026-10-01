"""
Run the load test against a running semantic cache.

    # What would a run involve, and roughly what would it cost? (sends nothing)
    uv run python -m loadtest.run --requests 2000 --dry-run

    # Run it against a fresh cache
    uv run python -m loadtest.run --requests 2000 --clear-cache

It sends the workload (see workload.py) at a steady rate, as several
concurrent clients would, and writes a report (see report.py) to
loadtest/results/<timestamp>/: results.json, summary.json and report.md.

RATE LIMITS
    LLM APIs limit requests per minute and per day (free tiers tightly).
    Going over them doesn't just slow a run down, it spoils the results:
    a rate-limited classifier falls back to the default policy.

    --rps           caps how fast requests are sent (every request makes
                    an embedding call).
    --llm-rpm       caps LLM calls per minute. Only misses call the LLM
                    (--llm-calls-per-miss each: the answer and the
                    classifier), so hits don't slow the run down.
    --llm-budget    stops the run cleanly before it makes more than this
                    many LLM calls in total, e.g. to stay inside a daily
                    quota. The report then covers the requests completed.

    A request that still gets 429 / 502 / 503 is retried with
    exponential backoff.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import httpx

from loadtest.report import (
    BYPASS,
    ERROR,
    HIT,
    Result,
    render_markdown,
    results_to_json,
    summarize,
)
from loadtest.workload import SYSTEM_PROMPT, Request, build_workload, describe
from semcache.config import PRICING_TABLE, estimate_cost_myr

RETRY_STATUSES = {429, 502, 503}

# Rough token counts per call, for the --dry-run cost estimate only.
_PROMPT_TOKENS, _COMPLETION_TOKENS = 25, 50
_CLASSIFIER_TOKENS, _EMBEDDING_TOKENS = 260, 12
_CLASSIFIER_MODEL, _EMBEDDING_MODEL = "gemini-3.5-flash-lite", "gemini-embedding-001"


class RateLimiter:
    """Spaces calls evenly: at most `per_second` starts per second."""

    def __init__(self, per_second: float) -> None:
        self._interval = 1 / per_second
        self._next_slot = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = asyncio.get_running_loop().time()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
        await asyncio.sleep(max(0.0, slot - now))


class LLMBudget:
    """
    Keeps the run's LLM calls under a per-minute limit and a total limit.

    Whether a request needs the LLM isn't known until it returns, so each
    request first RESERVES the calls a miss would make, then SETTLES with
    what it really used: a hit gives its reservation back.
    """

    def __init__(
        self,
        per_minute: int | None = None,
        total: int | None = None,
        calls_per_miss: int = 2,
        *,
        clock=time.monotonic,
        sleep=asyncio.sleep,
    ) -> None:
        if per_minute is not None and per_minute < calls_per_miss:
            raise ValueError("--llm-rpm must be at least --llm-calls-per-miss, or no miss could ever be sent.")
        self.calls_per_miss = calls_per_miss
        self.used = 0  # calls actually made so far
        self._per_minute, self._total = per_minute, total
        self._reserved = 0  # calls reserved by requests still in flight
        self._recent: deque[float] = deque()  # when each call of the last minute was reserved
        self._clock, self._sleep = clock, sleep
        self._lock = asyncio.Lock()

    async def reserve(self) -> bool:
        """
        Set aside the calls one miss would make, waiting for the per-minute limit.

        Returns False when the total budget can't cover another miss: stop sending.
        """
        need = self.calls_per_miss
        async with self._lock:
            if self._total is not None and self.used + self._reserved + need > self._total:
                return False
            if self._per_minute is not None:
                while True:
                    now = self._clock()
                    while self._recent and now - self._recent[0] >= 60:
                        self._recent.popleft()
                    excess = len(self._recent) + need - self._per_minute
                    if excess <= 0:
                        break
                    # Wait until enough of the oldest calls are a minute old.
                    await self._sleep(max(60 - (now - self._recent[excess - 1]), 0.01))
                self._recent.extend([now] * need)
            self._reserved += need
            return True

    def settle(self, calls_made: int) -> None:
        """Replace a reservation with the calls really made (0 for a hit)."""
        self._reserved -= self.calls_per_miss
        self.used += calls_made
        for _ in range(self.calls_per_miss - calls_made):
            if self._recent:
                self._recent.pop()  # give the unused calls back to this minute

    def calls_for(self, status: str) -> int:
        """LLM calls a finished request made: none for a hit, the answer only for a bypass."""
        if status == HIT:
            return 0
        if status == BYPASS:
            return 1
        return self.calls_per_miss  # a miss; for an error, assume the worst


def _backoff_seconds(response: httpx.Response | None, attempt: int) -> float:
    """How long to wait before retry number `attempt` (1-based): Retry-After, else 2, 4, 8… capped at 60."""
    if response is not None:
        retry_after = response.headers.get("retry-after", "")
        if retry_after.isdigit():
            return min(float(retry_after), 120.0)
    return min(2.0**attempt, 60.0)


async def send(
    client: httpx.AsyncClient, request: Request, model: str, *, max_retries: int = 5, sleep=asyncio.sleep
) -> Result:
    """Send one request, retrying while the server says it's overloaded or rate-limited."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.text},
        ],
    }
    base = {"index": request.index, "family": request.family, "kind": request.kind}
    error = "no attempt made"
    response: httpx.Response | None = None  # the previous attempt's response, if any
    for attempt in range(max_retries + 1):
        if attempt:
            await sleep(_backoff_seconds(response, attempt))
        response = None
        started = time.perf_counter()
        try:
            response = await client.post("/v1/chat/completions", json=body)
        except httpx.HTTPError as exc:
            error = f"{type(exc).__name__}: {exc}"
            continue
        latency = time.perf_counter() - started

        if response.status_code == 200:
            data = response.json()
            usage = data.get("usage") or {}
            similarity = response.headers.get("x-cache-similarity")
            return Result(
                **base,
                status=response.headers.get("x-cache-status", BYPASS),
                latency=latency,
                similarity=float(similarity) if similarity else None,
                answer=data["choices"][0]["message"]["content"],
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                retries=attempt,
            )
        error = f"HTTP {response.status_code}: {response.text[:200]}"
        if response.status_code not in RETRY_STATUSES:
            break
    return Result(**base, status=ERROR, latency=0.0, retries=attempt, error=error)


async def run(
    client: httpx.AsyncClient,
    requests: list[Request],
    model: str,
    *,
    concurrency: int,
    rps: float,
    budget: LLMBudget | None = None,
    progress_every: int = 100,
) -> list[Result]:
    """
    Send the requests, `concurrency` at a time, no faster than `rps` per second.

    With a `budget`, LLM calls also stay under its limits; if its total
    runs out, the run stops early and fewer results come back than requests.
    """
    limiter = RateLimiter(rps)
    out_of_budget = asyncio.Event()
    queue: asyncio.Queue[Request] = asyncio.Queue()
    for request in requests:
        queue.put_nowait(request)
    results: list[Result] = []

    async def worker() -> None:
        while not out_of_budget.is_set():
            try:
                request = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if budget is not None and not await budget.reserve():
                out_of_budget.set()
                return
            await limiter.wait()
            result = await send(client, request, model)
            if budget is not None:
                budget.settle(budget.calls_for(result.status))
            results.append(result)
            if progress_every and len(results) % progress_every == 0:
                hits = sum(1 for r in results if r.status == "HIT")
                errors = sum(1 for r in results if r.status == ERROR)
                print(f"  {len(results):>5}/{len(requests)}  hits so far: {hits}  errors: {errors}", flush=True)

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return results


def estimate(workload: dict, model: str) -> dict:
    """A rough cost range for a run (see the token assumptions at the top of this file)."""
    def cost(llm_calls: int) -> float | None:
        parts = [
            estimate_cost_myr(model, llm_calls * _PROMPT_TOKENS, llm_calls * _COMPLETION_TOKENS),
            estimate_cost_myr(_CLASSIFIER_MODEL, llm_calls * _CLASSIFIER_TOKENS),
            # The app remembers embeddings, so only each distinct wording costs an embedding call.
            estimate_cost_myr(_EMBEDDING_MODEL, workload["distinct_wordings"] * _EMBEDDING_TOKENS),
        ]
        return None if None in parts else sum(parts)

    low = workload["minimum_llm_calls"]
    high = min(workload["requests"], round(low * 1.6))  # rewordings that miss, expired entries
    return {
        "llm_calls": {"at_least": low, "likely_up_to": high},
        "embedding_calls": workload["distinct_wordings"],
        "cost_myr": {"at_least": cost(low), "likely_up_to": cost(high)},
        "without_cache_myr": cost(workload["requests"]),
        "usd_to_myr": PRICING_TABLE["usd_to_myr"],
    }


async def _main(args: argparse.Namespace) -> None:
    requests = build_workload(args.requests, seed=args.seed)
    workload = describe(requests)
    print("Workload:", json.dumps(workload, indent=2))

    if args.dry_run:
        print("Estimate:", json.dumps(estimate(workload, args.model), indent=2))
        minutes = args.requests / args.rps / 60
        print(f"At {args.rps} requests/second this takes about {minutes:.0f} minutes (more if rate-limited).")
        return

    headers = {"Authorization": f"Bearer {args.admin_token}"} if args.admin_token else {}
    async with httpx.AsyncClient(base_url=args.url, timeout=120, headers=headers) as client:
        if args.clear_cache:
            cleared = await client.post("/v1/cache/invalidate", json={"all": True})
            cleared.raise_for_status()
            print(f"Cleared {cleared.json()['deleted']} cached entries.")
        else:
            entries = (await client.get("/v1/cache/stats")).json().get("total_entries", 0)
            if entries:
                print(f"Note: the cache already holds {entries} entries, so early requests may hit. "
                      "Use --clear-cache for a cold start.")

        budget = None
        if args.llm_rpm or args.llm_budget:
            budget = LLMBudget(args.llm_rpm, args.llm_budget, args.llm_calls_per_miss)

        before = (await client.get("/v1/analytics")).json()
        started = time.time()
        results = await run(
            client, requests, args.model, concurrency=args.concurrency, rps=args.rps, budget=budget
        )
        elapsed = time.time() - started
        after = (await client.get("/v1/analytics")).json()

    summary = summarize(results)
    saved_myr = after["savings"]["saved_myr"] - before["savings"]["saved_myr"]
    settings = {
        "model": args.model, "requests": len(results), "concurrency": args.concurrency,
        "rps": args.rps, "seed": args.seed, "duration": f"{elapsed / 60:.1f} min",
    }
    if budget is not None:
        settings["llm calls"] = budget.used
        if args.llm_rpm:
            settings["llm rpm limit"] = args.llm_rpm
    if len(results) < len(requests):
        print(f"Stopped after {len(results)} of {len(requests)} requests: "
              f"the LLM call budget ({args.llm_budget}) was reached.")
    summary["settings"], summary["saved_myr"], summary["workload"] = settings, saved_myr, workload

    out = Path(args.out) / datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results_to_json(results), indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown = render_markdown(summary, title=f"Load test: {len(results)} requests", settings=settings, saved_myr=saved_myr)
    (out / "report.md").write_text(markdown, encoding="utf-8")

    print()
    print(markdown)
    print(f"Saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test the semantic cache.")
    parser.add_argument("--url", default="http://localhost:8000", help="Where the cache is running.")
    parser.add_argument("--model", default="gemini-3.5-flash-lite", help="Model to request.")
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=4, help="Requests in flight at once.")
    parser.add_argument("--rps", type=float, default=2.0, help="Most requests started per second.")
    parser.add_argument("--llm-rpm", type=int, default=None, help="Most LLM calls per minute (your rate limit).")
    parser.add_argument("--llm-budget", type=int, default=None, help="Stop before making more LLM calls than this.")
    parser.add_argument("--llm-calls-per-miss", type=int, default=2,
                        help="LLM calls a miss makes: the answer and the classifier.")
    parser.add_argument("--seed", type=int, default=7, help="Same seed, same workload.")
    parser.add_argument("--out", default="loadtest/results", help="Where to write the report.")
    parser.add_argument("--clear-cache", action="store_true", help="Empty the cache first (a cold start).")
    parser.add_argument("--admin-token", default=None, help="If the server sets ADMIN_TOKEN.")
    parser.add_argument("--dry-run", action="store_true", help="Describe the workload and estimate cost; send nothing.")
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
