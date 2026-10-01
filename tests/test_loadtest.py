"""
Tests for the load test tool: the workload, the runner, and the report.

The numbers in the report become the project's headline, so the arithmetic
behind them is tested here. The server is faked with httpx.MockTransport.
"""

import json
import time
from unittest.mock import AsyncMock

import httpx
import pytest

from loadtest.report import (
    BYPASS,
    ERROR,
    HIT,
    MISS,
    Result,
    find_wrong_hits,
    percentile,
    render_markdown,
    summarize,
)
from loadtest.run import LLMBudget, RateLimiter, estimate, run, send
from loadtest.workload import (
    SYSTEM_PROMPT,
    UNCACHED_KINDS,
    Request,
    build_workload,
    describe,
    families,
)

# ── Workload ────────────────────────────────────────────────

class TestWorkload:
    def test_same_seed_same_workload(self):
        assert build_workload(300, seed=7) == build_workload(300, seed=7)
        assert build_workload(300, seed=7) != build_workload(300, seed=8)

    def test_size_and_order(self):
        requests = build_workload(250)
        assert [r.index for r in requests] == list(range(250))

    def test_families_are_distinct_questions_with_several_wordings(self):
        all_families = families()
        assert len({f.id for f in all_families}) == len(all_families) > 100
        assert all(len(f.wordings) >= 3 for f in all_families)
        # No wording belongs to two families: a hit across families is always wrong.
        wordings = [w for f in all_families for w in f.wordings]
        assert len(wordings) == len(set(wordings))

    def test_popular_questions_repeat_and_are_reworded(self):
        requests = build_workload(1000)
        per_family: dict[str, list[str]] = {}
        for r in requests:
            per_family.setdefault(r.family, []).append(r.text)
        busiest = max(per_family.values(), key=len)

        assert len(busiest) > 50  # a long tail: a few questions dominate
        assert len(set(busiest)) > 1  # and they arrive in different wordings
        assert min(len(texts) for texts in per_family.values()) <= 3  # while others are rare

    def test_describe_counts_the_calls_no_cache_can_avoid(self):
        requests = [
            Request(0, "capital:Japan", "capital", "What is the capital of Japan?"),
            Request(1, "capital:Japan", "capital", "Which city is the capital of Japan?"),  # could hit
            Request(2, "creative:rain", "creative", "Write a haiku about rain."),
            Request(3, "creative:rain", "creative", "Write a haiku about rain."),  # never cached
        ]
        info = describe(requests)

        assert info["families"] == 2
        assert info["distinct_wordings"] == 3
        assert info["minimum_llm_calls"] == 3
        assert info["best_possible_hit_rate"] == 0.25
        assert "creative" in UNCACHED_KINDS


# ── Report ──────────────────────────────────────────────────

def _result(index, status, *, family="f1", kind="capital", latency=1.0, answer="A", tokens=(10, 20), **kwargs) -> Result:
    return Result(index=index, family=family, kind=kind, status=status, latency=latency, answer=answer,
                  prompt_tokens=tokens[0], completion_tokens=tokens[1], **kwargs)


class TestPercentile:
    def test_nearest_rank(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        assert percentile(values, 50) == 5.0
        assert percentile(values, 95) == 10.0
        assert percentile(values, 99) == 10.0
        assert percentile([3.0, 1.0, 2.0], 50) == 2.0  # order doesn't matter
        assert percentile([], 95) is None


class TestSummarize:
    def test_hit_rate_tokens_and_latency(self):
        results = [
            _result(0, MISS, latency=2.0),
            _result(1, HIT, latency=0.4),
            _result(2, HIT, latency=0.6),
            _result(3, MISS, family="f2", answer="B", latency=4.0),
            _result(4, BYPASS, latency=3.0),
            _result(5, ERROR, latency=0.0, tokens=(0, 0), error="HTTP 500"),
        ]
        summary = summarize(results)

        assert summary["outcomes"] == {HIT: 2, MISS: 2, BYPASS: 1, ERROR: 1}
        assert summary["hit_rate"] == 0.5  # hits / (hits + misses); bypasses and errors excluded
        # Each call used 30 tokens: 2 misses + 1 bypass spent; 2 hits saved.
        assert summary["tokens"] == {"spent": 90, "saved": 60, "without_cache": 150, "saved_share": 0.4}
        assert summary["latency_seconds"]["hit"]["p95"] == 0.6
        assert summary["latency_seconds"]["miss"]["p95"] == 4.0
        assert summary["latency_seconds"]["speedup"]["p95_reduction"] == pytest.approx(1 - 0.6 / 4.0)

    def test_convergence_windows_follow_request_order(self):
        # First 4 requests all miss, the next 4 all hit, given out of order.
        results = [_result(i, MISS if i < 4 else HIT) for i in range(8)]
        summary = summarize(list(reversed(results)), window=4)

        assert summary["convergence"] == [
            {"through_request": 4, "hit_rate": 0.0},
            {"through_request": 8, "hit_rate": 1.0},
        ]

    def test_by_kind(self):
        results = [
            _result(0, MISS, kind="capital"), _result(1, HIT, kind="capital"),
            _result(2, MISS, kind="creative", family="c1", answer="poem 1"),
            _result(3, MISS, kind="creative", family="c1", answer="poem 2"),
        ]
        by_kind = summarize(results)["by_kind"]

        assert by_kind["capital"] == {"requests": 2, "hit_rate": 0.5}
        assert by_kind["creative"] == {"requests": 2, "hit_rate": 0.0}

    def test_empty_run(self):
        summary = summarize([])
        assert summary["hit_rate"] is None
        assert summary["tokens"]["saved_share"] is None
        assert summary["latency_seconds"]["speedup"] is None


class TestWrongAnswers:
    def test_hit_with_another_familys_answer_is_wrong(self):
        results = [
            _result(0, MISS, family="capital:Austria", answer="Vienna."),
            _result(1, MISS, family="capital:Australia", answer="Canberra."),
            _result(2, HIT, family="capital:Austria", answer="Vienna."),  # right
            _result(3, HIT, family="capital:Australia", answer="Vienna.", similarity=0.96),  # WRONG
            _result(4, HIT, family="capital:Austria", answer="From an older run."),  # can't tell
        ]
        wrong, unverified = find_wrong_hits(results)

        assert [r.index for r in wrong] == [3]
        assert unverified == 1
        report = summarize(results)["wrong_answers"]
        assert report["count"] == 1
        assert report["share_of_hits"] == pytest.approx(1 / 3)
        assert report["examples"][0]["asked"] == "capital:Australia"

    def test_order_of_arrival_does_not_matter(self):
        """With concurrency, a hit can be recorded before the miss it came from."""
        results = [
            _result(1, HIT, family="f1", answer="A"),
            _result(0, MISS, family="f1", answer="A"),
        ]
        assert find_wrong_hits(results) == ([], 0)

    def test_the_matched_question_decides_not_the_answer_text(self):
        """Two reviews, one identical answer: only the matched question tells them apart."""
        battery = "Review: 'The battery died after two days.'"
        stopped = "Review: 'It stopped working within a week.'"
        negative = "This review is negative."
        results = [
            _result(0, MISS, family="sentiment:battery", prompt=battery, answer=negative),
            _result(1, MISS, family="sentiment:stopped", prompt=stopped, answer=negative),
            # Asked about the battery, answered from the battery entry: right.
            _result(2, HIT, family="sentiment:battery", prompt=battery, answer=negative, matched_prompt=battery),
            # Asked about the battery, answered from the OTHER review: wrong,
            # even though the answer happens to be correct.
            _result(3, HIT, family="sentiment:battery", prompt=battery, answer=negative,
                    matched_prompt=stopped, similarity=0.908),
        ]
        wrong, unverified = find_wrong_hits(results)

        assert [r.index for r in wrong] == [3]
        assert unverified == 0
        (example,) = summarize(results)["wrong_answers"]["examples"]
        assert (example["asked"], example["matched"]) == (battery, stopped)

    def test_without_the_match_a_shared_answer_text_cant_be_checked(self):
        """The old check guessed from the answer text, and counted these as wrong."""
        results = [
            _result(0, MISS, family="sentiment:a", answer="This review is negative."),
            _result(1, MISS, family="sentiment:b", answer="This review is negative."),
            _result(2, HIT, family="sentiment:a", answer="This review is negative."),
        ]
        assert find_wrong_hits(results) == ([], 1)

    def test_a_match_cached_before_the_run_cant_be_checked(self):
        results = [_result(0, HIT, family="f1", prompt="q", matched_prompt="asked in an earlier run")]
        assert find_wrong_hits(results) == ([], 1)

    def test_markdown_lists_the_wrong_matches(self):
        results = [
            _result(0, MISS, family="a", prompt="Question A"),
            _result(1, HIT, family="b", prompt="Question B", matched_prompt="Question A", similarity=0.9081),
        ]
        markdown = render_markdown(summarize(results), title="t", settings={})

        assert "| Wrong answers | 1 of 1 hits (100.0%) were answered from a different question |" in markdown
        assert "| Question B | Question A | 0.908 | A |" in markdown


def test_markdown_report_has_the_headline_rows():
    results = [_result(0, MISS, latency=2.0)] + [_result(i, HIT, latency=0.5) for i in range(1, 4)]
    markdown = render_markdown(summarize(results), title="Load test", settings={"model": "m"}, saved_myr=0.1234)

    assert "## Load test" in markdown
    assert "| **Hit rate** | **75.0%** |" in markdown
    assert "| **LLM tokens saved** | **75.0%** (90 of 120 without the cache) |" in markdown
    assert "| Money saved | RM 0.1234 |" in markdown
    assert "0.50 s cached vs 2.00 s uncached (**75.0% lower**)" in markdown
    assert "| Wrong answers | 0 of 3 hits (0.0%)" in markdown


# ── Runner ──────────────────────────────────────────────────

REQUEST = Request(index=0, family="capital:Japan", kind="capital", text="What is the capital of Japan?")


def _ok(status="MISS", similarity=None, lookup_id=None) -> httpx.Response:
    headers = {"x-cache-status": status}
    if lookup_id:
        headers["x-cache-lookup-id"] = lookup_id
    if similarity:
        headers["x-cache-similarity"] = similarity
    return httpx.Response(200, headers=headers, json={
        "choices": [{"message": {"content": "Tokyo."}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    })


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url="http://cache.test", transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_attach_matches_asks_the_server_what_each_hit_matched():
    from loadtest.run import attach_matches

    def handler(request):
        lookup_id = request.url.path.rsplit("/", 1)[-1]
        if lookup_id == "expired":
            return httpx.Response(404)
        return httpx.Response(200, json={"lookup": {"candidate_prompt": f"cached question for {lookup_id}"}})

    results = [
        _result(0, MISS, lookup_id="m0"),
        _result(1, HIT, lookup_id="h1"),
        _result(2, HIT, lookup_id="expired"),
        _result(3, HIT),  # no lookup id (e.g. lookup logging off)
    ]
    found = await attach_matches(_client(handler), results)

    assert found == 1
    assert [r.matched_prompt for r in results] == [None, "cached question for h1", None, None]


class TestSend:
    @pytest.mark.asyncio
    async def test_successful_request(self):
        sent = {}

        def handler(request):
            sent.update(json.loads(request.content))
            return _ok("HIT", "0.9731", lookup_id="abc123")

        result = await send(_client(handler), REQUEST, "gemini-3.5-flash-lite")

        assert sent["model"] == "gemini-3.5-flash-lite"
        assert sent["messages"] == [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is the capital of Japan?"},
        ]
        assert (result.status, result.similarity, result.answer) == (HIT, 0.9731, "Tokyo.")
        assert (result.prompt_tokens, result.completion_tokens, result.retries) == (12, 3, 0)
        assert result.family == "capital:Japan"
        assert result.prompt == "What is the capital of Japan?"
        assert result.lookup_id == "abc123"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("busy_status", [429, 502, 503])
    async def test_overloaded_server_is_retried_with_backoff(self, busy_status):
        responses = iter([httpx.Response(busy_status), httpx.Response(busy_status), _ok()])
        sleep = AsyncMock()

        result = await send(_client(lambda request: next(responses)), REQUEST, "m", sleep=sleep)

        assert (result.status, result.retries) == (MISS, 2)
        assert [call.args[0] for call in sleep.await_args_list] == [2.0, 4.0]  # exponential

    @pytest.mark.asyncio
    async def test_retry_after_header_is_honoured(self):
        responses = iter([httpx.Response(429, headers={"retry-after": "7"}), _ok()])
        sleep = AsyncMock()

        await send(_client(lambda request: next(responses)), REQUEST, "m", sleep=sleep)

        sleep.assert_awaited_once_with(7.0)

    @pytest.mark.asyncio
    async def test_other_errors_are_not_retried(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(400, json={"detail": "bad request"})

        result = await send(_client(handler), REQUEST, "m", sleep=AsyncMock())

        assert result.status == ERROR
        assert "HTTP 400" in result.error
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self):
        result = await send(
            _client(lambda request: httpx.Response(503)), REQUEST, "m", max_retries=2, sleep=AsyncMock()
        )
        assert (result.status, result.retries) == (ERROR, 2)

    @pytest.mark.asyncio
    async def test_network_error_is_retried(self):
        attempts = []

        def handler(request):
            attempts.append(1)
            if len(attempts) == 1:
                raise httpx.ConnectError("refused")
            return _ok()

        result = await send(_client(handler), REQUEST, "m", sleep=AsyncMock())

        assert (result.status, result.retries) == (MISS, 1)


@pytest.mark.asyncio
async def test_run_sends_every_request_once():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content)["messages"][1]["content"])
        return _ok()

    requests = build_workload(40)
    results = await run(_client(handler), requests, "m", concurrency=5, rps=1000, progress_every=0)

    assert sorted(r.index for r in results) == list(range(40))
    assert sorted(seen) == sorted(r.text for r in requests)


@pytest.mark.asyncio
async def test_rate_limiter_spaces_out_calls():
    limiter = RateLimiter(per_second=50)  # one every 20 ms
    started = time.perf_counter()
    for _ in range(6):
        await limiter.wait()
    elapsed = time.perf_counter() - started

    assert elapsed >= 0.08  # 5 gaps of 20 ms, with slack for timer resolution


def test_estimate_gives_a_cost_range():
    workload = describe(build_workload(500))
    est = estimate(workload, "gemini-3.5-flash-lite")

    assert est["llm_calls"]["at_least"] == workload["minimum_llm_calls"]
    assert est["llm_calls"]["at_least"] <= est["llm_calls"]["likely_up_to"] <= 500
    assert 0 < est["cost_myr"]["at_least"] <= est["cost_myr"]["likely_up_to"] < est["without_cache_myr"]
    # An unpriced model can't be estimated; say so instead of guessing.
    assert estimate(workload, "some-unknown-model")["cost_myr"]["at_least"] is None


# ── LLM call budget (rate limits and daily quotas) ──────────

class FakeTime:
    """A clock that only moves when something sleeps, so budget tests run instantly."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class TestLLMBudget:
    @pytest.mark.asyncio
    async def test_misses_are_held_to_the_per_minute_limit(self):
        fake = FakeTime()
        budget = LLMBudget(per_minute=6, calls_per_miss=2, clock=fake.clock, sleep=fake.sleep)

        for _ in range(3):  # 3 misses = 6 calls: exactly the limit
            assert await budget.reserve()
            budget.settle(2)
        assert fake.slept == []

        assert await budget.reserve()  # a 4th must wait for the first to be a minute old
        assert fake.slept == [60.0]
        assert budget.used == 6

    @pytest.mark.asyncio
    async def test_hits_give_their_reservation_back(self):
        fake = FakeTime()
        budget = LLMBudget(per_minute=2, calls_per_miss=2, clock=fake.clock, sleep=fake.sleep)

        for _ in range(50):  # hits never call the LLM, so they never wait
            assert await budget.reserve()
            budget.settle(budget.calls_for(HIT))

        assert fake.slept == []
        assert budget.used == 0

    @pytest.mark.asyncio
    async def test_total_budget_stops_before_it_is_exceeded(self):
        budget = LLMBudget(total=5, calls_per_miss=2)

        assert await budget.reserve()
        budget.settle(2)
        assert await budget.reserve()
        budget.settle(2)
        # 4 used: another miss would make 6, over the budget of 5.
        assert not await budget.reserve()
        assert budget.used == 4

    def test_calls_by_outcome(self):
        budget = LLMBudget(calls_per_miss=2)
        assert [budget.calls_for(s) for s in (HIT, MISS, BYPASS, ERROR)] == [0, 2, 1, 2]

    def test_a_limit_below_one_miss_is_rejected(self):
        with pytest.raises(ValueError, match="llm-rpm"):
            LLMBudget(per_minute=1, calls_per_miss=2)


@pytest.mark.asyncio
async def test_run_stops_cleanly_when_the_budget_runs_out():
    """Every request misses (2 calls each); a budget of 10 allows exactly 5."""
    requests = build_workload(40)
    budget = LLMBudget(total=10, calls_per_miss=2)

    results = await run(
        _client(lambda request: _ok("MISS")), requests, "m", concurrency=1, rps=1000, budget=budget, progress_every=0
    )

    assert len(results) == 5
    assert budget.used == 10


@pytest.mark.asyncio
async def test_hits_do_not_use_the_budget_in_a_run():
    requests = build_workload(40)
    budget = LLMBudget(total=10, calls_per_miss=2)

    results = await run(
        _client(lambda request: _ok("HIT", "0.99")), requests, "m", concurrency=3, rps=1000, budget=budget, progress_every=0
    )

    assert len(results) == 40
    assert budget.used == 0
