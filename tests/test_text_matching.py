"""
The number check behind cache matching (cache/text.py): a cached answer is
reused only when its question has the same numbers as the one asked.
"""

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Response

from semcache.api.chat import chat_completions
from semcache.cache.classifier import ClassifierResult, IntentClassifier
from semcache.cache.engine import CacheEngine
from semcache.cache.lookup_log import NEAR_MISS, LookupEvent
from semcache.cache.policy import TASK_POLICIES
from semcache.cache.store.base import CacheEntry
from semcache.cache.text import numbers_in, same_numbers
from semcache.cache.tuning import MIN_LABELS, could_match, learn_thresholds, tune
from semcache.metrics import metrics
from semcache.providers.base import LLMProvider
from semcache.schemas import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)
from tests.conftest import InMemoryLookupLog, InMemoryVectorStore, MockEmbedder


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


# ── Comparing numbers ───────────────────────────────────────

@pytest.mark.parametrize(("question", "other"), [
    ("What is 6906006 * 2032032?", "calculate 6906006 * 2032032"),
    ("if there were 6906006 apple * 2032032 how many apples do i have", "whats 6906006 * 2032032"),
    ("What is 2 + 3?", "What is 3 + 2?"),  # order doesn't matter
    ("What is 1,000,000 in words?", "What is 1000000 in words?"),
    ("Convert 2.50 dollars", "Convert 2.5 dollars"),
    ("Price of 12,500 units", "Price of 12500 units"),
    ("Half of 1,5 litres", "Half of 1.5 litres"),  # decimal comma
    ("Top 10 films of 2005.", "the 10 best films in 2005"),
    ("What is the capital of Malaysia?", "Which city is Malaysia's capital?"),  # no numbers at all
    # Numbers written as words
    ("what's fifteen percent of 200?", "What is 15% of 200?"),
    ("Summarise chapter 3", "Summarise chapter three"),
    ("What is twenty-five times four?", "what is 25 * 4"),
    ("a loan of two thousand five hundred", "a loan of 2,500"),
    ("one hundred and five divided by 5", "105 / 5"),
    ("population of 5 million", "population of 5,000,000"),
    ("cost of 1.5 million units", "cost of 1500000 units"),
    ("Explain two-factor authentication", "Explain 2-factor authentication"),
    # ... and in Malay
    ("berapa lima belas peratus daripada 200", "berapa 15% daripada 200"),
    ("dua puluh lima darab empat", "25 darab 4"),
    ("pinjaman dua ribu seratus ringgit", "pinjaman 2100 ringgit"),
    ("seribu sembilan ratus sembilan puluh", "1990"),
    ("gaji 15 ribu", "gaji 15000"),
    ("fifteen percent of 200", "lima belas peratus daripada 200"),  # across the two languages
    # "one" / "satu" alone is usually not a count
    ("Which one is bigger, 5 or 7?", "Which is bigger, 5 or 7?"),
    ("Apakah salah satu sebab banjir?", "Apakah sebab banjir?"),
])
def test_same_numbers(question, other):
    assert same_numbers(question, other)


@pytest.mark.parametrize(("question", "other"), [
    ("What is 6906006 * 2032032?", "What is 6906006 * 2032033?"),
    ("Ringgit to US dollar in 2005", "Ringgit to US dollar in 2015"),
    ("Value change since 1990 to 2026", "Value change in 2005"),
    ("What is 2 + 2?", "What is 2 + 2 + 2?"),  # how often a number appears counts
    ("Python 3.12 release notes", "Python 3.13 release notes"),
    ("Python 3.12.1 changes", "Python 3.12.2 changes"),
    ("What is 1.5 + 1?", "What is 15 + 1?"),
    ("Top 10 films", "Top films"),  # one has a number, the other doesn't
    # Numbers written as words
    ("What is fifty percent of 200?", "What is fifteen percent of 200?"),
    ("What is 50% of 200?", "what's fifteen percent of 200?"),
    ("What is two plus two?", "What is two plus three?"),
    ("loan of two thousand", "loan of two hundred"),
    ("dua puluh lima darab empat", "dua puluh enam darab empat"),
    ("Is it between one hundred and two hundred?", "Is it between 102 and 100?"),  # "and" between two numbers
    ("100 and 5", "105"),  # digits are never joined by "and"
    ("Summarise chapter 3", "Summarise the third chapter"),  # ordinals aren't recognised: no match
])
def test_different_numbers(question, other):
    assert not same_numbers(question, other)


def test_numbers_are_read_by_value():
    found = numbers_in("In 2005 it cost 1,234.50, not 1.234,50 or 007.")
    assert sorted(found.elements()) == [7, 1234.5, 1234.5, 2005]


@pytest.mark.parametrize(("text", "numbers"), [
    ("twenty one", [21]),
    ("ninety-nine", [99]),
    ("one, two, three", [2, 3]),  # a lone "one" isn't counted
    ("two three", [2, 3]),  # not a number together: two numbers
    ("ten five", [5, 10]),
    ("fifteen hundred", [1500]),
    ("a hundred percent", [100]),
    ("one hundred and one", [101]),
    ("one million two hundred thousand and five", [1200005]),
    ("three hundred thousand", [300000]),
    ("hundreds of thousands of people", []),  # plurals are not numbers
    ("someone, anyone, no one", []),
    ("sebelas, sepuluh ribu, tiga juta", [11, 10000, 3000000]),
    ("seratus dua puluh tiga", [123]),
    ("dua ratus lima belas", [215]),
    ("7-Eleven", [7, 11]),
    ("FIFTEEN", [15]),
])
def test_number_words_are_read_by_value(text, numbers):
    assert sorted(numbers_in(text).elements()) == numbers


# ── In the engine ───────────────────────────────────────────

def _engine(embedder=None) -> tuple[CacheEngine, InMemoryVectorStore, InMemoryLookupLog]:
    store, log = InMemoryVectorStore(), InMemoryLookupLog()
    return CacheEngine(embedder=embedder or MockEmbedder(), store=store, lookup_log=log), store, log


async def _seed(engine, store, embedder, entries) -> None:
    """Store cached questions at chosen similarities to the query vector [1, 0]."""
    namespace = (await engine.lookup(prompt="setup", model="m")).namespace
    for prompt, similarity in entries:
        vector = [similarity, math.sqrt(1 - similarity**2)]
        await store.store(vector, CacheEntry(
            prompt=prompt, response=f"answer to: {prompt}", model="m", namespace=namespace,
            required_similarity=0.95, intent="factual",
        ))


@pytest.mark.asyncio
async def test_the_question_is_embedded_exactly_as_asked():
    """Cleaning it first was measured to cost hits (docs/loadtest-2026-10-06.md)."""
    embedder = MockEmbedder()
    engine, _, _ = _engine(embedder)

    with patch.object(embedder, "embed", wraps=embedder.embed) as embed:
        await engine.lookup(prompt="Hi, please tell me the capital of Malaysia?", model="m")

    embed.assert_awaited_once_with("Hi, please tell me the capital of Malaysia?")


@pytest.mark.asyncio
async def test_a_question_with_different_numbers_is_never_served():
    embedder = MockEmbedder(dims=2)
    engine, store, log = _engine(embedder)
    await _seed(engine, store, embedder, [("What is 6906006 * 2032032?", 0.99)])

    with patch.object(embedder, "embed", return_value=[1.0, 0.0]):
        result = await engine.lookup(prompt="What is 6906006 * 2032033?", model="m")

    assert not result.hit  # 0.99 similar, but a different sum
    assert metrics.cache_number_blocks == 1
    assert metrics.cache_near_misses == 0  # and it's not a near miss to be labelled either
    latest = (await log.recent())[0]
    assert latest.outcome == "miss" and latest.candidate_prompt is None


@pytest.mark.asyncio
async def test_a_number_written_as_a_word_still_hits():
    embedder = MockEmbedder(dims=2)
    engine, store, _ = _engine(embedder)
    await _seed(engine, store, embedder, [("What is 50% of 200?", 0.98), ("What is 15% of 200?", 0.96)])

    with patch.object(embedder, "embed", return_value=[1.0, 0.0]):
        result = await engine.lookup(prompt="what's fifteen percent of 200?", model="m")

    assert result.hit and result.entry.prompt == "What is 15% of 200?"
    assert metrics.cache_number_blocks == 1  # the 50% answer would have been served


@pytest.mark.asyncio
async def test_the_entry_with_the_right_numbers_is_served_even_if_less_similar():
    embedder = MockEmbedder(dims=2)
    engine, store, _ = _engine(embedder)
    await _seed(engine, store, embedder, [
        ("Ringgit to US dollar rate in 2015", 0.99),  # closest, wrong year
        ("What was the ringgit to US dollar rate in 2005?", 0.96),
    ])

    with patch.object(embedder, "embed", return_value=[1.0, 0.0]):
        result = await engine.lookup(prompt="Ringgit to US dollar rate in 2005", model="m")

    assert result.hit
    assert result.entry.prompt == "What was the ringgit to US dollar rate in 2005?"
    assert result.similarity == pytest.approx(0.96)
    assert metrics.cache_number_blocks == 1  # the 2015 answer would have been served


@pytest.mark.asyncio
async def test_same_numbers_hit_as_before_and_nothing_is_counted_as_blocked():
    embedder = MockEmbedder(dims=2)
    engine, store, _ = _engine(embedder)
    await _seed(engine, store, embedder, [("whats 6906006 * 2032032", 0.97), ("What is 5 * 5?", 0.91)])

    with patch.object(embedder, "embed", return_value=[1.0, 0.0]):
        result = await engine.lookup(prompt="calculate 6906006 * 2032032", model="m")

    assert result.hit and result.entry.prompt == "whats 6906006 * 2032032"
    # The 5 * 5 entry was below its threshold anyway: dropping it prevented nothing.
    assert metrics.cache_number_blocks == 0


@pytest.mark.asyncio
async def test_a_near_miss_with_the_same_numbers_is_still_logged_for_labelling():
    embedder = MockEmbedder(dims=2)
    engine, store, log = _engine(embedder)
    await _seed(engine, store, embedder, [("whats 6906006 * 2032032", 0.92), ("whats 7 * 8", 0.94)])

    with patch.object(embedder, "embed", return_value=[1.0, 0.0]):
        result = await engine.lookup(
            prompt="if there were 6906006 apple * 2032032 how many apples do i have", model="m"
        )

    assert not result.hit
    latest = (await log.recent())[0]
    # The closest candidate that could match, not the closer one about 7 * 8.
    assert latest.outcome == NEAR_MISS and latest.candidate_prompt == "whats 6906006 * 2032032"
    assert metrics.cache_number_blocks == 0  # nothing would have been served


@pytest.mark.asyncio
async def test_through_the_api_the_llm_gets_the_question_exactly_as_asked():
    question = "Hi! Please tell me, What is 12 * 12?"
    provider = MagicMock(spec=LLMProvider)
    provider.generate = AsyncMock(side_effect=lambda request: ChatCompletionResponse(
        id="id", created=0, model=request.model,
        choices=[ChatCompletionChoice(message=ChatCompletionChoiceMessage(content="144"))],
    ))
    classifier = MagicMock(spec=IntentClassifier)
    classifier.classify_safe = AsyncMock(return_value=ClassifierResult(
        policy=TASK_POLICIES["factual"], tokens=5, is_fallback=False, intent="factual",
    ))
    engine, store, _ = _engine()

    def ask(text):
        request = ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content=text)])
        return chat_completions(request, Response(), engine, provider, classifier)

    await ask(question)
    served = await ask(question)  # asked again: a hit
    await ask("what is 12 * 13")  # a different question: the LLM is asked again

    assert provider.generate.await_args_list[0].args[0].messages[0].content == question
    assert classifier.classify_safe.await_args_list[0].args[0] == question
    assert served.choices[0].message.content == "144"
    assert provider.generate.await_count == 2
    assert [entry.prompt for _, entry in store._entries] == [question, "what is 12 * 13"]


# ── Tuning ignores pairs that can no longer match ───────────

def _event(prompt, candidate, similarity, good_match=None) -> LookupEvent:
    return LookupEvent(
        prompt=prompt, model="m", outcome=NEAR_MISS, similarity=similarity, required_similarity=0.95,
        intent="factual", candidate_prompt=candidate, good_match=good_match,
    )


def test_could_match():
    assert could_match(_event("what is 2 * 2", "calculate 2 * 2", 0.93))
    assert not could_match(_event("what is 2 * 2", "what is 2 * 3", 0.99))
    assert not could_match(LookupEvent(prompt="q", model="m", outcome="miss"))  # no candidate


def test_tuner_does_not_count_number_mismatches_as_would_be_hits():
    events = [
        _event("capital of Japan", "Japan's capital city", 0.96, good_match=True),
        _event("rate in 2005", "rate in 2015", 0.99, good_match=False),  # logged before the check existed
    ]

    row = next(r for r in tune(events) if r.threshold == 0.95)

    assert row.would_hit == 1 and row.wrong == 0


def test_old_number_mismatch_labels_do_not_make_thresholds_stricter():
    good = [_event(f"question {w}", f"the same question {w}", 0.93, good_match=True)
            for w in "abcdefghijkl"[:MIN_LABELS]]
    bad_numbers = [_event("rate in 2005", "rate in 2015", 0.99, good_match=False) for _ in range(5)]

    learned = learn_thresholds(good + bad_numbers)

    # Only the pairs that can still match teach the threshold: all good at 0.93.
    assert learned["factual"].threshold == 0.93
    assert learned["factual"].labels == MIN_LABELS
