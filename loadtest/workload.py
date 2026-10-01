"""
The load test's workload — a realistic, repeatable stream of questions.

WHAT "REALISTIC" MEANS HERE
    Real traffic to an LLM is not 2,000 different questions, and it's not
    one question 2,000 times. It's somewhere between:

      - A few questions are asked constantly, and most are asked rarely
        (a "long tail"). We model that with a Zipf distribution: the 2nd
        most popular question is asked about half as often as the 1st.
      - People phrase the same question differently. Each question has
        several wordings; every request picks one.
      - Some requests shouldn't be served from a cache at all: creative
        ones (a poem should be new each time) and time-sensitive ones.

FAMILIES
    A "family" is one underlying question with all its wordings, e.g.

        "What is the capital of Japan?"  /  "Which city is the capital of Japan?"

    Two requests from the same family should share a cached answer; two
    from different families must not. Because the workload knows every
    request's family, the report can count WRONG answers: hits that were
    served another family's answer.

REPEATABLE
    The same seed always produces the same requests in the same order, so
    runs can be compared.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Sent with every request. It keeps answers short (fewer tokens to pay
# for), and, being constant, keeps every request in one cache namespace.
SYSTEM_PROMPT = "Answer in one or two sentences."

# ── The questions ───────────────────────────────────────────
# kind → (wordings, subjects). Each subject makes one family.
_FAMILIES: dict[str, tuple[list[str], list[str]]] = {
    "capital": (
        [
            "What is the capital of {x}?",
            "Which city is the capital of {x}?",
            "What's the capital city of {x}?",
            "Tell me the capital of {x}.",
        ],
        [
            "Malaysia", "Japan", "France", "Brazil", "Canada", "Egypt", "Kenya", "Norway", "Peru", "Turkey",
            "Thailand", "Indonesia", "Vietnam", "Germany", "Spain", "Italy", "Portugal", "Argentina", "Chile",
            "Mexico", "India", "Pakistan", "Nepal", "Mongolia", "Poland", "Greece", "Hungary", "Finland",
            "Ireland", "Morocco", "Ghana", "Ethiopia", "Jordan", "Qatar", "New Zealand", "South Korea",
            "the Philippines", "Saudi Arabia", "the Netherlands", "Sweden",
        ],
    ),
    "concept": (
        [
            "What is {x}?",
            "Explain {x} briefly.",
            "Can you explain what {x} is?",
            "Give me a short definition of {x}.",
        ],
        [
            "photosynthesis", "inflation", "a black hole", "machine learning", "the greenhouse effect",
            "DNA", "blockchain", "a vaccine", "compound interest", "natural selection", "an API",
            "the water cycle", "supply and demand", "a neural network", "plate tectonics", "osmosis",
            "a recession", "gravity", "encryption", "the immune system", "a database index", "an atom",
            "cloud computing", "the stock market", "a virus", "renewable energy", "an algorithm",
            "the ozone layer", "a semiconductor", "democracy", "a carbon footprint", "a hash function",
            "quantum computing", "an operating system", "biodiversity", "a light year", "a protein",
            "the Doppler effect", "a vector database", "an embedding",
        ],
    ),
    "author": (
        [
            "Who wrote {x}?",
            "Who is the author of {x}?",
            "Which author wrote {x}?",
            "{x} was written by whom?",
        ],
        [
            "Pride and Prejudice", "1984", "The Great Gatsby", "Hamlet", "Don Quixote", "War and Peace",
            "The Odyssey", "Moby-Dick", "Frankenstein", "The Hobbit", "Jane Eyre", "Dracula",
            "The Old Man and the Sea", "Crime and Punishment", "To Kill a Mockingbird", "Les Misérables",
            "The Little Prince", "Brave New World", "The Alchemist", "Things Fall Apart",
        ],
    ),
    "how_to": (
        [
            "How do I {x} in Python?",
            "How can I {x} in Python?",
            "What's the way to {x} in Python?",
            "Show me how to {x} in Python.",
        ],
        [
            "reverse a list", "read a file line by line", "sort a dictionary by value", "remove duplicates from a list",
            "check if a string is a palindrome", "merge two dictionaries", "get the current date", "parse JSON",
            "write to a CSV file", "count words in a string", "flatten a nested list", "generate a random number",
            "convert a string to an integer", "find the largest item in a list", "loop over a dictionary",
            "create a virtual environment", "handle an exception", "format a string with variables",
            "make an HTTP request", "measure how long code takes", "read command-line arguments",
            "split a string by commas", "check if a file exists", "define a class", "use a list comprehension",
            "round a number to two decimals", "join a list into a string", "swap two variables",
            "open a file safely", "sort a list of tuples",
        ],
    ),
    "sentiment": (
        [
            "Is this review positive or negative? '{x}'",
            "Classify the sentiment of this review as positive or negative: '{x}'",
            "Positive or negative review? '{x}'",
        ],
        [
            "The battery died after two days and support never replied.",
            "Absolutely love it, works better than I expected!",
            "Arrived late and the box was damaged.",
            "Great value for the price, would buy again.",
            "The screen is dim and the speakers crackle.",
            "Fast delivery and the quality is excellent.",
            "It stopped working within a week.",
            "Comfortable, stylish and fits perfectly.",
            "Terrible customer service, I want a refund.",
            "Easy to set up and runs smoothly.",
            "Too expensive for what you get.",
            "My kids use it every day and it still looks new.",
        ],
    ),
    # Time-sensitive: cached only briefly (5 minutes), so repeats later in a long run miss again.
    "weather": (
        [
            "What's the weather in {x} today?",
            "How's the weather in {x} right now?",
            "Is it raining in {x} today?",
        ],
        ["Kuala Lumpur", "Penang", "Tokyo", "London", "Sydney", "Dubai", "Toronto", "Cape Town"],
    ),
    # Creative: never cached, so every one of these is a miss by design.
    "creative": (
        [
            "Write a haiku about {x}.",
            "Compose a short haiku on {x}.",
            "Give me an original haiku about {x}.",
        ],
        [
            "monsoon rain", "a quiet library", "the night market", "a sleeping cat", "the first day of school",
            "an empty beach", "city traffic", "a cup of teh tarik", "falling leaves", "a thunderstorm",
            "the sunrise", "an old bicycle",
        ],
    ),
}

# Kinds whose answers the cache is never expected to reuse.
UNCACHED_KINDS = {"creative"}


@dataclass(frozen=True)
class Family:
    """One underlying question and all the ways of asking it."""

    id: str  # e.g. "capital:Japan"
    kind: str
    wordings: tuple[str, ...]


@dataclass(frozen=True)
class Request:
    index: int  # position in the run
    family: str
    kind: str
    text: str


def families() -> list[Family]:
    return [
        Family(id=f"{kind}:{subject}", kind=kind, wordings=tuple(w.format(x=subject) for w in wordings))
        for kind, (wordings, subjects) in _FAMILIES.items()
        for subject in subjects
    ]


def build_workload(n: int, seed: int = 7, zipf_exponent: float = 0.9) -> list[Request]:
    """
    `n` requests drawn from the families, popular ones far more often.

    Families are shuffled (so popularity isn't tied to a kind), then the
    family at rank r is drawn with weight 1 / r^zipf_exponent. Each
    request uses the family's first wording half the time and another
    wording otherwise, since people do reword.
    """
    rng = random.Random(seed)
    pool = families()
    rng.shuffle(pool)
    weights = [1 / (rank**zipf_exponent) for rank in range(1, len(pool) + 1)]

    requests = []
    for index, family in enumerate(rng.choices(pool, weights=weights, k=n)):
        if rng.random() < 0.5:
            text = family.wordings[0]
        else:
            text = rng.choice(family.wordings)
        requests.append(Request(index=index, family=family.id, kind=family.kind, text=text))
    return requests


def describe(requests: list[Request]) -> dict:
    """What a workload contains, and the fewest LLM calls any cache could make for it."""
    seen: set[str] = set()
    forced_misses = 0  # requests no cache can serve: first sight of a family, or an uncached kind
    for request in requests:
        if request.kind in UNCACHED_KINDS or request.family not in seen:
            forced_misses += 1
        seen.add(request.family)

    by_kind: dict[str, int] = {}
    for request in requests:
        by_kind[request.kind] = by_kind.get(request.kind, 0) + 1

    n = len(requests)
    return {
        "requests": n,
        "families": len(seen),
        "distinct_wordings": len({r.text for r in requests}),
        "by_kind": dict(sorted(by_kind.items())),
        "minimum_llm_calls": forced_misses,
        "best_possible_hit_rate": (n - forced_misses) / n if n else 0.0,
    }
