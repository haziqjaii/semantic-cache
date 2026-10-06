"""
Threshold tuning — what-if analysis and learning, from the lookup log.

Both are plain functions over LookupEvents, so they're easy to test and to
reason about.

THE TUNER ("what if the threshold were t?")
    Each logged lookup remembers its closest cached question and their
    similarity. For a threshold t, a lookup *would have been a hit* if that
    similarity ≥ t. So for each t we can report:
      - hit rate:    share of lookups that would have hit
      - wrong rate:  among those, the share that people labelled as a bad
                     match (only counting lookups that were labelled)
    Lower t → more hits, but more wrong answers. That trade-off is the point.

LEARNING ("which threshold should each intent use?")
    For each intent (factual, how_to, ...), take its labelled lookups and
    pick the LOWEST threshold where at least TARGET_PRECISION of the
    labelled matches above it were good. Lowest = most hits; the precision
    target keeps wrong answers rare. Guardrails:
      - no learning until an intent has MIN_LABELS labels
      - the choice needs MIN_SUPPORT labels at or above it
      - never below the lowest similarity anyone has judged: if every
        label is at 0.92, we know nothing about 0.90, so we don't go there
      - results stay within [0.90, 0.99]: never looser than the search
        floor, never so strict that nothing can match
    If no threshold reaches the target, the intent gets the strictest (0.99).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from semcache.cache.lookup_log import LookupEvent
from semcache.cache.text import same_numbers

# Thresholds the tuner reports and learning chooses from.
TUNER_THRESHOLDS = [round(0.90 + i / 100, 2) for i in range(11)]  # 0.90 … 1.00
LEARNABLE_THRESHOLDS = TUNER_THRESHOLDS[:-1]  # 0.90 … 0.99

MIN_LABELS = 10  # labels an intent needs before its threshold is learned
MIN_SUPPORT = 5  # labels needed at or above the chosen threshold
TARGET_PRECISION = 0.95  # at most 1 in 20 matches may be wrong


def could_match(event: LookupEvent) -> bool:
    """
    True if the lookup's candidate could ever be served for its question.

    A candidate whose numbers differ from the question's is never served,
    whatever the threshold (see cache/text.py). The engine no longer logs
    such candidates; this keeps ones logged before that from being counted
    as would-be hits, or from teaching thresholds.
    """
    return event.has_candidate and same_numbers(event.prompt, event.candidate_prompt or "")


@dataclass
class TunerRow:
    threshold: float
    would_hit: int  # lookups whose closest candidate is at or above the threshold
    hit_rate: float  # would_hit / lookups analysed
    labelled: int  # of those, how many have feedback
    wrong: int  # of those, labelled as a bad match
    wrong_rate: float | None  # wrong / labelled; None without labels


def tune(events: list[LookupEvent], intent: str | None = None) -> list[TunerRow]:
    """
    Hit rate and wrong-answer rate at each threshold, from logged lookups.

    With `intent`, only lookups whose closest candidate had that intent are
    analysed (misses with no candidate have no intent, so they're excluded).
    """
    if intent is not None:
        events = [e for e in events if e.intent == intent]
    total = len(events)

    rows = []
    for t in TUNER_THRESHOLDS:
        hits = [e for e in events if could_match(e) and e.similarity >= t]
        labelled = [e for e in hits if e.good_match is not None]
        wrong = sum(1 for e in labelled if e.good_match is False)
        rows.append(TunerRow(
            threshold=t,
            would_hit=len(hits),
            hit_rate=len(hits) / total if total else 0.0,
            labelled=len(labelled),
            wrong=wrong,
            wrong_rate=wrong / len(labelled) if labelled else None,
        ))
    return rows


@dataclass
class LearnedThreshold:
    intent: str
    threshold: float
    labels: int  # labelled lookups for this intent
    precision: float | None  # share of good matches at or above the threshold


def learn_thresholds(labelled: list[LookupEvent]) -> dict[str, LearnedThreshold]:
    """Per-intent thresholds from labelled lookups (see module docstring)."""
    by_intent: dict[str, list[LookupEvent]] = defaultdict(list)
    for event in labelled:
        if event.intent and could_match(event) and event.good_match is not None:
            by_intent[event.intent].append(event)

    learned = {}
    for intent, events in by_intent.items():
        if len(events) < MIN_LABELS:
            continue
        # The lowest judged similarity, rounded down to the 0.01 grid: the
        # highest threshold that still includes that judged pair.
        lowest_judged = math.floor(min(e.similarity for e in events) * 100) / 100
        chosen, precision = LEARNABLE_THRESHOLDS[-1], None
        for t in LEARNABLE_THRESHOLDS:
            if t < lowest_judged:
                continue
            above = [e for e in events if e.similarity >= t]
            if len(above) < MIN_SUPPORT:
                continue
            good = sum(1 for e in above if e.good_match) / len(above)
            if good >= TARGET_PRECISION:
                chosen, precision = t, good
                break
        learned[intent] = LearnedThreshold(intent, chosen, len(events), precision)
    return learned
