"""
Two cheap text checks that make cache matching more accurate.

Both are plain string work: no API call, no model, microseconds per request.

1. clean_question — what gets embedded
    "Can you tell me what the capital of Malaysia is?" and "what is the
    capital of malaysia" ask the same thing, but capitals, punctuation and
    polite lead-ins push their embeddings apart. Cleaning them off before
    embedding lets wordings of one question land closer together.

    Only the text used for MATCHING is cleaned. The LLM always receives
    the question exactly as it was asked, and the cache stores and shows
    the original too.

2. same_numbers — a check embeddings can't do
    To an embedding model, "what is 6906006 * 2032032" and "what is
    6906006 * 2032033" are almost the same sentence (similarity ~0.99),
    but they have different answers. So do "the exchange rate in 2005" and
    "the exchange rate in 2015". A cached answer is only reused when the
    two questions contain exactly the same numbers.

    Numbers are compared by value ("1,000" = "1000", "2.50" = "2.5"), and
    their order doesn't matter. Numbers written as words ("two") aren't
    recognised; such a pair is simply treated as not matching, which costs
    a cache hit but never serves a wrong answer.
"""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation

# ── Cleaning a question for matching ────────────────────────

# Openers that carry no part of the question. Each pattern is tried at the
# start of the text, repeatedly, so "hi, can you please tell me ..." is
# stripped down to what is actually being asked.
#
# Deliberately short. "Can you ..." on its own is NOT stripped: in "can you
# eat raw eggs?" it is the question. Only openers that end in "tell me" /
# "know" are safe to drop.
_LEAD_INS = [re.compile(pattern) for pattern in (
    # A greeting only when punctuation sets it off ("Hi, ..."), so that
    # "hello world in Python" keeps its hello.
    r"(hi|hello|hey|hai|helo)( there)?\s*[,!.]+\s*",
    r"(please|pls|plz|kindly|tolong)\b[\s,]*",
    r"((can|could|would|will) (you|u) )?(please |kindly )?(tell me|let me know)\b[\s,:]*",
    r"(do|did) (you|u) know\b[\s,:]*",
    r"(i (want|need|wanna|would like)|i'd like) to know\b[\s,:]*",
    r"(may|can|could) i know\b[\s,:]*",
    # Malay
    r"(boleh (tak )?)?((awak|anda|kau|kamu) )?(tolong )?(beritahu|bagitahu|bagi tahu) saya\b[\s,:]*",
    r"saya (nak|ingin|mahu|hendak) tahu\b[\s,:]*",
    r"boleh saya tahu\b[\s,:]*",
)]

# Closers: a trailing "please" / "thanks" that is set off by punctuation
# ("..., please?" or "...? Thank you."), and end punctuation. Without the
# punctuation the word may be part of the question ("why do people say thanks").
_CLOSER = re.compile(r"\s*[,.?!;]+\s*(please|pls|plz|thanks|thank you|tq|terima kasih)[\s?.!]*$")
_END_PUNCTUATION = re.compile(r"[\s?.!]+$")
_WHITESPACE = re.compile(r"\s+")


def clean_question(text: str) -> str:
    """
    The question as it's embedded for matching: lower case, single spaces,
    without polite openers, closers or end punctuation.

    Never returns an empty string for a non-empty question: if cleaning
    would remove everything ("please"), the tidied original is used.
    """
    tidy = _WHITESPACE.sub(" ", text).strip().casefold()
    cleaned = tidy
    stripped = True
    while stripped:
        stripped = False
        for pattern in _LEAD_INS:
            match = pattern.match(cleaned)
            if match and match.end() > 0:
                cleaned = cleaned[match.end():]
                stripped = True
    cleaned = _CLOSER.sub("", cleaned)
    cleaned = _END_PUNCTUATION.sub("", cleaned).strip()
    return cleaned or _END_PUNCTUATION.sub("", tidy) or tidy


# ── Comparing the numbers in two questions ──────────────────

# Digits, with optional , or . between digit groups: 7, 2005, 1,000,000, 3.14
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _value(token: str) -> Decimal | str:
    """A number's value, so that 1,000 = 1000 and 2.50 = 2.5."""
    if "," in token and "." in token:
        # Both separators: the last one is the decimal point (1,234.5 or 1.234,5).
        last = max(token.rfind(","), token.rfind("."))
        token = re.sub(r"[.,]", "", token[:last]) + "." + token[last + 1:]
    elif token.count(",") > 1 or token.count(".") > 1:
        token = re.sub(r"[.,]", "", token)  # 1,000,000 (or a version like 3.12.1)
    elif "," in token:
        head, tail = token.split(",")
        # 12,500 is twelve thousand five hundred; 1,5 is a decimal comma.
        token = head + tail if len(tail) == 3 else head + "." + tail
    try:
        return Decimal(token)
    except InvalidOperation:
        return token


def numbers_in(text: str) -> Counter:
    """Every number in the text, by value, with how often it appears."""
    return Counter(_value(token) for token in _NUMBER.findall(text))


def same_numbers(question: str, other: str) -> bool:
    """True if both texts contain exactly the same numbers (in any order)."""
    return numbers_in(question) == numbers_in(other)
