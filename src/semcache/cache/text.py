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
    their order doesn't matter. Numbers written as words count too, in
    English and Malay: "fifteen" = "lima belas" = 15, "5 million" = 5000000.

    Anything this doesn't recognise (ordinals like "third", "half", "5k")
    makes the pair look different, which costs a cache hit but never
    serves a wrong answer.
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
_DIGITS = r"\d+(?:[.,]\d+)*"

# A text is read as digits, words, gaps (spaces and hyphens, which may sit
# inside a number: "twenty-five") and anything else (which ends a number).
_TOKEN = re.compile(rf"(?P<digits>{_DIGITS})|(?P<word>[^\W\d_]+)|(?P<gap>[\s-]+)|(?P<other>.)")

# Number words, English and Malay, by the part they play in a number.
_UNITS = {  # 1-9: can follow a ten ("twenty five")
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5, "enam": 6, "tujuh": 7, "lapan": 8,
    "delapan": 8, "sembilan": 9,
}
_TEENS = {  # complete on their own: nothing smaller can follow
    "zero": 0, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "sifar": 0, "sepuluh": 10, "sebelas": 11,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
_HUNDRED = {"hundred", "ratus"}
_SCALES = {
    "thousand": 10**3, "ribu": 10**3, "million": 10**6, "juta": 10**6, "billion": 10**9, "bilion": 10**9,
}
# Malay "se-" is "one": seratus = satu ratus (100), seribu = satu ribu (1000).
_ONE_OF = {"seratus": "ratus", "seribu": "ribu", "sejuta": "juta"}

# "one" and "satu" on their own are usually not a count ("which one", "no
# one", "salah satu"), so alone they are not read as 1.
_NOT_A_COUNT = {("one",), ("satu",)}


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


class _NumberReader:
    """
    Reads numbers out of a text one token at a time.

    A number is built the way it is said: "two thousand five hundred" is
    2 x 1000, then 5 x 100. `finished` is the part already multiplied by
    thousand / million, `group` is the part still being said (under 1000).
    """

    def __init__(self) -> None:
        self.found: list[dict] = []
        self._after_and = False  # was the last thing read an "and" right after a number?
        self._clear()

    def _clear(self) -> None:
        self._finished = self._group = self._unit = Decimal(0)
        self._last: str | None = None  # the kind of word just read; None = no number in progress
        self._words: list[str] = []
        self._joined = False

    def close(self, *, at_and: bool = False) -> None:
        """End the number in progress, if there is one."""
        was_number = self._last is not None
        if was_number:
            self.found.append({
                "value": self._finished + self._group,
                "last": self._last,
                "words": tuple(self._words),
                "joined": self._joined,
            })
        self._clear()
        self._after_and = at_and and was_number

    def _start(self) -> None:
        joined = self._after_and
        self.close()
        self._joined = joined

    def digits(self, token: str) -> None:
        value = _value(token)
        self._start()
        if isinstance(value, Decimal):
            self._group, self._last = value, "digits"  # "5 million": a scale word may follow
        else:
            self.found.append({"value": value, "last": "digits", "words": (), "joined": False})

    def word(self, word: str) -> None:
        if word in _ONE_OF:  # seratus = satu + ratus; "dua ribu seratus" = 2100
            self._small(1, "unit", follows=("scale",))
            self._multiply(_ONE_OF[word])
            self._words.append(word)
        elif word in _UNITS:
            self._small(_UNITS[word], "unit", follows=("tens", "hundred", "scale"))
            self._unit = Decimal(_UNITS[word])
            self._words.append(word)
        elif word in _TEENS:
            self._small(_TEENS[word], "teen", follows=("hundred", "scale"))
            self._words.append(word)
        elif word in _TENS:
            self._small(_TENS[word], "tens", follows=("hundred", "scale"))
            self._words.append(word)
        elif word == "belas" and self._last == "unit":  # dua belas = 12
            self._group += 10
            self._last = "teen"
            self._words.append(word)
        elif word == "puluh" and self._last == "unit":  # dua puluh = 20
            self._group += self._unit * 9
            self._last = "tens"
            self._words.append(word)
        elif word in _HUNDRED or word in _SCALES:
            self._multiply(word)
            self._words.append(word)
        else:
            self.close(at_and=word == "and")

    def _small(self, value: int, kind: str, follows: tuple[str, ...] = ()) -> None:
        """A word under 100: adds to the number in progress if it can follow it, else starts a new one."""
        if self._last not in follows:
            self._start()
        self._group += value
        self._last = kind

    def _multiply(self, word: str) -> None:
        if word in _HUNDRED:
            # "two hundred", "fifteen hundred", "3 hundred"; alone ("a hundred") it is 100.
            if self._last not in ("unit", "teen", "tens", "digits") or self._group >= 100:
                self._start()
                self._group = Decimal(1)
            self._group *= 100
            self._last = "hundred"
        else:
            # "two thousand", "5 million", "three hundred thousand"; alone it is 1000.
            if self._last in (None, "scale"):
                self._start()
                self._group = Decimal(1)
            self._finished += self._group * _SCALES[word]
            self._group = Decimal(0)
            self._last = "scale"


def numbers_in(text: str) -> Counter:
    """Every number in the text (digits or words), by value, with how often it appears."""
    reader = _NumberReader()
    for token in _TOKEN.finditer(text.casefold()):
        if token.lastgroup == "digits":
            reader.digits(token.group())
        elif token.lastgroup == "word":
            reader.word(token.group())
        elif token.lastgroup == "other":
            reader.close()
    reader.close()

    values: list[Decimal | str] = []
    previous: dict | None = None
    for number in reader.found:
        # "one hundred and five" is 105, but "100 and 5" stays two numbers.
        if (
            previous is not None
            and number["joined"]
            and previous["last"] in ("hundred", "scale")
            and number["words"]
            and number["value"] < 100
        ):
            values[-1] += number["value"]
            previous = None
            continue
        if number["words"] in _NOT_A_COUNT:
            previous = None
            continue
        values.append(number["value"])
        previous = number
    return Counter(values)


def same_numbers(question: str, other: str) -> bool:
    """True if both texts contain exactly the same numbers (in any order)."""
    return numbers_in(question) == numbers_in(other)
