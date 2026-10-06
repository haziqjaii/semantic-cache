"""
A check on cache matches that embeddings can't do: are the numbers the same?

Plain string work: no API call, no model, microseconds per request.

To an embedding model, "what is 6906006 * 2032032" and "what is 6906006 *
2032033" are almost the same sentence (similarity ~0.99), but they have
different answers. So do "the exchange rate in 2005" and "the exchange rate
in 2015". A cached answer is only reused when the two questions contain
exactly the same numbers.

Numbers are compared by value ("1,000" = "1000", "2.50" = "2.5"), and their
order doesn't matter. Numbers written as words count too, in English and
Malay: "fifteen" = "lima belas" = 15, "5 million" = 5000000.

Anything this doesn't recognise (ordinals like "third", "half", "5k") makes
the pair look different, which costs a cache hit but never serves a wrong
answer.
"""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation

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
