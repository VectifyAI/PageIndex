"""Character categories and per-run character statistics."""

from __future__ import annotations

import math
import unicodedata


# --------------------------------------------------------------------------- #
# Character classifier #
# --------------------------------------------------------------------------- #

# Character categories used by tokenization:
# 0 empty
# 1 number (digit / numeral)
# 2 uppercase letter (Lu, Lt)
# 3 lowercase letter (Ll)
# 4 other letter (Lo) -- CJK ideographs, syllabics, etc.
# 5 mark (Mc, Me, Mn)
# 6 sentence-end punct -- . ? ! ｡ 。 ？ ！ ．
# 7 connector / dash -- _ - — − ⁻ ₋ etc.
# 8 other punctuation
# 9 math symbol (Sm)
# 10 whitespace
# 11 other (symbols, format, control, unassigned)

_SENTENCE_END_CHARS = frozenset(".?!｡。？！．")
_MINUS_SIGN_CHARS = frozenset("−⁻₋")  # minus, superscript/subscript minus


def _max_nan_propagating(value: float, other_item: float) -> float:
    """propagates NaN (Python ``max`` swallows it)."""
    if math.isnan(value) or math.isnan(other_item):
        return math.nan
    return value if value >= other_item else other_item


def _min_nan_propagating(value: float, other_item: float) -> float:
    """propagates NaN (Python ``min`` swallows it)."""
    if math.isnan(value) or math.isnan(other_item):
        return math.nan
    return value if value <= other_item else other_item


def char_category(char_value: str) -> int:
    """Return the tokenizer character category code from Unicode General_Category."""
    if not char_value:
        return 0
    cat = unicodedata.category(char_value)
    # Letters ------------------------------------------------------------------
    if cat == "Ll":
        return 3
    if cat == "Lu" or cat == "Lt":
        return 2
    if cat == "Lo":
        return 4
    # Whitespace ---------------------------------------------------------------
    # The whitespace set is the Unicode WhiteSpace + LineTerminator set:
    # the C0 set \t\n\v\f\r, the BOM ﻿, and
    # Unicode Space/Line/Paragraph separators (Zs/Zl/Zp). NOT Python's
    # str.isspace, which also matches the C0 separators U+001C-U+001F and NEL
    # U+0085, which this tokenizer intentionally excludes, and misses ﻿.
    if char_value in "\t\n\x0b\x0c\r" or char_value == "\ufeff" or cat in ("Zs", "Zl", "Zp"):
        return 10
    # Sentence-end punctuation -------------------------------------------------
    if char_value in _SENTENCE_END_CHARS:
        return 6
    # Dash / connector punctuation ---------------------------------------------
    if cat in ("Pc", "Pd") or char_value in _MINUS_SIGN_CHARS:
        return 7
    # General punctuation ------------------------------------------------------
    if cat.startswith("P"):
        return 8
    # Number -------------------------------------------------------------------
    if cat.startswith("N"):
        return 1
    # Mark ---------------------------------------------------------------------
    if cat.startswith("M"):
        return 5
    # Math symbol --------------------------------------------------------------
    if cat == "Sm":
        return 9
    return 11


def is_word_category(number: int) -> bool:
    """is c a 'word-y' category (letter / digit / mark)?"""
    return number == 3 or number == 2 or number == 1 or number == 5


def is_punct_category(number: int) -> bool:
    """is c a punctuation-y category (dash / punct / sentence)?"""
    return number == 7 or number == 8 or number == 6


# Unicode trim strips the package whitespace set used by text parsing.
# Python str.strip uses a DIFFERENT set: it ALSO strips U+001C-001F and U+0085
# Trim keeps U+001C..U+001F and strips U+FEFF to match the intended whitespace set.
# (Same set as parser_pdfium_charlevel._UNICODE_WHITESPACE; defined here to avoid a
# circular import -- parser imports from model, not vice-versa.)
_UNICODE_WHITESPACE_CHARS = (
    "\t\n\x0b\x0c\r \xa0 "
    "           "
    "    　﻿"
)


def _trim_unicode_ws(text: str) -> str:
    """Strip the package whitespace set, not Python's broader ``str.strip`` set."""
    return text.strip(_UNICODE_WHITESPACE_CHARS)


# Unicode-compatible ``\s`` = WhiteSpace + LineTerminator = the same 25-cp set as
# _UNICODE_WHITESPACE_CHARS. Bare Python ``\s`` differs: stdlib ``re`` ``\s`` ALSO matches
# U+001C-U+001F and U+0085, the ``regex`` module ``\s`` matches U+0085, and
# NEITHER matches U+FEFF (which does). Splice this char-class BODY into
# regex definitions ("[" + _UNICODE_WHITESPACE_CLASS + "]") instead of a bare ``\s``.
_UNICODE_WHITESPACE_CLASS = r"\t\n\x0b\x0c\r\x20\xa0  -     　﻿"


def _round_half_up_to_int(value: float) -> int:
    """Round a non-negative finite number to an integer using exact half-up semantics. The ``floor(x + 0.5)`` idiom is not equivalent at the single double ``0.49999999999999994``: adding 0.5 rounds up to ``1.0`` so floor gives 1. Compute the fractional part directly (exact for x >= 0 by Sterbenz) and compare to 0.5."""
    score_value = math.floor(value)
    frac = value - score_value
    if frac < 0.5:
        return score_value
    return score_value + 1  # frac > 0.5, or an exact 0.5 tie


# --------------------------------------------------------------------------- #
# Per-string character-category accumulator #
# --------------------------------------------------------------------------- #


class CharStats:
    """Collect first/last character category, per-category counts, and total character count."""

    __slots__ = ("secondary_slot", "tertiary_slot", "primary_slot", "auxiliary_slot")

    def __init__(self, other_text: str):
        self.secondary_slot = 0
        self.tertiary_slot = 0
        self.primary_slot = [0] * 12
        self.auxiliary_slot = 0
        for secondary_item in other_text:
            cat = char_category(secondary_item)
            if self.secondary_slot == 0:
                self.secondary_slot = cat
            self.tertiary_slot = cat
            self.primary_slot[cat] += 1
            self.auxiliary_slot += 1


def merge_char_stats(char_stats: CharStats, other_char_stats: CharStats) -> None:
    """merge b into a in place."""
    if char_stats.secondary_slot == 0:
        char_stats.secondary_slot = other_char_stats.secondary_slot
    if other_char_stats.tertiary_slot != 0:
        char_stats.tertiary_slot = other_char_stats.tertiary_slot
    for candidate_item in range(12):
        char_stats.primary_slot[candidate_item] += other_char_stats.primary_slot[candidate_item]
    char_stats.auxiliary_slot += other_char_stats.auxiliary_slot


def letter_count(char_stats: CharStats) -> int:
    """count of letter-like chars (uppercase + lowercase + other-letter)."""
    return char_stats.primary_slot[3] + char_stats.primary_slot[2] + char_stats.primary_slot[4]


def punct_count(char_stats: CharStats) -> int:
    """count of sentence-punctuation chars (6 + 7 + 8)."""
    return char_stats.primary_slot[6] + char_stats.primary_slot[7] + char_stats.primary_slot[8]


def info_weight(char_stats: CharStats) -> float:
    """'informational' weight. ``letters + 2*other_letter + 0.5*(non-letter)`` -- biases towards alphabetic content; non-letter chars contribute half. """
    return char_stats.primary_slot[3] + char_stats.primary_slot[2] + 2 * char_stats.primary_slot[4] + 0.5 * (char_stats.auxiliary_slot - letter_count(char_stats))


def is_upper_dominant(char_stats: CharStats) -> bool:
    """uppercase-dominant string detector. True iff (uppercase chars) > max(letters*3/4, letters-4) and (uppercase chars) > max(3, total/3). """
    secondary_item = char_stats.primary_slot[2]
    candidate_item = letter_count(char_stats)
    return secondary_item > max(candidate_item * 3 / 4, candidate_item - 4) and secondary_item > max(3, char_stats.auxiliary_slot / 3)
