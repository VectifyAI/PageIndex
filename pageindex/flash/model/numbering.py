"""Numbering-prefix detection and numeric parsing."""

from __future__ import annotations

import math
import re
import unicodedata

import regex as regex_module  # supports Unicode \p{...} property classes

from .char_stats import (
    _trim_unicode_ws,
    _UNICODE_WHITESPACE_CLASS,
)
from .span_line import (
    Line,
    raw_text_of_line,
)


# --------------------------------------------------------------------------- #
# Numbering detection #
# --------------------------------------------------------------------------- #


# Uses Unicode property classes (\p{Number} / \P{Number}), compiled with the
# ``regex`` module (stdlib ``re`` can't express them). Matches:
# - leading roman or digit (group 1)
# - dotted lowercase a-h (group 2)
# - dotted lowercase ivx (group 3)
_NUMBERING_PREFIX_RE = regex_module.compile(
    r"^(?:"
    r"([IVX]+|[1-9１-９]\p{Number}?)(?:[.．｡。):]|-\P{Number}|-$|[" + _UNICODE_WHITESPACE_CLASS + r"]|$)"
    r"|(?:([A-Ha-h])|([ivx]))[.．｡。)]"
    r")"
)

# Bracketed numeric labels such as "[1]" or "(1)".
_BRACKETED_NUM_RE = re.compile(r"^[\[\(] *([1-9][0-9]?) *[\)\]]")


# string grammar (ToNumber). ASCII digits ONLY: Python's
# ``\d`` and ``float`` both accept Unicode decimal digits (e.g. Arabic-Indic
# ٢) and ``float`` also accepts ``1_000`` / ``inf`` / ``nan``, none of which
# ``Number`` accepts -- hence the explicit ``[0-9]`` classes.
_TO_NUMBER_DEC = re.compile(r"^[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
_TO_NUMBER_INF = re.compile(r"^[+-]?Infinity$")
_TO_NUMBER_HEX = re.compile(r"^0[xX][0-9a-fA-F]+$")
_TO_NUMBER_OCT = re.compile(r"^0[oO][0-7]+$")
_TO_NUMBER_BIN = re.compile(r"^0[bB][01]+$")


def to_number(text: str) -> float:
    """NFKC-normalized numeric conversion with decimal, exponent, hex, octal, binary, and Infinity forms."""
    if text is None:
        return math.nan
    token_value = _trim_unicode_ws(unicodedata.normalize("NFKC", text))
    if token_value == "":
        return 0.0
    if _TO_NUMBER_INF.match(token_value):
        return -math.inf if token_value[0] == "-" else math.inf
    if _TO_NUMBER_HEX.match(token_value):
        return float(int(token_value[2:], 16))
    if _TO_NUMBER_OCT.match(token_value):
        return float(int(token_value[2:], 8))
    if _TO_NUMBER_BIN.match(token_value):
        return float(int(token_value[2:], 2))
    if _TO_NUMBER_DEC.match(token_value):
        return float(token_value)
    return math.nan


def _detect_numbering(line: Line) -> None:
    """Detect leading section numbering and cache the numbering kind and text on the line."""
    if line.state_slot != -1:
        return                                           # already computed
    line.state_slot = 0
    if line.char_count() <= 0:
        return
    # Drop-capital / large-first-char detection (layout branch).
    # If first span is smaller, sits above the next non-empty span, and is
    # numeric -> use that span's text as the numbering.
    if len(line.primary_slot) > 1:
        secondary_item = line.primary_slot[0]
        candidate_item = line.primary_slot[2] if (line.primary_slot[1].char_count() <= 0 and len(line.primary_slot) > 2) else line.primary_slot[1]
        if (
            secondary_item.bbox_height() < candidate_item.bbox_height()
            and secondary_item.bottom_edge() > candidate_item.bottom_edge() + 0.05 * candidate_item.bbox_height()
            and not math.isnan(to_number(secondary_item.text))
        ):
            line.state_slot = 1
            line.style_slot = secondary_item.text
            return
    text = raw_text_of_line(line)
    measure_item = _NUMBERING_PREFIX_RE.match(text)
    if measure_item and measure_item.group(1) and "1" <= measure_item.group(1)[0] <= "9":
        line.state_slot = 1
        line.style_slot = measure_item.group(1)
        return
    if measure_item and (measure_item.group(1) or measure_item.group(3)):
        # Roman uppercase (group 1) or other -- both uppercase-ish
        line.state_slot = 2
        line.style_slot = measure_item.group(1) or measure_item.group(3)
        return
    if measure_item and measure_item.group(2):
        line.state_slot = 3
        line.style_slot = measure_item.group(2)
        return
    second_matrix = _BRACKETED_NUM_RE.match(text)
    if second_matrix:
        line.state_slot = 1
        line.style_slot = second_matrix.group(1)
        return


def numbering_text(line: Line) -> str:
    """get the cached numbering string."""
    _detect_numbering(line)
    return line.style_slot


def numbering_value(line: Line) -> float:
    """get numbering as a number, NaN if non-digit numbering."""
    text = numbering_text(line)
    return to_number(text) if line.state_slot == 1 else math.nan


def numbering_kind(line: Line) -> int:
    """get numbering type (0 none, 1 digit, 2 upper, 3 lower)."""
    _detect_numbering(line)
    return line.state_slot
