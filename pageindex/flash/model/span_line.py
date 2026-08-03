"""Span and Line types with text and style helpers."""

from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterator, Optional, Protocol

from .char_stats import (
    _trim_unicode_ws,
    CharStats,
    merge_char_stats,
    letter_count,
    info_weight,
)
from .rects import (
    Rect,
    EMPTY_RECT,
    Bounded,
    rect_union,
)


# --------------------------------------------------------------------------- #
# Text span #
# --------------------------------------------------------------------------- #


# Font-style detectors. Neither pattern is multiline or Unicode-aware: the end
# anchor binds at end of INPUT (Python's `$` would also match before a trailing
# newline, hence `\Z`), and case folding stays ASCII-only, so U+017F, U+0130
# and U+0131 do not fold onto "s"/"i". The digit classes are spelled out, so
# the ASCII flag touches nothing else here.
_bold_font_re = re.compile(r"(bold|timesb)", re.IGNORECASE | re.ASCII)
_italic_font_re = re.compile(r"(ital|it\Z|i[1-9][0-9]*\Z|obliq)", re.IGNORECASE | re.ASCII)
# Font-name canonicalization map.
_font_name_aliases = {
    "timesnewroman":  "Times",
    "times-new-roman": "Times",
    "timesroman":     "Times",
    "times-roman":    "Times",
    "timesnew":       "Times",
    "times-new":      "Times",
}

# subset prefix regex: 6 uppercase letters + plus sign
_subset_prefix_re = re.compile(r"^[A-Z]{6}\+")


class Span(Bounded):
    """Span emitted by one text-showing item. Stores raw and trimmed text, character statistics, skew, font family/name, font size, bold/italic flags, and bbox helpers."""

    __slots__ = (
        "text", "state_slot", "char_stats", "previous_slot", "font_family", "font_name", "font_size", "primary_slot", "measure_slot",
    )

    def __init__(
        self,
        bbox: Rect,
        text: str,
        font_name_raw: str,
        font_size: float,
        bold: bool,
        italic: bool,
        skew: float = 0.0,
        font_family: str = "",
    ):
        """Create a span from parser-normalized text, font, style, skew, and bounding-box fields."""
        super().__init__(bbox)
        self.text = text
        self.state_slot = _trim_unicode_ws(text)
        self.char_stats = CharStats(self.state_slot)
        self.previous_slot = skew
        self.font_family = font_family

        # ---- font name normalisation -----------
        reference_item = font_name_raw
        if _subset_prefix_re.match(reference_item):
            reference_item = reference_item[7:]
        reference_item = _font_name_aliases.get(reference_item.lower(), reference_item)

        self.font_name = reference_item

        self.font_size = font_size

        # Bold comes from the adapter flag or from the normalized font name.
        self.primary_slot = bool(bold) or bool(_bold_font_re.search(self.font_name))
        # Italic is name-derived only. The ``italic`` parameter is accepted
        # for adapter compatibility but is not consulted.
        del italic  # noqa: F841 -- explicitly drop the arg
        self.measure_slot = bool(_italic_font_re.search(self.font_name))

    def char_count(self) -> int:                                # type: ignore[override]
        """Span char count."""
        return self.char_stats.auxiliary_slot

    def font_style(self) -> str:
        """"<fontName> B" or "<fontName> R"."""
        return f"{self.font_name} {'B' if self.primary_slot else 'R'}"


# --------------------------------------------------------------------------- #
# Text line #
# --------------------------------------------------------------------------- #


class Line(Bounded):
    """A list of spans on roughly the same baseline, with line-wide character statistics, first letter-bearing span, weighted bold/italic/skew/font-size aggregates, cached text, numbering state, column index, and span list."""

    __slots__ = (
        "primary_slot", "char_stats", "alignment_slot", "weighted_ratio_primary", "weighted_ratio_secondary", "weighted_ratio_tertiary", "metric_slot", "previous_slot", "measure_slot", "marker_slot", "state_slot", "style_slot", "cache_slot",
    )

    def __init__(self):
        super().__init__(EMPTY_RECT)
        self.primary_slot: list[Span] = []
        self.char_stats: CharStats = CharStats("")
        self.alignment_slot: Optional[Span] = None
        self.weighted_ratio_primary: float = 0.0
        self.weighted_ratio_secondary: float = 0.0

        self.weighted_ratio_tertiary: float = 0.0
        self.metric_slot: float = 0.0
        self.previous_slot: float = 0.0
        # Column index assigned by the column pass; -1 means unassigned.
        self.measure_slot: int = -1
        self.marker_slot: Optional[str] = None
        self.state_slot: int = -1
        self.style_slot: str = ""
        self.cache_slot: float = 0.0

    def __iter__(self) -> Iterator[Span]:
        return iter(self.primary_slot)

    def char_count(self) -> int:                                # type: ignore[override]
        return self.char_stats.auxiliary_slot

    def avg_font_size(self) -> float:
        return self.metric_slot

    def bold_frac(self) -> float:
        return self.weighted_ratio_primary

    def skew_frac(self) -> float:
        return self.weighted_ratio_tertiary


def append_span(line: Line, other_span: Span) -> Line:
    """append span b into line a, updating weighted fields. Every aggregate field is updated in one pass so downstream line scoring sees the same weighted style, size, and geometry summaries. """
    line.primary_slot.append(other_span)
    span = info_weight(line.char_stats)
    added_weight = info_weight(other_span.char_stats)
    total_weight = span + added_weight
    if total_weight > 0:
        line.weighted_ratio_primary = (line.weighted_ratio_primary * span + (1 if other_span.primary_slot else 0) * added_weight) / total_weight
        line.weighted_ratio_secondary = (line.weighted_ratio_secondary * span + (1 if other_span.measure_slot else 0) * added_weight) / total_weight
        line.weighted_ratio_tertiary = (line.weighted_ratio_tertiary * span + other_span.previous_slot * added_weight) / total_weight
        line.metric_slot = (line.metric_slot * span + other_span.font_size * added_weight) / total_weight
    merge_char_stats(line.char_stats, other_span.char_stats)
    if line.alignment_slot is None and letter_count(other_span.char_stats) > 0:
        line.alignment_slot = other_span
    if other_span.char_count() <= 0:
        return line
    span = line.area()
    line.previous_slot = max(line.previous_slot, other_span.bbox_height())
    line.secondary_slot = rect_union(line.secondary_slot, other_span.secondary_slot)
    line.cache_slot = min(1.0, (line.cache_slot * span + other_span.area()) / max(1.0, line.area()))
    line.marker_slot = None
    line.state_slot = -1
    line.style_slot = ""
    return line


def last_span(line: Line) -> Span:
    """last span of line."""
    return line.primary_slot[-1]


class _HasCharCount(Protocol):
    """Anything with K (char count), A (width), N (height)."""

    def char_count(self) -> int: ...
    def bbox_width(self) -> float: ...
    def bbox_height(self) -> float: ...


def avg_char_width(primary_item: _HasCharCount) -> float:
    """Width per character. Returns 0 if there are no characters."""
    return 0.0 if primary_item.char_count() <= 0 else primary_item.bbox_width() / primary_item.char_count()


def raw_text_of_line(line: Line) -> str:
    """concatenate raw text of all spans (no trimming)."""
    parts = []
    for span in line.primary_slot:
        parts.append(span.text)
    return "".join(parts)


def text_of_line(line: Line) -> str:
    """cached trimmed line text."""
    if line.marker_slot is not None:
        return line.marker_slot
    line.marker_slot = _trim_unicode_ws(raw_text_of_line(line))
    return line.marker_slot


def avg_char_width2(primary_item: _HasCharCount) -> float:
    """Width per character. Returns 0 for empty text."""
    return 0.0 if primary_item.char_count() <= 0 else primary_item.bbox_width() / primary_item.char_count()


# --------------------------------------------------------------------------- #
# Text block #
# --------------------------------------------------------------------------- #


_ONE_DECIMAL_QUANTUM = Decimal("0.1")


def _format_half_up_one_decimal(value: float) -> str:
    """Round the exact double half-away-from-zero; the stats module uses the same helper."""
    return str(Decimal(value).quantize(_ONE_DECIMAL_QUANTUM, rounding=ROUND_HALF_UP))


def style_key(span: "Span") -> str:
    """Style hash ``"<fontName> <B|R> <size rounded to 0.1>"`` using shared half-up rounding."""
    return f"{span.font_style()} {_format_half_up_one_decimal(span.font_size)}"
