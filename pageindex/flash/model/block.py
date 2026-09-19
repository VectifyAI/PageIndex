"""Block type with text, style, and alignment helpers."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Iterator, Optional, Protocol

from .char_stats import (
    is_punct_category,
    CharStats,
    merge_char_stats,
    letter_count,
    punct_count,
    info_weight,
    is_upper_dominant,
)
from .rects import (
    EMPTY_RECT,
    Bounded,
    rect_union,
    left_aligned,
    right_aligned,
    center_aligned,
)
from .span_line import (
    Span,
    Line,
    text_of_line,
    style_key,
)


class Block(Bounded):
    """A vertically contiguous group of lines that share layout, such as a paragraph or heading run. Adding lines maintains weighted style, size, text, bbox, reading-order, classification, and cache fields."""

    __slots__ = (
        "primary_slot", "char_stats", "alignment_slot", "weighted_ratio_tertiary", "previous_slot", "weighted_skew", "weighted_font_size", "weighted_ratio_primary", "weighted_ratio_secondary", "style_slot",
        "style_char_counts", "size_char_counts", "reading_order_index", "orig_index", "type", "isolated_centered", "is_body_paragraph", "measure_slot", "used_as_heading",
        "state_slot", "marker_slot", "metric_slot",
        "dominant_style_cache", "dominant_size_cache", "token_text_cache", "deaccented_text_cache", "cache_slot", "tokens_cache",
    )

    def __init__(self):
        super().__init__(EMPTY_RECT)
        self.primary_slot: list = []
        self.char_stats: CharStats = CharStats("")
        self.alignment_slot: bool = True
        self.weighted_ratio_tertiary: float = 0.0
        self.previous_slot: float = 0.0
        self.weighted_skew: float = 0.0
        self.weighted_font_size: float = 0.0
        self.weighted_ratio_primary: float = 0.0
        self.weighted_ratio_secondary: float = 0.0
        self.style_slot: float = 0.0
        self.style_char_counts: dict = {}
        self.size_char_counts: dict = {}
        self.reading_order_index: int = 0
        self.orig_index: int = 0
        self.type: int = 0
        self.isolated_centered: bool = False
        self.is_body_paragraph: bool = False
        self.measure_slot: bool = False
        self.used_as_heading: bool = False
        self.state_slot: int = 0
        self.marker_slot: int = 0
        self.metric_slot: float = 0.0
        # caches, invalidated on every add_line
        self.dominant_style_cache: Optional[str] = None
        self.dominant_size_cache: Optional[float] = None
        self.token_text_cache: Optional[str] = None
        self.deaccented_text_cache: Optional[str] = None
        self.cache_slot: Optional[str] = None
        self.tokens_cache: Optional[Any] = None

    def __iter__(self):
        return iter(self.primary_slot)

    def line_count(self) -> int:
        """Line count -- ."""
        return len(self.primary_slot)

    def line(self):
        """First line -- ."""
        return self.primary_slot[0]

    def char_count(self) -> int:                                # type: ignore[override]
        """Total char count across all child lines."""
        return self.char_stats.auxiliary_slot

    def avg_font_size(self) -> float:
        """Weighted average font size -- ."""
        return self.weighted_font_size

    def bold_frac(self) -> float:
        """Weighted bold fraction -- ."""
        return self.weighted_ratio_tertiary

    def skew_frac(self) -> float:
        """Weighted skew fraction -- ."""
        return self.weighted_skew

    def add_line(self, other_line) -> "Block":
        """Add a line while maintaining weighted style, size, character, bbox, and per-style histograms."""
        self.alignment_slot = self.alignment_slot and (len(self.primary_slot) <= 0 or center_aligned(self, other_line, 1))
        self.primary_slot.append(other_line)
        line = info_weight(self.char_stats)
        added_weight = info_weight(other_line.char_stats)
        total_weight = line + added_weight
        if total_weight > 0:
            self.weighted_ratio_tertiary = (self.weighted_ratio_tertiary * line + other_line.bold_frac() * added_weight) / total_weight
            self.previous_slot = (self.previous_slot * line + other_line.weighted_ratio_secondary * added_weight) / total_weight
            self.weighted_skew = (self.weighted_skew * line + other_line.skew_frac() * added_weight) / total_weight
            self.weighted_font_size = (self.weighted_font_size * line + other_line.avg_font_size() * added_weight) / total_weight
            self.weighted_ratio_primary = (self.weighted_ratio_primary * line + other_line.cache_slot * added_weight) / total_weight
        merge_char_stats(self.char_stats, other_line.char_stats)
        if other_line.char_count() <= 0:
            return self
        line = self.area()                                   # area before union
        self.style_slot = max(self.style_slot, other_line.previous_slot)
        self.secondary_slot = rect_union(self.secondary_slot, other_line.secondary_slot)
        added_weight = self.area()                                   # area after union
        if added_weight > 0:
            self.weighted_ratio_secondary = (self.weighted_ratio_secondary * line + other_line.cache_slot * other_line.area()) / added_weight
        for span in other_line:
            sty = style_key(span)
            self.style_char_counts[sty] = self.style_char_counts.get(sty, 0) + span.char_count()
            # Font-size buckets use half-up rounding to one decimal place.
            # Python round is half-to-even, so use floor(x + 0.5) on the
            # scaled non-negative font size.
            size_key = math.floor(span.font_size * 10 + 0.5) / 10
            self.size_char_counts[size_key] = self.size_char_counts.get(size_key, 0) + span.char_count()
        # invalidate caches
        self.dominant_style_cache = self.dominant_size_cache = self.token_text_cache = self.deaccented_text_cache = self.cache_slot = self.tokens_cache = None
        self.metric_slot = 0.0
        return self


# Sorted child iterator.

def iter_sorted_children(primary_item):
    """Iterate a page-like object's sorted children as indexed item records."""
    for idx, item in enumerate(primary_item.secondary_slot):
        yield {"index": idx, "block": item}


# --------------------------------------------------------------------------- #
# Block-level accessors and derived text/style helpers #
# --------------------------------------------------------------------------- #


def argmax_key(items) -> Optional[str]:
    """return the key with max value. ``None`` if empty. ``items`` may be a ``dict`` (in which case we iterate ``.items``) or any iterable of ``(key, value)`` pairs. """
    pairs = items.items() if isinstance(items, dict) else items
    best: Optional[str] = None
    candidate_item = float("-inf")
    for reference_item, entry_item in pairs:
        if entry_item <= candidate_item:
            continue
        best = reference_item
        candidate_item = entry_item
    return best


def last_line_of(block: Block) -> Line:
    """last child line of a block."""
    return block.primary_slot[-1]


def first_span_of(block: Block) -> Span:
    """first span of a block's first line."""
    return block.line().primary_slot[0]


def dominant_style_of(block: Block) -> str:
    """Cached dominant style hash from the block's style histogram."""
    if block.dominant_style_cache is None:
        block.dominant_style_cache = argmax_key(block.style_char_counts) or ""
    return block.dominant_style_cache


def dominant_font_size(block: Block) -> float:
    """Return cached dominant font size from rounded-size character counts."""
    if block.dominant_size_cache is None:
        block.dominant_size_cache = float(argmax_key(block.size_char_counts) or 0)
    return block.dominant_size_cache


def is_caps_heavy(primary_item) -> bool:
    """Return True if a line or block is uppercase-dominant."""
    return is_upper_dominant(primary_item.char_stats) or primary_item.char_stats.primary_slot[2] >= max(2, primary_item.char_stats.auxiliary_slot)


def is_sentence_like(primary_item) -> bool:
    """Return whether a block looks like mixed-case body text rather than a heading. The test requires enough tokens, enough uppercase letters, and rejects long lowercase words."""
    from ..tokens import tokenize_block
    tokens = tokenize_block(primary_item)
    if tokens.length < 3 or is_caps_heavy(primary_item):
        return False
    upper_count = primary_item.char_stats.primary_slot[2]
    if upper_count <= 2 or upper_count < tokens.length / 10:
        return False
    match = 0
    for token in tokens:
        # Skip non-word tokens, short tokens, or g==4 (special)
        if token.type != 2 or len(token.str) <= 2 or token.primary_slot == 4:
            continue
        if token.primary_slot == 2:
            match += 1
        elif len(token.str) >= 5:
            return False
    return match >= 3


def heading_score(heading) -> float:
    """Line/block heading score: dominant font size plus caps-heavy and bold bonuses."""
    return dominant_font_size(heading) + (2 if is_caps_heavy(heading) else 0) + (1 if heading.weighted_ratio_tertiary > 0.5 else 0)


def case_signal(char_stats: CharStats) -> int:
    """Return an uppercase, lowercase, or neutral case signal from character statistics."""
    if is_upper_dominant(char_stats) and not is_punct_category(char_stats.secondary_slot) and letter_count(char_stats) > 3 * char_stats.auxiliary_slot / 4 and punct_count(char_stats) < 5:
        return 1
    if char_stats.primary_slot[3] > 0:
        return -1
    return 0


def alignment_code(primary_item) -> int:
    """cached block-level alignment code. Returns: 1 fully-justified (every line aligned with the block on left or right) 2 left-aligned (every line shares the block's left) 3 flag-set justified (the block center-alignment flag is set) 4 right-aligned 5 mixed / other """
    if primary_item.metric_slot != 0 or len(primary_item.primary_slot) <= 0:
        return primary_item.metric_slot
    left = True
    right = True
    any_value = True
    for score_value in primary_item.primary_slot:
        tolerance = max(1.0, score_value.bbox_width() / 20.0)
        line_left_aligned = left_aligned(primary_item, score_value, tolerance)
        line_right_aligned = right_aligned(primary_item, score_value, tolerance)
        if not line_left_aligned:
            left = False
        if not line_right_aligned:
            right = False
        if not (line_left_aligned or line_right_aligned):
            any_value = False
    if left and not right:
        primary_item.metric_slot = 2
    elif right and not left:
        primary_item.metric_slot = 4
    elif any_value:
        primary_item.metric_slot = 1
    elif primary_item.alignment_slot:
        primary_item.metric_slot = 3
    else:
        primary_item.metric_slot = 5
    return primary_item.metric_slot


def block_text(block: Block) -> str:
    """cached joined trimmed text of a block (space-separated)."""
    if block.cache_slot is not None:
        return block.cache_slot
    parts = []
    for line_index, line_value in enumerate(block.primary_slot):
        parts.append(text_of_line(line_value))
        if line_index < len(block.primary_slot) - 1:
            parts.append(" ")
    block.cache_slot = "".join(parts)
    return block.cache_slot


def deaccented_text(block: Block) -> str:
    """Cached diacritic-stripped block text; case and spacing are preserved."""
    if block.deaccented_text_cache is not None:
        return block.deaccented_text_cache
    block.deaccented_text_cache = _strip_diacritics(block_text(block))
    return block.deaccented_text_cache


_COMBINING_MARKS = re.compile("[̀-ͯ]")


def _strip_diacritics(text: str) -> str:
    """Strip combining diacritics only while preserving case and internal spacing."""
    return unicodedata.normalize(
        "NFC", _COMBINING_MARKS.sub("", unicodedata.normalize("NFD", text))
    )
