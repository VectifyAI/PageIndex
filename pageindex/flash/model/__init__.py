"""
Data model for rectangles, spans, lines, blocks, character categories, and
alignment predicates. Coordinate convention follows PDF (origin bottom-left, y increases upward).
``Rect`` is constructed as ``Rect(left, right, top, bottom)``. A few internal
storage fields are implementation details; public callers should use the semantic accessors.
"""

import math
import re
import unicodedata
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterator, Optional, Protocol

import regex as regex_module  # supports Unicode \p{...} property classes

from .char_stats import (
    _SENTENCE_END_CHARS,
    _MINUS_SIGN_CHARS,
    _max_nan_propagating,
    _min_nan_propagating,
    char_category,
    is_word_category,
    is_punct_category,
    _UNICODE_WHITESPACE_CHARS,
    _trim_unicode_ws,
    _UNICODE_WHITESPACE_CLASS,
    _round_half_up_to_int,
    CharStats,
    merge_char_stats,
    letter_count,
    punct_count,
    info_weight,
    is_upper_dominant,
)
from .rects import (
    RectLike,
    Rect,
    EMPTY_RECT,
    Bounded,
    rect_union,
    rect_intersection,
    extend_top_to,
    extend_bottom_to,
    cmp_left_edge,
    left_edge_key,
    cmp_reading_order,
    reading_order_key,
    cmp_bottom_edge,
    magnitude_ratio,
    same_x_extent,
    same_y_extent,
    intervals_overlap,
    y_overlaps,
    left_aligned,
    right_aligned,
    center_aligned,
    x_aligned,
    x_centers_close,
)
from .span_line import (
    _bold_font_re,
    _italic_font_re,
    _font_name_aliases,
    _subset_prefix_re,
    Span,
    Line,
    append_span,
    last_span,
    _HasCharCount,
    avg_char_width,
    raw_text_of_line,
    text_of_line,
    avg_char_width2,
    _ONE_DECIMAL_QUANTUM,
    _format_half_up_one_decimal,
    style_key,
)
from .block import (
    Block,
    iter_sorted_children,
    argmax_key,
    last_line_of,
    first_span_of,
    dominant_style_of,
    dominant_font_size,
    is_caps_heavy,
    is_sentence_like,
    heading_score,
    case_signal,
    alignment_code,
    block_text,
    deaccented_text,
    _COMBINING_MARKS,
    _strip_diacritics,
)
from .numbering import (
    _NUMBERING_PREFIX_RE,
    _BRACKETED_NUM_RE,
    _TO_NUMBER_DEC,
    _TO_NUMBER_INF,
    _TO_NUMBER_HEX,
    _TO_NUMBER_OCT,
    _TO_NUMBER_BIN,
    to_number,
    _detect_numbering,
    numbering_text,
    numbering_value,
    numbering_kind,
)

__all__ = [
    "RectLike", "Rect", "Bounded", "EMPTY_RECT", "rect_union", "rect_intersection", "extend_top_to", "extend_bottom_to",
    "CharStats", "merge_char_stats", "letter_count", "punct_count", "info_weight", "is_upper_dominant",
    "char_category", "is_word_category", "is_punct_category",
    "Span", "Line", "Block",
    "append_span", "last_span", "avg_char_width", "raw_text_of_line", "text_of_line", "avg_char_width2", "style_key", "iter_sorted_children",
    "magnitude_ratio", "same_x_extent", "same_y_extent", "intervals_overlap", "y_overlaps", "left_aligned", "right_aligned", "center_aligned", "x_aligned", "x_centers_close",
    "to_number", "numbering_text", "numbering_value", "numbering_kind",
    "argmax_key", "last_line_of", "first_span_of", "dominant_style_of", "dominant_font_size", "is_caps_heavy", "heading_score", "case_signal", "alignment_code", "block_text", "deaccented_text",
    "cmp_left_edge", "left_edge_key", "cmp_reading_order", "reading_order_key", "cmp_bottom_edge",
]
