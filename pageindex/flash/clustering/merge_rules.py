"""Span continuation and line-merge predicates."""

from __future__ import annotations

import re
from typing import Optional

from ..model import (
    _UNICODE_WHITESPACE_CLASS,
    avg_char_width2,
    Span,
    magnitude_ratio,
    same_x_extent,
    same_y_extent,
    append_span,
    last_span,
    avg_char_width,
    raw_text_of_line,
    text_of_line,
    reading_order_key,
    left_edge_key,
    numbering_kind,
    Line,
    letter_count,
    is_upper_dominant,
)


# Matches "...." dot-leader trails used in TOC entries: "Chapter 1 ........"
TRAILING_DOT_LEADER_RE = re.compile(r"([.][" + _UNICODE_WHITESPACE_CLASS + r"]*){4,}\Z")


# --------------------------------------------------------------------------- #
# In-line continuation predicate.
# --------------------------------------------------------------------------- #


def span_continues_line(line: Line, other_span: Span) -> bool:
    """Return whether ``span`` continues the current line. The test requires matching skew, overlapping vertical intervals, and a horizontal gap within a per-character tolerance that widens after sentence-ending punctuation."""
    if last_span(line).previous_slot != other_span.previous_slot:
        return False
    line_center_y = line.center_y()  # a's y-center
    span_center_y = other_span.center_y()  # b's y-center
    # Vertical disjointness check: if both centers fall outside the other box,
    # the spans are not on the same line.
    if (line_center_y > other_span.top_edge() or line_center_y < other_span.bottom_edge()) and (span_center_y > line.top_edge() or span_center_y < line.bottom_edge()):
        return False
    # tolerance from per-char height
    tolerance = min(5.0, max(0.1, avg_char_width(line), avg_char_width2(other_span)))
    wide_tolerance = 2.0 * tolerance
    # When a's last char is sentence-end punctuation, widen the tolerance
    if line.char_stats.tertiary_slot == 5:
        wide_tolerance *= 2.0
    return other_span.left_edge() > line.right_edge() - wide_tolerance and other_span.left_edge() < line.right_edge() + tolerance


# --------------------------------------------------------------------------- #
# Neighbor distance and picker.
# --------------------------------------------------------------------------- #


def vertical_distance_in_line_heights(line: Line, other_line: Line) -> float:
    """normalized vertical-center distance between two lines. ``|a.center_y - b.center_y| / max(a.bbox_height, b.bbox_height)``: how many line-heights apart the centres are. Returns 0 when centres coincide. """
    line_center_y = line.center_y()
    other_center_y = other_line.center_y()
    if line_center_y == other_center_y:
        return 0.0
    denom = max(line.bbox_height(), other_line.bbox_height())
    if denom == 0.0:
        # Empty lines carry an inverted-sentinel bbox. Preserve IEEE division
        # edge cases so the later distance comparison simply does not merge.
        diff = line_center_y - other_center_y
        return float("nan") if diff != diff else float("inf")
    return abs(line_center_y - other_center_y) / denom


def pick_closer_neighbor(
    line: Optional[Line],
    other_line: Optional[Line],
    candidate_line: Line,
    reference_item: float,
) -> Optional[Line]:
    """Pick the closer neighboring line to the current line when it falls within the merge tolerance. Returns the closer candidate when the distance is below the threshold, else ``None``. Either or both candidates may be ``None`` (e.g. c is at the top of the tree -> no predecessor). """
    if line is None and other_line is None:
        return None
    entry_item = vertical_distance_in_line_heights(line, candidate_line) if line is not None else float("inf")
    second_candidate = vertical_distance_in_line_heights(other_line, candidate_line) if other_line is not None else float("inf")
    if entry_item >= reference_item and second_candidate >= reference_item:
        return None
    return line if entry_item < second_candidate else other_line


# --------------------------------------------------------------------------- #
# Line merge predicate.
# --------------------------------------------------------------------------- #


def should_merge_lines(line: Line, other_line: Line, candidate_items: list) -> bool:
    """Return whether ``other_line`` should merge into ``line``. The decision compares the horizontal gap against a tolerance based on harmonic mean character width, then adjusts for style mismatch, script category, dot leaders, column membership, short continuations, bracketed starts, sentence endings, and uppercase dominance."""
    if line.char_count() > 0 and other_line.char_count() > 0:
        # Different skew/rotation -> never merge
        if magnitude_ratio(line.previous_slot, other_line.previous_slot) > 2 and abs(line.previous_slot - other_line.previous_slot) > 10:
            return False

    # Harmonic mean of character heights with no clamp. A zero char-height
    # contributes an infinite inverse, driving the merge tolerance to zero.
    line_projection = 1.0 / avg_char_width(line) if avg_char_width(line) != 0 else float("inf")
    other_projection = 1.0 / avg_char_width(other_line) if avg_char_width(other_line) != 0 else float("inf")
    harmonic_char_width = 2.0 / (line_projection + other_projection)
    horizontal_gap = other_line.left_edge() - line.right_edge()                                       # horizontal gap
    gap_factor = 2.0

    # italic mismatch
    italic = line.bold_frac() > 0
    other_italic = other_line.bold_frac() > 0
    if italic != other_italic:
        gap_factor /= 1.5

    # last-char category 4 = other-letter (Lo, CJK/syllabics)
    # OR more than half of a's chars are category 4
    if line.char_stats.tertiary_slot == 4 or line.char_stats.primary_slot[4] > line.char_count() / 2:
        gap_factor /= 2.0

    # sentence-end + all-digits + dot leader pattern -> TOC row, don't merge
    sent_end = line.char_stats.tertiary_slot == 6
    if sent_end:
        # candidate numeric-token test: the candidate has digits and all characters are digits

        all_digits = other_line.char_stats.auxiliary_slot > 0 and other_line.char_stats.auxiliary_slot == other_line.char_stats.primary_slot[1]
        if all_digits and TRAILING_DOT_LEADER_RE.search(raw_text_of_line(line)):
            gap_factor *= 3.0
    else:
        all_digits = False

    # Column-based bonuses ----------------------------------------------------
    if candidate_items and 0 <= line.measure_slot < len(candidate_items):
        line_column = candidate_items[line.measure_slot]
        col_left = line_column.get("left", float("inf"))
        col_right = line_column.get("right", float("-inf"))
    else:
        col_left = float("inf")
        col_right = float("-inf")

    inside_col = (
        line.left_edge() >= col_left
        and line.right_edge() <= col_right
        and other_line.left_edge() >= col_left
        and other_line.right_edge() <= col_right
    )
    if (line.char_count() < 40 or inside_col) and (
        same_y_extent(line, other_line, 0.1) or same_y_extent(last_span(line), other_line, 0.1)
    ):
        gap_factor *= 1.5
    if line.char_count() < 40 and inside_col:
        gap_factor *= 2.0

    # At-column-edge demotion
    if 0 <= other_line.measure_slot < len(candidate_items):
        other_column = candidate_items[other_line.measure_slot]
    else:
        other_column = None
    if (
        len(candidate_items) <= 0
        or (
            abs(line.right_edge() - col_right) < 5
            and (line.measure_slot >= len(candidate_items) - 1 or not other_column or abs(other_line.left_edge() - other_column.get("left", float("inf"))) < 5)
        )
    ):
        gap_factor /= 2.0

    # Very short leading line with continuation evidence: short, low aspect,
    # numbering-like, and followed by text with letters. The inside-column flag
    # controls whether this gets the stronger multiplier.
    if line.char_count() <= 8 and line.bbox_width() <= 10 * line.avg_font_size() and numbering_kind(line) != 0 and letter_count(other_line.char_stats) > 0:
        gap_factor *= 3.0 if inside_col else 2.0

    # Bracketed short line or uppercase sentence-period inside a column.
    if line.char_count() <= 10:
        text = text_of_line(line)
        if text.startswith("[") and text.endswith("]"):
            gap_factor *= 2.0
        elif inside_col and line.char_stats.secondary_slot == 2 and text.endswith("."):
            gap_factor *= 2.0

    # Both lines are uppercase-dominant inside the same column.

    if inside_col and is_upper_dominant(line.char_stats) and is_upper_dominant(other_line.char_stats):
        gap_factor *= 1.5

    return horizontal_gap <= gap_factor * harmonic_char_width
