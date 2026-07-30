"""Gutter-gap candidates and scoring for column detection."""

from __future__ import annotations

import math
from typing import Optional

from ..model import (
    Rect, rect_union, EMPTY_RECT, Line, info_weight, text_of_line, numbering_kind, numbering_value, _UNICODE_WHITESPACE_CLASS, _max_nan_propagating, _min_nan_propagating,
)


# Detect TOC dot leaders ("... 5", "....3"). Gutter scoring rejects a split
# candidate when too many dot-leader lines straddle the gap, because a TOC page
# should remain in one reading region.
# The regular expression is end-anchored only; use re.search rather than re.match.
import re as re_module


# --------------------------------------------------------------------------- #
# Sweep event. ``is_start=True`` means "line enters" at a start edge; False means
# "line leaves" at an end edge.
# --------------------------------------------------------------------------- #


class SweepEvent:
    __slots__ = ("line", "position", "is_start")

    def __init__(self, line: Line, position: float, is_start_flag: bool):
        self.line = line
        self.position = position
        self.is_start = is_start_flag


# --------------------------------------------------------------------------- #
# Column-split candidate. Direction 0 is a vertical sweep; direction 1 is a
# horizontal sweep. Higher score is better.
# --------------------------------------------------------------------------- #


class SplitCandidate:
    __slots__ = ("start", "end", "direction", "score")

    def __init__(self, start: float, end: float, direction_value: int, score: float):
        self.start = start
        self.end = end
        self.direction = direction_value
        self.score = score


# --------------------------------------------------------------------------- #
# Detection context. Thresholds derived from page geometry and page statistics.
# --------------------------------------------------------------------------- #


class ColumnDetectionContext:
    """Page-level thresholds used while recursively scoring gutter candidates."""

    __slots__ = ("secondary_slot", "primary_slot", "tertiary_slot", "state_slot", "auxiliary_slot", "option_slot", "measure_slot")

    def __init__(self, primary_item, secondary_item, candidate_item):
        self.secondary_slot = secondary_item
        self.primary_slot = candidate_item
        # ``log2(0)`` would be -inf -- guard against empty input.
        self.tertiary_slot = math.floor(2 * math.log2(len(candidate_item))) if candidate_item else 0
        self.state_slot = primary_item.bbox_width() / 6.0
        self.auxiliary_slot = _max_nan_propagating(0.5 * secondary_item.primary_slot, _min_nan_propagating(1.1 * (secondary_item.tertiary_slot - secondary_item.primary_slot), 3.0 * secondary_item.primary_slot))
        self.option_slot = secondary_item.measure_slot
        self.measure_slot = 1.5 * secondary_item.primary_slot
DOT_LEADER_RE = re_module.compile(r"([.][" + _UNICODE_WHITESPACE_CLASS + r"]*){5,}\Z")


# --------------------------------------------------------------------------- #
# Score split candidates in a sweep.
# --------------------------------------------------------------------------- #


def collect_gutter_candidates(
    context: ColumnDetectionContext,
    other_rect: Rect,                          # root rect (page bbox)
    candidate_rect: Rect,                          # current sub-rect
    events: list[SweepEvent],                    # sorted events
    direction: int,                         # direction: 0 vert sweep / 1 horiz sweep
    extent: float,                       # extent (height or width)
    min_gap: float,               # minimum gutter size
    out_candidates: list[SplitCandidate],                    # output: candidates to append to
) -> None:
    """Walk adjacent event pairs looking for column gutters. Each candidate gap receives a multiplicative score from line weight, font-size balance, indent/outdent structure, citation markers, and edge proximity; the best viable score wins."""
    active_count = 0
    for size_value in range(len(events) - 1):
        if events[size_value].is_start:
            active_count += 1
        else:
            active_count -= 1
        if active_count > 0:
            continue
        # The gap between adjacent event positions is a candidate gutter.
        score = _score_gutter_gap(context, other_rect, candidate_rect, events, direction, extent, min_gap, size_value)
        if score is not None:
            key_value = events[size_value].position
            score_value = events[size_value + 1].position
            out_candidates.append(SplitCandidate(key_value, score_value, direction, score))


def _score_gutter_gap(
    primary_item: ColumnDetectionContext,
    other_rect: Rect,                          # root rect
    candidate_rect: Rect,                          # current sub-rect (variable name p follows the extraction rule)

    reference_items: list[SweepEvent],                    # events
    next_number: int,                         # direction
    extent: float,                # extent
    min_gap: float,               # min gutter
    limit_number: int,                         # current event index
) -> Optional[float]:
    """Score one candidate gap at adjacent sweep events, or return None when it is not viable."""
    key_value = reference_items[limit_number].position
    score_value = reference_items[limit_number + 1].position
    item_value = score_value - key_value
    if item_value < min_gap:
        return None

    # --- backward pass: lines that close before this gap -----------------
    measure_item = preceding_max_width = 0
    secondary_item = reference_item = distance_accumulator = width_value = wide_accumulator = 0.0
    min_edge = math.inf
    preceding_max_trailing_edge = -math.inf
    min_edge_position = math.inf
    max_edge = -math.inf
    event_count = 0
    lower_accumulator = math.inf
    candidate_item = group_value = max_char_count = state_item = 0
    sample_item = limit_number
    while sample_item >= 0:
        sweep_event = reference_items[sample_item]
        event_position = sweep_event.position
        event_is_start = sweep_event.is_start
        sweep_line = sweep_event.line
        if event_position < key_value - item_value:
            break
        if event_is_start:
            sample_item -= 1
            continue
        measure_item += 1
        preceding_max_width = max(preceding_max_width, sweep_line.bbox_width())
        if next_number == 1:
            leading_edge, trailing_edge = sweep_line.bottom_edge(), sweep_line.top_edge()
        else:
            leading_edge, trailing_edge = sweep_line.left_edge(), sweep_line.right_edge()
        min_edge = min(min_edge, leading_edge)
        preceding_max_trailing_edge = max(preceding_max_trailing_edge, trailing_edge)
        entry_item = info_weight(sweep_line.char_stats)
        secondary_item += entry_item
        if sweep_line.avg_font_size() > reference_item:
            reference_item = sweep_line.avg_font_size()
            distance_accumulator = entry_item
        elif sweep_line.avg_font_size() == reference_item:
            distance_accumulator += entry_item
        if key_value - event_position < 1:
            width_value += 1
            wide_accumulator = max(wide_accumulator, sweep_line.bbox_width())
            min_edge_position = min(min_edge_position, leading_edge)
            max_edge = max(max_edge, trailing_edge)
        if numbering_kind(sweep_line) == 1:
            event_count += 1
            if numbering_value(sweep_line) == 1:
                lower_accumulator = min(lower_accumulator, sweep_line.left_edge())
        if next_number == 1:
            if sweep_line.char_count() <= 5 and numbering_kind(sweep_line) != 0:
                candidate_item += 1
            if sweep_line.char_count() <= 10:
                text = text_of_line(sweep_line)
                if (text.startswith("[") and text.endswith("]")) or (
                    sweep_line.char_stats.secondary_slot == 2 and text.endswith(".")
                ):
                    group_value += 1
            if DOT_LEADER_RE.search(text_of_line(sweep_line)):
                state_item += 1
        max_char_count = max(max_char_count, sweep_line.char_count())
        sample_item -= 1

    # Sanity gates
    if (
        candidate_item >= measure_item
        or candidate_item >= max(2, measure_item / 2)
        or group_value >= measure_item
        or state_item >= max(2, measure_item / 2)
        or (next_number == 1 and max_char_count <= 1)
    ):
        return None

    # --- forward pass: lines that open after this gap --------------------
    next_gap = 0.0
    following_line_count = 0
    other_gap = math.inf
    following_max_trailing_edge = -math.inf
    page_gap = following_max_font_size = after = 0
    quantity = 0
    numbering_score = following_numbering_count = following_edge_max_width = 0
    right_gap = 0.0
    count_item = limit_number + 1
    while count_item < len(reference_items):
        sweep_event = reference_items[count_item]
        event_position = sweep_event.position
        event_is_start = sweep_event.is_start
        sweep_line = sweep_event.line
        if event_position > score_value + item_value:
            break
        if not event_is_start:
            count_item += 1
            continue
        following_line_count += 1
        next_gap = max(next_gap, sweep_line.bbox_width())
        if next_number == 1:
            leading_edge, trailing_edge = sweep_line.bottom_edge(), sweep_line.top_edge()
        else:
            leading_edge, trailing_edge = sweep_line.left_edge(), sweep_line.right_edge()
        other_gap = min(other_gap, leading_edge)
        following_max_trailing_edge = max(following_max_trailing_edge, trailing_edge)
        width = info_weight(sweep_line.char_stats)
        after += width
        if sweep_line.avg_font_size() > following_max_font_size:
            following_max_font_size = sweep_line.avg_font_size()
            page_gap = width
        elif sweep_line.avg_font_size() == following_max_font_size:
            page_gap += width
        if event_position - score_value < 1:
            quantity += 1
            following_edge_max_width = max(following_edge_max_width, sweep_line.bbox_width())
        if numbering_kind(sweep_line) == 1:
            following_numbering_count += 1
            numbering_score = _max_nan_propagating(numbering_score, numbering_value(sweep_line))
            right_gap = _max_nan_propagating(right_gap, sweep_line.top_edge())
        count_item += 1

    if measure_item <= 0 or following_line_count <= 0:
        return None

    # Combined scoring mixes the root page rectangle for page-level thresholds
    # with the current recursive sub-rectangle for split geometry. Root and sub
    # coincide before the first split, but diverge on genuinely multi-column
    # pages; keeping both frames is part of the column decision model.
    root_width = other_rect.bbox_width()
    height = other_rect.bbox_height()
    root_center_x = other_rect.center_x()

    if next_number == 1:
        # vertical sweep: special pre-gate for single-line columns
        # The top-edge gate compares the sub-rectangle to the root page height.

        if width_value <= 1 and quantity <= 1 and not (
            candidate_rect.top_edge() < other_rect.bottom_edge() + 0.3 * height
            and lower_accumulator < math.inf
            and numbering_score <= 4
        ):
            return None
        if (secondary_item <= 100 and after <= 100) and (
            key_value < other_rect.left + 0.2 * root_width or score_value > other_rect.left + 0.8 * root_width
        ):
            return None

    mid = (reference_item + following_max_font_size) / 2

    if next_number == 0 and distance_accumulator >= 0.8 * secondary_item and page_gap >= 0.8 * after and (
        (
            abs(reference_item - following_max_font_size) < 0.1
            and reference_item >= primary_item.secondary_slot.primary_slot + 0.5
            and following_max_font_size >= primary_item.secondary_slot.primary_slot + 0.5
            and item_value < max(1.3 * mid, min_gap * 2)
        )
        or (
            reference_item >= primary_item.secondary_slot.primary_slot + 2
            and following_max_font_size >= primary_item.secondary_slot.primary_slot + 2
            and item_value < max(1.5 * mid, min_gap * 3)
        )
    ):
        return None

    line_value = extent * extent * item_value

    if next_number == 1:
        line_value *= min(width_value, quantity)
        if secondary_item <= 50 and measure_item <= 1:
            line_value /= 100
        candidate_height = candidate_rect.bbox_height()
        threshold = candidate_rect.top_edge() - 0.2 * candidate_height
        if following_numbering_count >= 3 and numbering_score >= 6 and right_gap < threshold:
            # Preserve IEEE division here: a zero denominator yields +inf and a
            # negative denominator is clamped below. The branch selection depends
            # on those numeric edge cases.
            denom = numbering_score - following_numbering_count
            inv = (1 / denom) if denom != 0 else math.inf
            factor = max(0.3, min(1.0, inv))
            factor *= factor
            line_value *= factor
        elif candidate_height > height / 2:
            factor = candidate_height / height
            factor *= factor
            line_value *= 1 + factor
        line_value *= max(1, 2 - abs(root_center_x - (key_value + score_value) / 2) / root_width * 10)

    if next_number == 0:
        candidate_width = candidate_rect.bbox_width()
        line_value *= max(wide_accumulator, following_edge_max_width) / candidate_width * (max(preceding_max_width, next_gap) / candidate_width)
        min_value = min(min_edge, other_gap)
        max_value = max(preceding_max_trailing_edge, following_max_trailing_edge)
        if min_value < root_center_x and max_value > root_center_x:
            left_center_distance = root_center_x - min_value
            value = max_value - root_center_x
            line_value *= 1 + min(left_center_distance, value) / max(left_center_distance, value)
        # Edge bands are measured from the root page rectangle.

        edge_top = other_rect.primary_slot + 0.2 * height
        edge_bot = other_rect.primary_slot + 0.8 * height
        if key_value < edge_top or score_value > edge_bot:
            line_value *= 4
            # The lower-edge boost uses forward-pass numbering and width counts,
            # because it is testing the material below the candidate gap.

            if (key_value < edge_top and event_count >= 1 and wide_accumulator < root_width / 4) or (
                score_value > edge_bot and following_numbering_count >= 1 and following_edge_max_width < root_width / 4
            ):
                line_value *= 9
        if 2 * width_value >= limit_number and reference_item > following_max_font_size + 0.5:
            line_value *= 100
        if lower_accumulator < math.inf:
            if lower_accumulator < root_center_x:
                line_value *= 100
        elif event_count >= 2 or following_numbering_count >= 2:
            line_value /= 4
        size = max(reference_item, following_max_font_size)
        # This test uses the full sweep extent, not the minimum gutter size.
        if (
            extent >= 0.99 * root_width
            and min_edge_position < root_center_x
            and max_edge > root_center_x
            and reference_item >= following_max_font_size + 0.5
            and item_value > size
        ):
            line_value *= item_value / size

    if secondary_item < 1 or after < 1:
        line_value *= 10

    return _max_nan_propagating(0.0, line_value)
