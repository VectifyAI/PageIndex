"""Recursive column splitting and column index assignment."""

from __future__ import annotations

import math
from typing import Optional

from ..model import (
    Rect, rect_union, EMPTY_RECT, Line, info_weight, text_of_line, numbering_kind, numbering_value, _UNICODE_WHITESPACE_CLASS, _max_nan_propagating, _min_nan_propagating,
)

from .gutters import (
    SweepEvent,
    SplitCandidate,
    ColumnDetectionContext,
    collect_gutter_candidates,
)


# --------------------------------------------------------------------------- #
# Assign column indexes to lines whose event is a start edge.
# --------------------------------------------------------------------------- #


def assign_column_index(items: list[SweepEvent], other_number: int) -> None:
    """Assign ``column_index`` to each gutter event that starts a column-owned line."""
    for column in items:
        if column.is_start:
            column.line.measure_slot = other_number


# --------------------------------------------------------------------------- #
# Recursive split driver.
# --------------------------------------------------------------------------- #


def recursive_split(
    context: ColumnDetectionContext,
    horizontal_events: list[SweepEvent],                    # horizontal events (sorted by F/L)
    vertical_events: list[SweepEvent],                    # vertical events (sorted by C/D)
    reference_rect: Rect,                          # root rect
    current_rect: Rect,                          # current sub-rect
    depth: int,                         # depth
    column_offset: int,                         # column-index offset
) -> list[Rect]:
    if depth >= context.tertiary_slot:
        assign_column_index(horizontal_events, column_offset)
        return [current_rect]

    split_candidates: list[SplitCandidate] = []
    if current_rect.bbox_height() >= context.measure_slot:
        collect_gutter_candidates(context, reference_rect, current_rect, horizontal_events, 1, current_rect.bbox_height(), context.option_slot, split_candidates)
    if current_rect.bbox_width() >= context.state_slot:
        collect_gutter_candidates(context, reference_rect, current_rect, vertical_events, 0, current_rect.bbox_width(), context.auxiliary_slot, split_candidates)

    if len(split_candidates) <= 0:
        if current_rect.bbox_width() < 0.8 * reference_rect.bbox_width():
            assign_column_index(horizontal_events, column_offset)
            return [current_rect]
        # Examine gaps in b for vertical gutters (fallback)
        key_value = active_overlap_count = 0
        measure_item = local = 0
        gap_indices: list[int] = []
        gap_scan_index = 0
        while gap_scan_index < len(horizontal_events) - 1:
            width_value = horizontal_events[gap_scan_index].position
            event_is_start = horizontal_events[gap_scan_index].is_start
            candidate_item = horizontal_events[gap_scan_index].line
            if width_value > reference_rect.left + reference_rect.bbox_width() * 5 / 6:
                break
            if event_is_start:
                active_overlap_count += 1
                key_value += info_weight(candidate_item.char_stats)
            else:
                active_overlap_count -= 1
                key_value -= info_weight(candidate_item.char_stats)
            local = max(local, active_overlap_count)
            measure_item = max(measure_item, key_value)
            if event_is_start or active_overlap_count > 2 or width_value < reference_rect.left + reference_rect.bbox_width() / 6:
                gap_scan_index += 1
                continue
            previous_gap_index = gap_indices[-1] if gap_indices else None
            if previous_gap_index is not None and width_value < horizontal_events[previous_gap_index].position + reference_rect.bbox_width() / 10:
                gap_indices[-1] = gap_scan_index
            elif local >= 8 and measure_item >= 100:
                gap_indices.append(gap_scan_index)
            local = measure_item = 0
            gap_scan_index += 1

        if len(gap_indices) <= 0 or len(gap_indices) > 2:
            assign_column_index(horizontal_events, column_offset)
            return [current_rect]
        if local < 4 or measure_item < 50:
            assign_column_index(horizontal_events, column_offset)
            return [current_rect]
        # Assign columns based on the discovered gaps
        out: list[Rect] = []
        cursor = 0
        for index in range(len(gap_indices) + 1):
            pos = gap_indices[index] if index < len(gap_indices) else len(horizontal_events)
            for event_index in range(cursor, pos):
                sweep_event = horizontal_events[event_index]
                if sweep_event.is_start:
                    sweep_event.line.measure_slot = column_offset + len(out)
            end = horizontal_events[pos].position if pos < len(horizontal_events) else current_rect.right_edge()
            out.append(Rect(horizontal_events[cursor].position, end, current_rect.top, current_rect.bottom_edge()))
            cursor = pos + 1
        return out

    # Pick best candidate split
    best: Optional[SplitCandidate] = None
    for count_item in split_candidates:
        if best is None or best.score < count_item.score:
            best = count_item
    assert best is not None  # h is non-empty here

    if best.direction == 0:
        # Horizontal split (vertical gutter): divide events into top / bottom halves
        upper_left, split_max = math.inf, -math.inf
        value, lower_right = math.inf, -math.inf
        upper_horizontal_events: list[SweepEvent] = []
        lower_horizontal_events: list[SweepEvent] = []
        for horizontal_event in horizontal_events:
            line = horizontal_event.line
            if line.top_edge() > best.start:
                upper_horizontal_events.append(horizontal_event)
                upper_left = min(upper_left, line.left_edge())
                split_max = max(split_max, line.right_edge())
            elif line.bottom_edge() < best.end:
                lower_horizontal_events.append(horizontal_event)
                value = min(value, line.left_edge())
                lower_right = max(lower_right, line.right_edge())
        upper_vertical_events: list[SweepEvent] = []
        split: list[SweepEvent] = []
        for vertical_event in vertical_events:
            if vertical_event.position > best.start:
                upper_vertical_events.append(vertical_event)
            elif vertical_event.position < best.end:
                split.append(vertical_event)
        upper = recursive_split(context, upper_horizontal_events, upper_vertical_events, reference_rect, Rect(upper_left, split_max, current_rect.top, best.end), depth + 1, column_offset)
        lower = recursive_split(
            context,
            lower_horizontal_events,
            split,
            reference_rect,
            Rect(value, lower_right, best.start, current_rect.bottom_edge()),
            depth + 1,
            column_offset + len(upper),
        )
        return upper + lower

    # Vertical split (horizontal gutter): divide events into left / right halves
    split_max, left_bottom = -math.inf, math.inf
    right_top, right_bottom = -math.inf, math.inf
    left_events: list[SweepEvent] = []
    right_events: list[SweepEvent] = []
    left_vert: list[SweepEvent] = []
    right_vert: list[SweepEvent] = []
    for split_event in horizontal_events:
        if split_event.position < best.end:
            left_events.append(split_event)
        elif split_event.position > best.start:
            right_events.append(split_event)
    for event in vertical_events:
        line = event.line
        if line.left_edge() < best.end:
            left_vert.append(event)
            split_max = max(split_max, line.top_edge())
            left_bottom = min(left_bottom, line.bottom_edge())
        elif line.right_edge() > best.start:
            right_vert.append(event)
            right_top = max(right_top, line.top_edge())
            right_bottom = min(right_bottom, line.bottom_edge())
    left = recursive_split(
        context, left_events, left_vert, reference_rect, Rect(current_rect.left, best.start, split_max, left_bottom), depth + 1, column_offset
    )
    right = recursive_split(
        context,
        right_events,
        right_vert,
        reference_rect,
        Rect(best.end, current_rect.right_edge(), right_top, right_bottom),
        depth + 1,
        column_offset + len(left),
    )
    return left + right


# --------------------------------------------------------------------------- #
# Build events, sort them, then start recursive splitting.
# --------------------------------------------------------------------------- #


def detect_columns(column: ColumnDetectionContext) -> list[Rect]:
    """Detect column rectangles and populate each line's column index."""
    horizontal_events: list[SweepEvent] = []
    vertical_events: list[SweepEvent] = []
    bbox = EMPTY_RECT
    for line in column.primary_slot:
        if line.bbox_width() <= 0 or line.bbox_height() <= 0:
            continue
        bbox = rect_union(bbox, line.secondary_slot)
        horizontal_events.append(SweepEvent(line, line.left_edge(), True))
        horizontal_events.append(SweepEvent(line, line.right_edge(), False))
        vertical_events.append(SweepEvent(line, line.bottom_edge(), True))
        vertical_events.append(SweepEvent(line, line.top_edge(), False))

    # Sort by position, start events before end events, then line width.
    horizontal_events.sort(key=lambda split_event: (split_event.position, 0 if split_event.is_start else 1, split_event.line.bbox_width()))
    # Sort by position, start events before end events, then line height.
    vertical_events.sort(key=lambda split_event: (split_event.position, 0 if split_event.is_start else 1, split_event.line.bbox_height()))

    return recursive_split(column, horizontal_events, vertical_events, bbox, bbox, 0, 0)


# --------------------------------------------------------------------------- #
# Public helper: produce the {left, right} dict list used by line merging #
# --------------------------------------------------------------------------- #


def columns_to_x_bounds(column_rects: list[Rect]) -> list[dict]:
    """Convert column rectangles to a ``[{left, right}, ...]`` table."""
    return [{"left": column.left, "right": column.right} for column in column_rects]
