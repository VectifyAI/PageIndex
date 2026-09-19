"""Builds initial lines and clusters them into merged lines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sortedcontainers import SortedKeyList

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

from .merge_rules import (
    span_continues_line,
    pick_closer_neighbor,
    should_merge_lines,
)


# --------------------------------------------------------------------------- #
# Initial line builder.
# --------------------------------------------------------------------------- #


def _skip_mark_only(span: Span, page_area: float) -> bool:
    """Return True for mark-heavy tiny glyphs whose area is below one part per million of the page area."""
    return (span.char_count() - span.char_stats.primary_slot[5]) > 1 and span.area() < page_area * 1e-6


def build_initial_lines(spans: list[Span], page_bbox) -> list[Line]:
    """Build initial lines from flat spans. Returns the list of initial lines. """
    line: list[Line] = []
    pending_line = Line()
    pending_span: Optional[Span] = None
    page_area = page_bbox.area()

    for span in spans:
        if span.text == "" or _skip_mark_only(span, page_area):
            continue
        if pending_span is not None:
            # Overstrike duplicate detection: same trimmed text, both edges +
            # both top/bottom within 10% of f's geometry -> f gets the bold
            # bit and h is discarded.
            if (
                pending_span.char_count() > 0
                and pending_span.state_slot == span.state_slot
                and same_x_extent(pending_span, span, 0.1 * pending_span.bbox_width())
                and same_y_extent(pending_span, span, 0.1 * pending_span.bbox_height())
            ):
                pending_span.primary_slot = True
                continue
            # End current line if e is non-empty AND tn says NOT to continue
            if not (len(pending_line.primary_slot) <= 0 or span_continues_line(pending_line, pending_span)):
                line.append(pending_line)
                pending_line = Line()
            append_span(pending_line, pending_span)
            pending_span = span
        else:
            pending_span = span

    if pending_span is not None:
        if not (len(pending_line.primary_slot) <= 0 or span_continues_line(pending_line, pending_span)):
            line.append(pending_line)
            pending_line = Line()
        append_span(pending_line, pending_span)
        line.append(pending_line)
    # If no current span exists, the pending line is intentionally dropped.
    # This path is currently unreachable from the loop logic.
    return line


# --------------------------------------------------------------------------- #
# xn -- line clustering driver #
# --------------------------------------------------------------------------- #


def _is_label_stack(line: Line, other_line: Line, body_ma: float) -> bool:
    """Detect a display-sized label stacked directly above the text it labels. The geometry must overlap horizontally while sitting on a different baseline; the upper piece must be display-sized relative to body text and larger than the lower text. This captures chapter numbers and drop caps that should be read before the title below them."""
    if body_ma <= 0:
        return False
    overlap = min(line.right_edge(), other_line.right_edge()) - max(line.left_edge(), other_line.left_edge())
    frac = overlap / max(1e-6, min(line.bbox_width(), other_line.bbox_width()))
    vertical_overlap = min(line.top_edge(), other_line.top_edge()) - max(line.bottom_edge(), other_line.bottom_edge())
    vertical_overlap_fraction = vertical_overlap / max(1e-6, min(line.bbox_height(), other_line.bbox_height()))
    if not (frac > 0.5 and vertical_overlap_fraction < 0.5):
        return False
    upper, lower = (line, other_line) if line.center_y() > other_line.center_y() else (other_line, line)
    # display-type (>= 2x body) AND larger than the text it sits above
    # (>= 1.5x lower): a leading label over smaller text. The second clause
    # drops same-size display stacks (e.g. chart axis numbers over each other).
    return upper.avg_font_size() >= 2.0 * body_ma and upper.avg_font_size() >= 1.5 * lower.avg_font_size()


@dataclass
class LinesContainer:
    """Mutable line container used by the clustering pass."""

    primary_slot: list[Line] = field(default_factory=list)


def _set_add(tree: SortedKeyList, line: Line) -> None:
    """Set-style insertion into the sorted line index. Lines with identical top, bottom, left, and right ordering keys are dropped instead of duplicated."""
    idx = tree.bisect_left(line)
    if idx < len(tree) and reading_order_key(tree[idx]) == reading_order_key(line):  # type: ignore[arg-type]
        return  # reading-order key collision -> sorted set insertion drops the element
    tree.add(line)


def cluster_lines(lines_container: LinesContainer, other_item: float, candidate_items: list) -> list[Line]:
    """Mutate the contained line list by merging nearby compatible lines."""
    # Sort input lines by reading order.
    lines_container.primary_slot.sort(key=left_edge_key)

    # Body-text reference for the display-size test in _is_label_stack: the
    # median glyph font size across the page (dominated by body text).
    merged_accent_spans = sorted(
        span_value.font_size for line in lines_container.primary_slot for span_value in line.primary_slot
        if getattr(span_value, "font_size", 0) > 0
    )
    body_ma = merged_accent_spans[len(merged_accent_spans) // 2] if merged_accent_spans else 0.0

    # Tree of lines, ordered by reading position (top desc, bottom desc, left, right).
    tree: SortedKeyList = SortedKeyList(key=reading_order_key)
    merged_lines: list[Line] = []                # output (lines that won't merge further)

    for candidate_line in lines_container.primary_slot:
        # Rotated / skewed lines: don't try to cluster, just emit
        if last_span(candidate_line).previous_slot > 1:
            merged_lines.append(candidate_line)
            continue

        # successor (just below f vertically) and predecessor (just above).
        # predecessor/successor search are INCLUSIVE floor/ceiling, so a reading-order-key-equal line already
        # in the tree is the zero-distance neighbour: successor = bisect_left
        # (first key >= f), predecessor = bisect_right - 1 (last key <= f).
        idx_succ = tree.bisect_left(candidate_line)
        successor_line = tree[idx_succ] if idx_succ < len(tree) else None
        idx_pred = tree.bisect_right(candidate_line)
        line_item = tree[idx_pred - 1] if idx_pred > 0 else None

        neighbor_line = pick_closer_neighbor(line_item, successor_line, candidate_line, other_item)
        if neighbor_line is None:
            _set_add(tree, candidate_line)
            continue

        tree.remove(neighbor_line)
        neighbor_last_span = last_span(neighbor_line)                                            # last span of k

        # Subscript / overstrike case (single-span f duplicating k's last span)
        if (
            len(candidate_line.primary_slot) == 1
            and len(neighbor_line.primary_slot) <= 5
            and neighbor_last_span.char_count() > 0
            and neighbor_last_span.state_slot == candidate_line.primary_slot[0].state_slot
            and same_x_extent(neighbor_last_span, candidate_line, 0.1 * neighbor_last_span.bbox_width())
            and same_y_extent(neighbor_last_span, candidate_line, 0.1 * neighbor_last_span.bbox_height())
        ):
            if abs(neighbor_last_span.left_edge() - candidate_line.left_edge()) < 0.01 and abs(neighbor_last_span.top_edge() - candidate_line.top_edge()) < 0.01:
                # exact duplicate -> keep the original line unchanged
                _set_add(tree, neighbor_line)
                continue
            # Otherwise create a new line carrying k's spans with m marked bold
            new_line = Line()
            neighbor_last_span.primary_slot = True
            for source_span in neighbor_line:
                append_span(new_line, source_span)
            _set_add(tree, new_line)
        elif should_merge_lines(neighbor_line, candidate_line, candidate_items):
            # Continuation merge. Normally append f after k (left-to-right).
            # If the candidate is a display-sized label stacked above the text,
            # reading order is top-to-bottom, so the label leads. Reorder spans
            # only; the merge and block/line structure stay unchanged.
            if _is_label_stack(neighbor_line, candidate_line, body_ma) and candidate_line.center_y() > neighbor_line.center_y():
                merged = Line()
                for span in candidate_line:
                    append_span(merged, span)
                for span in neighbor_line:
                    append_span(merged, span)
                _set_add(tree, merged)
            else:
                for span in candidate_line:
                    append_span(neighbor_line, span)
                _set_add(tree, neighbor_line)
        else:
            # Cannot merge: emit k as a finalized line, start fresh with f
            merged_lines.append(neighbor_line)
            _set_add(tree, candidate_line)

    # Drain remaining
    merged_lines.extend(tree)
    # Final sort by reading order.
    merged_lines.sort(key=reading_order_key)
    lines_container.primary_slot = merged_lines
    return merged_lines
