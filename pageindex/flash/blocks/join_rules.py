"""Line-to-block joining rules and the section-heading trie."""

from __future__ import annotations

from typing import Optional

import json
from pathlib import Path

from ..model import (
    style_key,
    magnitude_ratio,
    left_aligned,
    right_aligned,
    center_aligned,
    x_centers_close,
    Rect,
    last_span,
    avg_char_width,
    EMPTY_RECT,
    left_edge_key,
    reading_order_key,
    numbering_kind,
    Line,
    case_signal,
    last_line_of,
    first_span_of,
    letter_count,
    dominant_style_of,
    is_upper_dominant,
    Block,
    _max_nan_propagating,
)
from ..stats import DocStats, PageStats
from ..tokens import set_case_fold, TrieConfig, build_trie, tokenize_block


# Combined heading trie used to detect "first line is a section header" patterns
# when splitting two-line blocks.
_DICT_PATH = Path(__file__).parent.parent / "data" / "dictionaries.json"
_DICTS = json.loads(_DICT_PATH.read_text(encoding="utf-8"))
SECTION_HEADING_TRIE = build_trie(
    list(_DICTS.get("section_keywords", []))
    + list(_DICTS.get("abstract_keywords", []))
    + list(_DICTS.get("references", [])),
    set_case_fold(TrieConfig(), True),
)


# --------------------------------------------------------------------------- #
# Block-clustering context bundle #
# --------------------------------------------------------------------------- #


class BlockClusterContext:
    """Block-clustering context. Fields: j document statistics o page bbox g page statistics h lines to cluster v column rectangles """

    __slots__ = ("tertiary_slot", "auxiliary_slot", "primary_slot", "secondary_slot", "state_slot")

    def __init__(self, doc_stats: DocStats, page_bbox: Rect, page_stats: PageStats, lines: list, columns: list):
        self.tertiary_slot = doc_stats
        self.auxiliary_slot = page_bbox
        self.primary_slot = page_stats
        self.secondary_slot = lines
        self.state_slot = columns


# --------------------------------------------------------------------------- #
# Should a line join an existing block? #
# --------------------------------------------------------------------------- #


def should_join_line_to_block(
    block_cluster_ctx: BlockClusterContext,
    other_block: Block,

    candidate_line: Line,

    previous_line: Optional[Line],

    first_candidate_block: Block,

) -> bool:
    """Return True iff the candidate line should be appended to the current block."""
    # -- Step 1: reject incompatible skew ----------
    if abs(other_block.skew_frac() - candidate_line.skew_frac()) > 1:
        return False

    # -- Step 2: size + alignment gates ----------------------------------
    font_size_delta = candidate_line.avg_font_size() - other_block.avg_font_size()

    left_edges_aligned = left_aligned(other_block, candidate_line, 1)

    both_edges_aligned = left_edges_aligned or (other_block.line_count() == 1 and left_aligned(other_block, candidate_line, 8 * avg_char_width(other_block.line())))

    right_edges_aligned = right_aligned(other_block, candidate_line, 2)

    both_edges_aligned = both_edges_aligned and right_edges_aligned
    # m = min size-excess over page body; k = min size-excess over doc body
    page_body_font_delta = min(candidate_line.avg_font_size() - block_cluster_ctx.primary_slot.primary_slot, other_block.avg_font_size() - block_cluster_ctx.primary_slot.primary_slot)

    doc_body_font_delta = min(candidate_line.avg_font_size() - block_cluster_ctx.tertiary_slot.primary_slot, other_block.avg_font_size() - block_cluster_ctx.tertiary_slot.primary_slot)


    block_last_span = last_span(last_line_of(other_block))

    line_first_span = candidate_line.primary_slot[0]


    if (
        abs(font_size_delta) > page_body_font_delta
        and abs(font_size_delta) > doc_body_font_delta - 2
        and not (style_key(block_last_span) == style_key(line_first_span) and block_last_span.char_count() > 1 and line_first_span.char_count() > 1)
        and (
            font_size_delta > 2
            or (font_size_delta > 1 and not both_edges_aligned)
            or font_size_delta < -5
            or (font_size_delta < -2 and candidate_line.char_count() >= 5)
            or (font_size_delta < -1 and candidate_line.char_count() >= 20 and not both_edges_aligned)
        )
    ):
        return False

    # -- Step 3: font / bold mismatch ------------------------------------
    block_last_line = last_line_of(other_block)

    width_ratio = magnitude_ratio(other_block.bbox_width(), candidate_line.bbox_width())

    bold_mismatch = (block_last_span.primary_slot != line_first_span.primary_slot)

    font_mismatch = (
        block_last_span.font_name != line_first_span.font_name
        and dominant_style_of(other_block) != style_key(line_first_span)
    )


    if font_mismatch or bold_mismatch:
        if bold_mismatch and width_ratio > 2:
            return False
        if (block_last_line.char_stats.secondary_slot == 1 or block_last_line.char_stats.secondary_slot == 2) and (
            candidate_line.char_stats.secondary_slot == 2 or width_ratio > 4
        ):
            return False
        if block_last_line.char_stats.tertiary_slot == 6 or other_block.bbox_width() > 1.5 * block_last_line.bbox_width():
            return False

    if other_block.bold_frac() > 0.9 and candidate_line.bold_frac() < 0.8 and width_ratio > 2:
        return False

    # -- Step 4: spatial gates -------------------------------------------
    centers_aligned = center_aligned(other_block, candidate_line, 1)

    if not centers_aligned:
        vertical_gap = other_block.bottom_edge() - candidate_line.top_edge()

        horizontal_offset = candidate_line.left_edge() - other_block.left_edge()

        if (vertical_gap > -1 and horizontal_offset > 0.33 * other_block.bbox_width()) or horizontal_offset > 0.98 * other_block.bbox_width():
            return False
        if candidate_line.center_x() < other_block.left_edge():
            return False

    # -- Step 5: tolerance base ------------------------------------------
    bottom_edge_gap = other_block.bottom_edge() - candidate_line.bottom_edge()

    join_tolerance = (
        _max_nan_propagating(1.3 * (other_block.top_edge() - other_block.bottom_edge()) / other_block.line_count(), block_cluster_ctx.primary_slot.tertiary_slot)
        + 1.3 * other_block.avg_font_size()
    ) / 2.0


    # -- Step 6: case-flip "hanging indent" detector ---------------------
    block_case_signal = case_signal(other_block.char_stats)

    line_case_signal = case_signal(candidate_line.char_stats)

    # Capture the old block-last span before comparing both sides of the case
    # transition.
    case_signal_flip = (
        ((block_case_signal == 1 and line_case_signal == -1) or (line_case_signal == 1 and block_case_signal == -1))
        and letter_count(candidate_line.char_stats) >= 3
        and (is_upper_dominant(other_block.char_stats) != is_upper_dominant(line_first_span.char_stats) or letter_count(line_first_span.char_stats) < 3)
        and (is_upper_dominant(block_last_span.char_stats) != is_upper_dominant(candidate_line.char_stats) or letter_count(block_last_span.char_stats) < 3)
    )


    if (
        not font_mismatch and not bold_mismatch and not case_signal_flip
        and (width_ratio <= 1.2 or left_aligned(block_last_line, candidate_line, 0.1))
        # Preserve the no-guard width-ratio edge case: a zero-width block still
        # allows a positive-width last line to increase the join tolerance.
        and (block_last_line.bbox_width() / other_block.bbox_width() > 0.9 if other_block.bbox_width() != 0 else block_last_line.bbox_width() > 0)
    ):
        join_tolerance *= 1.3
    if page_body_font_delta > 0.5 * block_cluster_ctx.primary_slot.primary_slot and not case_signal_flip:
        join_tolerance *= 2

    # -- Step 7: column alignment ----------------------------------------
    column_rect = (block_cluster_ctx.state_slot[candidate_line.measure_slot] if (0 <= candidate_line.measure_slot < len(block_cluster_ctx.state_slot)) else None) or EMPTY_RECT

    line_left_aligned_to_column = left_aligned(candidate_line, column_rect, 4.5)

    line_right_aligned_to_column = right_aligned(candidate_line, column_rect, 4.5)

    block_left_aligned_to_column = left_aligned(other_block, column_rect, 4.5)

    block_right_aligned_to_column = right_aligned(other_block, column_rect, 4.5)

    block_column_justified = (
        block_left_aligned_to_column == block_right_aligned_to_column
        and other_block.alignment_slot
        and x_centers_close(block_cluster_ctx.auxiliary_slot, other_block)
    )

    line_column_centered = (
        line_left_aligned_to_column == line_right_aligned_to_column
        and (x_centers_close(block_cluster_ctx.auxiliary_slot, candidate_line) or (block_column_justified and centers_aligned))
    )


    # -- Step 8: alignment multipliers -----------------------------------
    if (
        block_column_justified and line_column_centered
        and other_block.bbox_width() > 0.5 * candidate_line.bbox_width()
        and (previous_line is None or candidate_line.bottom_edge() - previous_line.bottom_edge() >= bottom_edge_gap)
        and not font_mismatch
    ):
        join_tolerance *= 1.3
        if previous_line is not None and (
            (other_block.bold_frac() > previous_line.bold_frac() and candidate_line.bold_frac() > previous_line.bold_frac())
            or (other_block.avg_font_size() > previous_line.bbox_height() + 1 and candidate_line.bbox_height() > previous_line.bbox_height() + 1)
        ):
            join_tolerance = max(join_tolerance, candidate_line.bottom_edge() - previous_line.top_edge())
    elif block_right_aligned_to_column and line_left_aligned_to_column:
        join_tolerance *= 1.3 if other_block.line_count() <= 1 else 1.2
    elif block_left_aligned_to_column and line_left_aligned_to_column:
        join_tolerance *= 1.1
    elif block_right_aligned_to_column:
        if other_block.line_count() <= 1:
            join_tolerance *= 1.1
        if candidate_line.char_stats.secondary_slot == 3:
            join_tolerance *= 1.1
        if other_block.line_count() <= 1 and candidate_line.char_stats.secondary_slot == 3:
            join_tolerance *= 1.1
        if (
            candidate_line.left_edge() > other_block.left_edge()
            and candidate_line.left_edge() <= other_block.left_edge() + 0.1 * other_block.bbox_width()
            and (other_block.line_count() <= 1 or left_aligned(candidate_line, block_last_line, 1))
        ):
            join_tolerance *= 1.2
    elif candidate_line.bbox_width() < 0.9 * block_last_line.bbox_width() and center_aligned(other_block, candidate_line, 1):
        join_tolerance *= 1.1

    if left_edges_aligned and candidate_line.bbox_width() < 0.5 * other_block.bbox_width() and other_block.char_stats.tertiary_slot != 6 and candidate_line.char_stats.tertiary_slot == 6:
        join_tolerance *= 1.3

    # -- Step 9: numbering pattern checks --------------------------------
    block_numbering_kind = numbering_kind(other_block.line())

    block_has_numbering = (
        numbering_kind(other_block.line()) != 0
        and first_span_of(other_block).bbox_height() >= 0.8 * other_block.avg_font_size()
    )

    block_starts_with_digit = block_has_numbering and block_numbering_kind == 1

    line_numbering_kind = numbering_kind(candidate_line)

    line_has_numbering = (
        numbering_kind(candidate_line) != 0
        and candidate_line.primary_slot[0].bbox_height() >= 0.8 * candidate_line.avg_font_size()
    )

    line_starts_with_digit = line_has_numbering and line_numbering_kind == 1


    if block_starts_with_digit and not line_starts_with_digit and font_size_delta <= -0.5:
        join_tolerance /= 2
    elif (
        (block_starts_with_digit and (bold_mismatch or font_size_delta <= -0.5))
        or (line_starts_with_digit and (bold_mismatch or font_size_delta >= 0.5))
    ):
        join_tolerance /= 1.5
    elif block_starts_with_digit and candidate_line.left_edge() >= other_block.left_edge() and 0.9 * candidate_line.bbox_width() > other_block.bbox_width():
        join_tolerance /= 1.5
    elif block_has_numbering and candidate_line.left_edge() >= other_block.left_edge() and 0.9 * candidate_line.bbox_width() > other_block.bbox_width():
        join_tolerance /= 1.3
    elif (block_starts_with_digit and candidate_line.char_stats.secondary_slot != 3 or line_starts_with_digit) and font_mismatch:
        join_tolerance /= 1.3
    elif block_starts_with_digit and left_edges_aligned and candidate_line.char_stats.secondary_slot == 2:
        join_tolerance /= 1.3
    elif (
        (block_has_numbering and (font_mismatch or bold_mismatch or font_size_delta <= -0.5 or (left_edges_aligned and candidate_line.char_stats.secondary_slot == 2)))
        or (line_has_numbering and (font_mismatch or bold_mismatch or font_size_delta >= 0.5))
    ):
        join_tolerance /= 1.1

    if block_has_numbering and line_has_numbering:
        join_tolerance /= 1.3

    # -- Step 10: hanging-indent + neighbour patches ---------------------
    block_first_letter = other_block.line().alignment_slot

    if (
        block_numbering_kind == 1
        and line_numbering_kind != 1
        and not left_edges_aligned
        and block_first_letter is not None
        and left_aligned(block_first_letter, candidate_line, 1)
    ):
        join_tolerance *= 2

    if case_signal_flip:
        join_tolerance /= 1.1
        if other_block.line_count() == 1 or not left_edges_aligned:
            divisor = 3 if width_ratio > 3 else (1.5 if width_ratio > 1.5 else 1)
            join_tolerance /= divisor
        if (is_upper_dominant(other_block.char_stats) and block_has_numbering) or (is_upper_dominant(candidate_line.char_stats) and line_has_numbering):
            join_tolerance /= 2
        if font_mismatch or bold_mismatch:
            join_tolerance /= 1.5

    if other_block is not first_candidate_block and bottom_edge_gap > 1.1 * (first_candidate_block.bottom_edge() - candidate_line.bottom_edge()):
        join_tolerance /= 2

    return bottom_edge_gap <= join_tolerance
