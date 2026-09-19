"""Per-page block neighborhood maps and neighbor lookups."""

from __future__ import annotations

import math
from typing import Any, Optional
from ..model import (
    _UNICODE_WHITESPACE_CLASS,
    _strip_diacritics,
    _trim_unicode_ws,
    style_key, magnitude_ratio, same_x_extent, same_y_extent, y_overlaps, left_aligned, right_aligned, center_aligned, x_aligned, x_centers_close, to_number,
    last_span, avg_char_width, raw_text_of_line, heading_score, numbering_text, numbering_value, numbering_kind, Line, last_line_of, first_span_of, is_word_category, block_text, is_punct_category, deaccented_text, letter_count, punct_count, dominant_style_of,
    info_weight, dominant_font_size, is_upper_dominant, is_caps_heavy, CharStats, alignment_code, Block,
)

from .text_checks import clamp


# --------------------------------------------------------------------------- #
# Per-page neighbor map.
# --------------------------------------------------------------------------- #


class BlockNeighborCache:
    """Per-block neighbor cache populated by the page neighbor map."""

    __slots__ = ("state_slot", "tertiary_slot", "measure_slot", "auxiliary_slot", "primary_slot", "secondary_slot", "option_slot")

    def __init__(self):
        self.state_slot = False   # initialized flag
        self.tertiary_slot = None    # closest body block below
        self.measure_slot = None    # block 1-column-left
        self.auxiliary_slot = None    # next block to the right
        self.primary_slot = None    # earlier body block above
        self.secondary_slot = None    # nearest body block above
        self.option_slot = None    # block 1-column-right peer


def compute_bucket_span(neighbor_map, block) -> dict:
    """Compute the inclusive horizontal bucket span for a block."""
    start_bucket = int(clamp(math.floor(block.left_edge() / neighbor_map.tertiary_slot), 0, neighbor_map.secondary_slot - 1))
    end_bucket = int(clamp(math.ceil(block.right_edge() / neighbor_map.tertiary_slot), 0, neighbor_map.secondary_slot - 1))
    return {"start_bucket": start_bucket, "end_bucket": end_bucket}


def neighbor_above(neighbor_map, other_block: Block) -> Optional[Block]:
    """closest 'j' neighbor (block above)."""
    width_value = neighbor_map.primary_slot[other_block.orig_index] if other_block.orig_index < len(neighbor_map.primary_slot) else None
    return width_value.tertiary_slot if width_value is not None else None


def body_neighbor_above(neighbor_map, other_block: Block) -> Optional[Block]:
    """closest 'g' neighbor."""
    width_value = neighbor_map.primary_slot[other_block.orig_index] if other_block.orig_index < len(neighbor_map.primary_slot) else None
    return width_value.primary_slot if width_value is not None else None


def neighbor_right(neighbor_map, other_block: Block) -> Optional[Block]:
    """Closest right-side peer neighbor."""
    width_value = neighbor_map.primary_slot[other_block.orig_index] if other_block.orig_index < len(neighbor_map.primary_slot) else None
    return width_value.auxiliary_slot if width_value is not None else None


def neighbor_right_peer(neighbor_map, secondary_item):
    return neighbor_right(neighbor_map, secondary_item)


def closest_body_neighbor_above(neighbor_map, other_block: Block) -> Optional[Block]:
    """Closest stored neighbor above."""
    width_value = neighbor_map.primary_slot[other_block.orig_index] if other_block.orig_index < len(neighbor_map.primary_slot) else None
    return width_value.secondary_slot if width_value is not None else None


class PageNeighborMap:
    """Per-page horizontal-bucket neighbor map for constant-time nearby-block queries."""

    __slots__ = ("tertiary_slot", "secondary_slot", "primary_slot")

    def __init__(self, page):
        blocks = page.output_slot
        self.tertiary_slot = max(5, page.bounds.bbox_width() / 300)                  # bucket width
        self.secondary_slot = int(math.floor(page.bounds.bbox_width() / self.tertiary_slot))      # bucket count
        self.primary_slot: list[Optional[BlockNeighborCache]] = [None] * (max(len(blocks), 1) + 1)
        # mark buckets crossed by body-marked blocks
        marked = [False] * self.secondary_slot
        for candidate_item in blocks:
            if not candidate_item.is_body_paragraph:
                continue
            spans = compute_bucket_span(self, candidate_item)
            for index in range(spans["start_bucket"], spans["end_bucket"]):
                if 0 <= index < self.secondary_slot:
                    marked[index] = True
        recent_height: list[int] = [-1] * self.secondary_slot        # most recent body block height at bucket
        # Reads past the end of this list must behave like an unset slot: -1 is
        # falsy at the ``>= 0`` tests below just as a missing entry is, and a
        # write to it extends the list. On a degenerate page with zero buckets
        # every clamped index is 0, so one slot reproduces that growth.
        recent_block_index: list[int] = [-1] * max(self.secondary_slot, 1)  # most-recent block V (j-direction)
        recent: list[Optional[Block]] = [None] * self.secondary_slot    # most-recent block at bucket
        pending: list[list[int]] = [[] for _ in range(self.secondary_slot)]   # pending V's per bucket

        for block_index, current_block in enumerate(blocks):
            if (
                current_block.char_count() <= 0
                or current_block.skew_frac() > 1
                or current_block.type in (1, 2, 12)
            ):
                continue
            width = BlockNeighborCache()
            self.primary_slot[current_block.orig_index] = width
            spans = compute_bucket_span(self, current_block)
            left = spans["start_bucket"]
            right = spans["end_bucket"]
            # H field: block 1-column-left or right
            value = recent_block_index[left]
            adjacent_bucket_index = recent_block_index[
                left - 1 if (left > 0 and current_block.left_edge() < (left + 0.5) * self.tertiary_slot)
                else (left + 1 if left < self.secondary_slot - 1 else left)
            ]
            if value >= 0 or adjacent_bucket_index >= 0:
                same_bucket_block = blocks[value] if 0 <= value < len(blocks) else None
                adjacent_bucket_block = blocks[adjacent_bucket_index] if 0 <= adjacent_bucket_index < len(blocks) else None
                if same_bucket_block is not None and (adjacent_bucket_block is None or same_bucket_block.bottom_edge() < adjacent_bucket_block.bottom_edge()):
                    picked_index = value
                else:
                    picked_index = adjacent_bucket_index
                if 0 <= picked_index < len(blocks):
                    width.measure_slot = blocks[picked_index]
                    if self.primary_slot[picked_index] is not None:
                        self.primary_slot[picked_index].option_slot = current_block
            recent_block_index[left] = current_block.orig_index
            for col in range(left, right):
                if 0 <= col < self.secondary_slot:
                    width.state_slot = width.state_slot or marked[col]
                    recent_block = recent[col]
                    if recent_block is not None and (width.primary_slot is None or recent_block.bottom_edge() < width.primary_slot.bottom_edge()):
                        width.primary_slot = recent_block
                    previous_body_index = recent_height[col]
                    recent_height[col] = current_block.orig_index
                    if previous_body_index >= 0 and previous_body_index < len(blocks):
                        same_bucket_block = blocks[previous_body_index]
                        if width.tertiary_slot is None or same_bucket_block.bottom_edge() < width.tertiary_slot.bottom_edge():
                            width.tertiary_slot = same_bucket_block
                        previous_cache = self.primary_slot[previous_body_index]
                        if previous_cache is not None and (previous_cache.auxiliary_slot is None or current_block.top_edge() > previous_cache.auxiliary_slot.top_edge()):
                            previous_cache.auxiliary_slot = current_block
                    if current_block.is_body_paragraph:
                        for pending_index in pending[col]:
                            if 0 <= pending_index < len(self.primary_slot):
                                pending_neighbor = self.primary_slot[pending_index]
                                if pending_neighbor is not None and (pending_neighbor.secondary_slot is None or current_block.top_edge() > pending_neighbor.secondary_slot.top_edge()):
                                    pending_neighbor.secondary_slot = current_block
                        recent[col] = current_block
                        pending[col].clear()
                    pending[col].append(current_block.orig_index)
