"""Per-page processing driver and reading-order assignment."""

from __future__ import annotations

from typing import Optional

from ..clustering import LinesContainer, cluster_lines, build_initial_lines
from ..columns import detect_columns, ColumnDetectionContext, columns_to_x_bounds
from ..model import (
    Span,
    left_aligned,
    right_aligned,
    center_aligned,
    x_centers_close,
    to_number,
    Rect,
    append_span,
    avg_char_width,
    Line,
    info_weight,
)
from ..stats import column_index_of, PageStats, compute_page_stats

from .line_numbers import strip_line_numbers


# --------------------------------------------------------------------------- #
# Per-page reading order and paragraph-break flagging #
# --------------------------------------------------------------------------- #


class PageView:
    """Per-page mutable state carried through layout classification."""

    __slots__ = (
        "bounds", "output_slot", "secondary_slot", "measure_slot", "page_index", "primary_slot", "tertiary_slot", "lines", "blocks",
        "text", "previous_slot", "annotations",
        "auxiliary_slot", "state_slot", "style_slot", "option_slot", "viewport_box", "rot",
    )

    def __init__(self, page_num: int, page_bbox: Rect):
        self.bounds: Rect = page_bbox
        self.output_slot: list = []
        self.secondary_slot: list = []
        self.measure_slot: bool = False           # set when a labeled section appears
        self.page_index: int = page_num
        self.primary_slot: Optional[PageStats] = None
        self.tertiary_slot: list = []          # column rects
        self.lines: list = []
        self.blocks: list = []
        self.text: Optional[list] = None    # raw text items reconstructed by parser
        self.previous_slot = 0.0
        self.annotations = []
        # per-page fields used by heading detection and outline assembly:
        self.auxiliary_slot: bool = False                # marked as references page
        self.state_slot: bool = False                # has substantive body
        self.style_slot: set = set()                 # set of body-style hashes (sh)
        self.option_slot = None                       # reserved, unused here
        # Page viewport for heading coordinates: unrotated view box + /Rotate.
        # None -> fallback to the origin-0 upright shortcut.
        self.viewport_box: Optional[tuple] = None
        self.rot: int = 0


def assign_reading_order(primary_item: PageView, other_items: list) -> None:
    """Assign reading order and paragraph-break flags for a page. The column-aware path expects blocks, not raw lines, because the sort key reads the first child line's column index. Passing raw lines would read a different flag from the first span."""
    primary_item.output_slot = other_items
    for candidate_item in range(len(other_items)):
        setattr(other_items[candidate_item], "orig_index", candidate_item)

    primary_item.secondary_slot = list(other_items)
    primary_item.secondary_slot.sort(key=lambda sort_block: (column_index_of(sort_block), -sort_block.top_edge(), -sort_block.bottom_edge(), sort_block.left_edge(), sort_block.right_edge()))

    # Assign sorted index and paragraph/end-isolated flags to each item.
    for idx in range(len(primary_item.secondary_slot)):
        candidate_item = primary_item.secondary_slot[idx]
        candidate_item.reading_order_index = idx
        reference_item = primary_item.secondary_slot[idx + 1] if idx + 1 < len(primary_item.secondary_slot) else None
        # Isolated-centered is true when the item is centered on the page and
        # either has no successor, is vertically separated from it, or is not
        # left/right aligned with it. Non-page-centered items can still be
        # isolated if they are centered relative to a page-centered successor.
        if candidate_item.alignment_slot and x_centers_close(primary_item.bounds, candidate_item):
            candidate_item.isolated_centered = (not reference_item) or (reference_item.top_edge() > candidate_item.bottom_edge()) or (not left_aligned(candidate_item, reference_item, 1) and not right_aligned(candidate_item, reference_item, 1))
        else:
            candidate_item.isolated_centered = bool(
                candidate_item.alignment_slot and reference_item
                and not left_aligned(candidate_item, reference_item, 1) and not right_aligned(candidate_item, reference_item, 1)
                and center_aligned(candidate_item, reference_item, candidate_item.bbox_width() / 10) and x_centers_close(primary_item.bounds, reference_item)
            )


# --------------------------------------------------------------------------- #
# Per-page orchestrator #
# --------------------------------------------------------------------------- #


def process_page(spans: list[Span], page_num: int, page_bbox: Rect) -> PageView:
    """Run the full per-page pipeline on flat span input."""
    page = PageView(page_num, page_bbox)
    # Raw parser items are kept before clustering
    # so document statistics can accumulate the script-family histogram over them (the lines
    # below are merged + line-number-stripped, a different character multiset).
    page.text = spans
    # 1) Build initial lines.
    container = LinesContainer()
    container.primary_slot = build_initial_lines(spans, page_bbox)
    # 2) First clustering pass: no column info yet.
    cluster_lines(container, 0.75, [])
    # 3) Compute first-pass per-page stats.
    page.primary_slot = compute_page_stats(page_bbox, container.primary_slot)
    # 4) Detect column rectangles and assign each line's column index.
    column_context = ColumnDetectionContext(page_bbox, page.primary_slot, container.primary_slot)
    page.tertiary_slot = detect_columns(column_context)
    # 5) Second clustering pass: tighter tolerance with column info.
    cols = columns_to_x_bounds(page.tertiary_slot)
    cluster_lines(container, 0.5, cols)
    # 6) Strip line-number column if present.
    container.primary_slot = strip_line_numbers(page_bbox, container.primary_slot)
    # 7) Recompute stats on cleaned lines.
    page.primary_slot = compute_page_stats(page_bbox, container.primary_slot)
    page.lines = container.primary_slot
    return page
