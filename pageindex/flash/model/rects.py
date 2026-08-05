"""Rectangle types, geometry predicates, and ordering comparators."""

from __future__ import annotations

import math

from .char_stats import (
    _max_nan_propagating,
    _min_nan_propagating,
)


# --------------------------------------------------------------------------- #
# Rectangle model #
# --------------------------------------------------------------------------- #


class RectLike:
    """Empty base for objects that expose bbox accessors."""

    pass


class Rect(RectLike):
    """Axis-aligned bbox. PDF coordinates: top > bottom (y increases upward). """

    __slots__ = ("left", "right", "top", "primary_slot")

    def __init__(self, other_item: float, candidate_item: float, reference_item: float, next_item: float):
        self.left = other_item
        self.right = candidate_item
        self.top = reference_item
        self.primary_slot = next_item   # bottom

    # --- geometry accessors ----------------------------------

    def left_edge(self) -> float: return self.left
    def right_edge(self) -> float: return self.right
    def top_edge(self) -> float: return self.top
    def bottom_edge(self) -> float: return self.primary_slot                       # bottom
    def bbox_width(self) -> float: return _max_nan_propagating(0.0, self.right - self.left)  # width
    def bbox_height(self) -> float: return _max_nan_propagating(0.0, self.top - self.primary_slot)     # height
    def area(self) -> float: return self.bbox_width() * self.bbox_height()                # area
    def center_x(self) -> float: return (self.left + self.right) / 2       # x-center
    def center_y(self) -> float: return (self.top + self.primary_slot) / 2            # y-center

    def contains(self, other_rect: "Rect") -> bool:
        return (
            self.left <= other_rect.left
            and self.right >= other_rect.right
            and self.top >= other_rect.top
            and self.primary_slot <= other_rect.primary_slot
        )


# Shared empty / inverted rectangle used to initialize accumulators.
EMPTY_RECT = Rect(math.inf, -math.inf, -math.inf, math.inf)


class Bounded(RectLike):
    """Mixin-style wrapper around an owned ``Rect``."""

    __slots__ = ("secondary_slot",)

    def __init__(self, other_rect: Rect):
        self.secondary_slot = other_rect

    def left_edge(self) -> float: return self.secondary_slot.left
    def right_edge(self) -> float: return self.secondary_slot.right
    def top_edge(self) -> float: return self.secondary_slot.top
    def bottom_edge(self) -> float: return self.secondary_slot.primary_slot
    def bbox_width(self) -> float: return self.secondary_slot.bbox_width()
    def bbox_height(self) -> float: return self.secondary_slot.bbox_height()
    def area(self) -> float: return self.secondary_slot.area()
    def center_x(self) -> float: return self.secondary_slot.center_x()
    def center_y(self) -> float: return self.secondary_slot.center_y()


def rect_union(rect: Rect, other_rect: Rect) -> Rect:
    """bbox union."""
    return Rect(
        _min_nan_propagating(rect.left, other_rect.left),
        _max_nan_propagating(rect.right, other_rect.right),
        _max_nan_propagating(rect.top, other_rect.top),
        _min_nan_propagating(rect.primary_slot, other_rect.primary_slot),
    )


def rect_intersection(rect: Rect, other_rect: Rect) -> Rect:
    """bbox intersection; disjoint boxes may have inverted horizontal or vertical edges."""
    return Rect(
        _max_nan_propagating(rect.left, other_rect.left),
        _min_nan_propagating(rect.right, other_rect.right),
        _min_nan_propagating(rect.top, other_rect.top),
        _max_nan_propagating(rect.primary_slot, other_rect.primary_slot),
    )


def extend_top_to(rect: Rect, other_item: float) -> Rect:
    """Clip the rectangle top to be at least ``other_value``."""
    return Rect(rect.left, rect.right, _max_nan_propagating(rect.top, other_item), rect.primary_slot)


def extend_bottom_to(rect: Rect, other_item: float) -> Rect:
    """Clip the rectangle bottom to be at most ``other_value``."""
    return Rect(rect.left, rect.right, rect.top, _min_nan_propagating(rect.primary_slot, other_item))


# --------------------------------------------------------------------------- #
# Sort comparators #
# --------------------------------------------------------------------------- #


def cmp_left_edge(left_value: Bounded, right_value: Bounded) -> float:
    """Order by (left asc, right asc, top desc, bottom desc). Returns the raw delta, not a normalised -1/0/1, because callers only consume the sign."""
    if left_value.left_edge() != right_value.left_edge():
        return left_value.left_edge() - right_value.left_edge()
    if left_value.right_edge() != right_value.right_edge():
        return left_value.right_edge() - right_value.right_edge()
    if left_value.top_edge() != right_value.top_edge():
        return right_value.top_edge() - left_value.top_edge()
    return right_value.bottom_edge() - left_value.bottom_edge()


# Python's ``sorted`` accepts a key, not a cmp. Provide key functions too.
def left_edge_key(primary_item: Bounded) -> tuple:
    return (primary_item.left_edge(), primary_item.right_edge(), -primary_item.top_edge(), -primary_item.bottom_edge())


def cmp_reading_order(left_value: Bounded, right_value: Bounded) -> float:
    """Order by (top desc, bottom desc, left asc, right asc). Top-of-page rows come first; within a row, leftmost first. Returns the raw delta because callers only consume the sign."""
    if left_value.top_edge() != right_value.top_edge():
        return right_value.top_edge() - left_value.top_edge()
    if left_value.bottom_edge() != right_value.bottom_edge():
        return right_value.bottom_edge() - left_value.bottom_edge()
    if left_value.left_edge() != right_value.left_edge():
        return left_value.left_edge() - right_value.left_edge()
    return left_value.right_edge() - right_value.right_edge()


def reading_order_key(primary_item: Bounded) -> tuple:
    return (-primary_item.top_edge(), -primary_item.bottom_edge(), primary_item.left_edge(), primary_item.right_edge())


def cmp_bottom_edge(left_value: Bounded, right_value: Bounded) -> float:
    """Order by (bottom asc, top asc, left asc, right asc). Returns the raw delta because callers only consume the sign."""
    if left_value.bottom_edge() != right_value.bottom_edge():
        return left_value.bottom_edge() - right_value.bottom_edge()
    if left_value.top_edge() != right_value.top_edge():
        return left_value.top_edge() - right_value.top_edge()
    if left_value.left_edge() != right_value.left_edge():
        return left_value.left_edge() - right_value.left_edge()
    return left_value.right_edge() - right_value.right_edge()


# --------------------------------------------------------------------------- #
# Alignment / overlap predicates #
# --------------------------------------------------------------------------- #


def magnitude_ratio(value: float, other_item: float) -> float:
    """Return the larger-magnitude-over-smaller-magnitude ratio with IEEE-754 division semantics. Division by zero yields +/-Infinity for a nonzero non-NaN numerator and NaN for +/-0 over +/-0 and NaN over +/-0. Downstream threshold tests rely on signed infinity, so divide-by-zero must not be collapsed to NaN. """
    # NaN comparisons take the false arm, which selects ``other_value / value``.
    if abs(value) > abs(other_item):
        num, den = value, other_item
    else:
        num, den = other_item, value
    # raw `num/den`. Python raises ZeroDivisionError on den == +/-0, so the
    # IEEE cases are spelled out: x/±0 = ±Infinity with sign(x) XOR sign(±0)
    # 5/-0 = -Infinity, ±0/±0 = NaN, NaN/±0 = NaN. A NaN denominator passes
    # `den != 0` and divides through to NaN.
    if den != 0:
        return num / den
    if num == 0 or math.isnan(num):
        return math.nan
    return math.copysign(math.inf, num) * math.copysign(1.0, den)


def same_x_extent(primary_item: Bounded, secondary_item: Bounded, candidate_item: float) -> bool:
    """Return whether both horizontal edges are within the tolerance."""
    return abs(primary_item.left_edge() - secondary_item.left_edge()) <= candidate_item and abs(primary_item.right_edge() - secondary_item.right_edge()) <= candidate_item


def same_y_extent(primary_item: Bounded, secondary_item: Bounded, candidate_item: float) -> bool:
    """Return whether both vertical edges are within the tolerance."""
    return abs(primary_item.top_edge() - secondary_item.top_edge()) <= candidate_item and abs(primary_item.bottom_edge() - secondary_item.bottom_edge()) <= candidate_item


def intervals_overlap(value: float, other_item: float, candidate_item: float, reference_item: float) -> bool:
    """Return whether the two closed ranges overlap by either endpoint."""
    return (value <= candidate_item and candidate_item <= other_item) or (candidate_item <= value and value <= reference_item)


def y_overlaps(primary_item: Bounded, secondary_item: Bounded) -> bool:
    """Return whether the vertical intervals of two boxes overlap."""
    return intervals_overlap(primary_item.bottom_edge(), primary_item.top_edge(), secondary_item.bottom_edge(), secondary_item.top_edge())


def left_aligned(primary_item: Bounded, secondary_item: Bounded, candidate_item: float) -> bool:
    """Return whether left edges match within the tolerance."""
    return abs(primary_item.left_edge() - secondary_item.left_edge()) <= candidate_item


def right_aligned(primary_item: Bounded, secondary_item: Bounded, candidate_item: float) -> bool:
    """Return whether right edges match within the tolerance."""
    return abs(primary_item.right_edge() - secondary_item.right_edge()) <= candidate_item


def center_aligned(primary_item: Bounded, secondary_item: Bounded, candidate_item: float) -> bool:
    """Return whether two boxes are center-aligned within the tolerance. Their left and right edge offsets must have opposite signs, then pass the center-distance tolerance."""
    reference_item = primary_item.left_edge() - secondary_item.left_edge()
    entry_item = primary_item.right_edge() - secondary_item.right_edge()
    def sign(signed_delta):
        if signed_delta > 0: return 1
        if signed_delta < 0: return -1
        return 0
    if sign(reference_item) != -sign(entry_item):
        return False
    return abs(primary_item.center_x() - secondary_item.center_x()) <= max(candidate_item, min(abs(reference_item), abs(entry_item)) / 2)


def x_aligned(primary_item: Bounded, secondary_item: Bounded, candidate_item: float) -> bool:
    """any of left / right / center aligned."""
    return left_aligned(primary_item, secondary_item, candidate_item) or right_aligned(primary_item, secondary_item, candidate_item) or center_aligned(primary_item, secondary_item, candidate_item)


def x_centers_close(primary_item: Bounded, secondary_item: Bounded) -> bool:
    """Return whether x-centers match within the secondary box width tolerance."""
    return abs(secondary_item.center_x() - primary_item.center_x()) <= max(1, secondary_item.bbox_width() / 10)
