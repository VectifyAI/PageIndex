"""Page-level and document-level statistics aggregation."""

from __future__ import annotations

import functools
import math
from typing import Optional

from ..model import Span, _format_half_up_one_decimal, Line, info_weight, _max_nan_propagating

from .scripts import (
    ScriptHistogram,
    tally_scripts,
    dominant_script_family,
)


# --------------------------------------------------------------------------- #
# Weighted percentile.
# --------------------------------------------------------------------------- #


def _percentile_sample_cmp(values: tuple[float, float], other_values: tuple[float, float]) -> float:
    """Comparator for weighted percentile samples. NaN comparison results are treated as equal so insertion order is preserved for NaN-valued samples."""
    if values[0] != other_values[0]:
        return values[0] - other_values[0]
    return values[1] - other_values[1]


def weighted_percentile(values: list[tuple[float, float]], other_item: float) -> float:
    """Weighted percentile over ``(value, weight)`` samples. Returns ``NaN`` for empty input or an out-of-range percentile. Ties at the target weight return the average of current and previous values; overshoots return the current value."""
    if len(values) <= 0 or other_item < 0 or other_item > 100:
        return float("nan")
    samples = sorted(values, key=functools.cmp_to_key(_percentile_sample_cmp))  # type: ignore[arg-type]
    total = sum(page_value[1] for page_value in samples)
    target = total * other_item / 100.0
    candidate_item = 0.0
    reference_item: Optional[float] = None
    for value, weight in samples:
        if candidate_item == target:
            return value if reference_item is None else (reference_item + value) / 2.0
        reference_item = value
        candidate_item += weight
        if candidate_item > target:
            return value
    return float("nan") if reference_item is None else reference_item


# --------------------------------------------------------------------------- #
# Per-span style hash #
# --------------------------------------------------------------------------- #


def style_key(span: Span) -> str:
    """Return ``"<fontStyle> <size rounded to 0.1>"`` for same-style span histograms."""
    return f"{span.font_style()} {_format_half_up_one_decimal(span.font_size)}"


# --------------------------------------------------------------------------- #
# Per-page statistics #
# --------------------------------------------------------------------------- #


class PageStats:
    """Per-page layout statistics used by column detection and classification."""

    __slots__ = ("line_count", "secondary_slot", "tertiary_slot", "previous_slot", "style_slot", "cache_slot", "option_slot", "primary_slot", "measure_slot", "state_slot", "auxiliary_slot")

    def __init__(
        self,
        valid_line_count: int,
        total_line_weight: float,
        median_overlap_gap: float,
        median_line_width: float,
        median_char_count: float,
        median_area_metric: float,
        median_center_y: float,
        median_font_size: float,
        average_char_width: float,
        dominant_font: str,
        dominant_style: str,
    ):
        self.line_count = valid_line_count
        self.secondary_slot = total_line_weight
        self.tertiary_slot = median_overlap_gap
        self.previous_slot = median_line_width
        self.style_slot = median_char_count
        self.cache_slot = median_area_metric
        self.option_slot = median_center_y
        self.primary_slot = median_font_size
        self.measure_slot = average_char_width
        self.state_slot = dominant_font
        self.auxiliary_slot = dominant_style


# --------------------------------------------------------------------------- #
# Per-page statistics.
# --------------------------------------------------------------------------- #


def compute_page_stats(page, other_lines: list[Line]) -> PageStats:
    """Compute weighted medians plus dominant font/style for one page."""
    overlap_gap_samples: list[tuple[float, float]] = []   # bucket-overlap samples
    line_width_samples: list[tuple[float, float]] = []   # line-width samples
    char_count_samples: list[tuple[float, float]] = []   # line-char-count samples
    area_metric_samples: list[tuple[float, float]] = []   # line.U samples
    array: list[tuple[float, float]] = []   # y-center samples
    font_size_samples: list[tuple[float, float]] = []   # font-size samples

    font: dict[str, float] = {}       # font-name histogram (weighted)
    style: dict[str, float] = {}      # style-hash histogram (weighted)
    chars = 0                          # total char count across spans
    width = 0.0                        # total width across spans
    total = 0.0                        # total line weight (sum of tf)
    valid = 0                          # valid line count

    bucket_size = page.bbox_width() / 20.0           # page width / 20 buckets
    buckets: list[Optional[Line]] = [None] * 21   # 20 buckets, +1 guard

    for line in other_lines:
        if line.skew_frac() > 1:                   # rotated/skewed line: skip
            continue
        valid += 1
        for span in line:                      # for each span t in line u
            span_weight = info_weight(span.char_stats)
            span_weight = span_weight * span_weight * span.bbox_height()            # weight = tf^2 * height
            font[span.font_name] = font.get(span.font_name, 0.0) + span_weight
            sty = style_key(span)
            style[sty] = style.get(sty, 0.0) + span_weight
            chars += span.char_count()
            width += span.bbox_width()
        line_weight = info_weight(line.char_stats)
        total += line_weight
        sample = line_weight * line.avg_font_size()   # weight = line_weight * font_size
        font_size_samples.append((line.avg_font_size(), sample))
        line_width_samples.append((line.bbox_width(), sample))
        char_count_samples.append((line.char_count(), sample))
        area_metric_samples.append((line.cache_slot, line.area()))           # NB: this one is weighted by area
        array.append((line.center_y(), sample))

        # Vertical overlap with the most recent occupant of each horizontal
        # bucket. Infinite sentinel boxes skip overlap sampling.

        left = line.left_edge()
        right = line.right_edge()
        if not (left < float("inf") and right > float("-inf")):
            continue
        if bucket_size <= 0:
            # Degenerate zero-width pages skip overlap sampling; downstream
            # statistics still include font and line-width samples.
            continue
        # Clamp infinite sentinels before converting bucket indexes to integers.
        bucket_left = 0 if left == float("-inf") else max(0, int(left / bucket_size))
        bucket_right = 20 if right == float("inf") else min(20, math.ceil(right / bucket_size))
        best_gap = float("inf")
        best_prev: Optional[Line] = None
        idx = bucket_left
        while idx < bucket_right:
            prev_in_bucket = buckets[idx]
            buckets[idx] = line
            idx += 1
            if prev_in_bucket is None:
                continue
            gap = max(prev_in_bucket.bottom_edge(), line.top_edge()) - line.bottom_edge()
            if gap < best_gap:
                best_gap = gap
                best_prev = prev_in_bucket
        if best_gap < float("inf") and best_prev is not None:
            overlap_gap_samples.append((best_gap, info_weight(best_prev.char_stats) * line_weight))

    # Dominant values update only on strictly greater positive weight. This
    # keeps the empty value for all-zero pages and preserves first-seen ties.
    dominant_font = ""
    dominant_font_weight = 0.0
    for font_name, weight in font.items():
        if weight > dominant_font_weight:
            dominant_font_weight = weight
            dominant_font = font_name
    dominant_style = ""
    dominant_style_weight = 0.0
    for style_name, weight in style.items():
        if weight > dominant_style_weight:
            dominant_style_weight = weight
            dominant_style = style_name

    return PageStats(
        valid_line_count=valid,
        total_line_weight=total,
        median_overlap_gap=weighted_percentile(overlap_gap_samples, 50),
        median_line_width=weighted_percentile(line_width_samples, 50),
        median_char_count=weighted_percentile(char_count_samples, 50),
        median_area_metric=weighted_percentile(area_metric_samples, 50),
        median_center_y=weighted_percentile(array, 50),
        median_font_size=weighted_percentile(font_size_samples, 50),
        # Average char width with IEEE edge cases: no characters with positive
        # width yields +inf, and no characters with no width yields NaN.
        average_char_width=(width / chars) if chars != 0
        else (float("inf") if width > 0 else float("nan")),
        dominant_font=dominant_font,
        dominant_style=dominant_style,
    )


# --------------------------------------------------------------------------- #
# Document-level statistics #
# --------------------------------------------------------------------------- #


class DocStats:
    """Document-level layout statistics: dominant script family, landscape-page count, total valid lines, total line weight, max page line weight, median page total weight, width/height percentiles, center statistic, and median body font size."""

    __slots__ = ("tertiary_slot", "style_slot", "cache_slot", "state_slot", "previous_slot", "secondary_slot", "option_slot", "auxiliary_slot", "measure_slot", "primary_slot")

    def __init__(
        self,
        dominant_script: int,
        landscape_pages: int,
        total_lines: int,
        total_weight: float,
        max_page_weight: float,
        median_page_weight: float,
        median_line_width: float,
        upper_width_percentile: float,
        median_center_y: float,
        median_body_font_size: float,
    ):
        self.tertiary_slot = dominant_script
        self.style_slot = landscape_pages
        self.cache_slot = total_lines
        self.state_slot = total_weight
        self.previous_slot = max_page_weight
        self.secondary_slot = median_page_weight
        self.option_slot = median_line_width
        self.auxiliary_slot = upper_width_percentile
        self.measure_slot = median_center_y
        self.primary_slot = median_body_font_size


# --------------------------------------------------------------------------- #
# Document-level statistics.
# --------------------------------------------------------------------------- #


def compute_doc_stats(pages: list) -> DocStats:
    """Compute document-wide recurrence and script statistics from page records."""
    script = ScriptHistogram()
    landscape = 0
    total_lines = 0
    total_weight = 0.0
    max_weight = 0.0

    total_weights: list[tuple[float, float]] = []
    bucket_overlaps: list[tuple[float, float]] = []
    widths: list[tuple[float, float]] = []
    char_counts: list[tuple[float, float]] = []
    upper_samples: list[tuple[float, float]] = []
    centers: list[tuple[float, float]] = []
    font_sizes: list[tuple[float, float]] = []

    for query_value in pages:
        # Accumulate the script-family histogram over raw parser text items,
        # capped at 100k chars. Merged line text can omit line-number spans.
        if script.secondary_slot < 100_000:
            for span in (query_value.text or []):
                if script.secondary_slot >= 100_000:
                    break
                tally_scripts(script, span.text)
        if query_value.bounds.bbox_width() > query_value.bounds.bbox_height():
            landscape += 1
        stats: PageStats = query_value.primary_slot
        total_lines += stats.line_count
        weight = stats.secondary_slot
        total_weight += weight
        max_weight = _max_nan_propagating(max_weight, weight)
        if stats.line_count <= 0 or weight <= 0:
            continue
        per_page = min(100.0, weight / stats.line_count)
        total_weights.append((weight, per_page))
        bucket_overlaps.append((stats.tertiary_slot, per_page))
        widths.append((stats.previous_slot, per_page))
        char_counts.append((stats.style_slot, per_page))
        upper_samples.append((stats.cache_slot, per_page))
        centers.append((stats.option_slot, per_page))
        font_sizes.append((stats.primary_slot, per_page))

    return DocStats(
        dominant_script=dominant_script_family(script),                     # dominant script family
        landscape_pages=landscape,
        total_lines=total_lines,
        total_weight=total_weight,
        max_page_weight=max_weight,
        median_page_weight=weighted_percentile(total_weights, 50),
        # Width and uppercase samples are the document-wide outputs used later.
        median_line_width=weighted_percentile(widths, 50),
        upper_width_percentile=weighted_percentile(upper_samples, 80),              # NB: 80th percentile, not 50
        median_center_y=weighted_percentile(centers, 50),
        median_body_font_size=weighted_percentile(font_sizes, 50),
    )


# --------------------------------------------------------------------------- #
# Column-index accessor #
# --------------------------------------------------------------------------- #


def column_index_of(line: Line) -> int:
    """Return the column index stored on the first child line/span. In normal use this receives a block, so the first child is a line and its stored column index is returned. If a raw line is passed, the same field access still succeeds but reads a different flag; the block-clustering pipeline avoids that path for column-aware ordering."""
    if not line.primary_slot:
        return -1
    first = line.primary_slot[0]
    # If ``first`` is another container (block.g[0] is a line) consult its H.
    value = getattr(first, "measure_slot", None)
    return -1 if value is None else value
