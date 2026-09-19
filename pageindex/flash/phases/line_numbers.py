"""Line-number column detection and stripping."""

from __future__ import annotations

import math
from typing import Optional

from sortedcontainers import SortedKeyList
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


# --------------------------------------------------------------------------- #
# Line-number stripper #
# --------------------------------------------------------------------------- #


class LineNumberCluster:
    """Drop-cap or line-number cluster used to detect removable line numbers."""

    __slots__ = ("lines", "left", "secondary_slot", "primary_slot", "is_valid_sequence")

    def __init__(self, line: Line, candidate_item: float, valid_sequence_flag: bool):
        self.lines: list = [line]
        self.left: float = line.left_edge()
        self.secondary_slot: float = avg_char_width(line)
        self.primary_slot: float = candidate_item
        self.is_valid_sequence: bool = valid_sequence_flag


def init_line_number_cluster(line: Line) -> LineNumberCluster:
    """Build an initial line-number cluster for a candidate line."""
    line_number = to_number(line.primary_slot[0].state_slot)
    is_valid_integer = (line_number > 0 and line_number < 1e4 and not math.isnan(line_number) and line_number == math.floor(line_number))
    return LineNumberCluster(line, line_number, is_valid_integer)


def nearest_cluster(line: LineNumberCluster, other_line: Optional[LineNumberCluster], candidate_line: Optional[LineNumberCluster]) -> Optional[LineNumberCluster]:
    """Pick the nearer left or right cluster within two character heights."""
    distance = (line.left - other_line.left) if other_line is not None else math.inf
    candidate_distance = (candidate_line.left - line.left) if candidate_line is not None else math.inf
    tol = 2 * line.secondary_slot
    if distance > tol and candidate_distance > tol:
        return None
    return other_line if distance < candidate_distance else candidate_line


def validate_line_number_cluster(rect: Rect, other_lines: list[Line], candidate_line: LineNumberCluster) -> bool:
    """validate a candidate cluster (>= 5 lines, near left edge, bulk of body weight overlapping the cluster's vertical span)."""
    if len(candidate_line.lines) < 5:
        return False
    if candidate_line.left < 0.05 * rect.bbox_width():
        return True
    empty_line_count = 0
    flag = False
    top = -math.inf
    bot = math.inf
    min_gap = math.inf
    max_gap = -math.inf
    prev: Optional[Line] = None
    for cluster_line in candidate_line.lines:
        if cluster_line.char_count() - cluster_line.char_stats.primary_slot[1] <= 0:
            empty_line_count += 1
        first = cluster_line.alignment_slot
        if first and first.char_stats.secondary_slot == 3:
            flag = True
        top = max(top, cluster_line.top_edge())
        bot = min(bot, cluster_line.bottom_edge())
        if prev is not None:
            gap = prev.bottom_edge() - cluster_line.bottom_edge()
            min_gap = min(min_gap, gap)
            max_gap = max(max_gap, gap)
        prev = cluster_line
    # Preserve IEEE-754 division for the spacing-ratio test.

    if min_gap != 0:
        gap_ratio = max_gap / min_gap
    elif max_gap != 0:
        gap_ratio = math.copysign(math.inf, max_gap)
    else:
        gap_ratio = math.nan
    if empty_line_count < len(candidate_line.lines) / 2 and (not flag or gap_ratio > 1.3):
        return False
    total = 0.0
    covered = 0.0
    for line in other_lines:
        block_weight = info_weight(line.char_stats)
        total += block_weight
        if line.bottom_edge() < top and line.top_edge() > bot:
            covered += block_weight
    return covered >= 0.8 * total


def strip_line_numbers(rect: Rect, other_lines: list[Line]) -> list[Line]:
    """Detect a column of line numbers and strip it. Returns the original lines if no line-numbering pattern is detected."""
    # Cluster candidates by ``left`` x-position. SortedKeyList by left.
    cluster_tree: SortedKeyList = SortedKeyList(key=lambda line_key: line_key.left)
    for line in other_lines:
        if len(line.primary_slot) == 0 or len(line.primary_slot[0].state_slot) == 0:
            continue
        if line.left_edge() > 0.15 * rect.bbox_width():
            continue
        candidate_cluster = init_line_number_cluster(line)
        if not candidate_cluster.is_valid_sequence:
            continue
        # Equal-left clusters must merge, so predecessor/successor lookup is
        # inclusive: successor = first left >= current, predecessor = last left <=
        # current. Strict bisect would fragment a fixed-x line-number column.
        idx_succ = cluster_tree.bisect_left(candidate_cluster)
        successor_cluster: Optional[LineNumberCluster] = (
            cluster_tree[idx_succ] if idx_succ < len(cluster_tree) else None
        )  # type: ignore[assignment]
        idx_pred = cluster_tree.bisect_right(candidate_cluster)
        neighbor: Optional[LineNumberCluster] = (
            cluster_tree[idx_pred - 1] if idx_pred > 0 else None
        )  # type: ignore[assignment]
        match = nearest_cluster(candidate_cluster, neighbor, successor_cluster)
        if match is not None:
            if match.is_valid_sequence:
                match.is_valid_sequence = (candidate_cluster.primary_slot == match.primary_slot + 1)
            match.lines.append(line)
            match.primary_slot = candidate_cluster.primary_slot
        else:
            cluster_tree.add(candidate_cluster)

    # Find largest valid (Ua) cluster
    best: Optional[LineNumberCluster] = None
    for cluster in cluster_tree:
        if cluster.is_valid_sequence and (best is None or len(cluster.lines) > len(best.lines)):
            best = cluster
    if best is None or not validate_line_number_cluster(rect, other_lines, best):
        return other_lines

    # Build output: for each affected line, drop its first span
    affected = set(id(line) for line in best.lines)
    out: list[Line] = []
    for source_line in other_lines:
        if id(source_line) not in affected:
            out.append(source_line)
            continue
        new_line = Line()
        first_span = source_line.primary_slot[0]
        for span in source_line:
            if span is first_span:
                continue
            append_span(new_line, span)
        if new_line.char_count() <= 0:
            continue
        new_line.measure_slot = source_line.measure_slot
        out.append(new_line)
    return out
