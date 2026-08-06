"""Line clustering pipeline.

The initial pass walks spans in document order and groups them into lines using
an in-line continuation test, while also collapsing overstrike duplicates
(artificial-bold rendering where the same glyph is painted twice). The merge
pass inserts lines into a sorted structure keyed by top-desc reading order,
looks up predecessor/successor neighbors, and either merges the new line into a
neighbor or keeps it separate. Neighbor lookup is inclusive of an exact
reading-order key match, so the successor uses ``bisect_left`` and the
predecessor uses ``bisect_right - 1``.
"""

import re
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
    TRAILING_DOT_LEADER_RE,
    span_continues_line,
    vertical_distance_in_line_heights,
    pick_closer_neighbor,
    should_merge_lines,
)
from .build import (
    _skip_mark_only,
    build_initial_lines,
    _is_label_stack,
    LinesContainer,
    _set_add,
    cluster_lines,
)

# --------------------------------------------------------------------------- #
# Combined helper #
# --------------------------------------------------------------------------- #


__all__ = [
    "span_continues_line",
    "vertical_distance_in_line_heights",
    "pick_closer_neighbor",
    "should_merge_lines",
    "build_initial_lines",
    "cluster_lines",
    "TRAILING_DOT_LEADER_RE",
]
