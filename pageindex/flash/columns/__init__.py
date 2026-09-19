"""
Column detection via sweep-line gutter scoring and recursive page splitting.

The detector builds horizontal and vertical sweep events, scores candidate
gutters, assigns column indexes to lines, and returns column rectangles used by
the second line-clustering pass. Direction code 0 scans vertical positions to
find row breaks; direction code 1 scans horizontal positions to find column
breaks.
"""

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

from .gutters import (
    SweepEvent,
    SplitCandidate,
    ColumnDetectionContext,
    DOT_LEADER_RE,
    collect_gutter_candidates,
    _score_gutter_gap,
)
from .splitting import (
    assign_column_index,
    recursive_split,
    detect_columns,
    columns_to_x_bounds,
)

__all__ = [
    "SweepEvent",
    "SplitCandidate",
    "ColumnDetectionContext",
    "collect_gutter_candidates",
    "assign_column_index",
    "recursive_split",
    "detect_columns",
    "columns_to_x_bounds",
]
