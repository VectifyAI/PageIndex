"""Per-page pipeline orchestration. For each page, the extractor builds initial lines, computes page statistics,
detects columns, reclusters lines with column awareness, removes line-number
artifacts, recomputes statistics, and assigns reading order.
"""

import math
from typing import Optional

from sortedcontainers import SortedKeyList

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

from .page_view import (
    PageView,
    assign_reading_order,
    process_page,
)
from .line_numbers import (
    LineNumberCluster,
    init_line_number_cluster,
    nearest_cluster,
    validate_line_number_cluster,
    strip_line_numbers,
)

__all__ = [
    "assign_reading_order",
    "LineNumberCluster",
    "init_line_number_cluster",
    "nearest_cluster",
    "validate_line_number_cluster",
    "strip_line_numbers",
    "PageView",
    "process_page",
]
