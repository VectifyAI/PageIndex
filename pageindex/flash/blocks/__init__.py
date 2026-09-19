"""Block clustering. This module walks page lines in reading order, extends nearby compatible
blocks, starts a new block when no neighbor fits, and then splits simple
"heading + body" two-line blocks where the first line is a standalone section
heading. The clustering pass must return blocks, not raw lines. Reading-order assignment
then uses each block's first line to find the column index; doing that on raw
lines would read an unrelated first-span flag.
"""

from typing import Optional

from sortedcontainers import SortedKeyList

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

from .join_rules import (
    _DICT_PATH,
    _DICTS,
    SECTION_HEADING_TRIE,
    BlockClusterContext,
    should_join_line_to_block,
)
from .build import (
    split_heading_body_blocks,
    _set_add,
    cluster_lines_into_blocks,
)

__all__ = ["BlockClusterContext", "should_join_line_to_block", "cluster_lines_into_blocks", "split_heading_body_blocks", "SECTION_HEADING_TRIE"]
