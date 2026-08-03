"""Outline assembly chain. This module turns heading candidates and labeled section regions into the final
nested outline tree. It groups candidates by numbering depth, style signature,
script compatibility, document order, and local clusters, then serializes the
tree into the public PageIndex JSON shape.
"""

import math
from typing import Any, Callable, Optional

from sortedcontainers import SortedKeyList
from ..model import (
    style_key, left_aligned, right_aligned, center_aligned, x_aligned, rect_union,
    Rect, last_span, avg_char_width, raw_text_of_line, heading_score, numbering_text, numbering_value, numbering_kind,
    reading_order_key, left_edge_key, _trim_unicode_ws, _round_half_up_to_int, Line, last_line_of, first_span_of, block_text, deaccented_text, letter_count, dominant_style_of, info_weight, dominant_font_size, is_upper_dominant, is_caps_heavy, alignment_code, Block,
)
from ..stats import style_key as style_key_fn, column_index_of, tally_scripts, dominant_script_family, ScriptHistogram
from ..tokens import (
    Token, TokenView, wrap_tokens, enumerate_tokens, last_token, trie_prefix_match, first_token, set_case_fold, TrieConfig, build_trie, tokenize_block, avg_char_width as avg_char_width_fn, trie_full_match, first_anchor_span, is_char_token, is_word_token,
)


# --------------------------------------------------------------------------- #
# Numbering-pattern clique selection.
# --------------------------------------------------------------------------- #


# Section-keyword trie shared with outline filtering.
from ..outline import SECTION_KEYWORD_TRIE

from .candidates import (
    _viewport_y_fraction,
    HeadingCandidate,
    OutlineNode,
    compare_heading_order,
    _compare_block_order,
    heading_order_key,
    is_script_compatible,
    heading_signature,
    parent_signature,
    cached_signature,
    is_in_oo_range,
    has_style_neighbor,
)
from .style_context import (
    StyleCluster,
    pick_style_bucket,
    has_conflict_in_context,
    is_compatible_with_context,
    OutlineContext,
    NumberingTrie,
    insert_numbering,
    count_sibling_numberings,
    OutlineState,
    _apply_heading_to_state,
    compare_heading_depth,
)
from .cliques import (
    find_keyword_clique,
    CliqueTreeNode,
    find_ancestor_next_sibling,
    descend_to_deepest_last,
    append_tree_child,
    CliqueTreeBuilder,
    block_style_signature,
    is_member_of_tree,
    can_share_heading_style,
    compare_block_order,
    heading_precedes_line,
    CliqueFilterContext,
    detect_body_headings,
    partition_candidates,
    interleave_clusters,
)
from .selection import (
    min_font_distance,
    should_reject_heading,
    push_heading_to_state,
    HierarchyStack,
    find_parent_heading,
    is_appendix_nesting_ok,
    extract_sub_headings,
    extract_top_level_headings,
    is_outline_valid,
    is_chapter_outline_valid,
)
from .assembly import (
    mark_outline_block_types,
    compute_max_heading_gap,
    has_table_or_prominent,
    is_landscape_or_empty,
    build_heading_from_block,
    assemble_outline,
    _flatten_outline_nodes,
    _heading_appears_at_page_top,
    outline_to_dict_tree,
)

__all__ = [
    "HeadingCandidate", "OutlineNode",
    "compare_heading_order", "heading_order_key", "compare_heading_depth",
    "is_script_compatible", "heading_signature", "parent_signature", "cached_signature", "is_in_oo_range", "has_style_neighbor", "pick_style_bucket", "has_conflict_in_context", "is_compatible_with_context",
    "StyleCluster", "OutlineContext", "NumberingTrie", "insert_numbering", "count_sibling_numberings",
    "OutlineState",
    "find_keyword_clique", "detect_body_headings", "CliqueFilterContext",
    "partition_candidates", "interleave_clusters", "push_heading_to_state", "should_reject_heading", "find_parent_heading", "HierarchyStack", "extract_sub_headings", "min_font_distance",
    "extract_top_level_headings", "is_outline_valid", "is_chapter_outline_valid", "mark_outline_block_types", "compute_max_heading_gap", "has_table_or_prominent",
    "build_heading_from_block",
    "assemble_outline",
    "outline_to_dict_tree",
]
