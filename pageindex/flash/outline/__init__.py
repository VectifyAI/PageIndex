"""Heading predicates and section-keyword helpers. The full outline tree is assembled in ``outline_assembly``. This module keeps
the lower-level heading checks that decide whether a block is a plausible
outline heading based on numbering, style, geometry, and section-keyword tries.
"""

import re
from collections import defaultdict
from typing import Optional

from ..labels import extract_structural_number
from ..model import numbering_text, numbering_kind, block_text, is_caps_heavy, Block
from ..stats import column_index_of
from ..tokens import set_case_fold, TrieConfig, build_trie, tokenize_block, trie_full_match

from .filtering import (
    SECTION_KEYWORD_TRIE,
    heading_order_key,
    _NON_HEADING_TYPES,
    _DOT_LEADER_RE,
    _CAPTION_LABEL_RE,
    _EQUATION_LABEL_RE,
    _PAREN_FRAGMENT_RE,
    _looks_like_pseudo_code,
    is_heading_candidate,
    _style_key,
    _numbering_depth,
    collect_headings,
    _heading_signature,
    _matches_section_keywords,
    _PSEUDO_CODE_PATTERNS,
    _AUTHOR_PATTERNS,
    _BULLET_LIST_RE,
    filter_by_clique,
)
from .tree import (
    extract_top_level_headings,
    assign_levels,
    _heading_title,
    _heading_page_num,
    build_tree,
    validate,
)

__all__ = [
    "is_heading_candidate",
    "collect_headings",
    "filter_by_clique",
    "assign_levels",
    "build_tree",
    "validate",
    "extract_top_level_headings",
    "SECTION_KEYWORD_TRIE",
    "heading_order_key",
]
