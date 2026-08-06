"""Keyword-labeled section and caption-region detection. This module finds blocks that look like figure/table/chart labels or named
sections, then extends each label forward or backward to claim the associated
body blocks. The resulting regions are used by classification and outline
assembly to avoid treating captions or labeled content as ordinary headings.
"""

import regex as regex_module  # Unicode \p{...} property classes.
from typing import Optional

from ..classification import FIGURE_KEYWORDS_TRIE, TABLE_KEYWORDS_TRIE, CHART_KEYWORDS_TRIE
from ..model import (
    Rect, rect_union, extend_top_to, extend_bottom_to, EMPTY_RECT, Bounded,
    _trim_unicode_ws,
    center_aligned, last_span, heading_score, reading_order_key, numbering_text, Line, last_line_of, first_span_of, dominant_style_of, info_weight, Block,
)
from ..stats import column_index_of
from ..tokens import Token, TokenView, wrap_tokens, enumerate_tokens, last_token, trie_prefix_match, strip_leading_if_in, first_token, set_case_fold, TrieConfig, build_trie, tokenize_block, BuiltTrie, is_word_token

from .caption_text import (
    PERIOD_CHARS,
    STRUCTURAL_NUMBER_RE,
    is_number_separator,
    extract_structural_number,
    format_caption_label,
    REFERENCE_PHRASE_TRIE,
    is_uppercase_dominant,
    trie_matches_all,
    advance_past_line,
    skip_bracketed_word,
    token_case_signal,
    caption_outranks,
)
from .caption_regions import (
    CaptionedRegion,
    dedupe_caption_entries,
    extend_caption_region,
    build_caption_regions,
    CaptionEntry,
    CaptionContext,
    iter_page_blocks,
    detect_captions,
)

__all__ = [
    "PERIOD_CHARS", "STRUCTURAL_NUMBER_RE", "is_number_separator", "extract_structural_number", "format_caption_label", "REFERENCE_PHRASE_TRIE",
    "is_uppercase_dominant", "trie_matches_all", "advance_past_line", "skip_bracketed_word", "token_case_signal", "caption_outranks",
    "CaptionEntry", "CaptionedRegion", "CaptionContext", "iter_page_blocks", "detect_captions", "dedupe_caption_entries", "extend_caption_region", "build_caption_regions",
]
