"""
Block classification for header/footer, watermark, boilerplate, TOC-page, and
reference-list marking. The module combines recurrence hashes, page-number
patterns, body-paragraph gates, cross-page geometry, and numeric-column
clustering. The dot-leader and page-number gates intentionally use Unicode
number properties so fullwidth and non-Latin digits are handled consistently.
"""

import json
import math
import regex as regex_module  # Unicode \p{...} property classes
from pathlib import Path
from typing import Optional

from ..model import (
    _UNICODE_WHITESPACE_CLASS,
    _strip_diacritics,
    _round_half_up_to_int,
    magnitude_ratio,
    intervals_overlap,
    y_overlaps,
    center_aligned,
    to_number,
    last_span,
    heading_score,
    text_of_line,
    Line,
    last_line_of,
    first_span_of,
    is_word_category,
    block_text,
    deaccented_text,
    letter_count,
    dominant_style_of,
    punct_count,
    info_weight,
    is_upper_dominant,
    is_caps_heavy,
    alignment_code,
    Block,
)
from ..stats import style_key, DocStats, weighted_percentile, column_index_of, char_script_bucket
from ..tokens import (
    is_trimmable_token,
    token_numeric_value,
    Token,
    TokenView,
    wrap_tokens,
    enumerate_tokens,
    jenkins_hash,
    trie_prefix_match,
    strip_trie_match,
    strip_leading_if_in,
    COMMA_CHARS,
    strip_trailing_comma,
    trim_trailing_punct,
    set_case_fold,
    TrieConfig,
    build_trie,
    LineTokenizer,
    tokenize_block,
    BuiltTrie,
    trie_full_match,
    is_char_token,
    is_word_token,
)

from .keyword_tables import (
    _DICT_PATH,
    _DICTS,
    _dict_trie,
    COPYRIGHT_TRIE,
    VOLUME_WORDS_TRIE,
    TOC_TITLES_TRIE,
    FIGURE_KEYWORDS_TRIE,
    _TABLE_KEYWORDS_TRIE,
    TABLE_KEYWORDS_TRIE,
    _CHART_KEYWORDS_TRIE,
    CHART_KEYWORDS_TRIE,
    APPENDIX_SECTION_TRIE,
    INTRODUCTION_SECTION_TRIE,
    BOX_KEYWORD_TRIE,
    KEYWORDS_SECTION_TRIE,
    _BOILERPLATE_PHRASES_PATH,
    BOILERPLATE_TRIE,
    DOT_LEADER_ROW_RE,
    PAGE_NUMBER_ONLY_RE,
    _search_trie,
    _normalize_text_key,
)
from .body_text import (
    record_recurring_text,
    is_body_paragraph,
    span_style_text_key,
    normalized_block_text,
    _ROMAN_NUMERALS,
    span_page_number,
    longest_word_and_number,
)
from .header_footer import (
    PageMarkState,
    record_marked_block,
    is_header_positioned,
    has_adjacent_page_numbers,
    mark_header_footer,
    walk_from_page_edge,
    find_cross_page_match,
    HeaderFooterContext,
    bounded_edit_distance,
    detect_header_footer,
)
from .toc_boilerplate import (
    mark_watermarks,
    _institution_thesis_words,
    INSTITUTION_THESIS_TRIE,
    PROFESSOR_TITLES_TRIE,
    is_boilerplate_block,
    NumberColumnCluster,
    extract_number_column,
    pick_nearer_cluster,
    detect_toc_range,
    mark_toc_and_boilerplate,
)

__all__ = [
    "is_body_paragraph", "record_recurring_text", "span_style_text_key", "normalized_block_text", "span_page_number", "longest_word_and_number", "record_marked_block", "PageMarkState", "is_header_positioned", "has_adjacent_page_numbers", "mark_header_footer", "walk_from_page_edge", "find_cross_page_match",
    "HeaderFooterContext", "detect_header_footer", "mark_watermarks",
    "is_boilerplate_block", "NumberColumnCluster", "extract_number_column", "pick_nearer_cluster", "detect_toc_range", "mark_toc_and_boilerplate",
    "bounded_edit_distance",
    "COPYRIGHT_TRIE", "VOLUME_WORDS_TRIE", "TOC_TITLES_TRIE", "FIGURE_KEYWORDS_TRIE", "TABLE_KEYWORDS_TRIE", "CHART_KEYWORDS_TRIE", "APPENDIX_SECTION_TRIE", "INTRODUCTION_SECTION_TRIE", "BOX_KEYWORD_TRIE", "KEYWORDS_SECTION_TRIE",
]
