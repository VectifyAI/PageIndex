"""Per-page heading-candidate detection. This module builds and filters heading candidates from page blocks. It combines
numbering recognition, chapter/appendix keywords, local neighbor geometry,
font/style signals, cross-page rejection, and page-level candidate filtering
before handing candidates to outline assembly.
"""

import json
import math
import re
import regex as regex_module  # Unicode \p{...} property classes.
from pathlib import Path
from typing import Any, Optional
from ..outline_assembly import HeadingCandidate, OutlineNode
from ..labels import is_uppercase_dominant, trie_matches_all, advance_past_line, skip_bracketed_word, token_case_signal, format_caption_label, CaptionEntry, extract_structural_number
from ..model import (
    _UNICODE_WHITESPACE_CLASS,
    _strip_diacritics,
    _trim_unicode_ws,
    style_key, magnitude_ratio, same_x_extent, same_y_extent, y_overlaps, left_aligned, right_aligned, center_aligned, x_aligned, x_centers_close, to_number,
    last_span, avg_char_width, raw_text_of_line, heading_score, numbering_text, numbering_value, numbering_kind, Line, last_line_of, first_span_of, is_word_category, block_text, is_punct_category, deaccented_text, letter_count, punct_count, dominant_style_of,
    info_weight, dominant_font_size, is_upper_dominant, is_caps_heavy, CharStats, alignment_code, Block,
)
from ..tokens import (
    is_trimmable_token, token_numeric_value, Token, TokenView, wrap_tokens, enumerate_tokens, last_token, trie_prefix_match, strip_trie_match, strip_leading_if_in, COMMA_CHARS, strip_trailing_comma, first_token, trim_trailing_punct, set_case_fold, TrieConfig, build_trie, tokenize_block,
    trie_full_match, last_token_anchor, first_anchor_span, is_char_token, is_word_token,
)

from .keyword_tables import (
    _DICT_PATH,
    _DICTS,
    SECTION_KEYWORDS_TRIE,
    ABSTRACT_KEYWORDS_TRIE,
    REFERENCES_TRIE,
    APPENDIX_SECTION_TRIE,
    INTRODUCTION_SECTION_TRIE,
    BOX_KEYWORD_TRIE,
    KEYWORDS_SECTION_TRIE,
    CHAPTER_WORDS_TRIE,
    APPENDIX_KEYWORDS_TRIE,
    _normalize_text_key,
    ABSTRACT_KEYWORDS_SET,
    REFERENCES_SET,
    NUMBERED_PREFIX_RE,
    DEAD_DIGIT_RE,
    EQUATION_KEYWORDS_TRIE,
    ENGLISH_WORD_TO_NUMBER,
    ROMAN_NUMERAL_MAP,
    FORMULA_CHAR_WEIGHTS,
)
from .text_checks import (
    token_text_of_block,
    similar_style,
    is_heading_continuation,
    matches_abstract,
    matches_references,
    vertically_close,
    is_equation_adjacent_line,
    has_substantive_content,
    is_cover_page,
    clamp,
    token_to_number,
    letter_to_ordinal,
)
from .neighbors import (
    BlockNeighborCache,
    compute_bucket_span,
    neighbor_above,
    body_neighbor_above,
    neighbor_right,
    neighbor_right_peer,
    closest_body_neighbor_above,
    PageNeighborMap,
)
from .candidates import (
    PageScanState,
    push_candidate,
    make_heading_candidate,
    make_plain_candidate,
    make_body_heading_candidate,
    make_numbered_candidate,
    _di_count,
    _number_at_token_index,
)
from .detectors import (
    detect_numbered_heading,
    detect_labeled_heading,
    detect_chapter_appendix,
    detect_box_heading,
    classify_heading,
    is_acceptable_heading,
    safe_column_index,
    try_classify_heading,
    is_too_wide_for_heading,
    passes_neighbor_check,
    has_competing_labeled_heading,
    is_year_string,
    is_bibliography_entry,
)
from .style_detectors import (
    detect_font_heading,
    detect_heading_with_body,
)
from .page_scan import (
    scan_page_headings,
    DocCandidateCollector,
    filter_page_candidates,
    build_doc_heading_candidates,
    find_section_openers,
)

__all__ = [
    "SECTION_KEYWORDS_TRIE", "ABSTRACT_KEYWORDS_TRIE", "ABSTRACT_KEYWORDS_SET", "REFERENCES_TRIE", "REFERENCES_SET", "APPENDIX_SECTION_TRIE", "INTRODUCTION_SECTION_TRIE", "BOX_KEYWORD_TRIE", "KEYWORDS_SECTION_TRIE", "CHAPTER_WORDS_TRIE", "APPENDIX_KEYWORDS_TRIE",
    "ROMAN_NUMERAL_MAP", "ENGLISH_WORD_TO_NUMBER", "FORMULA_CHAR_WEIGHTS", "NUMBERED_PREFIX_RE", "DEAD_DIGIT_RE",
    "is_heading_continuation", "similar_style", "matches_abstract", "matches_references", "vertically_close", "is_equation_adjacent_line", "has_substantive_content", "is_cover_page", "token_to_number", "letter_to_ordinal",
    "BlockNeighborCache", "compute_bucket_span", "neighbor_above", "body_neighbor_above", "neighbor_right", "closest_body_neighbor_above",
    "PageNeighborMap", "PageScanState", "DocCandidateCollector", "filter_page_candidates",
    "classify_heading", "is_acceptable_heading", "try_classify_heading", "push_candidate", "detect_numbered_heading", "make_heading_candidate", "detect_labeled_heading", "make_body_heading_candidate", "detect_heading_with_body", "detect_chapter_appendix",
    "is_too_wide_for_heading", "passes_neighbor_check", "make_plain_candidate", "make_numbered_candidate", "has_competing_labeled_heading", "detect_font_heading", "scan_page_headings", "detect_box_heading",
    "build_doc_heading_candidates",
]
