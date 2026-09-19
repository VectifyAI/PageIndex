"""Dictionary-backed keyword tries and shared regexes."""

from __future__ import annotations

import json
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


# --------------------------------------------------------------------------- #
# Load dictionaries (built into tries on first use) #
# --------------------------------------------------------------------------- #


_DICT_PATH = Path(__file__).parent.parent / "data" / "dictionaries.json"
_DICTS = json.loads(_DICT_PATH.read_text(encoding="utf-8"))


def _dict_trie(key: str) -> BuiltTrie:
    """Build a case-folded trie from a dictionary entry."""
    return build_trie(_DICTS.get(key, []), set_case_fold(TrieConfig(), True))


COPYRIGHT_TRIE = build_trie(["Copyright", "©"], set_case_fold(TrieConfig(), True))    # inline list
VOLUME_WORDS_TRIE = _dict_trie("volume_words")
TOC_TITLES_TRIE = _dict_trie("toc_titles")
FIGURE_KEYWORDS_TRIE = _dict_trie("ai_section_keywords")
_TABLE_KEYWORDS_TRIE = _dict_trie("table_keywords")
TABLE_KEYWORDS_TRIE = _TABLE_KEYWORDS_TRIE

_CHART_KEYWORDS_TRIE = _dict_trie("chart_keywords")
CHART_KEYWORDS_TRIE = _CHART_KEYWORDS_TRIE
APPENDIX_SECTION_TRIE = _dict_trie("appendices_dict")
INTRODUCTION_SECTION_TRIE = _dict_trie("introduction_dict")
BOX_KEYWORD_TRIE = build_trie(["box"], set_case_fold(TrieConfig(), True))                # inline list
KEYWORDS_SECTION_TRIE = _dict_trie("keywords_dict")

# Multilingual boilerplate phrase trie: publisher and proceeding headers plus
# stock acknowledgement openers such as "First of all I would like to thank".
# Used by the body-paragraph gate to reject boilerplate as non-body.
# Phrase list stored as a data asset.
_BOILERPLATE_PHRASES_PATH = Path(__file__).parent.parent / "data" / "boilerplate_phrases.json"
BOILERPLATE_TRIE = build_trie(json.loads(_BOILERPLATE_PHRASES_PATH.read_text(encoding="utf-8")), set_case_fold(TrieConfig(), True))

# Regular expressions for the dot-leader and page-number gates (Unicode \p{Number} -> ``regex`` module).
# Leading class is ASCII 1-9 + fullwidth 1-9 (U+FF11-FF19); it must NOT admit
# fullwidth zero U+FF10, so it is [1-9１-９], not [1-9０-９].
DOT_LEADER_ROW_RE = regex_module.compile(r"([.][" + _UNICODE_WHITESPACE_CLASS + r"]*){5,}[" + _UNICODE_WHITESPACE_CLASS + r"]*[1-9１-９]\p{Number}*\Z")
PAGE_NUMBER_ONLY_RE = regex_module.compile(r"^[ |]*([1-9１-９]\p{Number}*)[ |]*\Z")


def _search_trie(trie: BuiltTrie, tokens) -> Optional[TokenView]:
    """Return the shortest earliest Aho-Corasick trie match for ``tokens``."""
    from ..tokens import aho_corasick_tokens as _real_bh
    return _real_bh(trie, tokens)


def _normalize_text_key(text: str) -> str:
    """Strip diacritics only; callers lowercase first when a case-folded key is needed."""
    return _strip_diacritics(text)
