"""Tokenizer subsystem. The tokenizer is character-driven: it walks each character of each line,
uses category-transition tolerances to decide when the current token can
extend, and closes tokens when script, punctuation, or spacing transitions
require a boundary. Tokens keep back-references to the contributing line and span offsets so the
visible text can be reconstructed and cross-line tokens, such as a word broken
by a hyphen across two lines, can be stitched.
"""

import unicodedata
from typing import Any, Iterable, Iterator, Optional

from ..model import (
    _strip_diacritics,
    avg_char_width2,
    intervals_overlap,
    to_number,
    rect_union,
    EMPTY_RECT,
    avg_char_width,
    Line,
    char_category,
    is_word_category,
    is_punct_category,
    letter_count,
    punct_count,
    info_weight,
    Block,
)

from .token_types import (
    SCRIPT_FAMILY_MAP,
    _build_gap_tolerance_grid,
    GAP_TOLERANCE_GRID,
    can_extend_token,
    TokenAnchor,
    last_token_anchor,
    first_anchor_span,
    is_char_token,
    is_word_token,
    is_trimmable_token,
    token_numeric_value,
    Token,
    TokenView,
    wrap_tokens,
    enumerate_tokens,
    first_token,
    last_token,
)
from .tokenizer import (
    LineTokenizer,
    tokenize_block,
    clamp_value,
    is_superscript_adjacent,
)
from .tries import (
    _de_norm,
    TrieConfig,
    BuiltTrie,
    set_reverse,
    set_case_fold,
    TrieNode,
    trie_insert_step,
    trie_walk_step,
    aho_corasick_match,
    aho_corasick_tokens,
    TrieBuilder,
    _trie_insert_entry,
    trie_bulk_insert,
    _trie_finalize,
    build_trie,
    trie_prefix_match,
    _trie_full_match,
    trie_full_match,
    strip_trie_match,
    strip_leading_if_in,
    COMMA_CHARS,
    strip_trailing_comma,
    is_comma_token,
    trim_trailing_punct,
)
from .hashing import (
    _FH_MASK,
    _to_uint32,
    _to_int32,
    _int32_xor,
    _int32_left_shift,
    _uint32_right_shift,
    _little_endian_signed_word,
    _utf8_bytes_from_utf16_units,
    _jenkins_mix,
    jenkins_hash,
)

__all__ = [
    # state machine
    "GAP_TOLERANCE_GRID", "can_extend_token", "TokenAnchor", "last_token_anchor", "first_anchor_span",
    "is_char_token", "is_word_token", "is_trimmable_token", "token_numeric_value", "Token",
    "TokenView", "wrap_tokens", "enumerate_tokens", "first_token", "last_token",
    "LineTokenizer", "tokenize_block",
    "clamp_value", "is_superscript_adjacent", "jenkins_hash",
    # trie
    "TrieConfig", "BuiltTrie", "set_reverse", "set_case_fold", "TrieNode", "trie_insert_step", "trie_walk_step", "build_trie", "trie_prefix_match", "trie_full_match",
    "strip_trie_match", "strip_leading_if_in", "strip_trailing_comma", "trim_trailing_punct", "COMMA_CHARS",
    "SCRIPT_FAMILY_MAP", "GAP_TOLERANCE_GRID",
]
