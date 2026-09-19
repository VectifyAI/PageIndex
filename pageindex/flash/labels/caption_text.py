"""Caption label text helpers and structural-number parsing."""

from __future__ import annotations

import regex as regex_module  # Unicode \p{...} property classes.
from typing import Optional
from ..model import (
    Rect, rect_union, extend_top_to, extend_bottom_to, EMPTY_RECT, Bounded,
    _trim_unicode_ws,
    center_aligned, last_span, heading_score, reading_order_key, numbering_text, Line, last_line_of, first_span_of, dominant_style_of, info_weight, Block,
)
from ..tokens import Token, TokenView, wrap_tokens, enumerate_tokens, last_token, trie_prefix_match, strip_leading_if_in, first_token, set_case_fold, TrieConfig, build_trie, tokenize_block, BuiltTrie, is_word_token


# --------------------------------------------------------------------------- #
# Helpers #
# --------------------------------------------------------------------------- #


PERIOD_CHARS = {".", "．", "｡", "。"}                  # period-character set


# Structural-number pattern: Unicode numeric code points, optional letter
# affixes, or Roman numerals. ``\Z`` anchors at the absolute end of string, not
# before a trailing newline.
STRUCTURAL_NUMBER_RE = regex_module.compile(
    r"^(?:[A-M]*\p{Number}+[A-Ma-m]?|[A-Ma-m]\p{Number}*|[IVX]+)\Z"
)


def is_number_separator(token: Optional[Token], other_flag: bool = True) -> bool:
    """Return whether the token is a structural-number separator candidate."""
    if token is None:
        return False
    if token.boundary_slot:
        return False
    if token.type == 3:
        return True
    if other_flag and token.type == 4:
        return True
    return False


def extract_structural_number(tokens: TokenView, other_flag: bool = True) -> Optional[TokenView]:
    """extract a leading structural-number prefix from tokens. Returns the matched prefix as a token-view slice, or None. """
    if tokens.length < 1:
        return None
    candidate_item = tokens
    first = tokens.token_at(0)
    if first is None:
        return None
    reference_item = first.str
    if len(reference_item) == 1 and "A" <= reference_item[0] <= "H":
        if not is_number_separator(tokens.token_at(1), other_flag):
            return None
        candidate_item = tokens.slice(2)
    if candidate_item.length < 1:
        return None
    head = first_token(candidate_item)
    if head is None or not STRUCTURAL_NUMBER_RE.match(head.str):
        return None
    candidate_item = candidate_item.slice(1)
    while candidate_item.length >= 2 and is_number_separator(candidate_item.token_at(0), other_flag) and STRUCTURAL_NUMBER_RE.match(candidate_item.token_at(1).str):  # type: ignore[union-attr]
        candidate_item = candidate_item.slice(2)
    return tokens.slice(0, tokens.length - candidate_item.length)


# - format code label
def format_caption_label(type_: int, num: Optional[TokenView]) -> str:
    """format the section-type letter prefix + number. type_ 4 -> "F", 5 -> "T", 11 -> "Q". Append the number string if any. """
    if type_ == 4:
        letter = "F"
    elif type_ == 5:
        letter = "T"
    elif type_ == 11:
        letter = "Q"
    else:
        return ""
    if num is not None:
        letter += _trim_unicode_ws(str(num))
    return letter


# - case-sensitive trie of phrases that indicate "this is a
# reference TO a figure/table, not a label OF one".
REFERENCE_PHRASE_TRIE = build_trie(["lists the", "presents", "show the", "showed the", "shows"], set_case_fold(TrieConfig(), False))


# --------------------------------------------------------------------------- #
# Token helpers for caption-entry ranking.
# --------------------------------------------------------------------------- #


def is_uppercase_dominant(tokens: TokenView) -> bool:
    """Return True when the token sequence is dominated by uppercase words. Multi-character lowercase-start words whose second character is not uppercase reject the sequence as body-like text."""
    from ..tokens import char_category
    secondary_item = candidate_item = 0
    for reference_item in tokens:
        if reference_item.type != 2:
            continue
        if reference_item.primary_slot == 2:
            secondary_item += 1
        elif reference_item.primary_slot == 3:
            if len(reference_item.str) > 4 and len(reference_item.str) >= 2 and char_category(reference_item.str[1]) != 2:
                return False
            candidate_item += 1
    return secondary_item > max(2, candidate_item)


def trie_matches_all(trie: BuiltTrie, tokens: TokenView) -> bool:
    """tokens fully match ``trie`` (or all but a final word-y token)."""
    match = trie_prefix_match(trie, tokens)
    if match is None:
        return False
    if match.length == tokens.length:
        return True
    if match.length == tokens.length - 1:
        last = last_token(tokens)
        return last is not None and is_word_token(last)
    return False


def advance_past_line(tokens: TokenView, line: Line, index: int) -> int:
    """Advance while the token at the current index belongs to ``line``."""
    while index < tokens.length:
        tok = tokens.token_at(index)
        if tok is None:
            break
        if tok.line() is not line:
            break
        index += 1
    return index


def skip_bracketed_word(tokens: TokenView, index: int) -> int:
    """advance over bracket-attached word token."""
    tok = tokens.token_at(index)
    if tok is not None and tok.boundary_slot and is_word_token(tok):
        return index + 1
    return index


def token_case_signal(token: Optional[Token]) -> int:
    """per-token "direction signal". Returns 2 if g==7/6 (sentence end), 1 if g==2 (uppercase), -1 if g==3 (lowercase), 0 otherwise. """
    if token is None:
        return 0
    token_kind = token.primary_slot
    if token_kind == 7 or token_kind == 6:
        return 2
    if token_kind == 2:
        return 1
    if token_kind == 3:
        return -1
    return 0


def caption_outranks(caption_entry: "CaptionEntry", other_caption_entry: "CaptionEntry") -> bool:
    """Return True when the first caption entry ranks better than the second."""
    caption = is_uppercase_dominant(tokenize_block(caption_entry.group_slot))
    other_is_uppercase = is_uppercase_dominant(tokenize_block(other_caption_entry.group_slot))
    if caption != other_is_uppercase:
        return caption
    caption_first_token = first_token(caption_entry.secondary_slot) if caption_entry.secondary_slot.length > 0 else None
    other_first_token = first_token(other_caption_entry.secondary_slot) if other_caption_entry.secondary_slot.length > 0 else None
    group = token_case_signal(caption_first_token)
    other_case_signal = token_case_signal(other_first_token)
    if group != other_case_signal:
        return group > other_case_signal
    if caption_entry.page_index != other_caption_entry.page_index:
        return caption_entry.page_index < other_caption_entry.page_index
    return caption_entry.group_slot.reading_order_index < other_caption_entry.group_slot.reading_order_index
