"""Page scan state and heading-candidate constructors."""

from __future__ import annotations

import math
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

from .text_checks import matches_references
from .neighbors import (
    neighbor_right,
    closest_body_neighbor_above,
    PageNeighborMap,
)


# --------------------------------------------------------------------------- #
# Main per-page heading state #
# --------------------------------------------------------------------------- #


class PageScanState:
    """Per-page heading scan state."""

    __slots__ = ("secondary_slot", "primary_slot", "state_slot", "auxiliary_slot", "tertiary_slot", "option_slot", "measure_slot")

    def __init__(self, doc, page):
        self.secondary_slot = doc                  # document state
        self.primary_slot = page
        self.state_slot = doc.primary_slot[page.page_index - 2] if page.page_index >= 2 else None    # prev page
        self.auxiliary_slot = page.output_slot               # blocks in original order
        self.tertiary_slot = PageNeighborMap(page)             # neighbor map
        self.option_slot: list[HeadingCandidate] = []         # output candidates
        self.measure_slot: set = set()           # set of block ids already pushed


# --------------------------------------------------------------------------- #
# Push heading candidate into page state #
# --------------------------------------------------------------------------- #


def push_candidate(page_scan: PageScanState, candidate: HeadingCandidate) -> None:
    """Push a candidate into the page scan state."""
    page_scan.option_slot.append(candidate)
    page_scan.measure_slot.add(candidate.group_slot)


# --------------------------------------------------------------------------- #
# Heading-candidate builder.
# --------------------------------------------------------------------------- #


def make_heading_candidate(page_scan: PageScanState, type_: int, block: Block, item_list: list[int],
       tokens: Optional[TokenView], title_tokens: Optional[TokenView], has_numbering_flag: bool = False) -> HeadingCandidate:
    """Build a heading candidate and apply the spatial promotion rule."""
    neighbor = page_scan.tertiary_slot
    right_neighbor = neighbor_right(neighbor, block)
    # Spatial promotion to structural numbering: if a right-side neighbour exists, the block
    # has high skew (real horizontal text), its title ends in a colon-like
    # symbol, and its last line is nearly as wide as and right-aligned to
    # the neighbour -> promote the flag to true.
    if (
        not has_numbering_flag and right_neighbor is not None
        and block.previous_slot > 0.9 and title_tokens is not None
    ):
        last_title_token = last_token(title_tokens)
        if last_title_token is not None and is_trimmable_token(last_title_token):
            mh_block = last_line_of(block)
            if (
                mh_block.bbox_width() > 0.7 * right_neighbor.bbox_width()
                and abs(mh_block.right_edge() - right_neighbor.right_edge()) < 2 * avg_char_width(mh_block)
            ):
                has_numbering_flag = True
    prominent_flag = (
        type_ == 7
        or (len(item_list) > 0 and title_tokens is not None and matches_references(title_tokens))
    )
    return HeadingCandidate(
        type_=type_,
        page=page_scan.primary_slot,
        group_value=block,
        anchor=closest_body_neighbor_above(neighbor, block),
        numbering_value=item_list,
        tokens=tokens,
        title_tokens=title_tokens,
        has_numbering_flag=has_numbering_flag,
        prominent_flag=prominent_flag,
    )


# --------------------------------------------------------------------------- #
# Shorthand heading-candidate builders #
# --------------------------------------------------------------------------- #


def make_plain_candidate(page_scan: PageScanState, type_: int, block: Block) -> HeadingCandidate:
    """Build a type-only candidate using the full block text."""
    return make_heading_candidate(page_scan, type_, block, [], None, tokenize_block(block), False)


def make_body_heading_candidate(page_scan: PageScanState, type_: int, block: Block, tokens: TokenView) -> HeadingCandidate:
    """Build a candidate from body-heading tokens."""
    return make_heading_candidate(page_scan, type_, block, [], None, trim_trailing_punct(tokens), True)


# --------------------------------------------------------------------------- #
# Composed-number heading-candidate builder #
# --------------------------------------------------------------------------- #


def make_numbered_candidate(page_scan: PageScanState, block: Block, item_list: list[int], tokens: TokenView, title_tokens: TokenView) -> Optional[HeadingCandidate]:
    """Build a numbered-heading candidate after the full reject-guard chain. The guard rejects empty numbering, weak single-token numbering, unsupported top-of-page continuations, alignment failures, and trailing-number continuation conflicts."""
    from ..labels import extract_structural_number  # numbering-prefix detector

    # Basic reject branch for empty, weak, or top-of-page continuation markers.
    if title_tokens.length <= 0:
        return None
    first_title_token = first_token(title_tokens)
    if (title_tokens.length == 1 and first_title_token is not None
            and first_title_token.primary_slot != 2 and first_title_token.primary_slot != 4 and first_title_token.secondary_slot != 2
            and not block.isolated_centered):
        return None
    if len(item_list) == 1 and item_list[0] == 1 and block.top_edge() < 0.3 * page_scan.primary_slot.bounds.bbox_height():
        from ..heading_detection import neighbor_right
        if neighbor_right(page_scan.tertiary_slot, block) is None:
            last_title_token = last_token(title_tokens)
            if last_title_token is not None and last_title_token.anchor_ranges and last_title_token.anchor_ranges[-1].line is last_line_of(block):
                return None

    # If basic guards didn't trigger, examine multi-line patterns.
    reject = False
    if block.line_count() > 1:
        second_line = block.primary_slot[1]
        first_number_token = first_token(tokens)
        first_title_token = first_token(title_tokens)
        if first_number_token is not None and first_title_token is not None:
            left = first_anchor_span(first_number_token).left_edge()
            title_left = first_anchor_span(first_title_token).left_edge()
            if not (left < title_left and second_line.left_edge() > (left + title_left) / 2):
                # Check trailing tokens for c+1 continuation
                trailing_tokens = tokenize_block(block)
                trailing_tokens = trailing_tokens.slice(_di_count(trailing_tokens, block.line()))
                trailing_tokens = extract_structural_number(trailing_tokens)
                if trailing_tokens is None or trailing_tokens.length <= 0:
                    reject = False
                elif block.measure_slot:
                    reject = True
                else:
                    if len(item_list) == 1 and trailing_tokens.length <= 2:
                        trailing_first_token = trailing_tokens.token_at(0)
                        if trailing_first_token is not None:
                            value = token_numeric_value(trailing_first_token)
                            # Strict equality on the raw Number, no truncation
                            # (a fractional value never
                            # equals the integer c[0]+1).
                            reject = (not math.isnan(value) and value == item_list[0] + 1)
                        else:
                            reject = False
                    else:
                        reject = False
    if reject:
        return None
    return make_heading_candidate(page_scan, 1, block, item_list, tokens, title_tokens, False)


def _di_count(tokens: TokenView, line) -> int:
    """count tokens belonging to ``line`` starting from index 0."""
    count_item = 0
    for index_value in range(tokens.length):
        token_value = tokens.token_at(index_value)
        if token_value is None or token_value.line() is not line:
            break
        count_item += 1
    return count_item


# --------------------------------------------------------------------------- #
# Numbered heading detector #
# --------------------------------------------------------------------------- #


def _number_at_token_index(tokens: TokenView, index: int) -> int:
    """Try to extract a numbering value at index ``b_idx`` of a token view. Returns 0 if not a number-followed-by-separator, else the number. """
    if tokens.length < index + 2:
        return 0
    token = tokens.token_at(index)
    if token is None or token.type != 1:
        return 0
    next_tok = tokens.token_at(index + 1)
    if next_tok is None:
        return 0
    from ..labels import PERIOD_CHARS as period_chars
    if not (
        next_tok.str in period_chars
        or next_tok.str in (")", "]", "．", "｡", "。", "）", "］", "】")
    ):
        return 0
    val = token_numeric_value(token)
    if math.isnan(val) or val <= 0 or val >= 1000:
        return 0
    return int(val)
