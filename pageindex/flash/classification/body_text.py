"""Body-paragraph classification and recurring-text recording."""

from __future__ import annotations

import math
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
    BOILERPLATE_TRIE,
    PAGE_NUMBER_ONLY_RE,
    _normalize_text_key,
)


# --------------------------------------------------------------------------- #
# Recurring-text histogram updater #
# --------------------------------------------------------------------------- #


def record_recurring_text(doc, other_text: str) -> None:
    """Increment the recurring-text histogram under the Jenkins lookup2 hash key. Empty normalized text is a valid key and must not be skipped."""
    key = jenkins_hash(other_text)
    doc.tertiary_slot[key] = doc.tertiary_slot.get(key, 0) + 1


# --------------------------------------------------------------------------- #
# Body-paragraph predicate #
# --------------------------------------------------------------------------- #


# The phrase gate is intentionally narrow. Broad Latin keyword matching
# over-rejects normal body paragraphs, for example sentences starting with
# "figure", and then lets figure captions be treated as headings.


def is_body_paragraph(doc_stats: DocStats, page, block: Block) -> bool:
    """Return whether ``block`` is a substantive body paragraph."""
    if block.weighted_ratio_primary < 0.6:
        return False
    width = info_weight(block.char_stats)
    lines = block.line_count()
    sentence_punct = block.char_stats.primary_slot[6]

    # The width-per-line ratio uses IEEE-style division. For a zero-line block,
    # d/e is +inf
    # (d>0) or NaN (d==0), so every ``d/e < k`` test is False and the block is
    # NOT rejected here (it falls through to the char-count gate below, which
    # rejects an empty block). This is not an early return.
    dw_per_line = (
        width / lines if lines != 0
        else (math.inf if width > 0 else math.nan)
    )
    if (
        dw_per_line < 15
        or (lines >= 10 and dw_per_line < 20)
        or (lines >= 10 and dw_per_line < 25 and sentence_punct < lines / 8)
        or (lines >= 20 and dw_per_line < 40 and sentence_punct < lines / 20)
    ):
        return False

    block_width = block.bbox_width()
    if lines >= 4:
        short = 0
        for state_item in block:
            if state_item.bbox_width() < 0.75 * block_width and not state_item.primary_slot[0].state_slot.startswith("•"):
                short += 1
        if short >= lines / 2 and sentence_punct < lines / 8:
            return False

    if block_width < page.bounds.bbox_width() / 7:
        return False

    chars = block.char_count()
    if chars < 40 or (lines >= 3 and alignment_code(block) == 3) or letter_count(block.char_stats) < 0.1 * chars:
        return False

    size = block.avg_font_size()
    body_size = min(page.primary_slot.primary_slot, doc_stats.primary_slot)
    min_value = min(doc_stats.primary_slot, max(page.bounds.bbox_height(), page.bounds.bbox_width()) / 60)
    min_value = min(0.7 * min_value, min_value - 3)
    # Boilerplate phrases are rejected as non-body even when they otherwise look
    # paragraph-like. This keeps acknowledgement/copyright/proceedings language
    # out of body-density calculations without broad keyword matching.

    if size < body_size - 2 or size < min_value or trie_prefix_match(BOILERPLATE_TRIE, tokenize_block(block)):
        return False

    if chars >= 250 and lines >= 4:
        return True
    if block_width < page.bounds.bbox_width() / 5 or size < body_size - 0.5:
        return False
    if chars >= 100 and lines >= 2 and sentence_punct >= 2:
        return True
    if (chars >= 100 or block.char_stats.tertiary_slot == 6) and (
        size >= page.primary_slot.primary_slot - 0.5 or size > doc_stats.primary_slot - 0.1
    ):
        return first_span_of(block).font_name == page.primary_slot.state_slot or last_span(last_line_of(block)).font_name == page.primary_slot.state_slot
    return False


# --------------------------------------------------------------------------- #
# Header/footer helper keys and predicates #
# --------------------------------------------------------------------------- #


def span_style_text_key(span) -> str:
    """Span style hash including text content: font name, rounded height, bold flag, lowercase text."""
    # Use exact half-up integer rounding; Python f"{x:.0f}" uses half-even.
    return f"{span.font_name} {_round_half_up_to_int(span.bbox_height())} {'B' if span.primary_slot else 'R'} {span.text.lower()}"


def normalized_block_text(block: Block) -> str:
    """block normalized-text hash."""
    out = []
    for token in tokenize_block(block):
        out.append(_normalize_text_key(token.str.lower()))
    return "".join(out)


# Roman numeral lookup used for page-number-like header/footer spans.
_ROMAN_NUMERALS = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
    "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13,
    "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20,
}


def span_page_number(span) -> Optional[int]:
    """Extract a page number from a span using a digit gate, then Roman numeral lookup."""
    text = span.text
    match = PAGE_NUMBER_ONLY_RE.match(text)
    if match:
        page_number = to_number(match.group(1))
        if not math.isnan(page_number) and page_number > 0 and page_number < 1e6 and page_number == math.ceil(page_number):
            return int(page_number)
        return None
    return _ROMAN_NUMERALS.get(text.upper())


def longest_word_and_number(block: Block) -> list[str]:
    """extract longest letter-word and longest digit-string. Returns a list of 0-2 strings: lowercased longest word (if >3 chars), then the longest digit-string (raw). """
    longest_word: Optional[str] = None
    longest_number: Optional[str] = None
    for tok in tokenize_block(block):
        if tok.type == 2:
            if longest_word is None or len(tok.str) > len(longest_word):
                longest_word = tok.str
        elif tok.type == 1:
            if longest_number is None or len(tok.str) > len(longest_number):
                longest_number = tok.str
    out: list[str] = []
    if longest_word and len(longest_word) > 3:
        out.append(_normalize_text_key(longest_word.lower()))
    if longest_number:
        out.append(longest_number)
    return out
