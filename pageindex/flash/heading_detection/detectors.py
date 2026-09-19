"""Numbered, labeled, chapter/appendix, and box heading detectors plus acceptability checks."""

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

from .keyword_tables import (
    APPENDIX_SECTION_TRIE,
    BOX_KEYWORD_TRIE,
    CHAPTER_WORDS_TRIE,
    APPENDIX_KEYWORDS_TRIE,
    ROMAN_NUMERAL_MAP,
)
from .text_checks import (
    similar_style,
    matches_abstract,
    matches_references,
    has_substantive_content,
    clamp,
    token_to_number,
    letter_to_ordinal,
)
from .neighbors import (
    neighbor_above,
    body_neighbor_above,
    neighbor_right,
    closest_body_neighbor_above,
)
from .candidates import (
    PageScanState,
    make_heading_candidate,
    make_plain_candidate,
    make_numbered_candidate,
)


def detect_numbered_heading(page_scan: PageScanState, block: Block, tokens: TokenView) -> Optional[HeadingCandidate]:
    """Identify "1.2.3" / "[1]" style numbered heading prefixes."""
    item_list: list[int] = []
    at_value = None
    for entry in enumerate_tokens(tokens):
        index = entry["index"]
        token = entry["token"]
        if token.type == 1:
            if len(item_list) >= 4:
                break
            val = token_numeric_value(token)
            if math.isnan(val) or val <= 0 or val >= 20:
                break
            if len(token.str) >= 3:
                break
            item_list.append(int(val))
            if not token.boundary_slot:
                continue
            at_value = tokens.token_at(index + 1)
            if (
                index + 2 < tokens.length
                and at_value is not None
                and (is_word_token(at_value) or at_value.type == 6)
                and tokens.token_at(index + 2) is not None
                and tokens.token_at(index + 2).type == 1
            ):
                break
            # Strong numbering, prominent style, or a viable separator token is
            # enough to build a numbered-heading candidate.
            if (
                len(item_list) > 1
                or heading_score(block) > page_scan.primary_slot.primary_slot.primary_slot + 1
                or (not block.measure_slot and at_value is not None and (
                    at_value.primary_slot in (2, 4) or at_value.secondary_slot == 2 or at_value.type == 4
                    or at_value.str == "." or at_value.str == "|"
                ))
            ):
                return make_numbered_candidate(
                    page_scan, block, item_list,
                    tokens.slice(0, index + 1),
                    tokens.slice(index + 1),
                )
            return None
        # Accept period-like punctuation or a symbol token as a numbering
        # separator.
        if token.str in (".", "．", "｡", "。") or token.type == 4:
            prev = tokens.token_at(index - 1)   # token_at(-1) returns None
            if prev is None or prev.type != 1:
                break
            if not token.boundary_slot:
                continue
            return make_numbered_candidate(
                page_scan, block, item_list,
                tokens.slice(0, index + 1), tokens.slice(index + 1),
            )
        if len(item_list) <= 0 or token.type != 2:
            break
        if token.primary_slot not in (2, 4):
            break
        if (
            len(item_list) > 1
            or len(token.str) >= 3
            or tokens.length - index >= 3
        ):
            return make_numbered_candidate(
                page_scan, block, item_list,
                tokens.slice(0, index), tokens.slice(index),
            )
        return None
    return None


# --------------------------------------------------------------------------- #
# Complex numbering format detector.
# --------------------------------------------------------------------------- #


def detect_labeled_heading(page_scan: PageScanState, block: Block, tokens: TokenView) -> Optional[HeadingCandidate]:
    """Detect Roman, letter, CJK, and mixed-numbering headings."""
    if tokens.length <= 1:
        return None
    first = tokens.token_at(0)
    second = tokens.token_at(1)
    if first is None or second is None:
        return None
    # Roman numeral path
    roman = ROMAN_NUMERAL_MAP.get(first.str)
    if roman is not None and is_word_token(second) and second.str in ".．｡。:)":
        prefix = tokens.slice(0, 2)
        return make_heading_candidate(page_scan, 2, block, [roman], prefix, tokens.slice(prefix.length))
    # CJK number path
    cjk_pos = "一二三四五六七八九十".find(first.str)
    if cjk_pos >= 0 and is_word_token(second):
        prefix = tokens.slice(0, 2)
        rest = tokens.slice(prefix.length)
        if rest.length <= 0:
            return None
        return make_heading_candidate(page_scan, 3, block, [cjk_pos + 1], prefix, rest)
    # Letter path
    if tokens.length <= 1 or (block.char_stats.secondary_slot == 3 and (block.line_count() > 1 or is_punct_category(block.char_stats.tertiary_slot))):
        return None
    letter_val = letter_to_ordinal(first.str)
    if letter_val is None:
        return None
    if second.str == "." or second.str == ")":
        value = letter_val
    else:
        first_anchor = first_anchor_span(first)
        second_anchor = first_anchor_span(second)
        if (
            not first.boundary_slot
            or first_anchor is second_anchor
            or second_anchor.left_edge() < first_anchor.right_edge() + first_anchor.bbox_width()
            or heading_score(block) < page_scan.primary_slot.primary_slot.primary_slot + 1
            or letter_count(block.char_stats) / tokens.length < 2
        ):
            return None
        value = letter_val
    item_list: list[int] = [value]
    prefix = tokens.slice(0, 2 if is_word_token(second) else 1)
    rest = tokens.slice(prefix.length)
    if second.str == "." and not second.boundary_slot and rest.length >= 2:
        first_rest = first_token(rest)
        if first_rest is not None and first_rest.type == 1:
            heading = token_numeric_value(first_rest)
            if math.isnan(heading) or heading <= 0 or heading >= 20:
                return None
            item_list.append(int(heading))
            rest = rest.slice(1)
            first_rest = first_token(rest)
            if rest.length > 0 and first_rest is not None and is_word_token(first_rest):
                rest = rest.slice(1)
    if rest.length <= 0:
        return None
    prefix = tokens.slice(0, tokens.length - rest.length)
    return make_heading_candidate(page_scan, 4, block, item_list, prefix, rest)


# --------------------------------------------------------------------------- #
# Chapter, appendix, and box-style dispatch.
# --------------------------------------------------------------------------- #


def detect_chapter_appendix(page_scan: PageScanState, other_block: Block) -> Optional[HeadingCandidate]:
    """. Match "Chapter X" / "Appendix X" / box-N / etc."""
    candidate_item = heading_score(other_block)
    if candidate_item <= page_scan.primary_slot.primary_slot.primary_slot + 0.1:
        return None
    flag = (
        other_block.isolated_centered or candidate_item > page_scan.secondary_slot.secondary_slot.primary_slot + 0.1
        and (other_block.bold_frac() > 0.9 or is_upper_dominant(other_block.char_stats) or candidate_item > 1.5 * page_scan.secondary_slot.secondary_slot.primary_slot)
    )
    tokens = tokenize_block(other_block)
    match = None
    if flag:
        match = trie_prefix_match(CHAPTER_WORDS_TRIE, tokens)
    if flag and match is not None:
        value = token_to_number(tokens.token_at(match.length))
        if value is None:
            return None
        prefix = tokens.slice(0, skip_bracketed_word(tokens, match.length + 1))
        return make_heading_candidate(page_scan, 8, other_block, [value], prefix, tokens.slice(prefix.length))
    if flag:
        match = trie_prefix_match(APPENDIX_SECTION_TRIE, tokens)
        if match is not None:
            prefix = tokens.slice(0, skip_bracketed_word(tokens, match.length))
            return make_heading_candidate(page_scan, 9, other_block, [], prefix, tokens.slice(prefix.length))
    match = trie_prefix_match(APPENDIX_KEYWORDS_TRIE, tokens)
    if match is not None:
        next_item = tokens.token_at(match.length)
        val = token_to_number(next_item) or (letter_to_ordinal(next_item.str) if next_item is not None else None)
        if not flag and val is None:
            return None
        item_list = [val] if val is not None else []
        prefix = tokens.slice(0, skip_bracketed_word(tokens, match.length + (1 if val is not None else 0)))
        return make_heading_candidate(page_scan, 10, other_block, item_list, prefix, tokens.slice(prefix.length))
    return None


# --------------------------------------------------------------------------- #
# Box-format heading.
# --------------------------------------------------------------------------- #


def detect_box_heading(page_scan: PageScanState, other_block: Block) -> Optional[HeadingCandidate]:
    """match "Box N" pattern."""
    tokens = tokenize_block(other_block)
    match = trie_prefix_match(BOX_KEYWORD_TRIE, tokens)
    if match is None:
        return None
    rest = tokens.slice(match.length)
    if rest.length <= 0 or rest.token_at(0).type != 1:
        return None
    val = token_numeric_value(rest.token_at(0))
    if math.isnan(val) or val <= 0:
        return None
    prefix = tokens.slice(0, skip_bracketed_word(tokens, match.length + 1))
    return make_heading_candidate(page_scan, 12, other_block, [int(val)], prefix, tokens.slice(prefix.length))


# --------------------------------------------------------------------------- #
# Heading-type dispatcher.
# --------------------------------------------------------------------------- #


def classify_heading(page_scan: PageScanState, other_block: Block) -> HeadingCandidate:
    """. Sequential dispatch through type detectors; fallback to the font-position classifier."""
    tokens = tokenize_block(other_block)
    heading = detect_chapter_appendix(page_scan, other_block)
    if heading is None:
        heading = detect_box_heading(page_scan, other_block)
    if heading is None:
        heading = detect_numbered_heading(page_scan, other_block, tokens)
    if heading is None:
        heading = detect_labeled_heading(page_scan, other_block, tokens)
    if heading is not None:
        return heading
    type_code = 7 if matches_references(tokens) else (5 if matches_abstract(tokens) else 0)
    return make_plain_candidate(page_scan, type_code, other_block)


# --------------------------------------------------------------------------- #
# Heading acceptance gate.
# --------------------------------------------------------------------------- #


def is_acceptable_heading(page_scan: PageScanState, other_heading_candidate: HeadingCandidate) -> bool:
    """. The big "is this an acceptable heading?" gate."""
    heading = other_heading_candidate.group_slot
    if heading.bbox_height() >= 2 * heading.bbox_width() or info_weight(heading.char_stats) <= 3 or heading.line_count() > 5 or heading.char_count() >= 300:
        return False
    page_height = page_scan.primary_slot.bounds.bbox_height()
    if heading.bottom_edge() > 0.95 * page_height:
        return False
    score = heading_score(heading)
    doc_group = page_scan.secondary_slot.secondary_slot.primary_slot
    if score <= page_scan.primary_slot.primary_slot.primary_slot + 0.5 and score <= doc_group + 0.5 and not other_heading_candidate.is_prominent:
        return False
    width = page_scan.primary_slot.bounds.bbox_width()
    if (
        (heading.left_edge() > 0.55 * width and score <= doc_group + 5)
        or heading.left_edge() > 0.75 * width
        or (
            heading.left_edge() > 0.4 * width
            and heading.center_x() > 0.6 * width
            and page_scan.primary_slot.primary_slot.secondary_slot > min(1000, page_scan.secondary_slot.secondary_slot.secondary_slot)
        )
    ):
        return False
    col_bottom = page_scan.primary_slot.tertiary_slot[safe_column_index(heading)] if 0 <= safe_column_index(heading) < len(page_scan.primary_slot.tertiary_slot) else None
    if (
        heading.bbox_width() < 0.2 * width and col_bottom is not None
        and col_bottom.bbox_width() < 0.2 * width and col_bottom.bbox_height() > 1.5 * col_bottom.bbox_width()
    ):
        return False
    neighbor = neighbor_right(page_scan.tertiary_slot, heading)
    gap = heading.bottom_edge() - neighbor.top_edge() if neighbor is not None else math.inf
    line_gap = page_scan.primary_slot.primary_slot.tertiary_slot - page_scan.primary_slot.primary_slot.primary_slot
    if gap < 0.9 * line_gap:
        return False
    above = neighbor_above(page_scan.tertiary_slot, heading)
    above_gap = above.bottom_edge() - heading.top_edge() if above is not None else math.inf
    if above_gap < 0.9 * line_gap:
        return False
    if (
        page_scan.state_slot is not None
        and not page_scan.state_slot.measure_slot
        and (
            page_scan.state_slot.primary_slot.secondary_slot < clamp(page_scan.secondary_slot.secondary_slot.secondary_slot, 200, 500)
            or not page_scan.state_slot.state_slot
        )
        and score > doc_group + 0.5
    ):
        return True
    doc_right_neighbor = page_scan.secondary_slot.secondary_slot.measure_slot
    previous_page_left_neighbor = page_scan.state_slot.primary_slot.option_slot if page_scan.state_slot is not None else math.nan
    previous_page_height = page_scan.state_slot.bounds.bbox_height() if page_scan.state_slot is not None else math.nan
    centered_flag = heading.isolated_centered
    if (
        previous_page_left_neighbor <= doc_right_neighbor
        and (score <= doc_group + 1.5 or (score <= doc_group + 5 and not centered_flag))
        or heading.weighted_ratio_primary < 0.5 * page_scan.secondary_slot.secondary_slot.auxiliary_slot
    ):
        return False
    body_neighbor = closest_body_neighbor_above(page_scan.tertiary_slot, heading)
    # Reject candidates that are separated from a classified above-neighbor, or
    # whose own content is more formula-like than heading-like.
    if body_neighbor is not None and heading.bottom_edge() - body_neighbor.top_edge() > 2 * heading.bbox_height() and body_neighbor.marker_slot != 0:
        return False
    previous_block = page_scan.auxiliary_slot[heading.orig_index - 1] if 0 <= heading.orig_index - 1 < len(page_scan.auxiliary_slot) else None
    next_block = page_scan.auxiliary_slot[heading.orig_index + 1] if 0 <= heading.orig_index + 1 < len(page_scan.auxiliary_slot) else None
    if has_substantive_content(heading, previous_block, next_block):
        return False
    return (
        (previous_page_left_neighbor > doc_right_neighbor + 0.1 * previous_page_height
         and (score > doc_group + 2
              or (previous_page_left_neighbor > doc_right_neighbor + 0.2 * previous_page_height
                  and body_neighbor is not None and neighbor is not None
                  and gap > neighbor.avg_font_size())))
        or (centered_flag and (neighbor is None or neighbor.marker_slot == 0))
        or score > 1.5 * doc_group
        or (len(other_heading_candidate.numbering) == 1 and other_heading_candidate.numbering[0] == 1 and above is None)
    )


def safe_column_index(block) -> int:
    """Safe wrapper for hh that handles missing H field."""
    from ..stats import column_index_of
    # Empty containers must return -1; returning 0 would index a real column.
    return column_index_of(block)


# --------------------------------------------------------------------------- #
# Heading classification + acceptability gate #
# --------------------------------------------------------------------------- #


def try_classify_heading(page_scan: PageScanState, other_block: Block) -> Optional[HeadingCandidate]:
    """Try to build a candidate for a block, then apply rejection gates."""
    candidate = classify_heading(page_scan, other_block)
    if len(candidate.numbering) > 1:
        return None
    if candidate.type in (8, 9, 10):
        return candidate
    if candidate.type == 12:
        return None
    return candidate if is_acceptable_heading(page_scan, candidate) else None


# --------------------------------------------------------------------------- #
# Additional heading detectors and neighbor gates.
# --------------------------------------------------------------------------- #


def is_too_wide_for_heading(page_scan: PageScanState, other_block: Block) -> bool:
    """. Block is too wide / central to be a heading."""
    width = other_block.bbox_width()
    if width > 0.7 * page_scan.primary_slot.bounds.bbox_width() / 2 or width > 0.7 * page_scan.secondary_slot.secondary_slot.option_slot:
        return True
    count = 0
    for heading in range(page_scan.primary_slot.page_index - 1, page_scan.primary_slot.page_index + 2):
        if 0 < heading <= len(page_scan.secondary_slot.primary_slot):
            page = page_scan.secondary_slot.primary_slot[heading - 1]
            if width > 0.7 * page.primary_slot.previous_slot:
                count += 1
    return count >= 2


def passes_neighbor_check(page_scan: PageScanState, other_block: Block) -> bool:
    """Block-level neighbor-aware acceptance gate. Returns True when the caller should reject the block."""
    if is_too_wide_for_heading(page_scan, other_block):
        return False
    blocks = page_scan.auxiliary_slot
    prev_idx = other_block.orig_index - 1
    candidate_item = blocks[prev_idx] if 0 <= prev_idx < len(blocks) else None
    overlap = candidate_item is not None and y_overlaps(other_block, candidate_item)
    if overlap and is_too_wide_for_heading(page_scan, candidate_item):
        return False
    next_idx = other_block.orig_index + 1
    candidate_item = blocks[next_idx] if 0 <= next_idx < len(blocks) else None
    next_overlap = candidate_item is not None and y_overlaps(other_block, candidate_item)
    if next_overlap and is_too_wide_for_heading(page_scan, candidate_item):
        return False
    if not overlap and not next_overlap:
        return False
    candidate_item = neighbor_right(page_scan.tertiary_slot, other_block)
    if candidate_item is not None and is_too_wide_for_heading(page_scan, candidate_item):
        return False
    if candidate_item is not None and not candidate_item.is_body_paragraph and candidate_item.line_count() > 3 and candidate_item.bbox_height() > 0.8 * candidate_item.bbox_width():
        return True
    # Compare against the closest above-neighbor with a width threshold derived
    # from this block's first line.
    above_or_overlap = closest_body_neighbor_above(page_scan.tertiary_slot, other_block)
    threshold = 4 * avg_char_width(other_block.line())
    if (above_or_overlap is not None and candidate_item is not above_or_overlap
            and x_aligned(other_block, above_or_overlap, threshold)
            and above_or_overlap.state_slot == 0
            and is_too_wide_for_heading(page_scan, above_or_overlap)):
        return False
    keyword_match = body_neighbor_above(page_scan.tertiary_slot, other_block)
    if (keyword_match is not None
            and x_aligned(other_block, keyword_match, threshold)
            and keyword_match.state_slot == 0
            and is_too_wide_for_heading(page_scan, keyword_match)):
        return False
    return True


def has_competing_labeled_heading(page_scan: PageScanState, other_heading_candidate: HeadingCandidate, candidate_block: Block) -> bool:
    """Cross-page reject check for competing labeled-heading siblings."""
    if candidate_block.type != 0 or candidate_block.char_count() >= 500:
        return False
    block = other_heading_candidate.group_slot
    if not similar_style(block, candidate_block) or abs(block.top_edge() - candidate_block.top_edge()) >= 5 * block.bbox_height():
        return False
    other_candidate = detect_labeled_heading(page_scan, candidate_block, tokenize_block(candidate_block))
    if other_candidate is None or other_heading_candidate.type != other_candidate.type:
        return False
    return abs(other_candidate.numbering[0] - other_heading_candidate.numbering[0]) >= 1


def is_year_string(text: str) -> bool:
    """True iff the text parses to a plausible year (1700..2100)."""
    value = to_number(text)
    return not math.isnan(value) and 1700 < value < 2100


def is_bibliography_entry(block: Block, other_number: int = -1) -> bool:
    """True iff ``block`` looks like a bibliography entry."""
    if other_number < 0:
        other_number = 0
        for line in block:
            reference_item = numbering_value(line)
            if not math.isnan(reference_item) and 0 < reference_item <= 9999:
                other_number += 1
    if other_number < 2 and block.char_count() / max(1, other_number) > 300:
        return False
    year = 0
    digit = 0
    word = 0
    period_after_word = 0
    word_state = 0
    tokens = tokenize_block(block)
    for entry in enumerate_tokens(tokens):
        state_item = entry["token"]
        if is_word_token(state_item):
            if state_item.type == 3 and word_state == 1:
                period_after_word += 1
            word_state = 0
        elif state_item.type == 1:
            key_value = token_numeric_value(state_item)
            if 0 < key_value < 1000:
                digit += 1
            elif is_year_string(state_item.str):
                year += 1
        elif state_item.type == 2:
            word += 1
            word_state += 1
    if word < 0.1 * tokens.length:
        return False
    return digit >= 1.5 * other_number or year >= 0.5 * other_number or period_after_word >= 0.5 * other_number
