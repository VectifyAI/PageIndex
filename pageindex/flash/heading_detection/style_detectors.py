"""Font-change and body-embedded heading detectors."""

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
    SECTION_KEYWORDS_TRIE,
    INTRODUCTION_SECTION_TRIE,
    KEYWORDS_SECTION_TRIE,
)
from .text_checks import (
    matches_abstract,
    vertically_close,
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
    make_body_heading_candidate,
)
from .detectors import (
    detect_numbered_heading,
    detect_labeled_heading,
    is_bibliography_entry,
)


def detect_font_heading(page_scan: PageScanState, other_block: Block) -> Optional[HeadingCandidate]:
    """Detailed font/position-based fallback heading classifier."""
    from ..model import x_aligned, last_span, last_line_of, first_span_of, letter_count, punct_count, dominant_style_of, is_upper_dominant, is_caps_heavy, is_sentence_like, alignment_code
    from ..tokens import last_token, is_comma_token

    above = neighbor_above(page_scan.tertiary_slot, other_block)
    top_gap = above.bottom_edge() - other_block.top_edge() if above is not None else math.inf
    predecessor = neighbor_right(page_scan.tertiary_slot, other_block)
    predecessor_gap = other_block.bottom_edge() - predecessor.top_edge() if predecessor is not None else math.inf
    keyword_match = body_neighbor_above(page_scan.tertiary_slot, other_block)
    above_or_overlap = closest_body_neighbor_above(page_scan.tertiary_slot, other_block)

    # Initial gate: one of On OR bold/centered tall block.
    if not (
        vertically_close(keyword_match, other_block) or vertically_close(above_or_overlap, other_block)
        or (other_block.bottom_edge() >= 0.8 * page_scan.primary_slot.bounds.bbox_height()
            and other_block.avg_font_size() >= page_scan.secondary_slot.secondary_slot.primary_slot + 1
            and other_block.bold_frac() > 0.9
            and (above is None or above.type == 1))
    ):
        return None

    page = page_scan.primary_slot.primary_slot                                  # page statistics
    far = 10 * min(other_block.avg_font_size(), page.tertiary_slot)
    if top_gap < math.inf and top_gap > far and above.state_slot == 0:
        return None
    if predecessor is not None and predecessor.state_slot != 0:
        return None

    # Compound rejection for candidates sitting above non-body predecessors.
    # Keep the explicit short-circuit structure: each inner predicate requires
    # the predecessor to exist.

    inner_reject = False
    if above is not None and above.state_slot != 0:
        inner_reject = (
            predecessor_gap > 5 * page.tertiary_slot
            or (predecessor is not None and not predecessor.is_body_paragraph)
            or (predecessor is not None and predecessor.bbox_width() < page_scan.primary_slot.bounds.bbox_width() / 5)
            or (predecessor is not None and predecessor.char_count() < 0.5 * other_block.char_count())
            or (predecessor is not None and predecessor.weighted_ratio_secondary < 0.33)
            or (predecessor is not None and predecessor.char_count() < 500
                and predecessor.weighted_ratio_secondary < 0.5 and alignment_code(predecessor) != 1)
            or (predecessor is not None and predecessor.char_count() < 250 and predecessor.weighted_ratio_secondary < 0.5)
        )
    if (inner_reject
            or (predecessor is not None and (
                predecessor.weighted_ratio_primary < 0.67 * page_scan.secondary_slot.secondary_slot.auxiliary_slot
                or (other_block.char_count() < 30 and predecessor.char_count() < 300
                    and predecessor.weighted_ratio_primary < 0.8 * page_scan.secondary_slot.secondary_slot.auxiliary_slot)))):
        return None
    last_tok = last_token(tokenize_block(other_block))
    if last_tok is not None and is_comma_token(last_tok):
        return None

    # Branch 1: tall first-line + big-font heading
    if (predecessor_gap < math.inf and predecessor_gap > 0
            and other_block.style_slot >= page_scan.secondary_slot.secondary_slot.primary_slot + 2
            and other_block.avg_font_size() >= page_scan.secondary_slot.secondary_slot.primary_slot + 1.5
            and other_block.avg_font_size() >= page.primary_slot + 0.5
            and (above is None or (other_block.style_slot >= above.style_slot and other_block.avg_font_size() >= above.avg_font_size()))
            and predecessor is not None
            and other_block.style_slot >= predecessor.style_slot and other_block.avg_font_size() >= predecessor.avg_font_size()):
        return make_plain_candidate(page_scan, 0, other_block)

    caps_heavy = is_caps_heavy(other_block)
    # Branch 2 reject: matches body-style and not all-caps, OR clearly
    # smaller font than predecessor near it.
    if ((dominant_style_of(other_block) in page_scan.primary_slot.style_slot and not caps_heavy
         and (page.auxiliary_slot == dominant_style_of(other_block)
              or (other_block.bold_frac() < 0.9 and other_block.previous_slot < 0.9
                  and alignment_code(other_block) != 3 and not is_sentence_like(other_block))))
            or (above is not None and predecessor is not None
                and other_block.avg_font_size() <= predecessor.avg_font_size()
                and top_gap < predecessor_gap / 4)):
        return None

    # Branch 3: medium-confidence font-size heading
    if (predecessor_gap < math.inf and predecessor_gap > 0
            and other_block.avg_font_size() >= page_scan.secondary_slot.secondary_slot.primary_slot + 0.5
            and predecessor is not None
            and other_block.style_slot >= predecessor.style_slot and other_block.avg_font_size() >= predecessor.avg_font_size()
            and predecessor.avg_font_size() >= page.primary_slot - 0.5
            and other_block.bbox_width() < 0.95 * predecessor.bbox_width()
            and predecessor.bbox_width() >= 0.25 * page_scan.primary_slot.bounds.bbox_width()):
        return make_plain_candidate(page_scan, 0, other_block)

    line_height = page.tertiary_slot - page.primary_slot
    # Branch 4: moderate-gap large-font heading
    if (predecessor_gap > line_height and predecessor_gap < 5 * line_height
            and (above is None or other_block.avg_font_size() >= above.avg_font_size() + 0.5)
            and predecessor is not None
            and other_block.avg_font_size() >= predecessor.avg_font_size() + 0.5
            and other_block.avg_font_size() >= page_scan.secondary_slot.secondary_slot.primary_slot - 0.5
            and other_block.bbox_width() < 0.95 * predecessor.bbox_width()
            and predecessor.is_body_paragraph
            and predecessor.avg_font_size() >= page.primary_slot - 0.5
            and predecessor.char_stats.secondary_slot != 1):
        return make_plain_candidate(page_scan, 0, other_block)

    # Reject: many letters with low density signals body para
    letters = other_block.char_stats.primary_slot[6]
    if other_block.char_stats.primary_slot[10] != 0:
        ratio = (letters + other_block.char_stats.primary_slot[8]) / other_block.char_stats.primary_slot[10]
    else:
        # IEEE division edge case: positive numerator over zero behaves as +inf,
        # which keeps the low-density rejection active.
        ratio = math.inf if (letters + other_block.char_stats.primary_slot[8]) > 0 else math.nan
    if letters > 1 and ratio > 0.3:
        return None

    symbol_count = punct_count(other_block.char_stats)
    letter_total = letter_count(other_block.char_stats)
    # Same IEEE division edge case as the letter-density ratio above.
    symbol_ratio = symbol_count / letter_total if letter_total != 0 else (math.inf if symbol_count > 0 else math.nan)
    if (symbol_count >= 5 and symbol_ratio > 0.2
            or top_gap < 0.2 * other_block.avg_font_size()
            or top_gap < min(other_block.avg_font_size(), 0.7 * predecessor_gap)):
        return None

    centered = other_block.char_stats.secondary_slot == 2
    neg = -0.2 * last_span(last_line_of(other_block)).bbox_height() if caps_heavy else 0

    # Branch A: tight criteria with neighbor analysis
    neighbor_heading_cue = (
        predecessor_gap < math.inf and predecessor_gap > neg
        and other_block.avg_font_size() >= page.primary_slot - 0.1
        and predecessor is not None and other_block.avg_font_size() >= predecessor.avg_font_size() - 0.1
        and other_block.avg_font_size() >= page_scan.secondary_slot.secondary_slot.primary_slot - 0.5
        and ((predecessor.is_body_paragraph and other_block.bbox_width() < 0.95 * predecessor.bbox_width()
              and x_aligned(other_block, predecessor, max(1, other_block.bbox_width() / 10))
              and predecessor_gap < 6 * other_block.bbox_height())
             or (top_gap < math.inf and above is not None and above.is_body_paragraph
                 and other_block.bbox_width() < 0.95 * above.bbox_width()
                 and x_aligned(other_block, above, max(1, other_block.bbox_width() / 10))
                 and top_gap < 6 * other_block.bbox_height()))
        and (centered or caps_heavy)
        and ((other_block.bold_frac() > predecessor.bold_frac() and other_block.bold_frac() > 0.5
              and (not first_span_of(predecessor).primary_slot
                   or (above is not None and other_block.bold_frac() > above.bold_frac())))
             or caps_heavy)
    )
    # Nearby body text with the dominant style changed is a strong heading cue.
    difference_style = bool(
        predecessor is not None and predecessor.is_body_paragraph
        and predecessor.avg_font_size() > page.primary_slot - 0.5
        and predecessor_gap > 0 and predecessor_gap < 3 * other_block.bbox_height()
        and dominant_style_of(other_block) != dominant_style_of(predecessor)
    )
    style_change_cue = (
        difference_style
        and above is not None and above.is_body_paragraph
        and top_gap > 0 and top_gap < 3 * other_block.bbox_height()
        and centered and dominant_style_of(above) == dominant_style_of(predecessor) if predecessor is not None else False
    )
    if neighbor_heading_cue or style_change_cue:
        return make_plain_candidate(page_scan, 0, other_block)

    # Top-like context: there is no above block, or the above block is already a
    # title/heading marker.
    topnum = above is None or above.type == 1
    branch_C1 = (
        topnum and centered and difference_style
        and predecessor_gap < other_block.bbox_height()
        and predecessor is not None and dominant_style_of(predecessor) == page.auxiliary_slot
    )
    branch_C2 = (
        topnum and centered
        and predecessor is not None and above_or_overlap is not None
        and predecessor is not above_or_overlap
        and predecessor.bottom_edge() - above_or_overlap.top_edge() < predecessor.avg_font_size()
        and dominant_style_of(above_or_overlap) == page.auxiliary_slot and dominant_style_of(other_block) != page.auxiliary_slot
        and (other_block.avg_font_size() >= predecessor.avg_font_size() + 0.5
             or (caps_heavy and not is_upper_dominant(predecessor.char_stats)))
    )
    branch_C3 = (
        above is not None
        and (above.used_as_heading or above in page_scan.measure_slot)
        and (above.avg_font_size() >= other_block.avg_font_size() + 0.5
             or (is_upper_dominant(above.char_stats) and not caps_heavy))
        and centered and difference_style
        and predecessor is not None and dominant_style_of(predecessor) == page.auxiliary_slot
    )
    if branch_C1 or branch_C2 or branch_C3:
        return make_plain_candidate(page_scan, 0, other_block)

    return None


def detect_heading_with_body(page_scan: PageScanState, other_block: Block) -> Optional[HeadingCandidate]:
    """. Detect heading-with-body 2-line patterns."""
    if other_block.line_count() < 2:
        return None
    tokens = tokenize_block(other_block)
    first_line = other_block.line()
    second_line = other_block.primary_slot[1]
    split = 0
    letter_count = 0
    font = first_line.primary_slot[0].font_name if first_line.primary_slot else ""
    if first_line.bold_frac() > 0 and first_line.bold_frac() < 1:
        for entry in enumerate_tokens(tokens):
            index = entry["index"]
            anchor_token = entry["token"]
            if anchor_token.line() is not first_line or not first_anchor_span(anchor_token).primary_slot:
                break
            if anchor_token.type == 2 and len(anchor_token.str) > 1:
                letter_count += 1
            split = index + 1
    elif last_span(last_line_of(other_block)).font_name != font:
        other_count = 0
        for candidate_line in other_block:
            if candidate_line is not first_line and candidate_line.primary_slot[0].font_name == font:
                other_count += 1
        if other_count > other_block.line_count() / 4:
            return None
        for entry in enumerate_tokens(tokens):
            index = entry["index"]
            anchor_token = entry["token"]
            line = anchor_token.line()
            if first_anchor_span(anchor_token).font_name != font or (line is not first_line and line is not second_line):
                break
            if anchor_token.type == 2 and (len(anchor_token.str) > 1 or anchor_token.primary_slot == 4):
                letter_count += 1
            split = index + 1
    if split <= 0 or split >= tokens.length:
        return None
    # Allow up to two punctuation-like tokens to stay with the prefix when they
    # remain on the same line and bracket attachment permits it.
    token = tokens.token_at(split - 1)
    next_token = tokens.token_at(split)
    for _ in range(2):
        if token is None or next_token is None:
            return None
        last_anchor = last_token_anchor(token)
        if not (is_word_token(next_token)
                and getattr(last_anchor, "line", None) is next_token.line()
                and (not token.boundary_slot or next_token.boundary_slot)):
            break
        split += 1
        token = next_token
        next_token = tokens.token_at(split)
        if token is None or next_token is None:
            return None
    if letter_count <= 0:
        return None
    prefix = tokens.slice(0, split)

    # First-token style check for prefix/body split confidence.
    first = prefix.token_at(0)
    first_anchor = first_anchor_span(first) if first is not None else None
    if first_anchor is not None:
        if not first_anchor.primary_slot and not first_anchor.measure_slot and other_block.previous_slot > 0.5:
            return None
        if (not first_anchor.primary_slot and first_anchor.font_size < other_block.avg_font_size() + 1):
            rest = tokens.slice(split)
            if rest.length <= 0 or (rest.token_at(0) is not None and rest.token_at(0).primary_slot == 3):
                return None

    # Reject prefixes that are only section keywords and contain no extra text.
    hn_match = trie_prefix_match(KEYWORDS_SECTION_TRIE, prefix)
    if hn_match is not None and len(hn_match) >= letter_count:
        return None

    # Reuse numbered-heading detection on the prefix.
    heading_kind = detect_numbered_heading(page_scan, other_block, prefix)
    if heading_kind is not None and len(heading_kind.numbering) > 1:
        return make_heading_candidate(page_scan, heading_kind.type, heading_kind.group_slot, heading_kind.numbering, heading_kind.secondary_slot, heading_kind.primary_slot, True)
    if heading_kind is not None and (trie_matches_all(INTRODUCTION_SECTION_TRIE, heading_kind.primary_slot) or (is_uppercase_dominant(heading_kind.primary_slot) and not is_bibliography_entry(other_block))):
        return make_heading_candidate(page_scan, heading_kind.type, heading_kind.group_slot, heading_kind.numbering, heading_kind.secondary_slot, heading_kind.primary_slot, True)

    # Reuse the labeled-heading detector on the prefix with body-heading status.
    if prefix.length > 3:
        heading_signature = detect_labeled_heading(page_scan, other_block, prefix)
        if heading_signature is not None:
            return make_heading_candidate(page_scan, heading_signature.type, heading_signature.group_slot, heading_signature.numbering, heading_signature.secondary_slot, trim_trailing_punct(heading_signature.primary_slot), True)

    if matches_abstract(prefix):
        return make_body_heading_candidate(page_scan, 5, other_block, prefix)
    if trie_matches_all(INTRODUCTION_SECTION_TRIE, prefix):
        return make_body_heading_candidate(page_scan, 11, other_block, prefix)

    # Final font-size and trailing-token reject gates.
    if first_line.avg_font_size() < page_scan.secondary_slot.secondary_slot.primary_slot - 2:
        return None
    if token is not None and is_word_token(token) and not is_trimmable_token(token):
        return None

    # Body paragraphs can still contain an all-caps heading prefix.
    if (not is_upper_dominant(other_block.char_stats) and other_block.is_body_paragraph and info_weight(other_block.char_stats) >= 100):
        all_caps_vf = CharStats(prefix.to_string())
        if is_upper_dominant(all_caps_vf) and all_caps_vf.primary_slot[2] <= other_block.char_stats.primary_slot[3]:
            if heading_kind is not None:
                return make_heading_candidate(page_scan, heading_kind.type, heading_kind.group_slot, heading_kind.numbering, heading_kind.secondary_slot, heading_kind.primary_slot, True)
            if trie_matches_all(SECTION_KEYWORDS_TRIE, prefix):
                return make_body_heading_candidate(page_scan, 6, other_block, prefix)
            return make_body_heading_candidate(page_scan, 0, other_block, prefix)

    # Centered two-line heading branch.
    above_neighbor = neighbor_above(page_scan.tertiary_slot, other_block)
    gap = (above_neighbor.bottom_edge() - first_line.top_edge()) if above_neighbor is not None else math.inf
    intersection = first_line.bottom_edge() - second_line.top_edge()
    per_char = avg_char_width(first_line)
    centered_flag = False
    # When there is no above block, the infinite gap is sufficient for this
    # branch and later above-block checks must remain guarded.
    if first_anchor is not None:
        cond_outer = (
            first_anchor.measure_slot
            and not first_anchor_span(next_token).measure_slot if next_token is not None else False
        )
        # First-line anchor, second-line anchor, gap, neighbor, and punctuation
        # checks together identify a centered heading prefix.
        if (first_anchor.measure_slot
                and next_token is not None and not first_anchor_span(next_token).measure_slot
                and (gap > 1.1 * intersection
                     or (last_token(tokenize_block(above_neighbor)) is not None and is_word_token(last_token(tokenize_block(above_neighbor))))
                     or last_line_of(above_neighbor).right_edge() < first_line.right_edge() - 8 * per_char)
                and (first_line.right_edge() > second_line.right_edge() - 4 * per_char
                     or first_line.char_stats.tertiary_slot != 6
                     or second_line.char_stats.secondary_slot == 3)):
            for prefix_token in prefix:
                if prefix_token.primary_slot == 2:
                    centered_flag = True
                    break
                if prefix_token.type == 2 or prefix_token.boundary_slot:
                    break
    if (centered_flag
            and token is not None and is_trimmable_token(token)
            and next_token is not None and next_token.primary_slot == 2):
        if trie_matches_all(SECTION_KEYWORDS_TRIE, prefix):
            return make_body_heading_candidate(page_scan, 6, other_block, prefix)
        if letter_count > 1:
            return make_body_heading_candidate(page_scan, 0, other_block, prefix)
    return None
