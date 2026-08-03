"""Whole-page heading scan and document-level candidate collection/filtering."""

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
)
from .text_checks import (
    is_heading_continuation,
    matches_abstract,
    matches_references,
    has_substantive_content,
    is_cover_page,
)
from .neighbors import (
    neighbor_above,
    neighbor_right,
    closest_body_neighbor_above,
)
from .candidates import (
    PageScanState,
    push_candidate,
    make_plain_candidate,
    make_body_heading_candidate,
)
from .detectors import (
    detect_numbered_heading,
    detect_labeled_heading,
    detect_chapter_appendix,
    try_classify_heading,
    passes_neighbor_check,
    has_competing_labeled_heading,
)
from .style_detectors import (
    detect_font_heading,
    detect_heading_with_body,
)


# --------------------------------------------------------------------------- #
# Main per-page heading scan #
# --------------------------------------------------------------------------- #


def scan_page_headings(page_scan: PageScanState) -> list[HeadingCandidate]:
    """Return heading candidates found on this page."""
    if is_cover_page(page_scan.secondary_slot, page_scan.primary_slot):
        return []
    page_scan.option_slot.clear()
    page_scan.measure_slot.clear()
    blocks = page_scan.auxiliary_slot
    for block in blocks:
        if block.char_count() <= 0 or block.skew_frac() > 1 or block.type != 0:
            continue
        if block.state_slot != 0:                              # already classified
            continue
        above = neighbor_above(page_scan.tertiary_slot, block)
        if above is not None and above.secondary_slot.contains(block.secondary_slot):
            continue
        if block.char_count() <= 1 and block.char_stats.secondary_slot != 4:
            continue
        wn_entry = page_scan.tertiary_slot.primary_slot[block.orig_index] if 0 <= block.orig_index < len(page_scan.tertiary_slot.primary_slot) else None
        has_da_above = wn_entry is not None and wn_entry.state_slot
        rows = block.line_count()
        # Try lo for 2-line heading-body patterns
        if has_da_above and rows > 1 and (
            (0 < block.bold_frac() < 1)
            or style_key(first_span_of(block)) != style_key(last_span(last_line_of(block)))
        ):
            lo_result = detect_heading_with_body(page_scan, block)
            if lo_result is not None:
                push_candidate(page_scan, lo_result)
                continue
        tokens = tokenize_block(block)
        first_line = block.line()
        first_line_tokens = tokens.slice(0, advance_past_line(tokens, first_line, 0))
        if has_da_above and not block.measure_slot and rows >= 3\
                and first_line.bbox_width() <= 0.2 * min(block.primary_slot[1].bbox_width(), block.primary_slot[2].bbox_width())\
                and matches_abstract(first_line_tokens):
            push_candidate(page_scan, make_body_heading_candidate(page_scan, 5, block, first_line_tokens))
            continue
        if block.char_count() >= 200:
            continue
        if block.char_count() >= 100 and rows > 1 and block.char_stats.primary_slot[6] - first_line.char_stats.primary_slot[6] > 1:
            continue
        size = block.avg_font_size()
        page_width = page_scan.primary_slot.bounds.bbox_width()
        lots_caps = block.char_stats.primary_slot[2] >= max(3, letter_count(block.char_stats) / 2)
        if rows > 4 or (rows >= 3 and not (size >= 1.5 * page_scan.primary_slot.primary_slot.primary_slot or lots_caps)):
            continue
        if block.weighted_ratio_primary < 0.5 * page_scan.secondary_slot.secondary_slot.auxiliary_slot:
            continue
        if letter_count(block.char_stats) <= 0:
            continue
        if block.bold_frac() < 0.1 and not lots_caps and size < page_scan.secondary_slot.secondary_slot.primary_slot - 2:
            continue
        layout_gate = detect_chapter_appendix(page_scan, block)
        if layout_gate is not None:
            push_candidate(page_scan, layout_gate)
            continue
        if passes_neighbor_check(page_scan, block):
            continue
        top_gap = above.bottom_edge() - block.top_edge() if above is not None else math.inf
        isolated = (
            not block.measure_slot and rows <= 2
            and (above is None or top_gap > 1.5 * block.avg_font_size()
                 or (above.state_slot != 5 and above.state_slot != 11))
        )
        if isolated:
            if matches_references(tokens):
                push_candidate(page_scan, make_plain_candidate(page_scan, 7, block))
                continue
            if has_da_above and trie_matches_all(INTRODUCTION_SECTION_TRIE, tokens):
                push_candidate(page_scan, make_plain_candidate(page_scan, 11, block))
                continue
        above_index = block.orig_index - 1
        below_index = block.orig_index + 1
        previous_block = blocks[above_index] if 0 <= above_index < len(blocks) else None
        below = blocks[below_index] if 0 <= below_index < len(blocks) else None
        if not has_da_above and (
            not (heading_score(block) >= page_scan.primary_slot.primary_slot.primary_slot + 1.5)
            or (above is not None and above.type != 1)
            or (previous_block is not None and previous_block.type != 1)
            or (below is not None and not (below.top_edge() < block.bottom_edge() - size))
        ):
            continue
        if block.bold_frac() < 0.1 and not lots_caps\
                and first_span_of(block).font_name == page_scan.primary_slot.primary_slot.state_slot\
                and size < page_scan.secondary_slot.secondary_slot.primary_slot - 0.5:
            continue
        predecessor = neighbor_right(page_scan.tertiary_slot, block)
        if predecessor is not None and predecessor.skew_frac() > 1:
            continue
        if top_gap < 0:
            continue
        predecessor_gap = block.bottom_edge() - predecessor.top_edge() if predecessor is not None else math.inf
        if predecessor_gap < -0.9 * block.bbox_height():
            continue
        line_gap = page_scan.primary_slot.primary_slot.tertiary_slot - page_scan.primary_slot.primary_slot.primary_slot
        if top_gap < line_gap and size < page_scan.primary_slot.primary_slot.primary_slot - 1:
            continue
        # Sibling/peer block pointers from the neighbor cache.
        right_neighbor_sib = wn_entry.measure_slot if wn_entry is not None else None
        left_neighbor_sib = wn_entry.option_slot if wn_entry is not None else None

        heading_kind = detect_numbered_heading(page_scan, block, tokens)
        if heading_kind is not None:
            from ..stats import column_index_of as _column_index
            col_idx = _column_index(block) if block.primary_slot else 0
            column_rect = page_scan.primary_slot.tertiary_slot[col_idx] if 0 <= col_idx < len(page_scan.primary_slot.tertiary_slot) else None
            # Narrow-column heading-vs-prev-numbering check
            if (size < page_scan.secondary_slot.secondary_slot.primary_slot - 1
                    and block.bbox_width() < 0.2 * page_width
                    and column_rect is not None
                    and column_rect.bbox_width() < 0.2 * page_width
                    and column_rect.bbox_height() > 1.5 * column_rect.bbox_width()):
                first_number = heading_kind.numbering[0] if heading_kind.numbering else 0
                if above is not None:
                    first = first_token(tokenize_block(above))
                    if first is not None and first.type == 1 and token_numeric_value(first) != first_number:
                        continue
                if predecessor is not None:
                    predecessor_first_token = first_token(tokenize_block(predecessor))
                    if predecessor_first_token is not None and predecessor_first_token.type == 1 and token_numeric_value(predecessor_first_token) != first_number:
                        continue
            # Prev-block continuation check via Kn (numbered-sequence test)
            first_token_value = tokens.token_at(0)
            second_token_value = tokens.token_at(1) if len(tokens) > 1 else None
            if len(heading_kind.numbering) <= 1 and (
                    (first_token_value is not None and first_token_value.boundary_slot)
                    or (second_token_value is not None and is_word_token(second_token_value))):
                first_number = heading_kind.numbering[0] if heading_kind.numbering else 0
                if above is not None and is_heading_continuation(above, block, first_number):
                    continue
                if (right_neighbor_sib is not None and above is not right_neighbor_sib
                        and not center_aligned(block, right_neighbor_sib, 1)
                        and (above.line_count() < 10 or above.char_count() < 300)
                        and is_heading_continuation(right_neighbor_sib, block, first_number)):
                    continue
                if predecessor is not None and is_heading_continuation(predecessor, block, first_number):
                    continue
                if (left_neighbor_sib is not None and predecessor is not left_neighbor_sib
                        and not center_aligned(block, left_neighbor_sib, 1)
                        and (predecessor.line_count() < 10 or predecessor.char_count() < 300)
                        and is_heading_continuation(left_neighbor_sib, block, first_number)):
                    continue
            # Top-of-page small-font footnote-marker rejection
            second = tokens.token_at(1) if len(tokens) > 1 else None
            if (block.top_edge() < page_scan.primary_slot.bounds.bbox_height() / 4
                    and size <= page_scan.primary_slot.primary_slot.primary_slot
                    and first_span_of(block).char_stats.secondary_slot == 1
                    and first_span_of(block).bbox_height() < size - 0.5
                    and len(heading_kind.numbering) <= 1
                    and second is not None and is_char_token(second)):
                continue
            push_candidate(page_scan, heading_kind)
            continue
        if block.measure_slot:
            continue
        if block.char_count() >= 120:
            continue
        if top_gap <= line_gap - 0.1:
            continue
        heading_signature = detect_labeled_heading(page_scan, block, tokens)
        if heading_signature is not None:
            if above is not None and has_competing_labeled_heading(page_scan, heading_signature, above):
                continue
            if (right_neighbor_sib is not None and above is not right_neighbor_sib
                    and has_competing_labeled_heading(page_scan, heading_signature, right_neighbor_sib)):
                continue
            if predecessor is not None and has_competing_labeled_heading(page_scan, heading_signature, predecessor):
                continue
            if (left_neighbor_sib is not None and predecessor is not left_neighbor_sib
                    and has_competing_labeled_heading(page_scan, heading_signature, left_neighbor_sib)):
                continue
            push_candidate(page_scan, heading_signature)
            continue
        if isolated:
            caps_heavy = size + 2 * block.bold_frac() >= page_scan.secondary_slot.secondary_slot.primary_slot + 4 or is_upper_dominant(block.char_stats)
            above_or_overlap = closest_body_neighbor_above(page_scan.tertiary_slot, block)
            page_width = page_scan.primary_slot.bounds.bbox_width()
            # Abstract-heading acceptance uses the closest above-overlap block
            # as the guard; when it exists, the predecessor exists too.
            type5_cond = caps_heavy or (
                above_or_overlap is not None
                and (block.bottom_edge() - above_or_overlap.top_edge() < 3 * (block.bottom_edge() - predecessor.top_edge())
                     or info_weight(predecessor.char_stats) >= 30)
            )
            if type5_cond and matches_abstract(tokens):
                push_candidate(page_scan, make_plain_candidate(page_scan, 5, block))
                continue
            type6_cond = (
                caps_heavy
                or (above is not None and x_aligned(block, above, 1)
                    and (above.bbox_width() >= page_width / 6
                         or above.bold_frac() > 0.9
                         or is_upper_dominant(above.char_stats)))
                or (predecessor is not None and x_aligned(block, predecessor, 1)
                    and (predecessor.bbox_width() >= page_width / 6
                         or predecessor.bold_frac() > 0.9
                         or is_upper_dominant(predecessor.char_stats)
                         or (block.previous_slot < 0.1 and predecessor.previous_slot > 0.9)))
            )
            if type6_cond and trie_full_match(SECTION_KEYWORDS_TRIE, tokens):
                push_candidate(page_scan, make_plain_candidate(page_scan, 6, block))
                continue
        if info_weight(block.char_stats) <= 3:
            continue
        if has_substantive_content(block, previous_block, below):
            continue
        outline_context = detect_font_heading(page_scan, block)
        if outline_context is not None:
            push_candidate(page_scan, outline_context)
    return page_scan.option_slot


# --------------------------------------------------------------------------- #
# Document-wide outline-candidate collector #
# --------------------------------------------------------------------------- #


class DocCandidateCollector:
    """Document-level state aggregating per-page heading candidates."""

    __slots__ = ("previous_slot", "measure_slot", "option_slot", "auxiliary_slot", "primary_slot", "tertiary_slot", "secondary_slot", "state_slot")

    def __init__(self, doc, labeled):
        self.previous_slot = doc
        self.measure_slot = labeled
        self.option_slot = 0
        self.auxiliary_slot = False
        self.primary_slot = 0
        self.secondary_slot = False
        self.tertiary_slot = False
        self.state_slot: list[HeadingCandidate] = []


def filter_page_candidates(doc_collector: DocCandidateCollector, page, page_candidates: list[HeadingCandidate]) -> None:
    """Per-page candidate filter for noisy pages, title overlap, page headers, and numbering continuity."""
    from ..outline_assembly import is_script_compatible
    from ..model import intervals_overlap, is_caps_heavy
    from ..stats import column_index_of

    # Advance document-level numbering state through outline entries up to this page.
    while doc_collector.option_slot < len(doc_collector.measure_slot):
        outline_entry = doc_collector.measure_slot[doc_collector.option_slot]
        current_candidate = outline_entry.heading
        if current_candidate.page.page_index > page.page_index:
            break
        if current_candidate.type == 2:
            if not doc_collector.tertiary_slot:
                doc_collector.tertiary_slot = (len(current_candidate.numbering) > 0 and current_candidate.numbering[0] == 1)
        elif current_candidate.type == 4:
            if not doc_collector.secondary_slot:
                doc_collector.secondary_slot = (len(current_candidate.numbering) > 0 and current_candidate.numbering[0] == 1)
        elif current_candidate.type == 1 and len(current_candidate.numbering) > 0:
            doc_collector.primary_slot = max(doc_collector.primary_slot, current_candidate.numbering[0])
        doc_collector.option_slot += 1

    count = len(page_candidates)
    if count >= 20:
        return

    # Sort candidates by column, vertical position, then horizontal position.
    def _ih_key(heading_candidate: HeadingCandidate):
        first_line = heading_candidate.group_slot.primary_slot[0] if heading_candidate.group_slot.primary_slot else None
        column_index = first_line.measure_slot if first_line is not None else -1
        return (column_index, -heading_candidate.group_slot.top_edge(), -heading_candidate.group_slot.bottom_edge(), heading_candidate.group_slot.left_edge(), heading_candidate.group_slot.right_edge())
    page_candidates.sort(key=_ih_key)

    accepted: list[HeadingCandidate] = []
    title: Optional[Block] = None
    if not doc_collector.auxiliary_slot and getattr(page, "auxiliary_slot", False):
        for block in page.output_slot:
            if block.type == 3:
                title = block
                break

    min_first_number = math.inf
    total_bottom = math.inf
    single_numbering_count = 0
    for page_candidate in page_candidates:
        if total_bottom == math.inf and page_candidate.type == 5:
            total_bottom = page_candidate.group_slot.top_edge() + page_candidate.group_slot.avg_font_size()
        if page_candidate.type == 1 and len(page_candidate.numbering) > 0:
            min_first_number = min(min_first_number, page_candidate.numbering[0])
            if len(page_candidate.numbering) == 1:
                single_numbering_count += 1

    max_first_number = 0
    for index in range(count):
        active_candidate = page_candidates[index]
        next_item = page_candidates[index + 1] if index + 1 < count else None

        if is_script_compatible(doc_collector.previous_slot.secondary_slot.tertiary_slot, active_candidate):
            continue

        if not doc_collector.auxiliary_slot and active_candidate.type != 11:
            bottom = active_candidate.group_slot.top_edge()
            if (title is not None and bottom > title.top_edge()
                    and intervals_overlap(active_candidate.group_slot.left_edge(), active_candidate.group_slot.right_edge(), title.left_edge(), title.right_edge())):
                continue
            if bottom > total_bottom:
                continue

        if (active_candidate.type == 0 and next_item is not None and next_item.type == 5
                and active_candidate.group_slot.left_edge() <= next_item.group_slot.right_edge() and active_candidate.group_slot.right_edge() >= next_item.group_slot.left_edge()
                and active_candidate.group_slot.bottom_edge() - next_item.group_slot.top_edge() < 2 * active_candidate.group_slot.bbox_height()
                and len(tokenize_block(active_candidate.group_slot)) > 1):
            continue

        if active_candidate.type == 2:
            if len(active_candidate.numbering) > 0 and active_candidate.numbering[0] == 1:
                doc_collector.tertiary_slot = True
            elif not doc_collector.tertiary_slot:
                continue
            accepted.append(active_candidate)
            continue

        if active_candidate.type == 4:
            if len(active_candidate.numbering) > 0 and active_candidate.numbering[0] == 1:
                doc_collector.secondary_slot = True
            elif not doc_collector.secondary_slot:
                continue
            accepted.append(active_candidate)
            continue

        if active_candidate.type != 1:
            accepted.append(active_candidate)
            continue

        # type == 1
        if single_numbering_count >= 5:
            continue
        first_number = active_candidate.numbering[0] if len(active_candidate.numbering) > 0 else 0
        if (active_candidate.group_slot.bold_frac() < 0.9 and not is_caps_heavy(active_candidate.group_slot)
                and active_candidate.group_slot.avg_font_size() < page.primary_slot.primary_slot + 1):
            if doc_collector.primary_slot <= 0 and min_first_number > 1 and len(active_candidate.numbering) <= 1:
                continue
            if first_number > 3 * page.page_index:
                continue
        max_first_number = max(max_first_number, first_number)
        accepted.append(active_candidate)

    doc_collector.state_slot.extend(accepted)
    doc_collector.auxiliary_slot = True
    doc_collector.primary_slot = max(doc_collector.primary_slot, max_first_number)


# --------------------------------------------------------------------------- #
# Public entry point: build heading candidates for the whole document #
# --------------------------------------------------------------------------- #


def build_doc_heading_candidates(doc, labeled: Optional[list] = None) -> list[HeadingCandidate]:
    """Run per-page heading detection across the document."""
    doc_collector = DocCandidateCollector(doc, labeled if labeled is not None else [])
    saw_body = False
    for page in doc.primary_slot:
        # Skip initial cover-like pages until the first body-like page is reached.
        from ..title import is_cover_like_page
        if not saw_body and is_cover_like_page(doc, page):
            continue
        saw_body = True
        page_vo = PageScanState(doc, page)
        page_candidates = scan_page_headings(page_vo)
        filter_page_candidates(doc_collector, page, page_candidates)
    return doc_collector.state_slot


def find_section_openers(doc, start_page_idx: int) -> list:
    """Find the first valid heading on each page, then clique-filter the result."""
    from ..outline_assembly import is_script_compatible, has_conflict_in_context, OutlineContext, OutlineNode

    item_list: list[HeadingCandidate] = []
    index = start_page_idx
    while index < len(doc.primary_slot):
        page = doc.primary_slot[index]
        current_candidate: Optional[HeadingCandidate] = None
        page_scan_state = PageScanState(doc, page)
        if not page_scan_state.primary_slot.auxiliary_slot:
            for block in page_scan_state.auxiliary_slot:
                if block.char_count() <= 0 or block.skew_frac() > 1 or block.type != 0:
                    continue
                if block.is_body_paragraph or block.top_edge() < 0.5 * page_scan_state.primary_slot.bounds.bbox_height():
                    break
                if block.marker_slot != 0:
                    break
                detected_candidate = try_classify_heading(page_scan_state, block)
                if detected_candidate is not None:
                    current_candidate = detected_candidate
                    break
                if block.line_count() > 2:
                    break
        if current_candidate is not None and not is_script_compatible(doc.secondary_slot.tertiary_slot, current_candidate):
            item_list.append(current_candidate)
        index += 1

    if len(item_list) <= 1:
        return []

    # Compare candidates against the full context and against the accepted subset.
    bundle = OutlineContext(item_list)
    accepted_context = OutlineContext([])
    out = []
    for current_candidate in item_list:
        if accepted_context.has_nearby_duplicate(current_candidate):
            current_candidate.group_slot.type = 12
            continue
        if has_conflict_in_context(bundle, current_candidate):
            continue
        accepted_context.add(current_candidate)
        out.append(OutlineNode(current_candidate))
        current_candidate.group_slot.type = 7
        current_candidate.group_slot.used_as_heading = True
    return out
