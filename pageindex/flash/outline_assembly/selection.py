"""Heading rejection rules, hierarchy stack, and sub/top-level heading extraction."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional
from ..model import (
    style_key, left_aligned, right_aligned, center_aligned, x_aligned, rect_union,
    Rect, last_span, avg_char_width, raw_text_of_line, heading_score, numbering_text, numbering_value, numbering_kind,
    reading_order_key, left_edge_key, _trim_unicode_ws, _round_half_up_to_int, Line, last_line_of, first_span_of, block_text, deaccented_text, letter_count, dominant_style_of, info_weight, dominant_font_size, is_upper_dominant, is_caps_heavy, alignment_code, Block,
)
from ..tokens import (
    Token, TokenView, wrap_tokens, enumerate_tokens, last_token, trie_prefix_match, first_token, set_case_fold, TrieConfig, build_trie, tokenize_block, avg_char_width as avg_char_width_fn, trie_full_match, first_anchor_span, is_char_token, is_word_token,
)

from .candidates import (
    HeadingCandidate,
    OutlineNode,
    heading_signature,
    parent_signature,
    is_in_oo_range,
    has_style_neighbor,
)
from .style_context import (
    StyleCluster,
    count_sibling_numberings,
    OutlineState,
    compare_heading_depth,
)


# --------------------------------------------------------------------------- #
# cp / bp -- state mutators (,) #
# --------------------------------------------------------------------------- #


def min_font_distance(state: OutlineState, other_heading_candidate: HeadingCandidate) -> float:
    """minimum font-distance between b and any other heading in the same fontStyle bucket within b's line."""
    min_value = math.inf
    line = other_heading_candidate.group_slot.line()
    for token_list in (other_heading_candidate.secondary_slot, other_heading_candidate.primary_slot):
        if token_list is None:
            continue
        for token in token_list:
            if token.type != 2:
                continue
            for anchor in token.anchor_ranges:
                if anchor.line is not line:
                    return min_value
                span = anchor.anchor_span
                tree = state.state_slot.get(span.font_style())
                if tree is None:
                    continue
                for entry in tree:
                    if entry["heading"] is other_heading_candidate:
                        continue
                    diff = abs(span.font_size - entry["size"])
                    if diff < min_value:
                        min_value = diff
                    if diff <= 0:
                        return 0
    return min_value


def should_reject_heading(state: OutlineState, other_heading_candidate: HeadingCandidate) -> bool:
    """should we REJECT heading b given current state? True = reject."""
    if other_heading_candidate.type == 0:
        for previous in state.style_slot:
            if previous is None:
                continue
            if compare_heading_depth(previous, other_heading_candidate) == 1:
                continue
            style_cluster = state.marker_slot.get(parent_signature(previous))
            if style_cluster is not None and is_in_oo_range(style_cluster, other_heading_candidate):
                return True
    if state.primary_slot is not None and state.primary_slot.is_prominent and other_heading_candidate.type == 0:
        count = 0
        for candidate_token in tokenize_block(other_heading_candidate.group_slot):
            if is_word_token(candidate_token) or candidate_token.type == 1:
                count += 1
                if count >= 3:
                    break
        if count >= 3:
            return True
    if (
        other_heading_candidate.type == 1 and len(other_heading_candidate.numbering) <= 1
        and (
            (0 if (other_heading_candidate.type != 1 or len(other_heading_candidate.numbering) <= 0) else count_sibling_numberings(state.cache_slot, other_heading_candidate, 0)) <= 1
        )
    ):
        return True
    if other_heading_candidate.type in (1, 5, 9, 10, 7):
        reject = False
    else:
        distance = min_font_distance(state, other_heading_candidate)
        if distance <= 0.9:
            reject = False
        elif distance >= math.inf:
            reject = True
        else:
            reject = not (is_caps_heavy(other_heading_candidate.group_slot) and other_heading_candidate.tertiary_slot is not None and other_heading_candidate.group_slot.bottom_edge() - other_heading_candidate.tertiary_slot.top_edge() < 5 * other_heading_candidate.group_slot.bbox_height())
    if reject:
        return True
    if other_heading_candidate.type == 1:
        first = other_heading_candidate.numbering[0]
        if (first < state.secondary_slot and first < state.tertiary_slot) or (state.secondary_slot > 0 and first > state.secondary_slot + 2):
            return True
        if len(other_heading_candidate.numbering) == 1 and state.auxiliary_slot is not None:
            if first == state.tertiary_slot:
                return True
            existing = state.auxiliary_slot.group_slot
            candidate_style = style_key(first_anchor_span(first_token(other_heading_candidate.primary_slot))) if other_heading_candidate.primary_slot is not None and first_token(other_heading_candidate.primary_slot) is not None else ""
            state_style = style_key(first_anchor_span(first_token(state.auxiliary_slot.primary_slot))) if state.auxiliary_slot.primary_slot is not None and first_token(state.auxiliary_slot.primary_slot) is not None else ""
            if candidate_style != state_style:
                # Bold-fraction comparison uses exact half-up integer rounding;
                # Python f-string rounding is half-even.
                if abs(dominant_font_size(other_heading_candidate.group_slot) - dominant_font_size(existing)) > 0.5 or _round_half_up_to_int(other_heading_candidate.group_slot.bold_frac()) != _round_half_up_to_int(existing.bold_frac()):
                    return True
    if (
        state.primary_slot is not None
        and other_heading_candidate.type == 4 and state.primary_slot.type == 4
        and len(state.primary_slot.numbering) > 0 and len(other_heading_candidate.numbering) > 0
        and (state.primary_slot.numbering[0] > other_heading_candidate.numbering[0] or (len(other_heading_candidate.numbering) == 1 and state.primary_slot.numbering[0] == other_heading_candidate.numbering[0]))
    ):
        return True
    if (state.primary_slot is not None and state.primary_slot.type == 8 and len(other_heading_candidate.numbering) <= 0):
        from ..model import _strip_diacritics
        candidate_tokens = other_heading_candidate.primary_slot or []
        tokens = state.primary_slot.primary_slot or []
        if len(candidate_tokens) == len(tokens):
            same = True
            for heading in range(len(candidate_tokens)):
                token = candidate_tokens[heading] if heading < len(candidate_tokens) else None
                state_token = tokens[heading] if heading < len(tokens) else None
                if token is None or state_token is None:
                    same = False
                    break
                if _strip_diacritics(token.str.lower()) != _strip_diacritics(state_token.str.lower()):
                    same = False
                    break
            if same:
                return True
    return False


def push_heading_to_state(state: OutlineState, other_heading_candidate: HeadingCandidate) -> None:
    """Push a heading into the outline state and update level trackers."""
    if len(other_heading_candidate.numbering) > 0:
        # Ensure S is long enough
        while len(state.style_slot) < len(other_heading_candidate.numbering):
            state.style_slot.append(None)
        state.style_slot[len(other_heading_candidate.numbering) - 1] = other_heading_candidate
    if other_heading_candidate.type == 1:
        first = other_heading_candidate.numbering[0]
        state.secondary_slot = max(state.secondary_slot, first)
        state.tertiary_slot = max(state.tertiary_slot, first)
        if len(other_heading_candidate.numbering) == 1:
            state.auxiliary_slot = other_heading_candidate
    elif other_heading_candidate.type in (8, 9):
        state.tertiary_slot = 0
    state.primary_slot = other_heading_candidate


# --------------------------------------------------------------------------- #
# Hierarchy-walk stack #
# --------------------------------------------------------------------------- #


class HierarchyStack:
    """Tree-walk stack of currently open outline nodes."""

    __slots__ = ("auxiliary_slot", "primary_slot", "secondary_slot", "tertiary_slot")

    def __init__(self, anchor):
        self.auxiliary_slot = anchor
        self.primary_slot: list[OutlineNode] = []
        self.secondary_slot = False
        self.tertiary_slot = False

    def pop(self) -> Optional[OutlineNode]:
        return self.primary_slot.pop() if self.primary_slot else None

    def push(self, other_outline_node: OutlineNode) -> None:
        self.primary_slot.append(other_outline_node)
        self.secondary_slot = self.secondary_slot or other_outline_node.heading.type == 4
        self.tertiary_slot = self.tertiary_slot or other_outline_node.heading.is_prominent


def find_parent_heading(stack: HierarchyStack, other_heading_candidate: HeadingCandidate) -> Optional[OutlineNode]:
    """Pop entries from the stack until a parent for the candidate is found."""
    heading: Optional[HeadingCandidate] = None
    while stack.primary_slot:
        stack_outline_node = stack.primary_slot[-1]
        state_candidate = stack_outline_node.heading
        if other_heading_candidate.is_prominent and len(other_heading_candidate.numbering) <= 1 and state_candidate.type != 8:
            stack.pop()
            heading = state_candidate
            continue
        if state_candidate.is_prominent and other_heading_candidate.type == 5:
            stack.pop()
            heading = state_candidate
            continue
        cmp = compare_heading_depth(state_candidate, other_heading_candidate, stack.auxiliary_slot)
        if cmp != -1:
            if cmp == 1:
                return stack_outline_node
            # Appendix and Roman/letter headings can nest under the current
            # parent only when the numbering sequence remains coherent.
            if (state_candidate.type != other_heading_candidate.type and other_heading_candidate.type in (4, 2) and not stack.tertiary_slot
                    and is_appendix_nesting_ok(stack, other_heading_candidate, heading)):
                first_number = other_heading_candidate.numbering[0] if other_heading_candidate.numbering else 0
                if heading is None:
                    if first_number == 1:
                        return stack_outline_node
                else:
                    # Empty numbering on the previous heading cannot establish
                    # an increasing appendix sequence.
                    if other_heading_candidate.type == heading.type and other_heading_candidate.numbering and heading.numbering and first_number > heading.numbering[0]:
                        return stack_outline_node
        stack.pop()
        heading = state_candidate
    return None


def is_appendix_nesting_ok(stack: HierarchyStack, other_heading_candidate: HeadingCandidate, candidate_heading_candidate: Optional[HeadingCandidate]) -> bool:
    """Return whether an appendix candidate may be nested under the current stack state. Non-appendix headings always pass; appendix headings pass when the stack is already in appendix mode, has no numbering context, or starts at appendix depth 1..3."""
    if other_heading_candidate.type != 4:
        return True
    if stack.secondary_slot:
        return True
    # Last heading info
    if not stack.primary_slot:
        return True
    entry_item = stack.primary_slot[-1].heading
    if len(entry_item.numbering) <= 0:
        return True
    return entry_item.numbering[0] <= 3


# --------------------------------------------------------------------------- #
# Sub-headings within a cluster #
# --------------------------------------------------------------------------- #


def extract_sub_headings(doc, state: OutlineState, parent_node: Optional[OutlineNode], cluster_candidates: list[HeadingCandidate]) -> list[OutlineNode]:
    """Walk a cluster's candidate list and emit subheadings. The input list is consumed in place so later passes do not reprocess headings already assigned to this cluster."""
    if not cluster_candidates:
        return []
    # Content cap: walk from the parent page to the first candidate page and
    # abort the cluster if accumulated body-block text exceeds 1000.
    from ..stats import info_weight as _info_weight
    first = cluster_candidates[0]
    page_index = (parent_node.heading.page.page_index - 1) if parent_node is not None else 0
    acc = 0
    end_pg = min(first.page.page_index, len(doc.primary_slot))
    while page_index < end_pg:
        heading_page = doc.primary_slot[page_index]
        if getattr(heading_page, "state_slot", False):
            for block in heading_page.output_slot:
                if page_index >= first.page.page_index - 1 and block.reading_order_index >= first.group_slot.reading_order_index:
                    break
                if getattr(block, "is_body_paragraph", None):
                    acc += _info_weight(block.char_stats)
                    if acc >= 1000:
                        return []
        page_index += 1
    out: list[OutlineNode] = []
    parent_anchor = parent_node if (parent_node is not None and parent_node.heading.type == 5) else None
    seen_signatures: set[str] = set()
    style_cluster = StyleCluster()
    saw_numbered = False
    index = 0
    while index < len(cluster_candidates):
        cluster_candidate = cluster_candidates[index]
        if not (
            cluster_candidate.type == 5
            or cluster_candidate.type == 6
            or (cluster_candidate.type == 11 and cluster_candidate.has_numbering and parent_anchor is not None and index <= 1)
        ):
            next_item = cluster_candidates[index + 1] if index + 1 < len(cluster_candidates) else None
            if next_item and next_item.type == 5 and next_item.page is cluster_candidate.page and next_item.tertiary_slot is cluster_candidate.tertiary_slot:
                index += 1
                continue
            break
        candidate_signature = heading_signature(cluster_candidate)
        if candidate_signature in seen_signatures:
            index += 1
            continue
        if should_reject_heading(state, cluster_candidate):
            index += 1
            continue
        push_heading_to_state(state, cluster_candidate)
        seen_signatures.add(candidate_signature)
        if cluster_candidate.has_numbering:
            saw_numbered = True
        elif saw_numbered:
            break
        if parent_anchor is None:
            parent_anchor = OutlineNode(cluster_candidate)
            out.append(parent_anchor)
            style_cluster.add(cluster_candidate)
            index += 1
            continue
        anchor_heading_candidate = parent_anchor.heading
        if cluster_candidate.page.page_index > anchor_heading_candidate.page.page_index:
            break
        cmp = compare_heading_depth(anchor_heading_candidate, cluster_candidate)
        if cmp != 1:
            if not has_style_neighbor(style_cluster, cluster_candidate, 1.0):
                break
            parent_anchor = OutlineNode(cluster_candidate)
            out.append(parent_anchor)
            style_cluster.add(cluster_candidate)
        index += 1
    # Remove processed items so the outline loop does not reprocess them.
    del cluster_candidates[:index]
    if (
        len(out) >= 3
        or (len(out) == 2 and out[0].heading.has_numbering and out[1].heading.has_numbering)
    ) and out[0].heading.type != 5:
        return []
    return out


# --------------------------------------------------------------------------- #
# Flatten outline to top-level headings #
# --------------------------------------------------------------------------- #


def extract_top_level_headings(item_list: list[OutlineNode]) -> list[OutlineNode]:
    """Walk the outline and emit top-level prominent headings."""
    out: list[OutlineNode] = []
    saw_prominent = False
    for heading in item_list:
        if heading.heading.is_prominent:
            if not saw_prominent:
                out.append(heading)
            saw_prominent = True
        else:
            saw_prominent = False
            out.extend(extract_top_level_headings(heading.child_nodes))
    return out


# --------------------------------------------------------------------------- #
# Outline validation.
# --------------------------------------------------------------------------- #


def is_outline_valid(doc, item_list: list[OutlineNode]) -> bool:
    """Return True when top-level headings span a meaningful fraction of the document."""
    top = extract_top_level_headings(item_list)
    if len(top) < 3:
        return False
    if len(top) >= 5:
        return True
    last_page = 1
    for top_node in top:
        line = top_node.heading.page.page_index
        if line - last_page > 0.5 * len(doc.primary_slot):
            return False
        last_page = line
    return True


def is_chapter_outline_valid(doc, item_list: list[OutlineNode]) -> bool:
    """Secondary validity check based on chapter count and inter-chapter span."""
    chapters = 0
    span = 0
    previous = -1
    for chapter_outline_node in item_list:
        chapter_page = chapter_outline_node.heading.page.page_index
        if previous >= 0:
            span += chapter_page - previous
            previous = -1
        if chapter_outline_node.heading.type == 8:
            chapters += 1
            previous = chapter_page
    if previous >= 0:
        span += len(doc.primary_slot) - previous + 1
    return (
        chapters >= 3
        and span >= 0.7 * len(doc.primary_slot)
        and span / max(1, chapters) < 100
    )
