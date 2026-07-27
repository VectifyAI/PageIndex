"""Heading candidate and outline node types plus ordering and signature helpers."""

from __future__ import annotations

from ..model import (
    style_key, left_aligned, right_aligned, center_aligned, x_aligned, rect_union,
    Rect, last_span, avg_char_width, raw_text_of_line, heading_score, numbering_text, numbering_value, numbering_kind,
    reading_order_key, left_edge_key, _trim_unicode_ws, _round_half_up_to_int, Line, last_line_of, first_span_of, block_text, deaccented_text, letter_count, dominant_style_of, info_weight, dominant_font_size, is_upper_dominant, is_caps_heavy, alignment_code, Block,
)
from ..stats import style_key as style_key_fn, column_index_of, tally_scripts, dominant_script_family, ScriptHistogram
from ..tokens import (
    Token, TokenView, wrap_tokens, enumerate_tokens, last_token, trie_prefix_match, first_token, set_case_fold, TrieConfig, build_trie, tokenize_block, avg_char_width as avg_char_width_fn, trie_full_match, first_anchor_span, is_char_token, is_word_token,
)


# --------------------------------------------------------------------------- #
# Heading candidate wrapper #
# --------------------------------------------------------------------------- #


def _viewport_y_fraction(viewport_box, rot: int, user_x: float, user_y: float) -> float:
    """Return viewport-normalized y coordinate for a PDF user-space point. Applies the same page ``/Rotate`` and the unrotated view box to a user-space point, then normalises the y component by the viewport height."""
    x_min, y_min, x_max, y_max = viewport_box
    center_x = (x_max + x_min) / 2.0
    center_y = (y_max + y_min) / 2.0
    rotation = rot % 360
    if rotation < 0:
        rotation += 360
    if rotation == 90:
        x_axis_scale, y_axis_scale = 1, 0
        x_axis_sign = 0
    elif rotation == 180:
        x_axis_scale, y_axis_scale = 0, 1
        x_axis_sign = -1
    elif rotation == 270:
        x_axis_scale, y_axis_scale = -1, 0
        x_axis_sign = 0
    else:
        x_axis_scale, y_axis_scale = 0, -1
        x_axis_sign = 1
    if x_axis_sign == 0:
        viewport_offset = abs(center_x - x_min)
        height = abs(x_max - x_min)
    else:
        viewport_offset = abs(center_y - y_min)
        height = abs(y_max - y_min)
    # transform[1]=b, transform[3]=d, transform[5]=off_y - b*cx - d*cy;
    # the viewport y-coordinate = b*x + d*y + transform[5].
    viewport_y = x_axis_scale * user_x + y_axis_scale * user_y + (viewport_offset - x_axis_scale * center_x - y_axis_scale * center_y)
    return viewport_y / (height or 1.0)


class HeadingCandidate:
    """One heading candidate. It stores the candidate type, page, underlying block, optional anchor block, numbering array, optional prefix tokens, title tokens, structural-numbering flag, prominence flag, dominant script family, and vertical page position."""

    __slots__ = ("type", "page", "group_slot", "tertiary_slot", "numbering", "secondary_slot", "primary_slot", "has_numbering", "is_prominent", "state_slot", "auxiliary_slot")

    def __init__(self, type_, page, group_value, anchor, numbering_value, tokens, title_tokens, has_numbering_flag, prominent_flag):
        self.type = type_
        self.page = page
        self.group_slot = group_value
        self.tertiary_slot = anchor
        self.numbering = numbering_value or []
        self.secondary_slot = tokens
        self.primary_slot = title_tokens
        self.has_numbering = has_numbering_flag
        self.is_prominent = prominent_flag
        # Compute the dominant script family over prefix and title tokens.
        acc = ScriptHistogram()
        if tokens is not None:
            for token_value in tokens:
                tally_scripts(acc, token_value.str)
        if title_tokens is not None:
            for token_value in title_tokens:
                tally_scripts(acc, token_value.str)
        self.state_slot = dominant_script_family(acc)
        # Compute vertical fraction on page. The viewport applies the page
        # /Rotate and view box; when that metadata is absent, fall back to the
        # origin-0 upright shortcut.
        viewport_box_value = getattr(page, "viewport_box", None)
        if viewport_box_value is not None:
            self.auxiliary_slot = _viewport_y_fraction(viewport_box_value, getattr(page, "rot", 0) or 0, group_value.left_edge(), group_value.top_edge())
        else:
            page_height = page.bounds.bbox_height() or 1.0
            self.auxiliary_slot = (page.bounds.top_edge() - group_value.top_edge()) / page_height

    def __repr__(self) -> str:  # diagnostic
        return f"<HeadingCandidate t={self.type} M={self.numbering} G={block_text(self.group_slot)[:30]!r}>"


# --------------------------------------------------------------------------- #
# Outline node #
# --------------------------------------------------------------------------- #


class OutlineNode:
    """Heading plus child outline nodes."""

    __slots__ = ("heading", "child_nodes")

    def __init__(self, heading: HeadingCandidate):
        self.heading = heading
        self.child_nodes: list["OutlineNode"] = []


# --------------------------------------------------------------------------- #
# Page and reading-position comparator.
# --------------------------------------------------------------------------- #


def compare_heading_order(heading_candidate: HeadingCandidate, other_heading_candidate: HeadingCandidate) -> float:
    """Order by page, then block reading position."""
    if heading_candidate.page.page_index != other_heading_candidate.page.page_index:
        return heading_candidate.page.page_index - other_heading_candidate.page.page_index
    return _compare_block_order(heading_candidate.group_slot, other_heading_candidate.group_slot)


def _compare_block_order(block: Block, other_block: Block) -> float:
    """Compare by column index first, then by reading position."""
    from ..model import cmp_reading_order
    from ..stats import column_index_of as _column_index
    heading_anchor = _column_index(block)
    other_column_index = _column_index(other_block)
    if heading_anchor != other_column_index:
        return heading_anchor - other_column_index
    return cmp_reading_order(block, other_block)


def heading_order_key(heading_candidate: HeadingCandidate) -> tuple:
    from ..stats import column_index_of as _column_index
    return (heading_candidate.page.page_index, _column_index(heading_candidate.group_slot), -heading_candidate.group_slot.top_edge(), -heading_candidate.group_slot.bottom_edge(), heading_candidate.group_slot.left_edge(), heading_candidate.group_slot.right_edge())


# --------------------------------------------------------------------------- #
# Candidate compatibility and style-cluster helpers #
# --------------------------------------------------------------------------- #


def is_script_compatible(number: int, other_heading_candidate: HeadingCandidate) -> bool:
    """Return whether candidate script/type is compatible with prior context. Args: a: integer previous script/type context b: heading candidate """
    candidate_item = other_heading_candidate.state_slot
    if candidate_item == 0 or candidate_item == 2 or candidate_item == 10:
        return True
    if number == candidate_item:
        return False
    if other_heading_candidate.type == 5:
        return False
    if other_heading_candidate.is_prominent:
        return False
    if len(other_heading_candidate.numbering) > 0:
        return False
    if number == 3 and candidate_item == 9:
        return False
    if number == 9 and candidate_item == 3:
        return False
    if number == 7 and candidate_item == 5:
        return False
    if number == 6 and candidate_item == 3:
        return False
    return True


def heading_signature(heading_candidate: HeadingCandidate) -> str:
    """Return a full heading signature including numbering or text."""
    if len(heading_candidate.numbering) > 0:
        # Numbering arrays are serialized as comma-joined values, not Python
        # list representations.
        return f"{heading_candidate.type}|{','.join(map(str, heading_candidate.numbering))}"
    heading = f"{heading_candidate.type}|"
    if heading_candidate.primary_slot is not None:
        for token in heading_candidate.primary_slot:
            if is_char_token(token):
                heading += token.str.lower()
    return heading


def parent_signature(heading_candidate: HeadingCandidate) -> str:
    """Return the signature of the candidate's parent numbering prefix."""
    secondary_item = f"{heading_candidate.type}|"
    for candidate_item in range(len(heading_candidate.numbering) - 1):
        if candidate_item > 0:
            secondary_item += ","
        secondary_item += str(heading_candidate.numbering[candidate_item])
    return secondary_item


def cached_signature(primary_item: "StyleCluster", other_heading_candidate: HeadingCandidate) -> str:
    """Cached heading-signature lookup. Keyed by the candidate object itself, not by object id, because addresses can be reused after a discarded object is collected."""
    candidate_item = primary_item.auxiliary_slot.get(other_heading_candidate)
    if candidate_item is not None:
        return candidate_item
    candidate_item = heading_signature(other_heading_candidate)
    primary_item.auxiliary_slot[other_heading_candidate] = candidate_item
    return candidate_item


def is_in_oo_range(primary_item: "StyleCluster", other_heading_candidate: HeadingCandidate) -> bool:
    """Return True if the candidate lies within a style cluster's order range."""
    if primary_item.primary_slot is None or primary_item.tertiary_slot is None:
        return False
    return compare_heading_order(other_heading_candidate, primary_item.primary_slot) >= 0 and compare_heading_order(other_heading_candidate, primary_item.tertiary_slot) <= 0


def has_style_neighbor(style: "StyleCluster", other_heading_candidate: HeadingCandidate, candidate_item: float) -> bool:
    """Return True if a candidate is close to a compatible style neighbor."""
    candidate_score = heading_score(other_heading_candidate.group_slot)

    def cmp_target():
        return {"z": candidate_score, "HeadingCandidate": other_heading_candidate}

    matched = [False]

    def fcheck(item):
        if abs(candidate_score - heading_score(item.group_slot)) >= candidate_item:
            return True
        # Within tolerance, check signature match:
        measure_item = other_heading_candidate.group_slot
        line_value = item.group_slot
        if abs(heading_score(measure_item) - heading_score(line_value)) >= candidate_item:
            state_item = False
        elif len(other_heading_candidate.numbering) > 0 and len(item.numbering) > 0:
            state_item = (other_heading_candidate.type == item.type and len(other_heading_candidate.numbering) == len(item.numbering))
        elif (len(other_heading_candidate.numbering) <= 0 and len(item.numbering) > 1) or (len(item.numbering) <= 0 and len(other_heading_candidate.numbering) > 1):
            state_item = False
        else:
            block = measure_item.isolated_centered
            other_centered = line_value.isolated_centered
            if block or other_centered:
                state_item = (block == other_centered)
            elif dominant_style_of(measure_item) == dominant_style_of(line_value):
                state_item = True
            else:
                if first_span_of(measure_item).font_style() != first_span_of(line_value).font_style():
                    state_item = False
                else:
                    state_item = abs(dominant_font_size(measure_item) - dominant_font_size(line_value)) < candidate_item
        if state_item:
            matched[0] = True
            return True
        return False

    # Walk sibling candidates in both directions from the candidate's page position
    if style.secondary_slot is None:
        return False
    # SortedKeyList walk
    target_key = (candidate_score, heading_order_key(other_heading_candidate))
    idx = style.secondary_slot.bisect_right(other_heading_candidate)
    for scan_index in range(idx, len(style.secondary_slot)):
        if fcheck(style.secondary_slot[scan_index]):
            break
    if not matched[0]:
        for scan_index in range(idx - 1, -1, -1):
            if fcheck(style.secondary_slot[scan_index]):
                break
    return matched[0]
