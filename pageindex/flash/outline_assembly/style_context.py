"""Style clusters, outline context/state, numbering trie, and depth comparison."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

from sortedcontainers import SortedKeyList
from ..model import (
    style_key, left_aligned, right_aligned, center_aligned, x_aligned, rect_union,
    Rect, last_span, avg_char_width, raw_text_of_line, heading_score, numbering_text, numbering_value, numbering_kind,
    reading_order_key, left_edge_key, _trim_unicode_ws, _round_half_up_to_int, Line, last_line_of, first_span_of, block_text, deaccented_text, letter_count, dominant_style_of, info_weight, dominant_font_size, is_upper_dominant, is_caps_heavy, alignment_code, Block,
)

from .candidates import (
    HeadingCandidate,
    compare_heading_order,
    heading_order_key,
    parent_signature,
    cached_signature,
    is_in_oo_range,
    has_style_neighbor,
)


# --------------------------------------------------------------------------- #
# Font/style-clustered heading group #
# --------------------------------------------------------------------------- #


class StyleCluster:
    """Group of headings sharing a font/style signature."""

    __slots__ = ("auxiliary_slot", "state_slot", "secondary_slot", "primary_slot", "tertiary_slot")

    def __init__(self):
        self.auxiliary_slot: dict[HeadingCandidate, str] = {}                               # candidate -> signature
        self.state_slot: dict[str, HeadingCandidate] = {}                               # signature -> candidate
        self.secondary_slot: SortedKeyList = SortedKeyList(
            key=lambda sort_node: (heading_score(sort_node.group_slot), heading_order_key(sort_node))
        )
        self.primary_slot: Optional[HeadingCandidate] = None                              # min by heading order
        self.tertiary_slot: Optional[HeadingCandidate] = None                              # max by heading order

    def size(self) -> int:
        return len(self.secondary_slot)

    def contains(self, other_heading_candidate: HeadingCandidate) -> bool:
        # Containment is key-based, not object identity. SortedKeyList's ``in``
        # tests identity among equal-key elements, so compare the sort keys.
        idx = self.secondary_slot.bisect_left(other_heading_candidate)
        return idx < len(self.secondary_slot) and self.secondary_slot.key(self.secondary_slot[idx]) == self.secondary_slot.key(other_heading_candidate)

    def add(self, other_heading_candidate: HeadingCandidate) -> None:
        self.state_slot[cached_signature(self, other_heading_candidate)] = other_heading_candidate
        # Keep set semantics over the sort key: equal-key elements are dropped,
        # while the signature and min/max heading-order state still update.
        idx = self.secondary_slot.bisect_left(other_heading_candidate)
        if idx >= len(self.secondary_slot) or self.secondary_slot.key(self.secondary_slot[idx]) != self.secondary_slot.key(other_heading_candidate):
            self.secondary_slot.add(other_heading_candidate)
        if self.primary_slot is None or compare_heading_order(other_heading_candidate, self.primary_slot) < 0:
            self.primary_slot = other_heading_candidate
        if self.tertiary_slot is None or compare_heading_order(other_heading_candidate, self.tertiary_slot) > 0:
            self.tertiary_slot = other_heading_candidate

    def has_nearby_duplicate(self, other_heading_candidate: HeadingCandidate) -> bool:
        """"have we seen a nearby matching signature within +/- 20 pages?"."""
        existing = self.state_slot.get(cached_signature(self, other_heading_candidate))
        return existing is not None and abs(other_heading_candidate.page.page_index - existing.page.page_index) < 20


# --------------------------------------------------------------------------- #
# Outline-context style-bucket operations #
# --------------------------------------------------------------------------- #


def pick_style_bucket(outline_context: "OutlineContext", other_heading_candidate: HeadingCandidate) -> StyleCluster:
    """Pick the right style bucket for a candidate."""
    if other_heading_candidate.type == 10:
        return outline_context.secondary_slot
    if other_heading_candidate.type == 8:
        return outline_context.primary_slot
    if len(other_heading_candidate.numbering) > 0:
        return outline_context.auxiliary_slot
    return outline_context.tertiary_slot


def has_conflict_in_context(outline_context: "OutlineContext", other_heading_candidate: HeadingCandidate) -> bool:
    """Return True if a candidate conflicts with the existing outline context."""
    if other_heading_candidate.type != 10 and is_in_oo_range(outline_context.secondary_slot, other_heading_candidate):
        return True
    if other_heading_candidate.type != 8 and is_in_oo_range(outline_context.primary_slot, other_heading_candidate):
        return True
    if len(other_heading_candidate.numbering) <= 0 and is_in_oo_range(outline_context.auxiliary_slot, other_heading_candidate):
        return True
    if other_heading_candidate.type != 8 and len(other_heading_candidate.numbering) > 0 and outline_context.primary_slot.size() > 0:
        return True
    return False


def is_compatible_with_context(outline_context: "OutlineContext", other_heading_candidate: HeadingCandidate) -> bool:
    """Return True iff a candidate can be added to the outline context."""
    if has_conflict_in_context(outline_context, other_heading_candidate):
        return False
    # Find the nearest predecessor by heading order.
    text: Optional[HeadingCandidate] = None
    for item in outline_context.state_slot:
        if compare_heading_order(item, other_heading_candidate) <= 0:
            if text is None or compare_heading_order(item, text) > 0:
                text = item
        else:
            break
    if text is not None:
        candidate_block = other_heading_candidate.group_slot
        previous_block = text.group_slot
        if previous_block.isolated_centered and not candidate_block.isolated_centered:
            return False
        if not other_heading_candidate.is_prominent and heading_score(previous_block) > heading_score(candidate_block) + 0.5:
            return False
        if text.is_prominent and not other_heading_candidate.is_prominent and heading_score(previous_block) > heading_score(candidate_block) - 0.5:
            return False
    style_cluster = pick_style_bucket(outline_context, other_heading_candidate)
    if not style_cluster.has_nearby_duplicate(other_heading_candidate) and has_style_neighbor(style_cluster, other_heading_candidate, 1.0):
        return True
    if len(other_heading_candidate.numbering) == 1 and has_style_neighbor(outline_context.tertiary_slot, other_heading_candidate, 1.0):
        return True
    return False


# --------------------------------------------------------------------------- #
# Outline-context group.
# --------------------------------------------------------------------------- #


class OutlineContext:
    """Bundles style clusters for chapter, appendix, numbered, and general headings."""

    __slots__ = ("secondary_slot", "primary_slot", "auxiliary_slot", "tertiary_slot", "state_slot")

    def __init__(self, headings: list[HeadingCandidate]):
        self.secondary_slot = StyleCluster()         # type == 10

        self.primary_slot = StyleCluster()         # type == 8

        self.auxiliary_slot = StyleCluster()         # has M (numbered)
        self.tertiary_slot = StyleCluster()         # everything else
        # The ordered heading list is set-like by heading-order key, with the
        # first equal-key candidate retained.
        self.state_slot: list[HeadingCandidate] = []
        seen_keys: set = set()
        for secondary_item in headings:
            self.add(secondary_item)
            key_value = heading_order_key(secondary_item)
            if key_value not in seen_keys:
                seen_keys.add(key_value)
                self.state_slot.append(secondary_item)
        self.state_slot.sort(key=heading_order_key)

    def add(self, other_heading_candidate: HeadingCandidate) -> None:
        pick_style_bucket(self, other_heading_candidate).add(other_heading_candidate)

    def has_nearby_duplicate(self, other_heading_candidate: HeadingCandidate) -> bool:
        return pick_style_bucket(self, other_heading_candidate).has_nearby_duplicate(other_heading_candidate)


# --------------------------------------------------------------------------- #
# Numbering-prefix tree.
# --------------------------------------------------------------------------- #


class NumberingTrie:
    """a recursive map for numbering prefixes."""

    __slots__ = ("primary_slot", "secondary_slot")

    def __init__(self):
        self.primary_slot: dict[int, "NumberingTrie"] = {}
        self.secondary_slot = 0


def insert_numbering(primary_item: NumberingTrie, other_heading_candidate: HeadingCandidate, index: int) -> None:
    """Insert the candidate numbering suffix into the trie."""
    if index == len(other_heading_candidate.numbering):
        primary_item.secondary_slot += 1
        return
    reference_item = primary_item.primary_slot.get(other_heading_candidate.numbering[index])
    if reference_item is None:
        reference_item = NumberingTrie()
        primary_item.primary_slot[other_heading_candidate.numbering[index]] = reference_item
    insert_numbering(reference_item, other_heading_candidate, index + 1)


def count_sibling_numberings(primary_item: NumberingTrie, other_heading_candidate: HeadingCandidate, index: int) -> int:
    """Count sibling numbering branches at the target depth."""
    if index >= len(other_heading_candidate.numbering) - 1:
        count = 0
        for reference_item in primary_item.primary_slot.values():
            if reference_item.secondary_slot > 0:
                count += 1
        return count
    reference_item = primary_item.primary_slot.get(other_heading_candidate.numbering[index])
    if reference_item is None:
        return 0
    return count_sibling_numberings(reference_item, other_heading_candidate, index + 1)


# --------------------------------------------------------------------------- #
# Global outline state.
# --------------------------------------------------------------------------- #


class OutlineState:
    """Global state for outline assembly walks."""

    __slots__ = ("state_slot", "cache_slot", "marker_slot", "previous_slot", "option_slot", "measure_slot", "style_slot", "secondary_slot", "auxiliary_slot", "tertiary_slot", "primary_slot")

    def __init__(self, clusters: list[dict]):
        self.state_slot: dict = {}
        self.cache_slot = NumberingTrie()
        self.marker_slot: dict[str, StyleCluster] = {}
        self.previous_slot = math.inf
        self.option_slot = -math.inf
        self.measure_slot = False
        clusters_by_parent_signature: dict[str, list[StyleCluster]] = {}
        for cluster in clusters:
            heading_branch = cluster.get("labeled_anchor")
            cluster_candidates = cluster.get("cluster_candidates", [])
            if heading_branch is not None:
                _apply_heading_to_state(self, heading_branch.heading)
            for state_candidate in cluster_candidates:
                _apply_heading_to_state(self, state_candidate)
                if len(state_candidate.numbering) <= 0:
                    continue
                key = parent_signature(state_candidate)
                item_list = clusters_by_parent_signature.get(key)
                if item_list is None:
                    item_list = []
                    clusters_by_parent_signature[key] = item_list
                placed = None
                for style_cluster in item_list:
                    if not style_cluster.has_nearby_duplicate(state_candidate) and has_style_neighbor(style_cluster, state_candidate, 1.0):
                        placed = style_cluster
                        break
                if placed is None and len(item_list) < 3:
                    placed = StyleCluster()
                    item_list.append(placed)
                if placed is not None:
                    placed.add(state_candidate)
        for key, group in clusters_by_parent_signature.items():
            group.sort(key=lambda bucket_group: -bucket_group.size())
            best = group[0]
            if best.size() <= 2:
                continue
            self.marker_slot[key] = best
        self.style_slot: list[Optional[HeadingCandidate]] = []
        self.secondary_slot = 0
        self.auxiliary_slot: Optional[HeadingCandidate] = None
        self.tertiary_slot = 0
        self.primary_slot: Optional[HeadingCandidate] = None


def _apply_heading_to_state(state: OutlineState, other_heading_candidate: HeadingCandidate) -> None:
    """Add a heading to the per-font tree and update document-level outline state."""
    seen: set[str] = set()
    line = other_heading_candidate.group_slot.line()
    for token_list in (other_heading_candidate.secondary_slot, other_heading_candidate.primary_slot):
        if token_list is None:
            continue
        for token in token_list:
            if token.line() is not line:
                break
            if token.type != 2:
                continue
            for anchor in token.anchor_ranges:
                if anchor.line is not line:
                    break
                span = anchor.anchor_span
                style = style_key(span)
                if style in seen:
                    continue
                seen.add(style)
                font_size = span.font_style()
                tree = state.state_slot.get(font_size)
                if tree is None:
                    tree = SortedKeyList(
                        key=lambda heading: (heading["size"], heading_order_key(heading["heading"]))
                    )
                    state.state_slot[font_size] = tree
                # Each per-font-size bucket is set-like by (size, heading-order)
                # key, retaining the first equal-key entry.
                entry = {"size": span.font_size, "heading": other_heading_candidate}
                idx = tree.bisect_left(entry)
                if idx >= len(tree) or tree.key(tree[idx]) != tree.key(entry):
                    tree.add(entry)
    if other_heading_candidate.type == 1 and len(other_heading_candidate.numbering) > 0:
        insert_numbering(state.cache_slot, other_heading_candidate, 0)
    state.previous_slot = min(state.previous_slot, other_heading_candidate.page.page_index)
    state.option_slot = max(state.option_slot, other_heading_candidate.page.page_index)
    if not state.measure_slot:
        state.measure_slot = other_heading_candidate.is_prominent


# --------------------------------------------------------------------------- #
# Pairwise heading-depth comparator #
# --------------------------------------------------------------------------- #


def compare_heading_depth(heading_candidate: HeadingCandidate, other_heading_candidate: HeadingCandidate, clique: Optional[StyleCluster] = None) -> int:
    """Compare two heading candidates for relative nesting depth. Returns ``-1`` when the first candidate should be shallower, ``1`` when it should be deeper, and ``0`` when both candidates should share a level. The decision combines special heading types, numbering depth, structural numbering, style prominence, centered layout, clique membership, and bold weight. """
    special = heading_candidate.type in (8, 9, 10)
    other_special = other_heading_candidate.type in (8, 9, 10)
    if special and other_special:
        return 0
    if special or other_special:
        return 1 if special else -1
    if (heading_candidate.type == 1 and other_heading_candidate.type == 1) or (heading_candidate.type == 4 and other_heading_candidate.type == 4):
        left_length = len(heading_candidate.numbering)
        right_length = len(other_heading_candidate.numbering)
        if left_length == right_length:
            return 0
        return 1 if left_length < right_length else -1
    if heading_candidate.type == 2 and other_heading_candidate.type == 2:
        return 0
    if heading_candidate.type == 11 and len(other_heading_candidate.numbering) == 1:
        return -1
    heading_block = heading_candidate.group_slot
    block = other_heading_candidate.group_slot
    if heading_candidate.has_numbering != other_heading_candidate.has_numbering:
        return -1 if heading_candidate.has_numbering else 1
    if heading_candidate.has_numbering and other_heading_candidate.has_numbering and abs(first_span_of(heading_block).font_size - first_span_of(block).font_size) < 0.9:
        return 0
    score = heading_score(heading_block)
    other_score = heading_score(block)
    heading_in_clique = clique is not None and clique.contains(heading_candidate)
    in_value = clique is not None and clique.contains(other_heading_candidate)
    # Z-based major-gap return
    if abs(score - other_score) > 1.9 or (abs(score - other_score) > 0.9 and (not heading_in_clique or not in_value)):
        return 1 if score > other_score else -1
    # uppercase-dominant comparison
    heading_caps_heavy = is_caps_heavy(heading_block)
    caps_heavy = is_caps_heavy(block)
    if heading_caps_heavy != caps_heavy:
        return 1 if heading_caps_heavy else -1
    # paragraph-end / isolated-centered comparison
    centered_flag = heading_block.isolated_centered
    if centered_flag != block.isolated_centered:
        return 1 if centered_flag else -1
    # skew (rotation) comparison -- skipped if both type 5
    if not (heading_candidate.type == 5 and other_heading_candidate.type == 5):
        heading_skewed = heading_block.previous_slot > 0.99
        skew = block.previous_slot > 0.99
        if heading_skewed != skew:
            return -1 if heading_skewed else 1
    # uppercase or both in clique -> tied
    if heading_caps_heavy or (heading_in_clique and in_value):
        return 0
    # clique containment asymmetric
    if (heading_in_clique and other_heading_candidate.type != 4) or (in_value and heading_candidate.type != 4):
        return 1 if heading_in_clique else -1
    # Bold comparison.
    bold = heading_block.bold_frac() > 0.5
    other_bold = block.bold_frac() > 0.5
    if bold != other_bold:
        return 1 if bold else -1
    return 0
