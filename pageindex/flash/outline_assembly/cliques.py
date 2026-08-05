"""Keyword cliques, clique trees, body-heading detection, and candidate partitioning."""

from __future__ import annotations

from typing import Any, Callable, Optional
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
# Numbering-pattern clique selection.
# --------------------------------------------------------------------------- #


# Section-keyword trie shared with outline filtering.
from ..outline import SECTION_KEYWORD_TRIE

from .candidates import (
    HeadingCandidate,
    OutlineNode,
    compare_heading_order,
    heading_order_key,
    has_style_neighbor,
)
from .style_context import (
    StyleCluster,
    is_compatible_with_context,
    OutlineContext,
)


def find_keyword_clique(heading_candidates: list[HeadingCandidate]) -> Optional[StyleCluster]:
    """Find the largest clique of section-keyword headings sharing a font signature."""
    buckets: dict[str, StyleCluster] = {}
    for candidate_item in heading_candidates:
        if candidate_item.primary_slot is None:
            continue
        if not trie_full_match(SECTION_KEYWORD_TRIE, candidate_item.primary_slot):
            continue
        first = first_token(candidate_item.primary_slot)
        if first is None or not first.anchor_ranges:
            continue
        font_size = first_anchor_span(first).font_style()
        style_cluster = buckets.get(font_size)
        if style_cluster is not None:
            if style_cluster.has_nearby_duplicate(candidate_item):
                return None                        # conflict -> abort
            if has_style_neighbor(style_cluster, candidate_item, 2.0):
                style_cluster.add(candidate_item)
        else:
            style_cluster = StyleCluster()
            buckets[font_size] = style_cluster
            style_cluster.add(candidate_item)
    winner: Optional[StyleCluster] = None
    max_size = 0
    for style_cluster in buckets.values():
        if style_cluster.size() > max_size:
            winner = style_cluster
            max_size = style_cluster.size()
    if winner is None or max_size <= 1:
        return None
    for entry_item in heading_candidates:
        if winner.contains(entry_item):
            continue
        if has_style_neighbor(winner, entry_item, 0.5):
            winner.add(entry_item)
    return winner


# --------------------------------------------------------------------------- #
# Clique-based clusters #
# --------------------------------------------------------------------------- #


class CliqueTreeNode:
    """Tree node used by clique-based heading filtering. Each node holds a heading candidate, parent pointer, child list, and sibling links. ``next`` walks the in-order successor."""

    __slots__ = ("heading", "parent", "primary_slot", "secondary_slot", "tertiary_slot")

    def __init__(self, heading, parent):
        self.heading = heading
        self.parent = parent if parent is not None else self
        self.primary_slot: list = []
        self.secondary_slot = None
        self.tertiary_slot = None

    def next(self):
        if self.primary_slot:
            return self.primary_slot[0]
        if self.secondary_slot is not None:
            return self.secondary_slot
        return find_ancestor_next_sibling(self.parent)


def find_ancestor_next_sibling(primary_item: CliqueTreeNode):
    """walk up parents until we find one with a next sibling."""
    if primary_item.parent is primary_item:
        return None
    return primary_item.secondary_slot or find_ancestor_next_sibling(primary_item.parent)


def descend_to_deepest_last(primary_item: CliqueTreeNode) -> CliqueTreeNode:
    """descend to deepest last-child."""
    while primary_item.primary_slot:
        primary_item = primary_item.primary_slot[-1]
    return primary_item


def append_tree_child(ao_tree, parent_node: CliqueTreeNode, heading) -> None:
    """Append a new clique-tree child and advance the builder cursor."""
    new_node = CliqueTreeNode(heading, parent_node)
    last = parent_node.primary_slot[-1] if parent_node.primary_slot else None
    if last is not None:
        last.secondary_slot = new_node
        new_node.tertiary_slot = last
    parent_node.primary_slot.append(new_node)
    ao_tree.primary_slot = new_node


class CliqueTreeBuilder:
    """(class at table entry). Builds a clique-tree from a heading list using a comparator. Each heading is placed by walking the cursor up/down based on comparator result. Depth capped at 8. """

    __slots__ = ("root", "primary_slot")

    def __init__(self, headings: list[HeadingCandidate], compare):
        self.root = CliqueTreeNode(None, None)
        self.primary_slot = self.root
        depth = 0
        for height in headings:
            while True:
                if self.primary_slot is self.root:
                    append_tree_child(self, self.primary_slot, height)
                    depth += 1
                    break
                comparison = compare(self.primary_slot.heading, height)
                if comparison < 0:
                    self.primary_slot = self.primary_slot.parent
                    depth -= 1
                else:
                    if comparison > 0 and depth < 8:
                        append_tree_child(self, self.primary_slot, height)
                        depth += 1
                    else:
                        append_tree_child(self, self.primary_slot.parent, height)
                    break


def block_style_signature(block) -> str:
    """Return a block-style signature combining dominant style and caps-heavy state."""
    from ..model import dominant_style_of, is_caps_heavy
    # The boolean portion is lower-case because the signature is used as an
    # opaque stable key.
    return f"{dominant_style_of(block)} {'true' if is_caps_heavy(block) else 'false'}"


def is_member_of_tree(doc, block, target_sig: str, sentence_like: bool, node: CliqueTreeNode) -> bool:
    """Return whether the target block is already represented by an ancestor in the candidate tree, using heading signature, body-text weight, and recursive parent traversal."""
    from ..model import is_sentence_like
    from ..stats import info_weight
    if node is None or node.parent is node:
        return False
    tree_parent_candidate = node.heading
    if tree_parent_candidate is None or tree_parent_candidate.type == 5 or tree_parent_candidate.is_prominent:
        return False
    if len(tree_parent_candidate.numbering) > 0:
        return is_member_of_tree(doc, block, target_sig, sentence_like, node.parent)
    parent_block = tree_parent_candidate.group_slot
    if target_sig != block_style_signature(parent_block) or (sentence_like and is_sentence_like(parent_block)):
        return is_member_of_tree(doc, block, target_sig, sentence_like, node.parent)
    if info_weight(block.char_stats) >= max(100, 4 * info_weight(parent_block.char_stats)):
        return is_member_of_tree(doc, block, target_sig, sentence_like, node.parent)
    return True


def can_share_heading_style(heading, other_heading, neighbor_map) -> bool:
    """Return whether two blocks can share a heading-style assignment after checking overlap, style signature, neighboring ambiguity, and predecessor consistency."""
    from ..model import y_overlaps, dominant_style_of
    from ..heading_detection import neighbor_right, neighbor_above
    if other_heading is None or not y_overlaps(heading, other_heading) or dominant_style_of(heading) != dominant_style_of(other_heading):
        return False
    heading_above = neighbor_above(neighbor_map, heading)
    other_above = neighbor_above(neighbor_map, other_heading)
    heading_right = neighbor_right(neighbor_map, heading)
    other_right = neighbor_right(neighbor_map, other_heading)
    if (heading_above is not None and heading_above.marker_slot != 0
            or other_above is not None and other_above.marker_slot != 0
            or heading_right is not None and heading_right.marker_slot != 0
            or other_right is not None and other_right.marker_slot != 0):
        return True
    if (heading_right is not other_right
            and (heading_right is not None and heading_right.is_body_paragraph)
            and (other_right is not None and other_right.is_body_paragraph)):
        return False
    return True


def compare_block_order(left_value, right_value) -> float:
    """Compare blocks or lines by column index first, then reading position."""
    from ..model import cmp_reading_order
    from ..stats import column_index_of
    left_column_index = column_index_of(left_value)
    right_column_index = column_index_of(right_value)
    if left_column_index != right_column_index:
        return left_column_index - right_column_index
    return cmp_reading_order(left_value, right_value)


def heading_precedes_line(line_heading_candidate: HeadingCandidate, page, line) -> bool:
    """Return whether the heading candidate sorts before the given page/line position."""
    if line_heading_candidate.page.page_index < page.page_index:
        return True
    if line_heading_candidate.page.page_index != page.page_index:
        return False
    return compare_block_order(line_heading_candidate.group_slot, line) < 0


class CliqueFilterContext:
    """State for clique-based body-heading discovery."""

    __slots__ = ("auxiliary_slot", "state_slot", "tertiary_slot", "measure_slot", "secondary_slot", "option_slot", "primary_slot", "candidates", "compare")

    def __init__(self, doc, candidates: list[HeadingCandidate], compare):
        self.auxiliary_slot = doc
        self.state_slot: set = set()
        self.tertiary_slot: dict = {}
        for reference_item in candidates:
            self.state_slot.add(reference_item.group_slot)
            if reference_item.has_numbering:
                continue
            if len(reference_item.numbering) > 0:
                continue
            sig = block_style_signature(reference_item.group_slot)
            self.tertiary_slot[sig] = self.tertiary_slot.get(sig, 0) + 1
        self.measure_slot = CliqueTreeBuilder(candidates, compare)
        self.secondary_slot = self.measure_slot.root
        self.option_slot = CliqueTreeBuilder(list(reversed(candidates)), compare)
        self.primary_slot = self.option_slot.primary_slot
        self.candidates = candidates
        self.compare = compare


def detect_body_headings(filter_context: CliqueFilterContext) -> list[HeadingCandidate]:
    """Discover body headings by comparing unvisited blocks against clique trees."""
    from ..model import style_key, dominant_style_of, last_span, last_line_of, first_span_of
    from ..heading_detection import neighbor_right, neighbor_above, closest_body_neighbor_above, PageNeighborMap as _bo_class, is_cover_page
    from ..tokens import first_token, tokenize_block

    out: list[HeadingCandidate] = []
    if not filter_context.candidates:
        return out

    # Reset cursors to root of forward tree / deepest of reversed tree.
    filter_context.secondary_slot = filter_context.measure_slot.root
    filter_context.primary_slot = filter_context.option_slot.primary_slot

    for page in filter_context.auxiliary_slot.primary_slot:
        if is_cover_page(filter_context.auxiliary_slot, page):
            continue
        all_blocks = page.output_slot
        if len(all_blocks) <= 0:
            continue
        neighbor_cache = _bo_class(page)
        for block in page.secondary_slot:
            # Advance the forward tree cursor while the next node is before
            # the current page and block in reading order.
            while True:
                next_item = filter_context.secondary_slot.next()
                if (next_item is None
                        or next_item.heading is None
                        or not heading_precedes_line(next_item.heading, page, block)):
                    break
                filter_context.secondary_slot = next_item
            # Advance the reverse tree cursor while the predecessor is before
            # cursor's heading is still before the current page and block.
            while filter_context.primary_slot.heading is not None and heading_precedes_line(filter_context.primary_slot.heading, page, block):
                left_sib = filter_context.primary_slot.tertiary_slot
                filter_context.primary_slot = descend_to_deepest_last(left_sib) if left_sib is not None else filter_context.primary_slot.parent
                if filter_context.primary_slot is filter_context.option_slot.root:
                    break

            if block in filter_context.state_slot:
                continue
            if filter_context.secondary_slot.heading is None:
                continue
            # Body-heading filters.
            if (block.char_count() <= 0 or block.skew_frac() > 1
                    or (block.char_count() <= 1 and block.char_stats.secondary_slot != 4)
                    or block.line_count() >= 5
                    or block.type != 0
                    or block.marker_slot != 0
                    or (block.char_stats.primary_slot[2] <= 0 and block.char_stats.primary_slot[4] <= 0)):
                continue
            if block.measure_slot:
                continue
            value = block.bold_frac()
            if 0.1 < value < 0.9:
                continue
            block_style = dominant_style_of(block)
            first_tok = first_token(tokenize_block(block))
            # Compare against the dominant style, first span, last token
            # anchor, and last span. The last anchor matters for wrapped tokens.
            anchor = first_tok.anchor_ranges[-1].anchor_span if (first_tok is not None and first_tok.anchor_ranges) else None
            if (block_style != style_key(first_span_of(block))
                    and (anchor is None or block_style != style_key(anchor))
                    and block_style != style_key(last_span(last_line_of(block)))):
                continue
            if block_style == page.primary_slot.auxiliary_slot:
                continue
            above = neighbor_above(neighbor_cache, block)
            if (above is not None
                    and above.bottom_edge() - block.top_edge() < 0.3 * block.avg_font_size()
                    and block.line_count() > 1):
                continue
            if above is not None and above.type == 3:
                continue
            sig = block_style_signature(block)
            pred_neigh = neighbor_right(neighbor_cache, block)
            # Reject when the block repeats the style signature of a close
            # vertical or right-side neighbor.
            if above is not None and sig == block_style_signature(above):
                continue
            if pred_neigh is not None and sig == block_style_signature(pred_neigh):
                continue
            previous_block = all_blocks[block.orig_index - 1] if 0 <= block.orig_index - 1 < len(all_blocks) else None
            next_block = all_blocks[block.orig_index + 1] if 0 <= block.orig_index + 1 < len(all_blocks) else None
            if can_share_heading_style(block, previous_block, neighbor_cache):
                continue
            if can_share_heading_style(block, next_block, neighbor_cache):
                continue
            if filter_context.tertiary_slot.get(sig, 0) < 3:
                continue
            # Sentence-like flag: enough long lowercase-leading word tokens make
            # a block look like body text rather than a heading.
            tok_total = 0
            tok_g3 = 0
            for token in tokenize_block(block):
                if token.type != 2 or len(token.str) < 5:
                    continue
                tok_total += 1
                if token.primary_slot == 3:
                    tok_g3 += 1
            sentence_like = tok_g3 >= max(2, tok_total / 2)
            # A block must fit either the forward or reverse clique cursor.
            if not (is_member_of_tree(filter_context, block, sig, sentence_like, filter_context.secondary_slot)
                    or is_member_of_tree(filter_context, block, sig, sentence_like, filter_context.primary_slot)):
                continue
            body_heading_candidate = HeadingCandidate(
                0, page, block,
                closest_body_neighbor_above(neighbor_cache, block),
                [], None, tokenize_block(block),
                False, False,
            )
            out.append(body_heading_candidate)
    return out


# --------------------------------------------------------------------------- #
# Partition candidates and interleave clusters #
# --------------------------------------------------------------------------- #


def partition_candidates(heading_candidates: list[HeadingCandidate], other_outline_nodes: list[OutlineNode]) -> dict:
    """Partition candidates into labeled-compatible and remaining groups."""
    labeled_headings = [entry_item.heading for entry_item in other_outline_nodes]
    accepted_context = OutlineContext(labeled_headings)
    remaining: list[HeadingCandidate] = []
    for entry_item in heading_candidates:
        if is_compatible_with_context(accepted_context, entry_item):
            other_outline_nodes.append(OutlineNode(entry_item))
            accepted_context.add(entry_item)
        else:
            remaining.append(entry_item)
    other_outline_nodes.sort(key=lambda sort_node: heading_order_key(sort_node.heading))
    return {"remaining": remaining, "labeled": other_outline_nodes}


def interleave_clusters(heading_candidates: list[HeadingCandidate], other_outline_nodes: list[OutlineNode]) -> list[dict]:
    """Interleave general candidates between successive labeled headings. Returns clusters with the labeled heading and intervening candidates. """
    out: list[dict] = []
    index = 0
    previous: Optional[OutlineNode] = None
    acc: list[HeadingCandidate] = []
    for labeled_outline_node in other_outline_nodes:
        boundary_candidate = labeled_outline_node.heading
        while index < len(heading_candidates) and compare_heading_order(heading_candidates[index], boundary_candidate) < 0:
            acc.append(heading_candidates[index])
            index += 1
        if acc or previous is not None:
            out.append({"labeled_anchor": previous, "cluster_candidates": acc})
        acc = []
        previous = labeled_outline_node
    while index < len(heading_candidates):
        acc.append(heading_candidates[index])
        index += 1
    out.append({"labeled_anchor": previous, "cluster_candidates": acc})
    return out
