"""Clusters lines into blocks and splits heading-body blocks."""

from __future__ import annotations

from sortedcontainers import SortedKeyList

from ..model import (
    style_key,
    magnitude_ratio,
    left_aligned,
    right_aligned,
    center_aligned,
    x_centers_close,
    Rect,
    last_span,
    avg_char_width,
    EMPTY_RECT,
    left_edge_key,
    reading_order_key,
    numbering_kind,
    Line,
    case_signal,
    last_line_of,
    first_span_of,
    letter_count,
    dominant_style_of,
    is_upper_dominant,
    Block,
    _max_nan_propagating,
)
from ..tokens import set_case_fold, TrieConfig, build_trie, tokenize_block

from .join_rules import (
    SECTION_HEADING_TRIE,
    BlockClusterContext,
    should_join_line_to_block,
)


# --------------------------------------------------------------------------- #
# Two-line block split post-process #
# --------------------------------------------------------------------------- #


def split_heading_body_blocks(input_blocks: list[Block]) -> list[Block]:
    """Split blocks whose first line is a section heading followed by body text."""
    from ..labels import trie_matches_all, advance_past_line
    split_output_blocks: list[Block] = []
    for input_block in input_blocks:
        first_line = input_block.line()                            # first line
        # Skip blocks that obviously aren't "heading + body":
        # - 1-line blocks
        # - small/short blocks
        # - first-span style == last-span style AND wide first line
        if (
            input_block.line_count() <= 1
            or (input_block.bbox_height() >= 0.6 * input_block.bbox_width() and input_block.char_count() < 20 * input_block.line_count())
            or (style_key(first_span_of(input_block)) == style_key(last_span(last_line_of(input_block))) and first_line.bbox_width() > 0.5 * input_block.bbox_width())
        ):
            split_output_blocks.append(input_block)
            continue
        block_tokens = tokenize_block(input_block)
        first_line_tokens = block_tokens.slice(0, advance_past_line(block_tokens, first_line, 0))
        split_token = block_tokens.token_at(first_line_tokens.length)
        if split_token is None or split_token.primary_slot == 3:
            split_output_blocks.append(input_block)
            continue
        if not trie_matches_all(SECTION_HEADING_TRIE, first_line_tokens):
            split_output_blocks.append(input_block)
            continue
        # Split: first block holds the heading line; second holds the rest.
        split_heading_block = Block()
        split_heading_block.add_line(first_line)
        split_body_block = Block()
        for line_idx in range(1, input_block.line_count()):
            split_body_block.add_line(input_block.primary_slot[line_idx])
        split_output_blocks.append(split_heading_block)
        split_output_blocks.append(split_body_block)
    return split_output_blocks


# --------------------------------------------------------------------------- #
# Block-clustering driver #
# --------------------------------------------------------------------------- #


def _set_add(tree: SortedKeyList, block: Block) -> None:
    """Sorted-set insertion semantics: when another block has the same left-edge ordering key, the new block is ignored instead of kept as a multiset duplicate."""
    idx = tree.bisect_left(block)
    if idx < len(tree) and left_edge_key(tree[idx]) == left_edge_key(block):  # type: ignore[arg-type]
        return  # key collision -> sorted set.add drops the element
    tree.add(block)


def cluster_lines_into_blocks(ctx: BlockClusterContext) -> list[Block]:
    """Walk lines, extend existing blocks when compatible, otherwise open a block. Returns blocks sorted bottom, then top, then left, then right before reading-order assignment."""
    # Tree of *blocks* sorted by (left, right, top desc, bottom desc)
    tree: SortedKeyList = SortedKeyList(key=left_edge_key)
    clustered_blocks: list[Block] = []

    lines = ctx.secondary_slot
    line_count = len(lines)
    for line_index in range(line_count):
        candidate_line = lines[line_index]
        next_line = lines[line_index + 1] if line_index + 1 < line_count else None

        # The new line wrapped as a block (used as the tree key for lookups).
        seed_block = Block().add_line(candidate_line)

        # Collect candidate blocks whose horizontal interval overlaps e_line.
        # * predecessors: walk backwards from g_seed_block's left, gather
        # blocks whose right edge >= e_line.left.
        # * successors: walk forwards, gather blocks whose left edge <= e_line.right.
        candidate_blocks: list[Block] = []
        # Predecessors by decreasing block-order key.
        # Predecessor walk starts at the largest key <= the seed key.
        idx_pred = tree.bisect_right(seed_block)
        block = idx_pred - 1
        while block >= 0:
            existing_block: Block = tree[block]                          # type: ignore[assignment]
            if existing_block.right_edge() < candidate_line.left_edge():
                break
            candidate_blocks.append(existing_block)
            block -= 1
        # Successors by increasing block-order key.
        # Successor walk starts at the smallest key >= the seed key. An exact
        # key-equal node is intentionally visited by both walks.
        idx_succ = tree.bisect_left(seed_block)
        block = idx_succ
        while block < len(tree):
            existing_block = tree[block]                              # type: ignore[assignment]
            if existing_block.left_edge() > candidate_line.right_edge():
                break
            candidate_blocks.append(existing_block)
            block += 1

        # Sort candidates by bottom, then top, left, and right.
        candidate_blocks.sort(key=lambda block: (block.bottom_edge(), block.top_edge(), block.left_edge(), block.right_edge()))

        did_join = False
        # Capture the first candidate (closest) before mutating the list
        first_candidate = candidate_blocks[0] if candidate_blocks else None
        for existing_block in candidate_blocks:
            if not did_join and first_candidate is not None and should_join_line_to_block(
                ctx, existing_block, candidate_line, next_line, first_candidate
            ):
                # Join: remove m from tree, extend with e_line, re-add.
                try:
                    tree.remove(existing_block)
                except ValueError:
                    pass
                existing_block.add_line(candidate_line)
                _set_add(tree, existing_block)
                did_join = True
            else:
                # Doesn't take this line -- block is "closed", emit it.
                clustered_blocks.append(existing_block)
                try:
                    tree.remove(existing_block)
                except ValueError:
                    pass
        if not did_join:
            _set_add(tree, seed_block)

    # Drain remaining open blocks
    for block in tree:
        clustered_blocks.append(block)

    # Post-process to split 2-line "heading+body" blocks when the first line
    # matches section, abstract, or references keywords.
    clustered_blocks = split_heading_body_blocks(clustered_blocks)
    clustered_blocks.sort(key=reading_order_key)
    return clustered_blocks
