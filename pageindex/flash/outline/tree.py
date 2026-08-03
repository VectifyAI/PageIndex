"""Level assignment and outline tree construction."""

from __future__ import annotations

import re
from collections import defaultdict
from ..model import numbering_text, numbering_kind, block_text, is_caps_heavy, Block

from .filtering import (
    _style_key,
    _numbering_depth,
)


def extract_top_level_headings(headings: list[Block], levels: dict[int, int]) -> list[Block]:
    """Flatten the heading tree, returning only top-level headings."""
    return [heading for heading in headings if levels.get(id(heading), 6) <= 1]


def assign_levels(headings: list[Block]) -> dict[int, int]:
    """Return ``{id(block) -> level}``. 1. Bucket by style key. 2. Rank styles by (size DESC, bold DESC) and assign level 1..6 in that order (anything below the 6th distinct style is clamped to 6). 3. If a heading has digit-numbering, its level is overridden to min(numbering_depth, style_level) -- numbering wins for deeper grouping but never promotes a heading above its style rank. """
    buckets: dict[tuple[str, float, bool], list[Block]] = defaultdict(list)
    for state_item in headings:
        buckets[_style_key(state_item)].append(state_item)
    ranked = sorted(buckets.keys(), key=lambda key_value: (-key_value[1], not key_value[2]))
    style_level = {key_value: min(index_value + 1, 6) for index_value, key_value in enumerate(ranked)}
    out: dict[int, int] = {}
    for state_item in headings:
        lvl = style_level.get(_style_key(state_item), 6)
        depth = _numbering_depth(state_item)
        if depth is not None:
            lvl = max(1, min(lvl, depth))
        out[id(state_item)] = lvl
    return out


# --------------------------------------------------------------------------- #
# Tree assembly #
# --------------------------------------------------------------------------- #


def _heading_title(block: Block) -> str:
    """Cleaned title text for output (no dot leaders, single-line)."""
    text = block_text(block).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _heading_page_num(block: Block, page_lookup) -> int:
    """Find the 1-based page number that owns this block. ``page_lookup`` is a dict ``{id(block) -> page.u}`` precomputed by the caller for O(1) lookup. """
    return page_lookup.get(id(block), 1)


def build_tree(headings: list[Block], levels: dict[int, int], page_lookup, total_pages: int) -> list[dict]:
    """Assemble nested ``{title, start_index, end_index, nodes}`` tree."""
    if not headings:
        return []

    root: list[dict] = []
    stack: list[tuple[int, dict]] = []
    for state_item in headings:
        title = _heading_title(state_item)
        if not title:
            continue
        node = {
            "title": title,
            "start_index": _heading_page_num(state_item, page_lookup),
            "end_index": _heading_page_num(state_item, page_lookup),
            "nodes": [],
        }
        lvl = levels.get(id(state_item), 6)
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        if not stack:
            root.append(node)
        else:
            stack[-1][1]["nodes"].append(node)
        stack.append((lvl, node))

    # Fill end_index in DFS order.
    flat: list[dict] = []

    def _walk_nodes(nodes: list[dict]) -> None:
        for count_item in nodes:
            flat.append(count_item)
            _walk_nodes(count_item["nodes"])

    _walk_nodes(root)
    for index_value, count_item in enumerate(flat):
        next_start = flat[index_value + 1]["start_index"] if index_value + 1 < len(flat) else total_pages
        count_item["end_index"] = max(count_item["start_index"], next_start - 1 if next_start > count_item["start_index"] else count_item["start_index"])
    if flat:
        flat[-1]["end_index"] = max(flat[-1]["start_index"], total_pages)

    # Promote parent end_index to the subtree maximum: end_index covers the
    # whole section, children included.  The leading segment stays derivable
    # from the first child's start_index.
    def _promote(nodes: list[dict]) -> int:
        end = 0
        for child in nodes:
            if child["nodes"]:
                child["end_index"] = max(child["end_index"], _promote(child["nodes"]))
            end = max(end, child["end_index"])
        return end

    _promote(root)

    # Drop empty children so the JSON matches the shape the rest of PageIndex emits.
    def _drop_empty_children(nodes: list[dict]) -> list[dict]:
        for count_item in nodes:
            if count_item["nodes"]:
                _drop_empty_children(count_item["nodes"])
            else:
                del count_item["nodes"]
        return nodes

    return _drop_empty_children(root)


# --------------------------------------------------------------------------- #
# Outline validation #
# --------------------------------------------------------------------------- #


def validate(headings: list[Block], levels: dict[int, int], doc) -> bool:
    """Return whether the outline has enough top-level headings spanning a meaningful fraction of the document."""
    top = [state_item for state_item in headings if levels.get(id(state_item), 6) <= 2]
    if len(top) < 3:
        return False
    if len(top) >= 5:
        return True
    last_page = 1
    for state_item in top:
        # Direct page lookup would need a page back-reference; we use the document
        # order proxy (top is already in reading order).
        # Find by scanning document pages for the page containing the block.
        page_num = 1
        for page in doc.primary_slot:
            if state_item in (page.secondary_slot or []):
                page_num = page.page_index
                break
        if page_num - last_page > 0.5 * len(doc.primary_slot):
            return False
        last_page = page_num
    return True
