"""Final outline assembly and conversion to the output dict tree."""

from __future__ import annotations

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
    heading_order_key,
)
from .style_context import (
    OutlineState,
    compare_heading_depth,
)
from .cliques import (
    find_keyword_clique,
    CliqueFilterContext,
    detect_body_headings,
    partition_candidates,
    interleave_clusters,
)
from .selection import (
    should_reject_heading,
    push_heading_to_state,
    HierarchyStack,
    find_parent_heading,
    extract_sub_headings,
)


def mark_outline_block_types(item_list: list[OutlineNode]) -> None:
    """Mark outline blocks as numbered or unnumbered headings."""
    for block in item_list:
        block.heading.group_slot.type = 8 if block.heading.has_numbering else 7
        mark_outline_block_types(block.child_nodes)


def compute_max_heading_gap(outline_nodes: list[OutlineNode], other_number: int) -> dict:
    """Compute the maximum page-position gap between outline nodes."""
    if not outline_nodes:
        return {"max_gap": 0, "last_page_position": other_number}
    heading = 0
    for stack_outline_node in outline_nodes:
        page_pos = stack_outline_node.heading.page.page_index + stack_outline_node.heading.auxiliary_slot
        heading = max(heading, page_pos - other_number)
        other_number = page_pos
        rec = compute_max_heading_gap(stack_outline_node.child_nodes, other_number)
        heading = max(heading, rec["max_gap"])
        other_number = rec["last_page_position"]
    return {"max_gap": heading, "last_page_position": other_number}


def has_table_or_prominent(outline_nodes: list[OutlineNode]) -> bool:
    """Return True if any heading is a table-like or prominent entry."""
    return any(secondary_item.heading.type == 5 or secondary_item.heading.is_prominent for secondary_item in outline_nodes)


def is_landscape_or_empty(doc) -> bool:
    """Return True for mostly-landscape or near-empty documents with little outline text."""
    if doc.secondary_slot.secondary_slot >= 1e3:
        return False
    secondary_item = 0
    candidate_item = 0.0
    for page in doc.primary_slot:
        if page.bounds.bbox_width() > page.bounds.bbox_height() and page.primary_slot.secondary_slot < 1e3:
            secondary_item += 1
            candidate_item += page.primary_slot.secondary_slot
    count_item = len(doc.primary_slot)
    return secondary_item >= 0.9 * count_item or (secondary_item >= 0.7 * count_item and candidate_item >= 0.5 * doc.secondary_slot.state_slot)


# --------------------------------------------------------------------------- #
# Build a heading candidate from a block #
# --------------------------------------------------------------------------- #


def build_heading_from_block(block: Block, page, anchor: Optional[Block] = None) -> HeadingCandidate:
    """Build a heading candidate wrapper for a heading block."""
    tokens = tokenize_block(block)
    # Extract structural numbering from the leading line.
    item_list: list[int] = []
    has_numbering = False
    prefix: Optional[TokenView] = None
    title: TokenView = tokens
    if numbering_kind(block.line()) == 1:
        num_str = numbering_text(block.line())
        if num_str:
            try:
                parts = [int(number_part) for number_part in num_str.replace("．", ".").split(".") if number_part.strip()]
                if all(0 <= number_part < 1000 for number_part in parts):
                    item_list = parts
                    has_numbering = True
                    # Strip the leading number tokens from g
                    skip = 0
                    while skip < tokens.length:
                        tok = tokens.token_at(skip)
                        if tok is None:
                            break
                        if tok.type == 1 or tok.str in ".．":
                            skip += 1
                        else:
                            break
                    title = tokens.slice(skip)
            except (ValueError, AttributeError):
                pass

    # Type from labeled-section classification or from numbering.
    marker_type = getattr(block, "marker_slot", 0) or 0
    if marker_type == 4:
        type_ = 4
    elif marker_type == 5:
        type_ = 5
    elif marker_type == 11:
        type_ = 11
    elif item_list:
        type_ = 1
    elif is_caps_heavy(block) and block.line_count() == 1:
        type_ = 2                  # uppercase short heading
    else:
        type_ = 0

    # Prominence flag: big font / bold-and-prominent.
    body_size_threshold = page.primary_slot.primary_slot + 0.5 if page.primary_slot else 0
    ja_flag = (
        block.avg_font_size() > body_size_threshold + 1.5
        or (block.bold_frac() > 0.5 and block.avg_font_size() >= body_size_threshold)
    )

    return HeadingCandidate(
        type_=type_,
        page=page,
        group_value=block,
        anchor=anchor,
        numbering_value=item_list,
        tokens=prefix,
        title_tokens=title,
        has_numbering_flag=has_numbering,
        prominent_flag=ja_flag,
    )


# --------------------------------------------------------------------------- #
# Main outline assembler #
# --------------------------------------------------------------------------- #


def assemble_outline(doc, labeled: list[OutlineNode]) -> list[OutlineNode]:
    """Produce the outline tree as a list of outline nodes. Arguments: ``doc`` is the document state; ``labeled`` is the list of outline nodes wrapping labeled headings. Output is a list of root outline nodes. Each node contains child nodes recursively. """
    # ----- Stage 1: collect general headings.
    from ..heading_detection import build_doc_heading_candidates
    # Labeled headings prime the type gates used by general heading filtering.
    general: list[HeadingCandidate] = build_doc_heading_candidates(doc, labeled)

    # ----- Stage 2: merge with labeled
    if len(labeled) + len(general) > 0:
        combined = list(general)
        for labeled_region_node in labeled:
            combined.append(labeled_region_node.heading)
        combined.sort(key=heading_order_key)
        # Build the keyword clique before body-heading filtering so the filter
        # can test whether a block is already represented in the candidate tree.
        clique = find_keyword_clique(combined)
        filtered = detect_body_headings(CliqueFilterContext(doc, combined, lambda line, other_line: compare_heading_depth(line, other_line, clique)))
        general.extend(filtered)
        # No dedup here: duplicate candidates that wrap the same block are
        # collapsed downstream by partitioning and already-placed-block checks.
        general = sorted(general, key=heading_order_key)

    # ----- Stage 3: partition + cluster
    if labeled:
        # No pre-filter: partitioning re-separates labeled vs general, so any
        # labeled block backfilled into the general list is handled there.
        result = partition_candidates(general, labeled)
        general = result["remaining"]
        labeled = result["labeled"]
        clusters = interleave_clusters(general, labeled)
    else:
        clusters = [{"labeled_anchor": None, "cluster_candidates": general}]

    # ----- Stage 4: assemble tree
    state = OutlineState(clusters)
    if not state.measure_slot and state.option_slot <= state.previous_slot:
        return []

    out: list[OutlineNode] = []
    for cluster in clusters:
        cluster_anchor = cluster.get("labeled_anchor")
        cluster_candidates = cluster.get("cluster_candidates", [])
        if cluster_anchor is not None:
            push_heading_to_state(state, cluster_anchor.heading)
            out.append(cluster_anchor)
        sub = extract_sub_headings(doc, state, cluster_anchor, cluster_candidates)
        target = cluster_anchor.child_nodes if cluster_anchor is not None else out
        target.extend(sub)

        sub_clique = find_keyword_clique(cluster_candidates) if cluster_candidates else None
        stack = HierarchyStack(sub_clique)
        for insertion_candidate in cluster_candidates:
            if should_reject_heading(state, insertion_candidate):
                continue
            push_heading_to_state(state, insertion_candidate)
            insertion_candidate.group_slot.used_as_heading = True
            stack_outline_node = OutlineNode(insertion_candidate)
            parent = find_parent_heading(stack, insertion_candidate)
            if parent is not None:
                parent.child_nodes.append(stack_outline_node)
            elif cluster_anchor is not None:
                cluster_anchor.child_nodes.append(stack_outline_node)
            else:
                out.append(stack_outline_node)
            stack.push(stack_outline_node)
    return out


# --------------------------------------------------------------------------- #
# Outline tree -> PageIndex dict tree #
# --------------------------------------------------------------------------- #


def _flatten_outline_nodes(outline_node_list: list[OutlineNode]) -> list[OutlineNode]:
    """Walk an outline tree DFS to a flat list, preserving order."""
    out: list[OutlineNode] = []

    def _walk_nodes(items: list[OutlineNode]) -> None:
        for item in items:
            out.append(item)
            if item.child_nodes:
                _walk_nodes(item.child_nodes)

    _walk_nodes(outline_node_list)
    return out


def _heading_appears_at_page_top(heading: HeadingCandidate) -> bool:
    """Return whether a heading begins its page with no flowing content above it."""
    top_heading = heading.group_slot
    page = heading.page
    if top_heading is None or page is None:
        return True
    group_index = getattr(top_heading, "reading_order_index", 0)
    for block in (page.secondary_slot or []):
        if block is top_heading or getattr(block, "reading_order_index", 0) >= group_index:
            continue                     # only blocks before the heading
        if block.char_count() <= 0:
            continue                     # no text
        if block.type in (1, 2, 12):       # header / footer / watermark
            continue
        return False                     # real content precedes the heading
    return True


def outline_to_dict_tree(outline_node_list: list[OutlineNode], total_pages: int) -> list[dict]:
    """Convert the outline tree directly to PageIndex JSON shape. Preserves the natural outline nesting without font-overlay rewriting. """
    flat_nodes: list[dict] = []

    def _walk_nodes(items: list[OutlineNode]) -> list[dict]:
        result: list[dict] = []
        for item in items:
            # Title text is the numbering prefix plus the heading tokens, but
            # the two are carried as separate fields and trimmed one by one,
            # then rejoined with a single space and only for a non-empty
            # prefix. A prefix's string form ends in a space after every
            # space-flagged token, so trimming the parts separately is what
            # keeps that space out of the join.
            # Trim with the Unicode WhiteSpace+LineTerminator set, not Python's
            # str.strip set: they differ on U+FEFF, U+0085, and U+001C-1F.
            prefix_tokens = item.heading.secondary_slot
            token = item.heading.primary_slot
            child = _trim_unicode_ws(str(prefix_tokens)) if prefix_tokens is not None else ""
            node = _trim_unicode_ws(str(token)) if token is not None else ""
            title = (child + " " if child else "") + node
            if not title:
                if item.child_nodes:
                    result.extend(_walk_nodes(item.child_nodes))
                continue
            node = {
                "title": title,
                "node_id": "",
                "start_index": item.heading.page.page_index,
                "end_index": item.heading.page.page_index,
                "nodes": _walk_nodes(item.child_nodes) if item.child_nodes else [],
                "_appear_start": _heading_appears_at_page_top(item.heading),
            }
            flat_nodes.append(node)
            result.append(node)
        return result

    root = _walk_nodes(outline_node_list)

    # Fill end_index via DFS-order next-start - 1; last node extends to doc end.
    flat: list[dict] = []

    def _collect(nodes: list[dict]) -> None:
        for count_item in nodes:
            flat.append(count_item)
            _collect(count_item["nodes"])

    _collect(root)
    for line, outline_entry in enumerate(flat):
        if line + 1 < len(flat):
            nxt = flat[line + 1]
            # page_index post_processing (utils.post_processing): if the next
            # heading starts at the top of its page, this section ends the page
            # before it; otherwise the next heading sits below this section's
            # tail, so the two share that boundary page and the end extends onto
            # it.
            boundary = (
                nxt["start_index"] - 1
                if nxt["_appear_start"]
                else nxt["start_index"]
            )
        else:
            boundary = total_pages
        outline_entry["end_index"] = max(
            outline_entry["start_index"],
            boundary if boundary > outline_entry["start_index"] else outline_entry["start_index"],
        )
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

    # Stable DFS pre-order node ids, zero-padded to 4 (PageIndex convention;
    # uses zero-padded depth-first ids). Drop the
    # transient appear_start marker now that end_index is settled.
    for line, outline_entry in enumerate(flat):
        outline_entry["node_id"] = str(line).zfill(4)
        del outline_entry["_appear_start"]

    def _drop_empty_children(nodes: list[dict]) -> list[dict]:
        for count_item in nodes:
            if count_item["nodes"]:
                _drop_empty_children(count_item["nodes"])
            else:
                del count_item["nodes"]
        return nodes

    return _drop_empty_children(root)
