"""Caption region growth, deduplication, and detection."""

from __future__ import annotations

from typing import Optional

from ..classification import FIGURE_KEYWORDS_TRIE, TABLE_KEYWORDS_TRIE, CHART_KEYWORDS_TRIE
from ..model import (
    Rect, rect_union, extend_top_to, extend_bottom_to, EMPTY_RECT, Bounded,
    _trim_unicode_ws,
    center_aligned, last_span, heading_score, reading_order_key, numbering_text, Line, last_line_of, first_span_of, dominant_style_of, info_weight, Block,
)
from ..stats import column_index_of
from ..tokens import Token, TokenView, wrap_tokens, enumerate_tokens, last_token, trie_prefix_match, strip_leading_if_in, first_token, set_case_fold, TrieConfig, build_trie, tokenize_block, BuiltTrie, is_word_token

from .caption_text import (
    PERIOD_CHARS,
    extract_structural_number,
    format_caption_label,
    REFERENCE_PHRASE_TRIE,
    caption_outranks,
)


# --------------------------------------------------------------------------- #
# Captioned/labeled region wrapper #
# --------------------------------------------------------------------------- #


class CaptionedRegion(Bounded):
    """Captioned or labeled region plus its body blocks. The region stores the document context, page, heading block, body blocks, neighboring block reference, label flag, label type, and an area-weighted score used to choose forward vs backward extension."""

    __slots__ = ("weighted_ratio_primary", "page", "primary_slot", "output_slot", "state_slot", "alignment_slot", "type", "score")

    def __init__(self, primary_item, secondary_item, candidate_item, bbox: Rect, blocks, next_item, flag):
        super().__init__(bbox)
        self.weighted_ratio_primary = primary_item
        self.page = secondary_item
        self.primary_slot = candidate_item                              # the original heading block
        self.output_slot = blocks                       # list of body blocks
        self.state_slot = next_item
        self.alignment_slot = flag
        # Caption label type is carried by the heading block marker.
        # ``Block.type`` is a later classification label and is still zero here.
        self.type = candidate_item.marker_slot
        # Region score formula.
        area_pct = 100.0 * self.area() / self.page.bounds.area() if self.page.bounds.area() > 0 else 0.0
        if area_pct <= 0:
            score = 0.0
        else:
            if (self.state_slot is not None
                    and self.state_slot.top_edge() < self.top_edge()
                    and self.state_slot.right_edge() > self.left_edge()
                    and self.alignment_slot):
                area_pct /= 5.0
            if self.type == 4:
                inner = 0.0
                for block in self.output_slot:
                    if block.skew_frac() > 1:
                        continue
                    inner += block.area()
                score = area_pct * max(0.1, 1 - inner / self.area()) if self.area() > 0 else 0.0
            else:
                # Span text is a string, so every span contributes its character
                # count to the caption-region score.
                count = 1.0
                for block in self.output_slot:
                    for line in block:
                        for span in line:
                            count += span.char_count()
                score = count * area_pct
        self.score = score


# --------------------------------------------------------------------------- #
# Deduplicate caption entries and keep the best entry for each label.
# --------------------------------------------------------------------------- #


def dedupe_caption_entries(caption_context: "CaptionContext") -> list["CaptionEntry"]:
    """Deduplicate structural-number entries by label while preserving page order."""
    if not caption_context.state_slot:
        return caption_context.auxiliary_slot
    captions_by_label: dict[str, CaptionEntry] = {}
    for caption in caption_context.auxiliary_slot:
        if len(caption.primary_slot) <= 1:
            continue
        existing = captions_by_label.get(caption.primary_slot)
        if existing is None or caption_outranks(caption, existing):
            captions_by_label[caption.primary_slot] = caption
    out = list(captions_by_label.values())
    out.sort(key=lambda caption_sort_key: (caption_sort_key.page_index, caption_sort_key.group_slot.reading_order_index))
    return out


# --------------------------------------------------------------------------- #
# Extend a labeled section forward or backward.
# --------------------------------------------------------------------------- #


def extend_caption_region(
    caption_context: "CaptionContext",
    entry: "CaptionEntry",
    prior_regions: list,
    page_set: Optional[set],
    direction: int,
) -> Optional[CaptionedRegion]:
    """Walk page blocks forward or backward from a labeled entry, accumulating a region until an already-classified block, claimed block, deep body block, fresh top-level heading, or size/gap boundary is reached."""
    page = caption_context.primary_slot.primary_slot[entry.page_index - 1]
    origin = entry.group_slot
    anchor = origin.bottom_edge() if direction > 0 else origin.top_edge()
    bbox = Rect(origin.left_edge(), origin.right_edge(), anchor, anchor)
    blocks: list[Block] = []
    sorted_value = page.secondary_slot
    index = entry.group_slot.reading_order_index + direction
    previous: Block = origin

    while 0 <= index < len(sorted_value):
        caption = sorted_value[index]
        caption_column = column_index_of(caption)
        if caption_column < 0:
            break
        # layout branch: when crossing the column band, walk page.j (column
        # rects) to the nearest column that horizontally overlaps the
        # bbox and extend the bbox vertically to that column's edge.
        if direction < 0 and caption_column < column_index_of(entry.group_slot) and caption.bottom_edge() < anchor:
            col_idx = caption_column - 1
            column_rect = page.tertiary_slot[col_idx] if 0 <= col_idx < len(page.tertiary_slot) else None
            while column_rect is not None and (
                column_rect.bottom_edge() < bbox.top_edge()
                or column_rect.right_edge() < bbox.left_edge()
                or column_rect.left_edge() > bbox.right_edge()
            ):
                col_idx -= 1
                column_rect = page.tertiary_slot[col_idx] if 0 <= col_idx < len(page.tertiary_slot) else None
            if column_rect is not None:
                bbox = extend_top_to(bbox, column_rect.bottom_edge())
            else:
                bbox = extend_top_to(bbox, page.bounds.top_edge())
            break
        if direction > 0 and caption_column > column_index_of(entry.group_slot) and caption.top_edge() > anchor:
            col_idx = caption_column + 1
            column_rect = page.tertiary_slot[col_idx] if 0 <= col_idx < len(page.tertiary_slot) else None
            while column_rect is not None and (
                column_rect.top_edge() > bbox.bottom_edge()
                or column_rect.right_edge() < bbox.left_edge()
                or column_rect.left_edge() > bbox.right_edge()
            ):
                col_idx += 1
                column_rect = page.tertiary_slot[col_idx] if 0 <= col_idx < len(page.tertiary_slot) else None
            if column_rect is not None:
                bbox = extend_bottom_to(bbox, column_rect.top_edge())
            else:
                bbox = extend_bottom_to(bbox, page.bounds.bottom_edge())
            break
        # Grow the bbox to include n
        if direction < 0:
            bbox = extend_top_to(bbox, caption.bottom_edge())
        else:
            bbox = extend_bottom_to(bbox, caption.top_edge())
        # Stop conditions
        if caption.type != 0 or caption.reading_order_index in caption_context.secondary_slot:
            break
        if page_set is not None and index in page_set:
            break
        size = min(caption_context.primary_slot.secondary_slot.primary_slot, entry.group_slot.avg_font_size())
        if caption.is_body_paragraph and caption.avg_font_size() > min(0.9 * size, size - 1.5):
            break
        next_block = sorted_value[index + 1] if index + 1 < len(sorted_value) else None
        gap = previous.bottom_edge() - caption.top_edge() if direction > 0 else 0
        line_gap = page.primary_slot.tertiary_slot - page.primary_slot.primary_slot
        if (
            direction > 0 and next_block is not None and caption.line_count() <= 4 and caption.char_stats.secondary_slot != 3
            and gap > line_gap
            and (previous is entry.group_slot or gap > min(3 * line_gap, caption.bottom_edge() - next_block.top_edge()))
        ):
            next_item = sorted_value[index + 2] if index + 2 < len(sorted_value) else None
            if heading_score(caption) >= heading_score(previous) + 0.5 and (next_block.is_body_paragraph or (next_item is not None and next_item.is_body_paragraph)):
                break
            # A numbering-like line with enough trailing text can stop this
            # backward body-paragraph scan.
            line_text = numbering_text(caption.line())
            if (line_text
                    and caption.char_stats.secondary_slot == 2
                    and heading_score(caption) >= size
                    and gap > 2 * caption.avg_font_size()
                    and caption.char_count() - len(line_text) > 2):
                break
        blocks.append(caption)
        bbox = rect_union(bbox, caption.secondary_slot)
        index += direction
        previous = caption

    if direction < 0 and index < 0:
        bbox = extend_top_to(bbox, page.bounds.top_edge())
    elif direction > 0 and index >= len(sorted_value):
        bbox = extend_bottom_to(bbox, page.bounds.bottom_edge())

    # When a backward extension expands the region, also consume forward
    # neighbours whose geometric center sits inside the grown bbox.
    if direction < 0:
        fwd_idx = entry.group_slot.reading_order_index + 1
        while fwd_idx < len(sorted_value):
            block = sorted_value[fwd_idx]
            center_x = block.center_x()
            center_y = block.center_y()
            if (center_x < bbox.left_edge() or center_x > bbox.right_edge()
                    or center_y < bbox.bottom_edge() or center_y > bbox.top_edge()):
                break
            blocks.append(block)
            bbox = rect_union(bbox, block.secondary_slot)
            fwd_idx += 1

    area = bbox.area()
    if area <= 0:
        return None

    # Check overlap with prior regions; if heavy overlap, reject.
    for prior in prior_regions:
        overlap_area = max(
            0.0,
            min(bbox.right, prior.secondary_slot.right) - max(bbox.left, prior.secondary_slot.left),
        ) * max(
            0.0,
            min(bbox.top, prior.secondary_slot.top) - max(bbox.primary_slot, prior.secondary_slot.primary_slot),
        )
        if overlap_area >= 0.25 * min(area, prior.area()):
            return None

    next_block = sorted_value[index] if 0 <= index < len(sorted_value) else None
    on_page_set = page_set is not None and index in page_set
    return CaptionedRegion(
        primary_item=caption_context.primary_slot, secondary_item=page, candidate_item=entry.group_slot,
        bbox=bbox, blocks=blocks, next_item=next_block, flag=on_page_set,
    )


# --------------------------------------------------------------------------- #
# Extend all deduplicated labeled-section entries.
# --------------------------------------------------------------------------- #


def build_caption_regions(caption_context: "CaptionContext") -> list[CaptionedRegion]:
    """Build caption regions by extending each labeled entry in both directions."""
    caption_context.tertiary_slot.clear()
    caption_context.secondary_slot.clear()
    entries = dedupe_caption_entries(caption_context)
    for caption in entries:
        set_value = caption_context.tertiary_slot.get(caption.page_index)
        if set_value is None:
            set_value = set()
            caption_context.tertiary_slot[caption.page_index] = set_value
        set_value.add(caption.group_slot.reading_order_index)
    out: list[CaptionedRegion] = []
    page = 0
    prior_regions: list[CaptionedRegion] = []
    for entry in entries:
        if entry.page_index != page:
            prior_regions = []
            caption_context.secondary_slot.clear()
            page = entry.page_index
        if len(prior_regions) >= 8:
            continue
        page_set = caption_context.tertiary_slot.get(entry.page_index)
        back = extend_caption_region(caption_context, entry, prior_regions, page_set, -1)
        forward = extend_caption_region(caption_context, entry, prior_regions, page_set, 1)
        winner = (
            back if (back is not None and (forward is None or back.score > forward.score))
            else forward
        )
        if winner is not None:
            for body_block in winner.output_slot:
                caption_context.secondary_slot.add(body_block.reading_order_index)
            prior_regions.append(winner)
            out.append(winner)
    return out


# --------------------------------------------------------------------------- #
# Labeled-section entry.
# --------------------------------------------------------------------------- #


class CaptionEntry:
    """One labeled-section entry with label, type, page, block, and remainder tokens."""

    __slots__ = ("primary_slot", "type", "page_index", "group_slot", "secondary_slot")

    def __init__(self, label: str, type_: int, page: int, block: Block, remainder: TokenView):
        self.primary_slot = label
        self.type = type_
        self.page_index = page
        self.group_slot = block
        self.secondary_slot = remainder


# --------------------------------------------------------------------------- #
# Labeled-section context.
# --------------------------------------------------------------------------- #


class CaptionContext:
    """Document-level state for labeled-section detection."""

    __slots__ = ("primary_slot", "auxiliary_slot", "state_slot", "tertiary_slot", "secondary_slot")

    def __init__(self, doc):
        self.primary_slot = doc
        self.auxiliary_slot: list[CaptionEntry] = []
        self.state_slot: bool = False
        self.tertiary_slot: dict = {}            # page -> set of heading-block ga
        self.secondary_slot: set = set()          # set of heading-block ga across doc


# --------------------------------------------------------------------------- #
# Document-wide (page, block) iterator.
# --------------------------------------------------------------------------- #


def iter_page_blocks(doc):
    """Yield ``{'page': page, 'G': block}`` records in reading order."""
    for page in doc.primary_slot:
        for block in (page.secondary_slot or []):
            yield {"page": page, "block": block}


# --------------------------------------------------------------------------- #
# Labeled-section detection driver.
# --------------------------------------------------------------------------- #


def detect_captions(caption_context: CaptionContext) -> None:
    """Find figure, table, and chart labels and record their structural prefixes."""
    for entry in iter_page_blocks(caption_context.primary_slot):
        page = entry["page"]
        block = entry["block"]
        if block.type != 0:
            continue
        tokens = tokenize_block(block)
        type_value: Optional[int] = None
        prefix = trie_prefix_match(FIGURE_KEYWORDS_TRIE, tokens)
        if prefix is not None:
            type_value = 4
        else:
            prefix = trie_prefix_match(TABLE_KEYWORDS_TRIE, tokens)
            if prefix is not None:
                type_value = 5
            else:
                prefix = trie_prefix_match(CHART_KEYWORDS_TRIE, tokens)
                if prefix is not None:
                    type_value = 11
        if type_value is None:
            continue

        remainder = strip_leading_if_in(tokens.slice(prefix.length), PERIOD_CHARS)
        number = extract_structural_number(remainder)
        label = format_caption_label(type_value, number)
        if number is not None:
            caption_context.state_slot = True
            remainder = remainder.slice(number.length)
        if trie_prefix_match(REFERENCE_PHRASE_TRIE, remainder) is not None:
            continue
        page.measure_slot = True
        caption_context.auxiliary_slot.append(CaptionEntry(label, type_value, page.page_index, block, remainder))
        # Mark the block's Y category (used by outline.py heading filter)
        block.marker_slot = type_value
