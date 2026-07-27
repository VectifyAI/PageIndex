"""Header and footer detection via cross-page recurrence."""

from __future__ import annotations

import math
from typing import Optional

from ..model import (
    _UNICODE_WHITESPACE_CLASS,
    _strip_diacritics,
    _round_half_up_to_int,
    magnitude_ratio,
    intervals_overlap,
    y_overlaps,
    center_aligned,
    to_number,
    last_span,
    heading_score,
    text_of_line,
    Line,
    last_line_of,
    first_span_of,
    is_word_category,
    block_text,
    deaccented_text,
    letter_count,
    dominant_style_of,
    punct_count,
    info_weight,
    is_upper_dominant,
    is_caps_heavy,
    alignment_code,
    Block,
)
from ..stats import style_key, DocStats, weighted_percentile, column_index_of, char_script_bucket
from ..tokens import (
    is_trimmable_token,
    token_numeric_value,
    Token,
    TokenView,
    wrap_tokens,
    enumerate_tokens,
    jenkins_hash,
    trie_prefix_match,
    strip_trie_match,
    strip_leading_if_in,
    COMMA_CHARS,
    strip_trailing_comma,
    trim_trailing_punct,
    set_case_fold,
    TrieConfig,
    build_trie,
    LineTokenizer,
    tokenize_block,
    BuiltTrie,
    trie_full_match,
    is_char_token,
    is_word_token,
)

from .keyword_tables import (
    COPYRIGHT_TRIE,
    VOLUME_WORDS_TRIE,
    FIGURE_KEYWORDS_TRIE,
    TABLE_KEYWORDS_TRIE,
    CHART_KEYWORDS_TRIE,
    _search_trie,
)
from .body_text import (
    record_recurring_text,
    is_body_paragraph,
    span_style_text_key,
    normalized_block_text,
    span_page_number,
    longest_word_and_number,
)


class PageMarkState:
    """Per-page classification state: first classified index, max heading score, and classified character count."""

    __slots__ = ("primary_slot", "secondary_slot", "tertiary_slot")

    def __init__(self):
        self.primary_slot = -1
        self.secondary_slot = 0
        self.tertiary_slot = 0


def record_marked_block(state: PageMarkState, idx: int, block: Block) -> None:
    """Update per-page state after classifying ``block``."""
    state.primary_slot = idx
    state.secondary_slot = max(state.secondary_slot, heading_score(block))
    state.tertiary_slot += block.char_count()


def is_header_positioned(ctx, other_block: Block, candidate_block: Optional[Block]) -> bool:
    """Return whether a block is header-positioned relative to the reference block, with content-density gates."""
    if candidate_block is None:
        cond = True
    elif ctx.primary_slot == 1:
        cond = other_block.top_edge() > candidate_block.bottom_edge()
    else:
        cond = other_block.bottom_edge() < candidate_block.top_edge()
    return cond and other_block.line_count() == 1 and info_weight(other_block.char_stats) >= 8 and letter_count(other_block.char_stats) >= 5 and other_block.char_stats.primary_slot[1] >= 1


def has_adjacent_page_numbers(ctx, page: int, candidate_number: int, reference_flag: bool) -> bool:
    """Return whether nearby pages show a strong ``n±1`` / ``n±2`` / ``n±4`` page-number pattern."""
    page_index = page - 1
    page_count = len(ctx.secondary_slot.primary_slot)
    adjacent_one = (
        (page_index - 1 >= 0 and (candidate_number - 1) in ctx.tertiary_slot[page_index - 1])
        or (page_index + 1 < page_count and (candidate_number + 1) in ctx.tertiary_slot[page_index + 1])
    )
    adjacent_two = (
        (page_index - 2 >= 0 and (candidate_number - 2) in ctx.tertiary_slot[page_index - 2])
        or (page_index + 2 < page_count and (candidate_number + 2) in ctx.tertiary_slot[page_index + 2])
    )
    if not adjacent_one and not adjacent_two:
        return False
    if adjacent_one and adjacent_two:
        return True
    adjacent_four = (
        (page_index - 4 >= 0 and (candidate_number - 4) in ctx.tertiary_slot[page_index - 4])
        or (page_index + 4 < page_count and (candidate_number + 4) in ctx.tertiary_slot[page_index + 4])
    )
    if candidate_number > page / 2 - 30:
        return adjacent_one or (not reference_flag and adjacent_two) or (adjacent_two and adjacent_four)
    return bool(adjacent_two and adjacent_four)


def mark_header_footer(ctx, other_block: Block) -> None:
    """mark block as classified + bump ghost-text count."""
    record_recurring_text(ctx.secondary_slot, deaccented_text(other_block))
    other_block.type = ctx.primary_slot


def walk_from_page_edge(ctx, blocks: list[Block], callback) -> None:
    """direction-aware iteration. HEADER (g=1) walks blocks in normal order from top; FOOTER (g=2) walks in reverse from bottom. ``callback`` returns True to halt. """
    if ctx.primary_slot == 1:
        for page in blocks:
            if callback(page):
                break
    else:
        block_index = len(blocks) - 1
        while block_index >= 0:
            if callback(blocks[block_index]):
                break
            block_index -= 1


def find_cross_page_match(ctx, page, block: Block, text_key: str, ref: Block) -> Optional[Block]:
    """Find a matching block on a nearby page by exact normalized text, then by longest word/number pieces."""
    entries = ctx.auxiliary_slot.get(text_key) or []
    for entry in entries:
        entry_page_index = entry["page_index"]
        entry_block: Block = entry["block"]
        if entry_page_index < page.page_index - 3:
            continue
        if entry_page_index == page.page_index:
            continue
        if entry_page_index > page.page_index + 3:
            break
        distance_sq = entry_block.left_edge() - block.left_edge()
        left_delta = entry_block.top_edge() - block.top_edge()
        right_delta = entry_block.right_edge() - block.right_edge()
        bottom_delta = entry_block.bottom_edge() - block.bottom_edge()
        distance_sq = distance_sq * distance_sq + left_delta * left_delta + right_delta * right_delta + bottom_delta * bottom_delta
        size = page.primary_slot.primary_slot
        if not (
            distance_sq >= 100
            or (distance_sq >= 1 and (
                (page.page_index == 1 and heading_score(block) >= size + 0.5)
                or (entry_page_index == 1 and heading_score(entry_block) >= size + 0.5)
            ))
        ):
            return entry_block

    if is_header_positioned(ctx, block, ref):
        for key in longest_word_and_number(block):
            map_value = ctx.measure_slot.get(key)
            if map_value is None or len(map_value) < max(4, len(ctx.secondary_slot.primary_slot) / 4):
                continue
            target = heading_score(block)
            for nearby_page_index in range(page.page_index - 2, page.page_index + 3):
                if nearby_page_index == page.page_index:
                    continue
                nearby_entry = map_value.get(nearby_page_index)
                if nearby_entry is None:
                    continue
                body_font_size = page.primary_slot.primary_slot
                if (abs(target - heading_score(nearby_entry["block"])) > 1
                        or (page.page_index == 1 and target >= body_font_size + 0.5)
                        or (nearby_page_index == 1 and heading_score(nearby_entry["block"]) >= body_font_size + 0.5)):
                    continue
                threshold = min(len(text_key), len(nearby_entry["text_key"])) / 5
                if bounded_edit_distance(text_key, nearby_entry["text_key"], threshold) >= threshold:
                    continue
                return nearby_entry["block"]
    return None


# --------------------------------------------------------------------------- #
# Header/footer detection context #
# --------------------------------------------------------------------------- #


class HeaderFooterContext:
    """Per-pass header/footer state."""

    __slots__ = ("secondary_slot", "primary_slot", "previous_slot", "option_slot", "tertiary_slot", "auxiliary_slot", "measure_slot", "state_slot")

    def __init__(self, doc, candidate_number: int):
        self.secondary_slot = doc
        self.primary_slot = candidate_number
        self.previous_slot = "HEADER" if candidate_number == 1 else "FOOTER"
        self.option_slot: dict[str, int] = {}             # span style/text key -> page count
        self.tertiary_slot: list[set[int]] = []             # per-page page-number set
        self.auxiliary_slot: dict[str, list[dict]] = {}      # normalized text key -> location/block records
        self.measure_slot: dict[str, dict[int, dict]] = {} # word/number key -> page -> text/block record
        self.state_slot: list[list[Block]] = []             # per-page candidate blocks


# --------------------------------------------------------------------------- #
# Bounded edit distance for fuzzy block-key comparison. #
# --------------------------------------------------------------------------- #


def bounded_edit_distance(text: str, other_text: str, candidate_item: float) -> float:
    """Bounded banded edit distance. Returns the limit when the strings differ by more than that many edits; otherwise returns the exact Levenshtein distance."""
    candidate_item = max(len(text), len(other_text)) if candidate_item <= 0 else math.ceil(candidate_item)
    if len(text) <= 0:
        return min(len(other_text), candidate_item)
    if len(other_text) <= 0:
        return min(len(text), candidate_item)
    if len(text) < len(other_text):
        text, other_text = other_text, text                       # a is the longer string (columns)
    if len(text) - len(other_text) >= candidate_item:
        return candidate_item
    reference_item = 0                                 # leftmost band column
    entry_item = 0                                 # rightmost band column
    score_value = [0] * (len(text) + 1)                # previous row
    group_value = [0] * (len(text) + 1)                # current row
    for state_item in range(len(text) + 1):           # seed row 0, but only out to column c
        score_value[state_item] = state_item
        if state_item > candidate_item:
            break
        entry_item = state_item
    for state_item in range(1, len(other_text) + 1):
        compare_char = other_text[state_item - 1]
        key_value = len(text)                        # leftmost column kept < c this row
        measure_item = 0                             # rightmost column kept < c this row
        for line_value in range(reference_item, min(entry_item + 1, len(text)) + 1):
            if line_value == reference_item:
                group_value[line_value] = 1 + score_value[line_value]
            elif text[line_value - 1] == compare_char:
                group_value[line_value] = score_value[line_value - 1]
            else:
                group_value[line_value] = 1 + min(group_value[line_value - 1], score_value[line_value - 1])
                if line_value <= entry_item:
                    group_value[line_value] = min(group_value[line_value], 1 + score_value[line_value])
            if group_value[line_value] < candidate_item:
                key_value = min(key_value, line_value)
                measure_item = line_value
        if key_value > measure_item:                         # whole band reached c -> distance >= c
            return candidate_item
        score_value, group_value = group_value, score_value
        reference_item = key_value
        entry_item = measure_item
    return min(score_value[entry_item] + len(text) - entry_item, candidate_item)


# --------------------------------------------------------------------------- #
# Header / footer detection #
# --------------------------------------------------------------------------- #


def detect_header_footer(ctx: HeaderFooterContext) -> None:
    """Run the three-pass header/footer detector."""
    # ----- Pass 1: per-page candidate collection -----------------------
    for page in ctx.secondary_slot.primary_slot:
        seen_style_keys: set[str] = set()
        page_numbers: set[int] = set()
        ctx.tertiary_slot.append(page_numbers)
        page_candidates: list[Block] = []
        ctx.state_slot.append(page_candidates)
        first_substantive_ref: list[Optional[Block]] = [None]   # closure-friendly

        def walk_cb(block: Block) -> bool:
            if block.skew_frac() >= 1 or block.area() <= 0:
                return False
            # Page height should be positive. If a degenerate page appears, keep
            # IEEE-style Infinity/NaN behavior so the comparisons below stay inert.
            den = page.bounds.bbox_height()
            num = block.top_edge() if ctx.primary_slot == 1 else block.bottom_edge()
            relative = (num / den) if den else (math.copysign(math.inf, num) if num else math.nan)
            if (ctx.primary_slot == 1 and relative < 0.8) or (ctx.primary_slot == 2 and relative > 0.2):
                pass_value = False
            else:
                tokens = tokenize_block(block)
                if _search_trie(COPYRIGHT_TRIE, tokens):
                    pass_value = True
                elif (
                    block.line_count() >= 3
                    or info_weight(block.char_stats) * (1 + block.bold_frac()) >= 200
                    or trie_prefix_match(FIGURE_KEYWORDS_TRIE, tokens)
                    or trie_prefix_match(CHART_KEYWORDS_TRIE, tokens)
                    or trie_prefix_match(TABLE_KEYWORDS_TRIE, tokens)
                ):
                    pass_value = False
                else:
                    pass_value = True
            if not pass_value:
                return True
            if letter_count(block.char_stats) >= 5 and first_substantive_ref[0] is None:
                first_substantive_ref[0] = block
            ref = first_substantive_ref[0]
            if block.type == 0 and block.char_count() > 0:
                page_candidates.append(block)
                if letter_count(block.char_stats) >= 5:
                    text_key = normalized_block_text(block)
                    item_list = ctx.auxiliary_slot.get(text_key)
                    if item_list is None:
                        item_list = []
                        ctx.auxiliary_slot[text_key] = item_list
                    item_list.append({"page_index": page.page_index, "block": block})
                    if is_header_positioned(ctx, block, ref):
                        for key in longest_word_and_number(block):
                            inner = ctx.measure_slot.get(key)
                            if inner is None:
                                inner = {}
                                ctx.measure_slot[key] = inner
                            if page.page_index not in inner:
                                inner[page.page_index] = {"text_key": text_key, "block": block}
                for line in block:
                    for span in line:
                        if span.char_count() <= 0:
                            continue
                        ok = span_style_text_key(span)
                        if block.char_count() >= 4 and ok not in seen_style_keys:
                            ctx.option_slot[ok] = ctx.option_slot.get(ok, 0) + 1
                            seen_style_keys.add(ok)
                        detected_page_number = span_page_number(span)
                        if detected_page_number is not None:
                            page_numbers.add(detected_page_number)
            return False

        walk_from_page_edge(ctx, page.output_slot, walk_cb)

    # ----- Pass 2: per-page rejection sweep ----------------------------
    text_counts: dict[str, int] = {}
    samples: list[tuple[float, float]] = []

    for page in ctx.secondary_slot.primary_slot:
        candidates = ctx.state_slot[page.page_index - 1]
        state = PageMarkState()
        seen_page_number = False
        first_substantive: list[Optional[Block]] = [None]
        for candidate_index in range(len(candidates)):
            candidate_block = candidates[candidate_index]
            if candidate_block.char_count() <= 0:
                continue
            if candidate_block.type == ctx.primary_slot:
                record_marked_block(state, candidate_index, candidate_block)
                continue
            if candidate_block.type != 0:
                continue
            candidate_tokens = tokenize_block(candidate_block)
            # Copyright terms must appear at the start of the block, not merely
            # anywhere inside it.
            if candidate_tokens.length < 10 and trie_prefix_match(COPYRIGHT_TRIE, candidate_tokens):
                mark_header_footer(ctx, candidate_block)
                record_marked_block(state, candidate_index, candidate_block)
                continue
            if letter_count(candidate_block.char_stats) >= 5:
                if first_substantive[0] is None:
                    first_substantive[0] = candidate_block
                pk_hash = normalized_block_text(candidate_block)
                match = find_cross_page_match(ctx, page, candidate_block, pk_hash, first_substantive[0])
                if match is not None:
                    mark_header_footer(ctx, candidate_block)
                    record_marked_block(state, candidate_index, candidate_block)
                    other_unmarked = match.type != ctx.primary_slot
                    if other_unmarked:
                        mark_header_footer(ctx, match)
                    if len(pk_hash) >= 5:
                        previous_count = text_counts.get(pk_hash, 0)
                        text_counts[pk_hash] = 2 if (previous_count or other_unmarked) else 1
                    continue
            style_threshold = max(2.0, min(len(ctx.secondary_slot.primary_slot) / 3.0, 5.0))
            chars = 0
            for candidate_line in candidate_block:
                for candidate_span in candidate_line:
                    if candidate_span.char_count() <= 0:
                        continue
                    style_hash = span_style_text_key(candidate_span)
                    if ctx.option_slot.get(style_hash, 0) >= style_threshold:
                        chars += candidate_span.char_count()
                        continue
                    page_number = span_page_number(candidate_span)
                    if page_number is not None and has_adjacent_page_numbers(ctx, page.page_index, page_number, seen_page_number):
                        seen_page_number = True
                        chars += candidate_span.char_count()
            if chars >= candidate_block.char_count():
                mark_header_footer(ctx, candidate_block)
                record_marked_block(state, candidate_index, candidate_block)
                pk_again = normalized_block_text(candidate_block)
                if len(pk_again) >= 5:
                    text_counts[pk_again] = text_counts.get(pk_again, 0) + 1

        if state.primary_slot < 0:
            continue
        first_block = candidates[state.primary_slot]
        samples.append((first_block.center_y(), float(state.tertiary_slot)))

        # Also classify earlier non-confirmed blocks
        for index in range(state.primary_slot):
            block = candidates[index]
            if block.type == ctx.primary_slot:
                continue
            if ctx.primary_slot == 1 and block.bottom_edge() < first_block.bottom_edge():
                continue
            if block.bbox_width() >= page.bounds.bbox_width() / 2:
                continue
            if is_body_paragraph(ctx.secondary_slot.secondary_slot, page, block):
                continue
            if letter_count(block.char_stats) > 0 and heading_score(block) >= state.secondary_slot + 1:
                continue
            block.type = ctx.primary_slot
            record_recurring_text(ctx.secondary_slot, deaccented_text(block))

    # ----- Pass 3: cutoff line + top-3 text-hash sweep -----------------
    if len(samples) < len(ctx.secondary_slot.primary_slot) / 20:
        return
    cutoff = weighted_percentile(samples, 20 if ctx.primary_slot == 1 else 80)

    top: list[tuple[str, int]] = []
    for text_hash, count in text_counts.items():
        if count < len(ctx.secondary_slot.primary_slot) / 20:
            continue
        top.append((text_hash, count))
    if not top:
        return
    top.sort(key=lambda item_pair: -item_pair[1])
    if len(top) > 3:
        top = top[:3]

    for page in ctx.secondary_slot.primary_slot:
        for recurring_block in ctx.state_slot[page.page_index - 1]:
            if ctx.primary_slot == 1 and recurring_block.top_edge() < cutoff:
                break
            if ctx.primary_slot == 2 and recurring_block.bottom_edge() > cutoff:
                break
            if recurring_block.type != 0:
                continue
            recurring_tokens = tokenize_block(recurring_block)
            stripped = _search_trie(VOLUME_WORDS_TRIE, recurring_tokens)
            if stripped is not None:
                # ``stripped.end`` is absolute in the forward token view, so this
                # drops the matched volume phrase and keeps the tail.
                tail = recurring_tokens.slice(stripped.end)
                head_tok = tail.token_at(0) if tail.length > 0 else None
                if head_tok is not None and head_tok.type == 1:
                    recurring_block.type = ctx.primary_slot
                    record_recurring_text(ctx.secondary_slot, deaccented_text(recurring_block))
                    # Deliberately fall through: the same block can also match the
                    # top recurring-text sweep below.
            if letter_count(recurring_block.char_stats) < 5:
                continue
            if page.page_index <= 1 and heading_score(recurring_block) > ctx.secondary_slot.secondary_slot.primary_slot + 1:
                continue
            text_key = normalized_block_text(recurring_block)
            for text_hash, _ in top:
                threshold = min(len(text_key), len(text_hash)) / 2.0
                if bounded_edit_distance(text_key, text_hash, threshold) >= threshold:
                    continue
                # No break: every sufficiently similar recurring key contributes
                # to the recurring-text histogram.
                recurring_block.type = ctx.primary_slot
                record_recurring_text(ctx.secondary_slot, deaccented_text(recurring_block))
