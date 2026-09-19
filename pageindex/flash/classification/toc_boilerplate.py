"""Watermark, boilerplate, and TOC-range detection."""

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
    _DICTS,
    _dict_trie,
    TOC_TITLES_TRIE,
    DOT_LEADER_ROW_RE,
    _search_trie,
)
from .body_text import (
    is_body_paragraph,
    normalized_block_text,
)


# --------------------------------------------------------------------------- #
# Side-rail watermark detector #
# --------------------------------------------------------------------------- #


def mark_watermarks(doc) -> None:
    """Bucket skewed side-rail blocks by normalized text; recurring groups are marked as watermarks."""
    buckets: dict[str, list[Block]] = {}
    for page in doc.primary_slot:
        for block in page.output_slot:
            if block.skew_frac() < 1:
                continue
            if letter_count(block.char_stats) < 5:
                continue
            # Skip blocks in the central 80% of the page width.
            horizontal_offset = block.center_x()
            page_width = page.bounds.bbox_width()
            if 0.1 * page_width < horizontal_offset < 0.9 * page_width:
                continue
            # No empty-string guard: empty normalized-text keys bucket together.
            key = normalized_block_text(block)
            buckets.setdefault(key, []).append(block)
    for group in buckets.values():
        if len(group) < 3:
            continue
        for block in group:
            block.type = 12


# --------------------------------------------------------------------------- #
# Boilerplate block predicate #
# --------------------------------------------------------------------------- #


# Institution/thesis trie combines institution words with thesis-specific terms.
_institution_thesis_words = list(_DICTS.get("institution_words", [])) + list(_DICTS.get("nk_thesis_words", []))
INSTITUTION_THESIS_TRIE = build_trie(_institution_thesis_words, set_case_fold(TrieConfig(), True))
PROFESSOR_TITLES_TRIE = _dict_trie("professor_titles")


def is_boilerplate_block(block: Block) -> bool:
    """line/block looks like boilerplate (committee members, author affiliations, journal volume info etc.)."""
    tokens = tokenize_block(block)
    if info_weight(block.char_stats) >= 200 or tokens.length >= 100:
        return False
    if _search_trie(INSTITUTION_THESIS_TRIE, tokens):
        return True
    # Strip leading lines that match professor/title boilerplate.
    while tokens.length > 0:
        # Count tokens belonging to the first token's line and strip that line.
        first_line = tokens.token_at(0).line() if tokens.token_at(0) else None
        if first_line is None:
            break
        line_end = 0
        while line_end < tokens.length:
            tok = tokens.token_at(line_end)
            if tok is None or tok.line() is not first_line:
                break
            line_end += 1
        if not trie_prefix_match(PROFESSOR_TITLES_TRIE, tokens.slice(0, line_end)):
            return False
        tokens = tokens.slice(line_end)
    return True


# --------------------------------------------------------------------------- #
# TOC-page detection chain #
# --------------------------------------------------------------------------- #


class NumberColumnCluster:
    """numeric-leading-token cluster."""

    __slots__ = ("anchor_x", "width", "secondary_slot", "primary_slot", "length", "tertiary_slot")

    def __init__(self, anchor_x_value: float, width: float, reference_number: int, next_number: int, length: int, limit_flag: bool):
        self.anchor_x = anchor_x_value        # anchor x-position
        self.width = width  # cluster typical width
        self.secondary_slot = reference_number          # first value seen
        self.primary_slot = next_number          # last value seen
        self.length = length
        self.tertiary_slot = limit_flag          # is increasing


def extract_number_column(block) -> Optional[NumberColumnCluster]:
    """Extract a numeric-leading cluster if block lines form an increasing page-number sequence."""
    column = 0
    last_number = 0
    sequence_length = 0
    for line in block:
        line_number = to_number(text_of_line(line))
        if math.isnan(line_number):
            return None
        if not (line_number > 0 and line_number < 1e6 and line_number == math.ceil(line_number)) or line_number >= 1e4 or last_number > line_number:
            return NumberColumnCluster(block.center_x(), block.bbox_width(), column, last_number, sequence_length, False)
        if column <= 0:
            column = int(line_number)
        last_number = int(line_number)
        sequence_length += 1
    return NumberColumnCluster(block.center_x(), block.bbox_width(), column, last_number, sequence_length, True)


def pick_nearer_cluster(cluster: NumberColumnCluster, other_cluster: Optional[NumberColumnCluster], other: Optional[NumberColumnCluster]) -> Optional[NumberColumnCluster]:
    """Pick the closer neighbor cluster within the current cluster width."""
    distance = (cluster.anchor_x - other_cluster.anchor_x) if other_cluster is not None else math.inf
    candidate_distance = (other.anchor_x - cluster.anchor_x) if other is not None else math.inf
    if distance > cluster.width and candidate_distance > cluster.width:
        return None
    return other_cluster if distance < candidate_distance else other


def detect_toc_range(doc, page, index) -> Optional[dict]:
    """Detect a TOC-like block range within ``page``. The detector combines dot-leader rows, blocks ending in dot-leader page numbers, contents-like titles, and same-x-range numeric clusters. Returns a ``{start_index, end_index}`` range or ``None``."""
    blocks = page.output_slot
    lines = 0
    dot_leader_blocks = 0
    weight = 0.0
    last_multiline = -1
    contents = -1
    pre_contents = -1
    last_toc = -1
    seen_body = False
    body_stop_y = page.bounds.top_edge()
    is_last_page = (index == page.page_index - 1) if isinstance(index, int) and index >= 0 else False
    clusters: list[NumberColumnCluster] = []  # sorted by anchor_x

    def _add_cluster(cluster: NumberColumnCluster) -> None:
        # Sorted-set semantics: an equal anchor_x is a no-op.
        import bisect
        keys = [existing_cluster.anchor_x for existing_cluster in clusters]
        insert_index = bisect.bisect_left(keys, cluster.anchor_x)
        if insert_index < len(clusters) and clusters[insert_index].anchor_x == cluster.anchor_x:
            return
        clusters.insert(insert_index, cluster)

    def _next_number_column_cluster(cluster: NumberColumnCluster) -> Optional[NumberColumnCluster]:
        # Non-strict successor: an equal anchor_x entry is returned.
        import bisect
        keys = [column.anchor_x for column in clusters]
        cluster_index = bisect.bisect_left(keys, cluster.anchor_x)
        return clusters[cluster_index] if cluster_index < len(clusters) else None

    def _prev_number_column_cluster(cluster: NumberColumnCluster) -> Optional[NumberColumnCluster]:
        # Non-strict predecessor: an equal anchor_x entry is returned.
        import bisect
        keys = [column.anchor_x for column in clusters]
        cluster_index = bisect.bisect_right(keys, cluster.anchor_x)
        return clusters[cluster_index - 1] if cluster_index > 0 else None

    def _remove(cluster: NumberColumnCluster) -> None:
        try:
            clusters.remove(cluster)
        except ValueError:
            pass

    for codepoint, block in enumerate(blocks):
        is_toc = False
        # Count dot-leader rows across all lines in the block.

        for line in block:
            if DOT_LEADER_ROW_RE.search(text_of_line(line)):
                is_toc = True
                if last_toc >= 0:
                    last_toc = codepoint
                else:
                    lines += 1
                    if lines >= 5 or (lines >= 3 and is_last_page):
                        last_toc = codepoint
        # A dot-leader on the last line also starts or extends the TOC range.

        last_line = block.primary_slot[-1] if block.primary_slot else None
        if last_line is not None and DOT_LEADER_ROW_RE.search(text_of_line(last_line)):
            is_toc = True
            if last_toc >= 0:
                last_toc = codepoint
                continue
            else:
                dot_leader_blocks += 1
                weight += info_weight(block.char_stats)
                if is_last_page and dot_leader_blocks >= 2 and weight >= 0.8 * page.primary_slot.secondary_slot:
                    last_toc = codepoint
        # Body block tracking
        if not is_toc and is_body_paragraph(doc.secondary_slot, page, block):
            seen_body = True
            body_stop_y = min(body_stop_y, block.bottom_edge())
        if block.line_count() > 1 and not is_toc:
            last_multiline = codepoint
        # "Contents"-like title must consume the whole block, not just a prefix.
        if contents < 0 and block.line_count() <= 1 and trie_full_match(TOC_TITLES_TRIE, tokenize_block(block)):
            contents = codepoint
            pre_contents = last_multiline
            if not seen_body and block.top_edge() > 3 * page.bounds.bbox_height() / 4:
                return {"start_index": pre_contents + 1, "end_index": len(blocks) - 1}
        # Numeric-column clustering on unclassified blocks below the body line.
        if block.right_edge() < page.bounds.center_x():
            continue
        if block.top_edge() > body_stop_y:
            continue
        # Extract even from an empty-looking block; the extractor decides whether
        # a usable numeric sequence exists.
        cluster = extract_number_column(block)
        if cluster is not None:
            successor = _next_number_column_cluster(cluster)
            predecessor = _prev_number_column_cluster(cluster)
            picked = pick_nearer_cluster(cluster, predecessor, successor)
            if picked is not None:
                _remove(picked)
                new_value = NumberColumnCluster(
                    picked.anchor_x,
                    picked.width,
                    picked.secondary_slot,
                    cluster.primary_slot,
                    picked.length + cluster.length,
                    picked.tertiary_slot and cluster.tertiary_slot and picked.primary_slot <= cluster.secondary_slot,
                )
            else:
                new_value = cluster
            _add_cluster(new_value)
            if (new_value.tertiary_slot
                    and (new_value.length >= 10 or (new_value.length >= 5 and is_last_page))
                    and new_value.primary_slot - new_value.secondary_slot > 0.01 * new_value.primary_slot):
                last_toc = codepoint

    if last_toc < 0:
        return None
    return {"start_index": pre_contents + 1 if pre_contents >= 0 else 0, "end_index": last_toc}


# --------------------------------------------------------------------------- #
# TOC pages, references lists, and figure/table captions #
# --------------------------------------------------------------------------- #


def mark_toc_and_boilerplate(doc) -> None:
    """Mark TOC blocks as type=9 and captions/boilerplate as type=12."""
    previous_toc_page = -math.inf
    for page in doc.primary_slot:
        if (
            page.page_index - 1 >= len(doc.primary_slot) / 2
            and page.primary_slot.secondary_slot >= 0.9 * doc.secondary_slot.secondary_slot
        ):
            continue
        # "Most-boilerplate" check for front-matter pages.

        if (
            page.page_index > 1
            and page.page_index < 50
            and page.primary_slot.secondary_slot < max(200, min(0.75 * doc.secondary_slot.secondary_slot, 1000))
        ):
            total = 0.0
            body_paragraph = 0.0
            body_paragraph_lines = 0
            for line_or_block in page.output_slot:
                if line_or_block.type != 0 or line_or_block.skew_frac() >= 1:
                    continue
                width_value = info_weight(line_or_block.char_stats) * heading_score(line_or_block)
                total += width_value
                if is_boilerplate_block(line_or_block):
                    body_paragraph += width_value
                    body_paragraph_lines += 1
            if body_paragraph >= 0.8 * total and body_paragraph_lines >= 3:
                for block in page.output_slot:
                    block.type = 12
                continue
        toc = detect_toc_range(doc, page, previous_toc_page)
        if toc is None:
            continue
        previous_toc_page = page.page_index
        start = toc["start_index"]
        end = toc["end_index"]
        blocks = page.output_slot
        # Track centered-block count, weighted font sum, total weight, and the
        # running bottom edge used by walk-forward break conditions.
        centered_flag = 0
        width_flag = 0.0
        width = 0.0
        walk_break_y = page.bounds.top_edge()
        for idx in range(start, len(blocks)):
            block = blocks[idx]
            score = heading_score(block)
            if idx <= end:
                if block.isolated_centered:
                    centered_flag += 1
                heading_weight = info_weight(block.char_stats)
                width_flag += block.avg_font_size() * heading_weight
                width += heading_weight
                walk_break_y = min(walk_break_y, block.bottom_edge())
                block.type = 9
                continue
            # Walk forward with four break conditions: prominent heading, centered
            # block, dense body text, or a large vertical gap to a prominent block.
            if block.skew_frac() < 1 and score > doc.secondary_slot.primary_slot + 4 and score > page.primary_slot.primary_slot + 4:
                break
            if centered_flag <= 1 and block.isolated_centered:
                break
            if info_weight(block.char_stats) > 300 and block.char_stats.primary_slot[6] > 2 and block.weighted_ratio_secondary > 0.5:
                break
            if width > 0:
                average = width_flag / width
                if walk_break_y - block.top_edge() > average and block.skew_frac() < 1 and score > average + 1.5:
                    break
            walk_break_y = min(walk_break_y, block.bottom_edge())
            block.type = 9
