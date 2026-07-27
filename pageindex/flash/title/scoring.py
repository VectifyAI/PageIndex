"""Title-candidate scoring."""

from __future__ import annotations

import math

from ..model import (
    _trim_unicode_ws,
    left_aligned,
    right_aligned,
    center_aligned,
    Rect,
    last_span,
    heading_score,
    Line,
    last_line_of,
    first_span_of,
    block_text,
    deaccented_text,
    letter_count,
    dominant_style_of,
    info_weight,
    is_upper_dominant,
    alignment_code,
    Block,
)
from ..stats import DocStats, column_index_of, tally_scripts, dominant_script_family, ScriptHistogram
from ..tokens import is_superscript_adjacent, clamp_value, enumerate_tokens, jenkins_hash, trie_prefix_match, set_case_fold, TrieConfig, build_trie, tokenize_block, _de_norm, BuiltTrie, is_word_token

from .dicts import (
    INSTITUTION_WORDS,
    TITLE_LABEL_TRIE,
)


# --------------------------------------------------------------------------- #
# Title candidate state container #
# --------------------------------------------------------------------------- #


class TitleCandidate:
    """Best title candidate so far: page, contributing blocks, and score."""

    __slots__ = ("page", "output_slot", "score")

    def __init__(self, page, blocks: list[Block], score_value: float):
        self.page = page
        self.output_slot = blocks
        self.score = score_value

    def to_string(self) -> str:
        """Join contributing blocks into the displayed title string, inserting one inter-block space only after the accumulator is non-empty."""
        primary_item = ""
        for block in self.output_slot:
            if primary_item:
                primary_item += " "
            primary_item += _trim_unicode_ws(tokenize_block(block).to_string())

        return primary_item

    def __str__(self) -> str:
        return self.to_string()


# --------------------------------------------------------------------------- #
# Cover-like page predicate #
# --------------------------------------------------------------------------- #


def is_cover_like_page(doc, page) -> bool:
    """Return whether a page is sparse enough to behave like a cover page."""
    if getattr(page, "measure_slot", False):
        return False
    threshold = 0.5 * min(doc.secondary_slot.secondary_slot, 5e3)
    if page.page_index <= 1 and page.primary_slot.secondary_slot < threshold:
        return True
    early_limit = 1 + min(15, len(doc.primary_slot) / 5)
    return page.page_index < early_limit and page.primary_slot.secondary_slot < 0.8 * threshold


# --------------------------------------------------------------------------- #
# xp: candidate-block filter #
# --------------------------------------------------------------------------- #


def is_title_candidate_block(block: Block) -> bool:
    """Return whether ``block`` can be considered as a document-title candidate."""
    return (
        letter_count(block.char_stats) > 0
        and block.skew_frac() < 1
        and block.type == 0
        and block.char_count() < 400
        and block.bbox_height() < 2 * block.bbox_width()
    )


# --------------------------------------------------------------------------- #
# yp: multiplicative scoring for a candidate group #
# --------------------------------------------------------------------------- #


def score_title_candidate(zp_state, page, index: int) -> None:
    """Score a candidate block group and update the title-search state."""
    doc = zp_state.tertiary_slot
    blocks = page.secondary_slot                                   # sorted blocks
    title_block = blocks[index]
    title_group: list[Block] = [title_block]

    # Try to extend with next block if alignment / style / vertical proximity match
    if index + 1 < len(blocks):
        next_item = blocks[index + 1]
        title_score = heading_score(title_block)
        height = title_block.avg_font_size()
        # Two acceptance conditions:
        if (
            (abs(title_score - heading_score(next_item)) < 0.1
                and dominant_style_of(title_block) == dominant_style_of(next_item)
                and title_block.bottom_edge() - next_item.top_edge() < height)
            or (
                title_score > doc.secondary_slot.primary_slot + 5
                and title_score > page.primary_slot.primary_slot + 1
                and abs(height - next_item.avg_font_size()) < 0.1
                and title_block.bottom_edge() - next_item.top_edge() < 0.5 * height
            )
        ):
            tolerance = 0.1 * height
            align_value = alignment_code(title_block)
            next_alignment = alignment_code(next_item)
            if (
                (left_aligned(title_block, next_item, tolerance)
                    and align_value in (1, 2) and next_alignment in (1, 2))
                or (right_aligned(title_block, next_item, tolerance)
                    and align_value in (1, 4) and next_alignment in (1, 4))
                or (center_aligned(title_block, next_item, tolerance)
                    and title_block.alignment_slot and next_item.alignment_slot)
            ):
                title_group.append(next_item)

    group = title_group
    for measure_item in group:
        zp_state.primary_slot.add(id(measure_item))

    doc_state = zp_state.tertiary_slot
    previous_block = blocks[index - 1] if index - 1 >= 0 else None

    # Accumulate statistics over the title group
    max_heading_score = 0
    max_width = 0.0
    consecutive = 0
    max_consecutive = 0
    bracket_count = 0
    total_tokens = 0
    right_pen = 1.0
    email_count = 0

    for result_value in group:
        max_heading_score = max(max_heading_score, heading_score(result_value))
        max_width = max(max_width, result_value.bbox_width())
        title_tokens_view = tokenize_block(result_value)
        for entry in enumerate_tokens(title_tokens_view):
            sample_item = entry["token"]
            total_tokens += 1
            if is_word_token(sample_item):
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
                if sample_item.boundary_slot:
                    bracket_count += 1
                # email detection: "@" followed by word "." word (4 tokens)
                if sample_item.str == "@" and entry["index"] + 3 < title_tokens_view.length:
                    next_token = title_tokens_view.token_at(entry["index"] + 1)
                    dot = title_tokens_view.token_at(entry["index"] + 2)
                    after = title_tokens_view.token_at(entry["index"] + 3)
                    if (
                        next_token is not None and dot is not None and after is not None
                        and next_token.type == 2 and dot.str == "." and after.type == 2
                    ):
                        email_count += 1
            else:
                consecutive = 0
        if alignment_code(result_value) == 4:
            # The line count is structurally positive here. Keep the fallback so
            # a degenerate line cannot raise during title scoring.

            right_pen /= result_value.line_count() or 1

    if total_tokens <= 0:
        return

    # Multiplicative factors
    len_value = clamp_value(total_tokens * total_tokens / 16.0, 0.5, 1.0)
    # Page width should be positive. Keep IEEE-style Infinity/NaN behavior for
    # degenerate pages instead of raising during scoring.
    width_ratio_sq = (max_width / page.bounds.bbox_width()) if page.bounds.bbox_width() else (math.inf if max_width > 0 else math.nan)
    width_ratio_sq *= width_ratio_sq
    bracket = bracket_count / total_tokens
    bracket_factor = max(0.1, 1 - 9 * bracket * bracket) / max(1, max_consecutive - 2)
    page_pos = max(0.1, 1 - 2 * (page.page_index - 1) / max(1, len(doc_state.primary_slot)))
    # Doc-wide height is positive in normal inputs. The epsilon prevents a
    # degenerate input from raising and still yields the minimum density factor.
    page_density_ratio = page.primary_slot.secondary_slot / max(1e-6, doc_state.secondary_slot.secondary_slot)
    density_factor = max(0.5, 1 - page_density_ratio * page_density_ratio) * (1 + clamp_value((0.25 - page_density_ratio) / 0.15, 0, 1))
    # Page top/height is positive in normal inputs. Degenerate pages take the
    # minimum top-position factor instead of raising.

    top = max(0.1, group[0].top_edge() / page.bounds.top_edge()) if page.bounds.top_edge() else 0.1

    # Abbreviation penalty: count adjacent single-char + delimiter pairs
    abbrev = 0
    for block in group:
        tokens = tokenize_block(block)
        previous = None
        for token in tokens:
            if previous is not None and len(token.str) <= 1 and is_superscript_adjacent(previous, token):
                abbrev += 1
            previous = token
    factor = clamp_value(1.0 / max(1, abbrev), 0.3, 1.0)

    # Recurrence penalty: first block's normalized text appears how often?
    # The histogram uses the same normalized text hash as the document-wide
    # ghost-text map.
    norm_text = jenkins_hash(deaccented_text(group[0]))
    recurrence_count = doc_state.tertiary_slot.get(norm_text, 0) if hasattr(doc_state, "tertiary_slot") and isinstance(doc_state.tertiary_slot, dict) else 0
    ratio = recurrence_count / max(1, len(doc_state.primary_slot))
    adj = total_tokens - 3
    recurrence_factor = 1 - 0.5 * clamp_value(ratio / 0.3, 0, 1) * (1 / max(1, adj * adj))

    # Institution-word penalty (non-first-page)
    institution = 1.0
    if is_cover_like_page(doc_state, page):
        inst_hits = 0
        for block in group:
            for token in tokenize_block(block):
                # single-token
                # Match using the same lowercase + diacritic-stripped form as
                # the institution-word set.
                if _de_norm(token.str, True) in INSTITUTION_WORDS:
                    inst_hits += 1
        institution = 1.0 / (1 + inst_hits)

    # "Title:" label bonus from previous block
    label = 1.0
    if previous_block is not None:
        prev_tokens = tokenize_block(previous_block)
        if prev_tokens.length <= 3 and trie_prefix_match(TITLE_LABEL_TRIE, prev_tokens) is not None:
            label = 3.0

    # Email penalty
    email = 1.0 / ((1 + email_count) ** 2)

    # Script-family match: build a script histogram over the candidate group's text and
    # compare the candidate script family against the document script family.
    script_acc = ScriptHistogram()
    for result_value in group:
        tally_scripts(script_acc, block_text(result_value))
    script = 1.0 if dominant_script_family(script_acc) == doc_state.secondary_slot.tertiary_slot else 0.5

    score = (
        max_heading_score * len_value * width_ratio_sq * right_pen * bracket_factor * page_pos
        * density_factor * top * factor * recurrence_factor * institution * label
        * email * script
    )

    if zp_state.secondary_slot is None or score > zp_state.secondary_slot.score:
        zp_state.secondary_slot = TitleCandidate(page, group, score)
