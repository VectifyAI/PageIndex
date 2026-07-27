"""Line tokenization into word, char, and number tokens."""

from __future__ import annotations

import unicodedata
from typing import Any, Iterable, Iterator, Optional

from ..model import (
    _strip_diacritics,
    avg_char_width2,
    intervals_overlap,
    to_number,
    rect_union,
    EMPTY_RECT,
    avg_char_width,
    Line,
    char_category,
    is_word_category,
    is_punct_category,
    letter_count,
    punct_count,
    info_weight,
    Block,
)

from .token_types import (
    SCRIPT_FAMILY_MAP,
    GAP_TOLERANCE_GRID,
    can_extend_token,
    TokenAnchor,
    last_token_anchor,
    first_anchor_span,
    Token,
    TokenView,
    wrap_tokens,
)


# --------------------------------------------------------------------------- #
# Line tokenizer state machine #
# --------------------------------------------------------------------------- #


class LineTokenizer:
    """Line-tokenizer state machine with line/span anchors for reconstruction."""

    __slots__ = ("tertiary_slot", "secondary_slot", "cache_slot", "auxiliary_slot", "option_slot", "marker_slot", "primary_slot", "previous_slot", "state_slot", "style_slot", "measure_slot")

    def __init__(self):
        self.tertiary_slot: list[Token] = []
        self.secondary_slot = None       # last anchor span
        self.cache_slot: Optional[Line] = None
        self.auxiliary_slot = -1
        self.option_slot = -1
        self.marker_slot: list[TokenAnchor] = []
        self.primary_slot = ""
        self.previous_slot = False
        self.state_slot = 0          # last-char category
        self.style_slot = 0          # first-char category
        self.measure_slot = 0          # type-hint accumulator

    # --- inner state ops --------------------------------------------------

    def _close_anchor_range(self) -> None:
        """Close the current anchor range into the in-flight token and reset offsets."""
        self.marker_slot.append(TokenAnchor(self.cache_slot, self.secondary_slot, self.auxiliary_slot, self.option_slot))
        self.auxiliary_slot = self.option_slot = -1

    def _close_token(self, boundary_flag: bool) -> None:
        """Close the in-flight token into the token list."""
        if self.auxiliary_slot >= 0:
            self._close_anchor_range()
        self.tertiary_slot.append(Token(self.measure_slot, self.primary_slot, self.marker_slot, boundary_flag, self.style_slot, self.state_slot))
        self.marker_slot = []
        self.primary_slot = ""
        self.measure_slot = 0
        self.style_slot = 0
        self.state_slot = 0

    def _accumulate_char(self, other_text: str, candidate_number: int) -> None:
        """Append a character and update the in-flight token kind from the category map."""
        if len(self.primary_slot) == 1 and self.state_slot == 5:
            # If the in-flight token is a single mark, attach it before the new
            # character so combining marks bind to the following letter.
            self.primary_slot = other_text + self.primary_slot
            self.style_slot = candidate_number
        else:
            if not self.primary_slot:
                self.style_slot = candidate_number
            self.primary_slot += other_text
            self.state_slot = candidate_number
        cat = SCRIPT_FAMILY_MAP[candidate_number]
        if self.measure_slot == 0:
            self.measure_slot = cat
        elif self.measure_slot == 1 and cat != 1:
            self.measure_slot = 2
        self.previous_slot = False

    def _advance_char(self, other_text: str, candidate_number: int) -> None:
        """Advance the tokenizer with one character. Whitespace sets the pending-boundary flag; non-whitespace either extends or closes the current token."""
        reference_item = char_category(other_text)
        if reference_item == 10:
            # whitespace
            self.previous_slot = True
            return

        if self.previous_slot and self.primary_slot:
            # If the last non-whitespace category and the current category cannot
            # belong to the same word-like token, close the current token.

            if not (reference_item == 5 and is_word_category(self.state_slot)):
                self._close_token(True)

        # Soft-hyphen rejoin across lines: if there is no in-flight token, the
        # current char is lowercase, and the previous tokens were a word plus
        # "-" ending on another line, undo the split and continue that word.
        if not self.primary_slot and len(self.tertiary_slot) >= 2 and reference_item == 3:
            entry_item = self.tertiary_slot[-1]
            token = self.tertiary_slot[-2]
            if (
                token.secondary_slot == 3
                and not token.boundary_slot
                and entry_item.str == "-"
                and last_token_anchor(entry_item).line is not self.cache_slot
            ):
                self.tertiary_slot.pop()                            # drop "-"
                entry_item = self.tertiary_slot.pop()                        # pop word
                self.measure_slot = entry_item.type
                self.primary_slot = entry_item.str
                self.marker_slot = entry_item.anchor_ranges
                self.style_slot = entry_item.primary_slot
                self.state_slot = entry_item.secondary_slot
                self.previous_slot = False
                self._accumulate_char(other_text, reference_item)
                self.auxiliary_slot = self.option_slot = candidate_number
                return

        if self.primary_slot:
            if can_extend_token(self.state_slot, reference_item, other_text):
                self._accumulate_char(other_text, reference_item)
                if self.auxiliary_slot < 0:
                    self.auxiliary_slot = candidate_number
                self.option_slot = candidate_number
            else:
                self._close_token(False)
                self._accumulate_char(other_text, reference_item)
                self.auxiliary_slot = self.option_slot = candidate_number
        else:
            self._accumulate_char(other_text, reference_item)
            self.auxiliary_slot = self.option_slot = candidate_number

    # --- public API -------------------------------------------------------

    def add_line(self, other_line: Line) -> "LineTokenizer":
        """Walk one line and append its token contribution."""
        line = self.tertiary_slot[-1] if self.tertiary_slot else None
        if self.primary_slot:
            # Close in-flight; a trailing hyphen can glue to the next line only
            # when the previous token was not already bracket-attached.
            self._close_token(self.primary_slot != "-" or line is None or line.boundary_slot)
        self.cache_slot = other_line

        # Single-codepoint pending combining mark.
        pending = None  # type: Optional[Any]
        for index in range(len(other_line.primary_slot)):
            span = other_line.primary_slot[index]
            if span.char_count() <= 0:
                continue
            # Drop solitary combining marks (last-character category is 5)

            if pending is None and span.char_count() == 1 and span.char_stats.secondary_slot == 5:
                pending = span
                continue
            # Drop the bullet-then-content kerning glitch (layout branch:
            # single-character token, previous category is 11, and next span overlaps horizontally)

            if (
                index + 1 < len(other_line.primary_slot)
                and span.char_count() == 1
                and span.char_stats.secondary_slot == 11
                and span.left_edge() >= other_line.primary_slot[index + 1].left_edge()
                and span.center_x() < other_line.primary_slot[index + 1].right_edge()
            ):
                continue

            if self.secondary_slot is not None and self.primary_slot:
                # Decide whether the new span continues the same token
                if (
                    span.left_edge() <= self.secondary_slot.right_edge() + 0.1 * avg_char_width2(self.secondary_slot)
                    and (
                        abs(span.bottom_edge() - self.secondary_slot.bottom_edge()) < 0.1
                        or abs(span.center_y() - self.secondary_slot.center_y()) < 0.1
                    )
                    and self.secondary_slot.primary_slot == span.primary_slot
                ):
                    # Continue: close the current cross-line anchor entry and switch anchor.
                    self._close_anchor_range()
                    self.secondary_slot = span
                    self.previous_slot = False
                else:
                    gap_tolerance = (GAP_TOLERANCE_GRID[self.secondary_slot.char_stats.tertiary_slot][span.char_stats.secondary_slot] or 0.12) * avg_char_width(self.cache_slot)
                    close = (
                        self.previous_slot
                        or abs(self.secondary_slot.bottom_edge() - span.bottom_edge()) > 1
                        or span.left_edge() < self.secondary_slot.right_edge() - 1
                        or span.left_edge() > self.secondary_slot.right_edge() + gap_tolerance
                    )
                    self._close_token(close)
                    self.secondary_slot = span
            else:
                self.secondary_slot = span

            for char_index in range(len(span.text)):
                char_value = span.text[char_index]
                if (
                    char_index == 0
                    and pending is not None
                    and intervals_overlap(pending.left_edge(), pending.right_edge(), span.left_edge(), span.right_edge())
                ):
                    # Compose with the pending combining mark
                    combined = unicodedata.normalize("NFC", char_value + pending.state_slot[0])
                    self._advance_char(combined[0], 0)
                else:
                    self._advance_char(char_value, char_index)
                pending = None
        return self

    def tokens(self) -> TokenView:
        """Finalize and return a token view."""
        if self.primary_slot:
            self._close_token(True)
        return wrap_tokens(self.tertiary_slot)


# --------------------------------------------------------------------------- #
# X(block) -- cached token list for a block #
# --------------------------------------------------------------------------- #


def tokenize_block(block: Block) -> TokenView:
    """tokenize all lines of a block, cached on the block token cache."""
    if block.tokens_cache is not None:
        return block.tokens_cache  # type: ignore[return-value]
    token = LineTokenizer()
    for line in block.primary_slot:
        token.add_line(line)
    block.tokens_cache = token.tokens()  # type: ignore[assignment]
    return block.tokens_cache  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Utility helpers.
# --------------------------------------------------------------------------- #


def clamp_value(value: float, lower_bound: float, upper_bound: float) -> float:
    """Clamp a value between lower and upper bounds. The lower bound wins when the bounds are inverted, and NaN propagates."""
    measure_item = upper_bound if upper_bound < value else value
    return lower_bound if lower_bound > measure_item else measure_item


def is_superscript_adjacent(token: Token, other_token: Token) -> bool:
    """Return whether the next token is a raised, shorter marker on the same line."""
    candidate_item = last_token_anchor(token).anchor_span
    reference_item = first_anchor_span(other_token)
    return (
        reference_item is not candidate_item
        and last_token_anchor(token).line is other_token.line()
        and reference_item.bbox_height() < candidate_item.bbox_height()
        and reference_item.bottom_edge() > candidate_item.bottom_edge() + 0.1 * candidate_item.bbox_height()
    )
