"""Token types, anchors, and token-view utilities."""

from __future__ import annotations

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


# --------------------------------------------------------------------------- #
# Character-category transition table #
# --------------------------------------------------------------------------- #


# Map 12 character categories down to script-family buckets used by statistics.
SCRIPT_FAMILY_MAP = [0, 1, 2, 2, 2, 2, 3, 4, 5, 6, 7, 7, 8]


def _build_gap_tolerance_grid() -> list[list[float]]:
    """Return the fractional gap tolerance for adjacent character categories."""
    token_value = [[0.0] * 12 for _ in range(12)]
    # Small punctuation-to-mark transition weights.
    for candidate_item in (1, 2, 3, 4):
        token_value[candidate_item][5] = 0.16
        token_value[candidate_item][6] = 0.16
    token_value[3][2] = 0.1
    token_value[6][2] = 0.1
    token_value[6][3] = 0.1
    token_value[8][2] = 0.1
    token_value[8][3] = 0.1
    return token_value


GAP_TOLERANCE_GRID = _build_gap_tolerance_grid()


def can_extend_token(number: int, other_number: int, candidate_text: str) -> bool:
    """Return whether the current token can extend with ``candidate_text``."""
    from ..stats import char_script_bucket
    if other_number == 4 and char_script_bucket(candidate_text) == 5:
        return False
    if other_number == number and not is_punct_category(other_number):
        return True
    if is_word_category(number) and is_word_category(other_number):
        return True
    return False


# --------------------------------------------------------------------------- #
# Token span anchors.
# --------------------------------------------------------------------------- #


class TokenAnchor:
    """Cross-line anchor range attached to a token."""

    __slots__ = ("line", "anchor_span", "start_offset", "primary_slot")

    def __init__(self, line: Line, anchor_span_value, start_offset_value: int, next_number: int):
        self.line = line
        self.anchor_span = anchor_span_value
        self.start_offset = start_offset_value
        self.primary_slot = next_number


def last_token_anchor(token: "Token") -> TokenAnchor:
    """Return the token's last cross-line anchor entry."""
    return token.anchor_ranges[-1]


def first_anchor_span(token: "Token"):
    """Return the anchor span from the token's first cross-line entry."""
    return token.anchor_ranges[0].anchor_span


# --------------------------------------------------------------------------- #
# Token kind predicates #
# --------------------------------------------------------------------------- #


def is_char_token(token: "Token") -> bool:
    """Return True for digit or letter tokens."""
    return token.type == 1 or token.type == 2


def is_word_token(token: "Token") -> bool:
    """Return True for merged word-like tokens: word, number-word, or symbolic token kinds."""
    return token.type in (3, 4, 5)


def is_trimmable_token(token: "Token") -> bool:
    """Return True for word, number-word, or colon tokens that can be trimmed from phrase edges."""
    return token.type == 3 or token.type == 4 or token.str == ":"


def token_numeric_value(token: "Token") -> float:
    """numeric value of token, NaN if non-numeric."""
    return to_number(token.str)


# --------------------------------------------------------------------------- #
# Token #
# --------------------------------------------------------------------------- #


class Token:
    """One token. It stores the token kind, raw text, contributing line/span anchors, bracket attachment flag, and first/last character categories."""

    __slots__ = ("type", "str", "anchor_ranges", "boundary_slot", "primary_slot", "secondary_slot")

    def __init__(self, type_: int, candidate_text: str, anchor_ranges_value: list[TokenAnchor], boundary_flag: bool, previous_number: int, limit_number: int):
        self.type = type_
        self.str = candidate_text
        self.anchor_ranges = anchor_ranges_value
        self.boundary_slot = boundary_flag
        self.primary_slot = previous_number
        self.secondary_slot = limit_number

    def line(self) -> Line:
        """Line of the first origin-span back-reference."""
        return self.anchor_ranges[0].line

    def __repr__(self) -> str:  # diagnostic
        return f"<Token t={self.type} {self.str!r} ba={self.boundary_slot}>"


# --------------------------------------------------------------------------- #
# Directional token view #
# --------------------------------------------------------------------------- #


class TokenView:
    """Sliceable, directional view over a token array. Supports forward / reverse iteration via ``dir`` = +1 / -1. ``slice`` and ``reverse`` produce new views without copying. """

    __slots__ = ("primary_slot", "start", "end", "dir", "length")

    def __init__(self, other_tokens: list[Token], start: int, end: int, dir_: int):
        self.primary_slot = other_tokens
        self.start = start
        self.end = end
        self.dir = dir_
        self.length = (end - start) // dir_ if dir_ != 0 else 0

    def __iter__(self) -> Iterator[Token]:
        secondary_item = self.start
        while secondary_item != self.end:
            yield self.primary_slot[secondary_item]
            secondary_item += self.dir

    def token_at(self, other_number: int) -> Optional[Token]:
        if other_number < 0 or other_number >= self.length:
            return None
        return self.primary_slot[self.start + other_number * self.dir]

    def __getitem__(self, other_number: int) -> Optional[Token]:
        return self.token_at(other_number)

    def __len__(self) -> int:
        return self.length

    def __bool__(self) -> bool:
        return self.length > 0

    def __str__(self) -> str:
        parts: list[str] = []
        for token in self:
            parts.append(token.str)
            if token.boundary_slot:
                parts.append(" ")
        return "".join(parts)

    def slice(self, other_number: int = 0, candidate_number: int = 0) -> "TokenView":
        """Bounds-clamped directional slice. Args follow Unicode-compatible semantics: a > 0 -> from index a a < 0 -> from end-relative a = 0 -> from start b > 0 -> to index b b < 0 -> end-relative b = 0 -> to end """
        if other_number > 0:
            slice_start = self.start + other_number * self.dir
        elif other_number < 0:
            slice_start = self.end + other_number * self.dir
        else:
            slice_start = self.start
        if slice_start * self.dir < self.start * self.dir:
            slice_start = self.start
        if slice_start * self.dir > self.end * self.dir:
            slice_start = self.end
        if candidate_number > 0:
            slice_end = self.start + candidate_number * self.dir
        elif candidate_number < 0:
            slice_end = self.end + candidate_number * self.dir
        else:
            slice_end = self.end
        if slice_end * self.dir < slice_start * self.dir:
            slice_end = slice_start
        if slice_end * self.dir > self.end * self.dir:
            slice_end = self.end
        return TokenView(self.primary_slot, slice_start, slice_end, self.dir)

    def reverse(self) -> "TokenView":
        return TokenView(self.primary_slot, self.end - self.dir, self.start - self.dir, -self.dir)

    def to_string(self) -> str:
        return str(self)


def wrap_tokens(tokens: list[Token]) -> TokenView:
    """Wrap a list of tokens as a forward token view."""
    return TokenView(tokens, 0, len(tokens), 1)


def enumerate_tokens(tokens: TokenView) -> Iterator[dict]:
    """Enumerate a token view yielding indexed token records."""
    token = tokens.primary_slot
    start = tokens.start
    end = tokens.end
    step = tokens.dir
    cursor = start
    while cursor != end:
        yield {"index": (cursor - start) // step, "token": token[cursor]}
        cursor += step


def first_token(tokens: TokenView) -> Optional[Token]:
    """Return the first token, or None."""
    return tokens.primary_slot[tokens.start] if tokens.length > 0 else None


def last_token(tokens: TokenView) -> Optional[Token]:
    """Return the last token, or None."""
    return tokens.primary_slot[tokens.end - tokens.dir] if tokens.length > 0 else None
