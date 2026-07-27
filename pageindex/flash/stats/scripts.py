"""Script bucket tables and script histogram helpers."""

from __future__ import annotations

import json
from pathlib import Path


# --------------------------------------------------------------------------- #
# Script-family detector and bucket table.
# --------------------------------------------------------------------------- #

_SCRIPT_BUCKET_TABLE_PATH = Path(__file__).parent.parent / "data" / "script_bucket_table.json"
SCRIPT_BUCKET_TABLE: list[int] = json.loads(_SCRIPT_BUCKET_TABLE_PATH.read_text(encoding="utf-8"))


def char_script_bucket(text: str) -> int:
    """Return the script-bucket id for a single character: empty/multi-character, ASCII punctuation/digit, control, ASCII letter, or a table-driven non-Latin script bucket."""
    if not text:
        return 0
    if len(text) != 1:
        return 10
    # Astral code points are classified as the multi-unit script bucket.
    if ord(text) > 0xFFFF:
        return 10
    if ("a" <= text <= "z") or ("A" <= text <= "Z"):
        return 3
    if "\x00" < text < " ":
        return 2
    if text < "":
        return 1
    idx = ord(text) >> 4
    if 0 <= idx < len(SCRIPT_BUCKET_TABLE):
        return SCRIPT_BUCKET_TABLE[idx]
    return 0


# map: each output category -> contributing bucket weights.
# Fixed weights for collapsing script buckets into document script families.
SCRIPT_FAMILY_WEIGHTS: dict[int, list[tuple[int, int]]] = {
    2: [(2, 10)],
    0: [(0, 1), (2, 1)],
    3: [(3, 1), (4, -3), (5, -3), (6, -3), (7, -3), (8, -3), (9, -10)],
    4: [(4, 1)],
    5: [(5, 1), (6, -10), (7, -10)],
    6: [(6, 1)],
    7: [(7, 1)],
    8: [(8, 1)],
    9: [(9, 1)],
    10: [(10, 1)],
}


class ScriptHistogram:
    """Script-bucket accumulator with total character count and per-bucket histogram."""

    __slots__ = ("secondary_slot", "primary_slot")

    def __init__(self):
        self.secondary_slot: int = 0
        self.primary_slot: list[int] = [0] * 11


def tally_scripts(primary_item: ScriptHistogram, other_text: str) -> None:
    """feed a string into the bucket accumulator."""
    for candidate_item in other_text:
        primary_item.primary_slot[char_script_bucket(candidate_item)] += 1
        primary_item.secondary_slot += 1


def dominant_script_family(primary_item: ScriptHistogram) -> int:
    """best-scoring script-family for the accumulator. Returns the output category 0..10 with the highest weighted score. """
    secondary_item = 0
    candidate_item = 0
    for reference_item in range(11):
        entry_item = SCRIPT_FAMILY_WEIGHTS.get(reference_item)
        if not entry_item:
            continue
        score_value = 0
        for (script_index, weight) in entry_item:
            score_value += weight * primary_item.primary_slot[script_index]
        if score_value > candidate_item:
            secondary_item = reference_item
            candidate_item = score_value
    return secondary_item
