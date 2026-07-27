"""Unicode normalization tables, whitespace classes, spacing factors, and bidi reordering."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path


_DROP_CHARS = str.maketrans({
    # U+FFFE is PDFium's "no unicode mapping" textpage sentinel. The
    # patch pipeline (_apply_font_unicode) replaces it with decoded text
    # wherever the map walk succeeds; a REMAINING U+FFFE means the guarded
    # walk gave up for that run, so deleting it keeps PDFium noise out of the
    # spans. A pathological ToUnicode map that intentionally emits literal
    # U+FFFE is indistinguishable from this sentinel here and is dropped.
    "￾": None,
    "\t": " ",
    "\n": " ",
    "\r": " ",
    # text extraction maps a glyph whose unicode lands on U+00AD to U+002D. (An
    # earlier "\x02" -> "-" entry here compensated PDFium decoding
    # re-encoded hyphens (charcode 2, ToUnicode gap) as U+0002; that decode
    # is now handled by _apply_font_unicode: mapped soft hyphens emit '-' where the
    # font's Differences name the glyph, and keeps the raw \x02 where they
    # don't, e.g. math-heavy page body ligature codes.
    "­": "-",
})


# The normalized Unicode table is a fixed, sparse per-glyph
# lookup table; a char absent from it is emitted unchanged. This is NOT Unicode
# NFKC: NFKC over-normalises (fullwidth→ASCII, superscripts→digits, ohm→omega,
# nbsp→space) exactly where this table leaves the glyph untouched. Apply the
# table per code point.
_NORMALIZED_UNICODES: dict[str, str] = json.loads(
    (Path(__file__).parent.parent / "data" / "normalized_unicodes.json")
    .read_text(encoding="utf-8")
)


def _normalize_unicodes(text: str) -> str:
    """Apply the per-glyph normalized-Unicode substitution table to a text item. The table is keyed by single code points and never introduces table keys, so applying it to the already-joined LTR item string preserves per-glyph substitution after the text item is joined."""
    unit_count = _NORMALIZED_UNICODES
    if not any(candidate_item in unit_count for candidate_item in text):
        return text
    return "".join(unit_count.get(candidate_item, candidate_item) for candidate_item in text)


# span merger
TRACKING_SPACE_FACTOR = 0.1
NON_SPACE_GAP_FACTOR = 0.03
NEGATIVE_SPACE_FACTOR = -0.2
SPACE_IN_FLOW_MIN_FACTOR = 0.1
SPACE_IN_FLOW_MAX_FACTOR = 0.6


# Whitespace classification uses the Unicode WhiteSpace + LineTerminator set.
# Python's str.isspace is not the same set: it omits U+FEFF and adds
# U+001C-U+001F and U+0085. Use the explicit code points so the
# whitespace-skip branch fires on the intended glyphs.
_WHITESPACE_CODEPOINTS = frozenset({
    0x9, 0xA, 0xB, 0xC, 0xD, 0x20, 0xA0, 0x1680,
    0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007,
    0x2008, 0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
    0xFEFF,
})


def _is_whitespace(number: int) -> bool:
    """Return whether a glyph code point is classified as whitespace."""
    return number in _WHITESPACE_CODEPOINTS


# Character classification checks whitespace before marks/formats, so a code
# point such as U+FEFF that is also Cf is treated as whitespace, not as an
# invisible format mark.
def _is_zero_width_diacritic(number: int) -> bool:
    """text extraction zero-width diacritic classification (group 2 = ``\\p{Mn}``)."""
    return number not in _WHITESPACE_CODEPOINTS and unicodedata.category(chr(number)) == "Mn"


def _is_invisible_format_mark(number: int) -> bool:
    """text extraction invisible format-mark classification (group 3 = ``\\p{Cf}``)."""
    return number not in _WHITESPACE_CODEPOINTS and unicodedata.category(chr(number)) == "Cf"


# Bidirectional character-type tables. base bidi type table covers
# U+0000..U+00FF; Arabic bidi type table covers U+0600..U+06FF indexed by the low byte
# (the "" at 0x1D follows the extraction rule placeholder for nonexistent U+061D).

_BIDI_BASE_TYPES = (
    "BN BN BN BN BN BN BN BN BN S B S WS B BN BN BN BN BN BN BN BN BN BN BN BN "
    "BN BN B B B S WS ON ON ET ET ET ON ON ON ON ON ES CS ES CS CS EN EN EN EN "
    "EN EN EN EN EN EN CS ON ON ON ON ON ON L L L L L L L L L L L L L L L L L L "
    "L L L L L L L L ON ON ON ON ON ON L L L L L L L L L L L L L L L L L L L L "
    "L L L L L L ON ON ON ON BN BN BN BN BN BN B BN BN BN BN BN BN BN BN BN BN "
    "BN BN BN BN BN BN BN BN BN BN BN BN BN BN BN BN CS ON ET ET ET ET ON ON ON "
    "ON L ON ON BN ON ON ET ET EN EN ON L ON ON ON EN L ON ON ON ON ON L L L L "
    "L L L L L L L L L L L L L L L L L L L ON L L L L L L L L L L L L L L L L L "
    "L L L L L L L L L L L L L L ON L L L L L L L L "
).split()
assert len(_BIDI_BASE_TYPES) == 256
_BIDI_ARABIC_TYPES = [
    "" if bidi_type == "~" else bidi_type for bidi_type in (
        "AN AN AN AN AN AN ON ON AL ET ET AL CS AL ON ON NSM NSM NSM NSM NSM NSM "
        "NSM NSM NSM NSM NSM AL AL ~ AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL "
        "AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL "
        "AL AL AL AL AL NSM NSM NSM NSM NSM NSM NSM NSM NSM NSM NSM NSM NSM NSM NSM "
        "NSM NSM NSM NSM NSM NSM AN AN AN AN AN AN AN AN AN AN ET AN AN AL AL AL "
        "NSM AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL "
        "AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL "
        "AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL "
        "AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL AL "
        "AL AL AL NSM NSM NSM NSM NSM NSM NSM AN ON NSM NSM NSM NSM NSM NSM AL AL "
        "NSM NSM ON NSM NSM NSM NSM AL AL EN EN EN EN EN EN EN EN EN EN AL AL AL AL "
        "AL AL "
    ).split()
]
assert len(_BIDI_ARABIC_TYPES) == 256


def _apply_bidi_reordering(text: str, start_level: int = -1, vertical: bool = False) -> str:
    """Apply the simplified single-line UAX#9 pass used for flushed PDF text items. Empty, vertical, and purely LTR text pass through. Otherwise the pass resolves W1-W7/N1-N2/I1-I2 levels from the tables above, reverses runs, and strips literal '<'/'>'. Astral-codepoint handling follows Python strings; surrogate pairs are not corrupted because both halves classify L at equal levels and reversal spans restore the pair."""
    if not text or vertical:
        return text
    count_item = len(text)
    chars = list(text)
    types: list[str] = [""] * count_item
    num_bidi = 0
    for index_value, char in enumerate(chars):
        codepoint = ord(char)
        token_value = "L"
        if codepoint <= 0xFF:
            token_value = _BIDI_BASE_TYPES[codepoint]
        elif 0x0590 <= codepoint <= 0x05F4:
            token_value = "R"
        elif 0x0600 <= codepoint <= 0x06FF:
            token_value = _BIDI_ARABIC_TYPES[codepoint & 0xFF]
        elif 0x0700 <= codepoint <= 0x08AC:
            token_value = "AL"
        if token_value in ("R", "AL", "AN"):
            num_bidi += 1
        types[index_value] = token_value
    if num_bidi == 0:
        return text
    if start_level == -1:
        if num_bidi / count_item < 0.3 and count_item > 4:
            start_level = 0
        else:
            start_level = 1
    levels = [start_level] * count_item
    entry_item = "R" if (start_level & 1) else "L"
    sor = entry_item
    eor = sor
    # W1: NSM takes the type of the previous character (sor at run start).
    last = sor
    for index_value in range(count_item):
        if types[index_value] == "NSM":
            types[index_value] = last
        else:
            last = types[index_value]
    # W2: EN after an AL (searching back to the first strong type) becomes AN.
    last = sor
    for index_value in range(count_item):
        token_value = types[index_value]
        if token_value == "EN":
            types[index_value] = "AN" if last == "AL" else "EN"
        elif token_value in ("R", "L", "AL"):
            last = token_value
    # W3: AL -> R.
    for index_value in range(count_item):
        if types[index_value] == "AL":
            types[index_value] = "R"
    # W4: single ES between ENs -> EN; single CS between same-type numbers.
    for index_value in range(1, count_item - 1):
        if types[index_value] == "ES" and types[index_value - 1] == "EN" and types[index_value + 1] == "EN":
            types[index_value] = "EN"
        if (types[index_value] == "CS" and types[index_value - 1] in ("EN", "AN")
                and types[index_value + 1] == types[index_value - 1]):
            types[index_value] = types[index_value - 1]
    # W5: ET runs adjacent to EN -> EN.
    for index_value in range(count_item):
        if types[index_value] == "EN":
            for state_item in range(index_value - 1, -1, -1):
                if types[state_item] != "ET":
                    break
                types[state_item] = "EN"
            for state_item in range(index_value + 1, count_item):
                if types[state_item] != "ET":
                    break
                types[state_item] = "EN"
    # W6: WS/ES/ET/CS -> ON.
    for index_value in range(count_item):
        if types[index_value] in ("WS", "ES", "ET", "CS"):
            types[index_value] = "ON"
    # W7: EN after an L (searching back to the first strong type) -> L.
    last = sor
    for index_value in range(count_item):
        token_value = types[index_value]
        if token_value == "EN":
            types[index_value] = "L" if last == "L" else "EN"
        elif token_value in ("R", "L"):
            last = token_value
    # N1: neutrals between same-direction strongs take that direction
    # (numbers count as R); N2: leftovers take the embedding direction.
    index_value = 0
    while index_value < count_item:
        if types[index_value] == "ON":
            end = index_value + 1
            while end < count_item and types[end] == "ON":
                end += 1
            before = types[index_value - 1] if index_value > 0 else sor
            after = types[end + 1] if end + 1 < count_item else eor
            if before != "L":
                before = "R"
            if after != "L":
                after = "R"
            if before == after:
                for state_item in range(index_value, end):
                    types[state_item] = before
            index_value = end - 1
        index_value += 1
    for index_value in range(count_item):
        if types[index_value] == "ON":
            types[index_value] = entry_item
    # I1/I2: level bumps.
    for index_value in range(count_item):
        token_value = types[index_value]
        if levels[index_value] % 2 == 0:
            if token_value == "R":
                levels[index_value] += 1
            elif token_value in ("AN", "EN"):
                levels[index_value] += 2
        else:
            if token_value in ("L", "AN", "EN"):
                levels[index_value] += 1
    #: reverse contiguous runs from the highest level down to the lowest
    # odd level.
    highest = -1
    lowest_odd = 99
    for layout_value in levels:
        if layout_value > highest:
            highest = layout_value
        if layout_value < lowest_odd and (layout_value & 1):
            lowest_odd = layout_value
    for level in range(highest, lowest_odd - 1, -1):
        start = -1
        for index_value in range(count_item):
            if levels[index_value] < level:
                if start >= 0:
                    chars[start:index_value] = chars[start:index_value][::-1]
                    start = -1
            elif start < 0:
                start = index_value
        if start >= 0:
            chars[start:count_item] = chars[start:count_item][::-1]
    # text extraction final loop: literal '<' and '>' are dropped (numBidi > 0 only).
    return "".join("" if char in "<>" else char for char in chars)


def _rtl_sign(char: str) -> int:
    """+1 for LTR runs, -1 for a strong right-to-left char (bidi class R/AL, e.g. Hebrew/Arabic). PDFium reports RTL text in logical order with decreasing char origins, so the LTR ``advance = ox - prev_text_x`` model (prev_text_x = ox+glyph_w, a right edge) yields a large negative advance. For RTL chunks the x-axis is signed with ``sign*ox`` so the reading-direction advance is positive and the existing LTR merge logic applies unchanged."""
    return -1 if unicodedata.bidirectional(char) in ("R", "AL") else 1


def _reverse_if_rtl(chars: str) -> str:
    """span merger ``RTL ligature reversal`` : reverse a multi-char (Arabic/Hebrew ligature) value when its FIRST code unit is in the Hebrew ``[0x0590,0x05ff)`` or Arabic ``[0x0600,0x06ff)`` range (Unicode range table[11]/ [13], ``right-to-left range test`` uses ``>= begin  and  < end``, so the range end is EXCLUSIVE). text extraction wraps every glyph's ``normalized Unicode`` in this, so a table value like "\u0626\u062c" emitted for U+FC00 is reversed to "\u062c\u0626"; a single-char value (length <= 1) is returned as-is."""
    if len(chars) <= 1:
        return chars
    first_codepoint = ord(chars[0])
    if (0x0590 <= first_codepoint < 0x05FF) or (0x0600 <= first_codepoint < 0x06FF):
        return chars[::-1]
    return chars


def _read_end(mapping: dict, sign: int) -> float:
    """The reading-direction FAR edge of a glyph (the edge facing the next char). PDFium reports the origin (ox) as the glyph's LEFT edge in both directions; the glyph extends RIGHT by glyph_w. So: * LTR (reading right): far edge = right edge = max(ox+glyph_w, ink right). * RTL (reading left): far edge = LEFT edge = ox (the origin itself). The next char's gap is then measured to its NEAR edge -- ox for LTR, ox+glyph_w for RTL -- in ``_read_gap`` below. (Earlier this added glyph_w on the RTL side too, which used the PREVIOUS glyph's width and injected spurious spaces.)"""
    if sign > 0:
        return max(mapping["ox"] + mapping["glyph_w"], mapping["right"])
    return mapping["ox"]


def _read_gap(prev_far: float, other_mapping: dict, sign: int) -> float:
    """Reading-direction gap between the previous glyph's far edge and the current glyph's NEAR edge. LTR near edge = ox (left); RTL near edge = ox+glyph_w (right). ==0 for adjacent glyphs, >0 for a word gap, <0 for a backward jump."""
    if sign > 0:
        return other_mapping["ox"] - prev_far
    return prev_far - (other_mapping["ox"] + other_mapping["glyph_w"])
