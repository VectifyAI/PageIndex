"""PostScript number parsing and ToUnicode CMap interpretation."""

from __future__ import annotations

import math
import re

from .pdf_objects import (
    _PDF_WHITESPACE_BYTES,
    _PDF_DELIMITER_BYTES,
    _PDF_STRING_ESCAPE_BYTES,
)
from .text_normalize import _WHITESPACE_CODEPOINTS


def _utf16be_units_to_str(units: list[int]) -> str:
    """Decode UTF-16BE token bytes into text. Odd trailing bytes pair with 0. A unit can exceed 0xFF during range carry, and no byte mask is applied before surrogate handling, so a composed value may exceed 0xFFFF and become an astral character."""
    if len(units) % 2:
        units = units + [0]
    out: list[int] = []
    key_value = 0
    while key_value < len(units):
        width_one = (units[key_value] << 8) | units[key_value + 1]
        key_value += 2
        if (width_one & 0xF800) != 0xD800:
            out.append(width_one)
            continue
        width_two = 0
        if key_value < len(units):
            width_two = (units[key_value] << 8) | units[key_value + 1]
            key_value += 2
        out.append(((width_one & 0x3FF) << 10) + (width_two & 0x3FF) + 0x10000)
    return "".join(chr(candidate_item) for candidate_item in out)


# ASCII-only numeric grammar used for PDF numeric-name heuristics. It uses the
# same decimal grammar as model.to_number but without NFKC normalization. Trim set is the
# Unicode WhiteSpace + LineTerminator set, not Python's str.strip set.
_NUM_DECIMAL_RE = re.compile(r"^[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
_NUM_INFINITY_RE = re.compile(r"^[+-]?Infinity$")
_NUM_HEX_RE = re.compile(r"^0[xX][0-9a-fA-F]+$")
_NUM_OCTAL_RE = re.compile(r"^0[oO][0-7]+$")
_NUM_BINARY_RE = re.compile(r"^0[bB][01]+$")
_WHITESPACE_STRIP = "".join(chr(unit_value) for unit_value in _WHITESPACE_CODEPOINTS)


def _ieee_div(value: float, other_item: float) -> float:
    """IEEE-754 division, no ZeroDivisionError (``0/0-> NaN, ``x/±0-> ±Inf with the usual sign rules)."""
    if other_item != 0.0:
        return value / other_item
    if value == 0.0 or value != value:
        return math.nan
    return math.inf if (value > 0.0) == (math.copysign(1.0, other_item) > 0.0) else -math.inf


def _compute_skew(mtx: tuple) -> float:
    """Return the text matrix skew score for an item. transform's rotation/shear ratios, no zero guard (cardinal rotation -> Inf, upright -> 0). Degenerate case: the matrix-size path folds font size into the transform, so ``Tf 0`` text gives 0/0 = NaN there; the PDFium object matrix keeps font size separate and yields finite ratios (degenerate invisible text only)."""
    primary_item, secondary_item, candidate_item, reference_item = mtx
    quad_one = _ieee_div(secondary_item, primary_item)
    quad_two = _ieee_div(candidate_item, reference_item)
    return quad_one * quad_one + quad_two * quad_two


def _to_number(text: str) -> float:
    """/ ``numeric conversion`` (no NFKC): trim parser whitespace, ``""-> 0, then the numeric literal grammar (decimal/exponent, ``0x``/``0o``/``0b``, ``+-Infinity``); anything else -> NaN."""
    token_value = text.strip(_WHITESPACE_STRIP)
    if token_value == "":
        return 0.0
    if _NUM_INFINITY_RE.match(token_value):
        return -math.inf if token_value[0] == "-" else math.inf
    if _NUM_HEX_RE.match(token_value):
        return float(int(token_value[2:], 16))
    if _NUM_OCTAL_RE.match(token_value):
        return float(int(token_value[2:], 8))
    if _NUM_BINARY_RE.match(token_value):
        return float(int(token_value[2:], 2))
    if _NUM_DECIMAL_RE.match(token_value):
        return float(token_value)
    return math.nan


def _parse_int(text: str, radix: int) -> float:
    """skip leading parser whitespace, an optional sign, an optional ``0x`` prefix when ``radix == 16``, then the leading run of radix digits. Returns ``NaN`` (as in the heading heuristics) when no digit is consumed."""
    token_value = text.lstrip(_WHITESPACE_STRIP)
    index_value = 0
    neg = False
    if index_value < len(token_value) and token_value[index_value] in "+-":
        neg = token_value[index_value] == "-"
        index_value += 1
    if radix == 16 and token_value[index_value:index_value + 2] in ("0x", "0X"):
        index_value += 2
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"[:radix]
    start = index_value
    val = 0
    while index_value < len(token_value) and token_value[index_value].lower() in digits:
        val = val * radix + digits.index(token_value[index_value].lower())
        index_value += 1
    if index_value == start:
        return math.nan
    return float(-val if neg else val)


def _cmap_str_to_int(seq) -> int:
    """Accumulate CMap definition-code bytes with 32-bit unsigned wrap."""
    primary_item = 0
    for codepoint in seq:
        primary_item = ((primary_item << 8) | codepoint) & 0xFFFFFFFF
    return primary_item


def _parse_tounicode_cmap(data: bytes) -> dict[int, str]:
    """CMap reader for ToUnicode streams, following text extraction CMap parsing + ToUnicode parsing: bfchar/bfrange with hex, literal-string, and (bfrange dst / array elements) integer tokens, plus cidchar/cidrange (numeric entries -> code-point conversion, the numeric-CID class). Structural junk is contained per block like CMap parsing's warn-and-continue catch (the block is dropped, the map survives); only decode-level errors (chr on a code-point conversion-invalid value) propagate so the caller reaches span merger ToUnicode parsing rejection path (-> no included map)."""
    tokens: list = []
    index_value, count_item = 0, len(data)
    while index_value < count_item:
        candidate_item = data[index_value]
        if candidate_item in _PDF_WHITESPACE_BYTES:
            index_value += 1
        elif candidate_item == 0x25:  # comment
            while index_value < count_item and data[index_value] not in b"\r\n":
                index_value += 1
        elif candidate_item == 0x3C:  # << dict-open (skip) or <hex>
            if index_value + 1 < count_item and data[index_value + 1] == 0x3C:
                index_value += 2
                continue
            state_item = data.find(b">", index_value)
            if state_item < 0:
                break  # unterminated hex string: stop and keep tokens already read
            hex_values = "".join(chr(secondary_item) for secondary_item in data[index_value + 1:state_item]
                           if chr(secondary_item) in "0123456789abcdefABCDEF")
            if len(hex_values) % 2:
                hex_values = hex_values[:-1]  # drop a lone trailing hex digit
            tokens.append(("hex", tuple(bytes.fromhex(hex_values))))
            index_value = state_item + 1
        elif candidate_item == 0x3E:  # >> dict-close (skip)
            index_value += 2 if (index_value + 1 < count_item and data[index_value + 1] == 0x3E) else 1
        elif candidate_item in b"[]":
            tokens.append(("delim", chr(candidate_item)))
            index_value += 1
        elif candidate_item == 0x2F:  # /name
            state_item = index_value + 1
            while state_item < count_item and data[state_item] not in _PDF_WHITESPACE_BYTES and data[state_item] not in _PDF_DELIMITER_BYTES:
                state_item += 1
            tokens.append(("name", data[index_value + 1:state_item].decode("latin-1")))
            index_value = state_item
        elif candidate_item == 0x28:  # (string) -- literal-string lexer code units (dst values)
            depth = 0
            unicode_scalar: list[int] = []
            while index_value < count_item:
                byte_value = data[index_value]
                if byte_value == 0x5C:
                    if index_value + 1 >= count_item:
                        index_value += 1
                        break
                    entry_item = data[index_value + 1]
                    if entry_item in _PDF_STRING_ESCAPE_BYTES:
                        unicode_scalar.append(_PDF_STRING_ESCAPE_BYTES[entry_item])
                        index_value += 2
                    elif 0x30 <= entry_item <= 0x37:
                        state_item = index_value + 1
                        val = 0
                        while state_item < count_item and state_item - index_value <= 3 and 0x30 <= data[state_item] <= 0x37:
                            val = (val << 3) | (data[state_item] - 0x30)
                            state_item += 1
                        unicode_scalar.append(val)
                        index_value = state_item
                    elif entry_item in (0x0D, 0x0A):
                        index_value += 2
                        if entry_item == 0x0D and index_value < count_item and data[index_value] == 0x0A:
                            index_value += 1
                    else:
                        unicode_scalar.append(entry_item)
                        index_value += 2
                    continue
                if byte_value == 0x28:
                    if depth:
                        unicode_scalar.append(byte_value)
                    depth += 1
                elif byte_value == 0x29:
                    depth -= 1
                    if depth == 0:
                        index_value += 1
                        break
                    unicode_scalar.append(byte_value)
                else:
                    unicode_scalar.append(byte_value)
                index_value += 1
            tokens.append(("hex", tuple(unicode_scalar)))
        else:
            state_item = index_value
            while state_item < count_item and data[state_item] not in _PDF_WHITESPACE_BYTES and data[state_item] not in _PDF_DELIMITER_BYTES:
                state_item += 1
            word = data[index_value:state_item].decode("latin-1")
            if (0x30 <= data[index_value] <= 0x39) or data[index_value] in b"+-.":
                try:
                    numeric_value = float(word)
                except ValueError:
                    numeric_value = 0.0
                tokens.append(("num", numeric_value))
            else:
                tokens.append(("op", word))
            index_value = state_item

    out: dict[int, str] = {}

    def codepoint_to_string(numeric_value: float) -> str:
        # ToUnicode parsing numeric entry: code-point conversion(token) -- its
        # RangeError (non-integer / out of range) kills the whole map, so
        # chr's ValueError propagate.
        codepoint = int(numeric_value)
        if codepoint != numeric_value:
            raise ValueError("code-point conversion non-integer")
        return chr(codepoint)

    def is_int(numeric_value: float) -> bool:
        # The integer test that guards the numeric-entry check and selects the
        # destination branch rejects +-Infinity, NaN AND any fractional value.
        return math.isfinite(numeric_value) and numeric_value == int(numeric_value)

    def map_range_units(range_start: int, range_end: int, units: list[int]) -> None:
        # text extraction CMap.bf-range mapping : ``last byte`` is FIXED to
        # the ORIGINAL dst length-1; only THAT byte index is incremented. On
        # 0xFF overflow it carries into byte last byte-1 (byte-to-character conversion ToUint16
        # == the & 0xFFFF) and sets the tail to 0x00; the next non-overflow
        # step is substring(0,last byte)+chr(next), so a 1-byte dst collapses
        # back to ONE byte. A 1-byte 0xFF overflow gives "\x00\x00"
        # Empty destinations yield "" for the first code and "\x00" for each
        # subsequent code after carry.
        last_byte = len(units) - 1
        for code in range(range_start, range_end + 1):
            out[code] = _utf16be_units_to_str(units)
            if last_byte < 0:
                units = [0x00]
                continue
            cur = units[last_byte] if last_byte < len(units) else 0
            nxt = cur + 1
            if nxt > 0xFF:
                if last_byte - 1 >= 0:
                    units = (units[:last_byte - 1]
                             + [(units[last_byte - 1] + 1) & 0xFFFF, 0x00])
                else:
                    units = [0x00, 0x00]
            else:
                units = units[:last_byte] + [nxt]

    key_value = 0
    while key_value < len(tokens):
        kind, val = tokens[key_value]
        if kind == "op" and val == "beginbfchar":
            key_value += 1
            while key_value + 1 < len(tokens) and tokens[key_value][0] == "hex":
                src = _cmap_str_to_int(tokens[key_value][1])
                if tokens[key_value + 1][0] != "hex":
                    # the heading heuristics string-operand check throws -> CMap parsing catch drops the
                    # rest of the block, map survives.
                    key_value += 2
                    break
                out[src] = _utf16be_units_to_str(list(tokens[key_value + 1][1]))
                key_value += 2
        elif kind == "op" and val == "beginbfrange":
            key_value += 1
            while (key_value + 1 < len(tokens) and tokens[key_value][0] == "hex"
                   and tokens[key_value + 1][0] == "hex"):
                src_start = _cmap_str_to_int(tokens[key_value][1])
                src_end = _cmap_str_to_int(tokens[key_value + 1][1])
                key_value += 2
                if src_end - src_start > 0xFFFFFF:
                    # The range-limit throw is raised from INSIDE the bf-range
                    # mapping itself, i.e. from inside the call that CMap
                    # parsing wraps, so the rest of the block goes with it (the
                    # destination has already been lexed -- for an array, up to
                    # and including the "]").
                    if key_value < len(tokens) and tokens[key_value] == ("delim", "["):
                        while key_value < len(tokens) and tokens[key_value] != ("delim", "]"):
                            key_value += 1
                        key_value += 1
                    elif key_value < len(tokens) and tokens[key_value][0] in ("hex", "num"):
                        key_value += 1
                    break
                if key_value < len(tokens) and tokens[key_value] == ("delim", "["):
                    key_value += 1
                    code = src_start
                    # The array form stores EVERY lexed object up to "]" or end
                    # of input; the UTF-16BE walk over a value that has no
                    # length (a name, an operator) runs zero times and yields
                    # the empty string.
                    while key_value < len(tokens) and tokens[key_value] != ("delim", "]"):
                        if code <= src_end:
                            dst_token = tokens[key_value]
                            if dst_token[0] == "hex":
                                out[code] = _utf16be_units_to_str(list(dst_token[1]))
                            elif dst_token[0] == "num":
                                out[code] = codepoint_to_string(dst_token[1])
                            else:
                                out[code] = ""
                        code += 1
                        key_value += 1
                    if key_value < len(tokens):
                        key_value += 1
                elif key_value < len(tokens) and tokens[key_value][0] == "hex":
                    units = list(tokens[key_value][1])
                    key_value += 1
                    map_range_units(src_start, src_end, units)
                elif key_value < len(tokens) and tokens[key_value][0] == "num" and is_int(tokens[key_value][1]):
                    # Integer destinations are one UTF-16 unit, then the normal
                    # increment walk applies. A non-integer number is neither an
                    # integer nor a string nor "[", so it falls through to the
                    # `else` arm below.
                    units = [int(tokens[key_value][1]) & 0xFFFF]
                    key_value += 1
                    map_range_units(src_start, src_end, units)
                else:
                    break  # parse error -> contained: drop the block
        elif kind == "op" and val == "begincidchar":
            key_value += 1
            while (key_value + 1 < len(tokens) and tokens[key_value][0] == "hex"
                   and tokens[key_value + 1][0] == "num"):
                if not is_int(tokens[key_value + 1][1]):
                    # The integer check throws -> the CMap parsing catch drops
                    # the rest of the block, map survives.
                    key_value += 2
                    break
                out[_cmap_str_to_int(tokens[key_value][1])] = codepoint_to_string(tokens[key_value + 1][1])
                key_value += 2
        elif kind == "op" and val == "begincidrange":
            key_value += 1
            while (key_value + 2 < len(tokens) and tokens[key_value][0] == "hex"
                   and tokens[key_value + 1][0] == "hex" and tokens[key_value + 2][0] == "num"):
                src_start = _cmap_str_to_int(tokens[key_value][1])
                src_end = _cmap_str_to_int(tokens[key_value + 1][1])
                start = tokens[key_value + 2][1]
                key_value += 3
                if not is_int(start):
                    break  # the integer check precedes CID-range mapping: block dropped
                if src_end - src_start > 0xFFFFFF:
                    break  # CID-range range-limit: the block is dropped too
                for code in range(src_start, src_end + 1):
                    out[code] = codepoint_to_string(start + (code - src_start))
        else:
            key_value += 1
    return out
