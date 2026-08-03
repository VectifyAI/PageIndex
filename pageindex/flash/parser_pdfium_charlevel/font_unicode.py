"""Simple-font encoding resolution and per-font Unicode map construction."""

from __future__ import annotations

import re

from .glyph_tables import (
    _load_glyph_tables,
    _get_unicode_for_glyph,
    _from_char_code,
)
from .cmap_parse import (
    _to_number,
    _parse_int,
    _parse_tounicode_cmap,
)


_TYPE1_SPECIAL_BYTES = b"/[]{}()"
# content stream tokenizer tokenises with PDF parser whitespace = {SP, TAB, CR, LF}
# ONLY -- narrower than the content-stream/CMap lexer's whitespace-byte set (no 0x0C, no 0x00).
_TYPE1_WHITESPACE_BYTES = frozenset(b" \t\r\n")


def _type1_builtin_encoding(font_file: bytes):
    """content stream tokenizer font-header extraction's /Encoding case, run over the cleartext segment of an embedded Type1 font file. Returns ("named", encoding-name) | ("array", {code: glyphname}) | None."""
    end = font_file.find(b"eexec")
    head = font_file[: end if end >= 0 else len(font_file)]

    header_tokens: list[bytes] = []
    index_value, count_item = 0, len(head)
    while index_value < count_item:
        candidate_item = head[index_value]
        if candidate_item in _TYPE1_WHITESPACE_BYTES:
            index_value += 1
        elif candidate_item == 0x25:  # % comment runs to EOL (PDF token reader's comment eater)
            while index_value < count_item and head[index_value] not in b"\r\n":
                index_value += 1
        elif candidate_item in _TYPE1_SPECIAL_BYTES:
            header_tokens.append(head[index_value:index_value + 1])
            index_value += 1
        else:
            state_item = index_value
            while state_item < count_item and head[state_item] not in _TYPE1_WHITESPACE_BYTES and head[state_item] not in _TYPE1_SPECIAL_BYTES:
                state_item += 1
            header_tokens.append(head[index_value:state_item])
            index_value = state_item

    def _header_token(number: int) -> bytes | None:
        return header_tokens[number] if number < len(header_tokens) else None

    # font-header extraction consumes "/"+name PAIRS and keeps scanning after each
    # case, so a LATER /Encoding overwrites an earlier one (last wins), and a
    # "//Encoding" pair is consumed whole (its bare "Encoding" never matches).
    result: tuple | None = None
    page_value = 0
    while page_value < len(header_tokens):
        if header_tokens[page_value] != b"/":
            page_value += 1
            continue
        name_tok = _header_token(page_value + 1)
        page_value += 2  # the name scanner advances past the slash unconditionally after a '/'
        if name_tok != b"Encoding":
            continue
        arg = _header_token(page_value)
        if arg is None:
            # Detail: encoding lookup(null) -> null assigned to built-in encoding.

            result = None
            break
        if not arg.isdigit():
            # named encoding: encoding lookup(name) -- null when unknown
            # Overwrite any previous result; later encoding declarations win.
            glyph_name, encs = _load_glyph_tables()
            name = arg.decode("latin-1")
            result = ("named", name) if name in encs else None
            page_value += 1
            continue
        # Decimal integer count is parsed through float64, then coerced to int32.
        # Huge digit strings may round or overflow to Infinity before coercion.
        array_size_float = float(arg)
        size = 0 if array_size_float == float("inf") else ((int(array_size_float) + 2**31) % 2**32) - 2**31
        page_value += 1  # at 'array'
        enc: dict[int, str] = {}
        for _ in range(size):
            token_value = _header_token(page_value)
            while token_value is not None and token_value not in (b"dup", b"def"):
                page_value += 1
                token_value = _header_token(page_value)
            if token_value is None:
                # Invalid headers abort the scan and keep any previous encoding.
                return result
            if token_value == b"def":
                break
            page_value += 1  # past 'dup'
            # Malformed integer tokens coerce to 0 and do not abort the entry.
            token_value = _header_token(page_value)
            try:
                value = _parse_int(token_value.decode("latin-1"), 10) if token_value is not None else 0.0
            except OverflowError:
                value = float("inf")     # huge digit run
            if value != value or value == float("inf") or value == -float("inf"):
                value = 0.0              # ToInt32(NaN / ±Infinity) = 0
            idx = ((int(value) + 2**31) % 2**32) - 2**31
            page_value += 1
            page_value += 1  # '/' slot consumed blindly
            group_value = _header_token(page_value)
            page_value += 1
            if group_value is not None:
                enc[idx] = group_value.decode("latin-1")
            page_value += 1  # 'put' slot consumed blindly
        result = ("array", enc)  # keep scanning: a later /Encoding wins
    return result


def _simple_font_to_unicode(
    default_enc: list[str],
    base_encoding_name: str | None,
    differences: dict[int, str],
    force_glyphs: bool = False,
) -> dict[int, str]:
    """content stream tokenizer simple-font Unicode-map construction, detailed behavior (including the byte-to-character conversion 16-bit truncation on glyphlist hits, the Gxx/g00xx/Cdd/cdd/u heuristics, the base encoding correction branch, and the forced glyph-name pass re-parse when a Cdd name turns out hexadecimal)."""
    glyphs, encs = _load_glyph_tables()
    encoding: dict[int, str] = {font: glyph_name_value for font, glyph_name_value in enumerate(default_enc)}
    for font, glyph_name_value in differences.items():
        if glyph_name_value == ".notdef":
            continue  # text extraction skips .notdef (.notdef entries)
        encoding[font] = glyph_name_value

    to_unicode: dict[int, str] = {}
    for charcode in sorted(encoding):
        glyph_name = encoding[charcode]
        if glyph_name == "":
            continue
        codepoint = glyphs.get(glyph_name)
        if codepoint is not None:
            to_unicode[charcode] = _from_char_code(codepoint)
            continue
        code = 0
        glyph_prefix = glyph_name[0]
        if glyph_prefix == "G":  # Gxx
            if len(glyph_name) == 3:
                parsed_integer = _parse_int(glyph_name[1:], 16)
                code = int(parsed_integer) if parsed_integer == parsed_integer else 0        # pi==pi: not NaN
        elif glyph_prefix == "g":  # g00xx
            if len(glyph_name) == 5:
                parsed_integer = _parse_int(glyph_name[1:], 16)
                code = int(parsed_integer) if parsed_integer == parsed_integer else 0
        elif glyph_prefix in ("C", "c"):  # Cdd{d} / cdd{d}
            if 3 <= len(glyph_name) <= 4:
                code_str = glyph_name[1:]
                if force_glyphs:
                    parsed_integer = _parse_int(code_str, 16)
                    code = int(parsed_integer) if parsed_integer == parsed_integer else 0
                else:
                    # First try the full numeric grammar. Only when that is NaN
                    # and tolerant base-16 parsing succeeds do we re-parse the
                    # whole encoding as base-16. Non-integer numeric values pass
                    # through and then fail the integer gate below.
                    num = _to_number(code_str)
                    if num != num:                       # NaN
                        parsed_integer = _parse_int(code_str, 16)
                        if parsed_integer == parsed_integer:
                            return _simple_font_to_unicode(
                                default_enc, base_encoding_name,
                                differences, force_glyphs=True)
                        code = 0
                    elif num.is_integer():
                        code = int(num)
                    else:
                        code = 0
        elif glyph_prefix == "u":
            unicode_unit = _get_unicode_for_glyph(glyph_name, glyphs)
            if unicode_unit != -1:
                code = unicode_unit
        if 0 < code <= 0x10FFFF:
            # Prefer the base encoding glyph when code == charcode
            if base_encoding_name and code == charcode:
                base = encs.get(base_encoding_name)
                # the heading heuristics base encoding[charcode] for charcode > 255 is undefined
                # (falsy) -- fall through instead of IndexError.
                if base and 0 <= charcode < len(base) and base[charcode]:
                    to_unicode[charcode] = _from_char_code(
                        glyphs.get(base[charcode], 0))
                    continue
            to_unicode[charcode] = chr(code)  # code-point conversion
    return to_unicode


def _font_unicode_map(pdf_doc, xref: int) -> tuple[int, dict[int, str]] | None:
    """Return the final per-charcode glyph-unicode map for one font as ``(bytes_per_code, {charcode: unicode})``. Simple fonts use 1-byte codes; Identity-H/V composite fonts use 2-byte codes with the included ToUnicode map. ``None`` means uncovered input such as non-Identity composite CMaps or unreadable dictionaries; callers then skip the page patch walk and keep PDFium's output."""
    glyphs, encs = _load_glyph_tables()

    def _xref_key(number: int, other_text: str) -> tuple[str, str]:
        return pdf_doc.xref_get_key(number, other_text)

    pdf_value_type, pdf_value = _xref_key(xref, "Subtype")
    subtype = pdf_value.lstrip("/") if pdf_value_type == "name" else ""
    if subtype == "Type0":
        pdf_value_type, pdf_value = _xref_key(xref, "Encoding")
        if pdf_value_type != "name" or pdf_value.lstrip("/") not in ("Identity-H", "Identity-V"):
            return None
        # text extraction reads ToUnicode from the DESCENDANT dict first, then the
        # Type0 dict (the composite-font prepass uses the descendant for composites).
        desc_xref = 0
        delta_top, delta_value = _xref_key(xref, "DescendantFonts")
        if delta_top == "xref":
            delta_value = pdf_doc.xref_object(int(delta_value.split()[0]), compressed=True)
            delta_top = "array"
        if delta_top == "array":
            delta_matrix = re.search(r"(\d+)\s+\d+\s+R", delta_value)
            if delta_matrix:
                desc_xref = int(delta_matrix.group(1))
        pdf_value_type, pdf_value = ("null", "null")
        if desc_xref:
            pdf_value_type, pdf_value = _xref_key(desc_xref, "ToUnicode")
        if pdf_value_type != "xref":
            pdf_value_type, pdf_value = _xref_key(xref, "ToUnicode")
        tu_map: dict[int, str] | None = None
        if pdf_value_type == "xref":
            try:
                tu_map = _parse_tounicode_cmap(
                    pdf_doc.xref_stream(int(pdf_value.split()[0])))
            except Exception:
                tu_map = None  # ToUnicode parsing rejects -> no ToUnicode map
        # The "font carries a ToUnicode map" flag is set only for a present,
        # accepted and NON-EMPTY map. A missing, rejected or empty ToUnicode all
        # leave it false, so all three take the composite branch below.
        if tu_map:
            return 2, tu_map
        # No usable ToUnicode: predefined-collection Unicode-map construction
        # maps Adobe-{GB1,CNS1,Japan1,Korea1} CIDSystemInfo through the shipped
        # Adobe-XX-UCS2 bcmap (real unicode per cid) -- not implemented.
        # Returning identity chr(cid) would actively CORRUPT PDFium's
        # table-driven decode for that class, so keep the guarded None (PDFium
        # output). Every other registry/ordering IS the identity fallback.
        if desc_xref:
            right_type, right_value_local = _xref_key(desc_xref, "CIDSystemInfo/Registry")
            other_type, other_value_local = _xref_key(desc_xref, "CIDSystemInfo/Ordering")
            reg = re.sub(r"[()\s]", "", right_value_local) if right_type != "null" else ""
            ordering = re.sub(r"[()\s]", "", other_value_local) if other_type != "null" else ""
            if reg == "Adobe" and ordering in ("GB1", "CNS1", "Japan1", "Korea1"):
                return None
        return 2, {}  # identity Unicode map: unicode == chr(cid)
    pdf_value_type, pdf_value = _xref_key(xref, "BaseFont")
    base_font = pdf_value.lstrip("/") if pdf_value_type == "name" else ""

    flags = 0
    fd_xref = 0
    has_descriptor = False
    pdf_value_type, pdf_value = _xref_key(xref, "FontDescriptor")
    if pdf_value_type == "xref":
        fd_xref = int(pdf_value.split()[0])
        has_descriptor = True
        font_token, font_value = _xref_key(fd_xref, "Flags")
        if font_token == "int":
            flags = int(font_value)
    elif pdf_value_type == "dict":
        has_descriptor = True
        flags_match = re.search(r"/Flags\s+([+-]?\d+)", pdf_value)
        if flags_match:
            flags = int(flags_match.group(1))
    if not has_descriptor and subtype != "Type3":
        # font loading's simulated descriptor (span merger `if (!descriptor)`,
        # non-Type3 branch): flags come from the BaseFont name with the style
        # suffix stripped -- Symbol/Dingbats/ZapfDingbats get Symbolic, all
        # else Nonsymbolic. (the heading heuristics also sets Serif/FixedPitch there; nothing in
        # this implementation consults those bits, so they are not simulated.) A missing
        # BaseFont makes the heading heuristics throw parse error -> fallback font, i.e. text extraction DROPS
        # that font's text entirely; returning None keeps PDFium's decode
        # instead -- the implementation's conservative boundary, not the same branch. Type3
        # takes the OTHER the heading heuristics arm:
        # a barebones descriptor with NO flags and NO BaseFont requirement
        # (dvips bitmap fonts have neither), so flags stay 0 there.
        if not base_font:
            return None
        base_wo_style = re.sub(r"[,_]", "-", base_font).split("-")[0]
        flags = 4 if base_wo_style in ("Symbol", "Dingbats", "ZapfDingbats") else 32

    file_key = None
    if fd_xref:
        for char_code in ("FontFile", "FontFile2", "FontFile3"):
            font_token, font_value = _xref_key(fd_xref, char_code)
            if font_token == "xref":
                file_key = (char_code, int(font_value.split()[0]))
                break

    # --- encoding and Differences extraction: /Encoding -> base encodingName + differences
    differences: dict[int, str] = {}
    base_encoding_name: str | None = None
    pdf_value_type, pdf_value = _xref_key(xref, "Encoding")
    enc_obj: str | None = None
    if pdf_value_type == "name":
        base_encoding_name = pdf_value.lstrip("/")
    elif pdf_value_type == "xref":
        enc_obj = pdf_doc.xref_object(int(pdf_value.split()[0]), compressed=True)
    elif pdf_value_type == "dict":
        enc_obj = pdf_value
    if enc_obj is not None:
        flags_match = re.search(r"/BaseEncoding\s*/([^\s/\[\]<>()]+)", enc_obj)
        if flags_match:
            base_encoding_name = flags_match.group(1)
        flags_match = re.search(r"/Differences\s*\[", enc_obj)
        if flags_match:
            depth = 1
            scan_index = flags_match.end()
            while scan_index < len(enc_obj) and depth:
                if enc_obj[scan_index] == "[":
                    depth += 1
                elif enc_obj[scan_index] == "]":
                    depth -= 1
                scan_index += 1
            idx = 0
            for token_match in re.findall(r"/([^\s/\[\]<>()]+)|(\d+)", enc_obj[flags_match.end():scan_index - 1]):
                if token_match[1]:
                    idx = int(token_match[1])
                else:
                    name_value = re.sub(
                        r"#([0-9a-fA-F]{2})",
                        lambda encoding_key: chr(int(encoding_key.group(1), 16)), token_match[0])
                    differences[idx] = name_value
                    idx += 1
    # Table 114: a named base encoding must be one of these three.
    if base_encoding_name not in ("MacRomanEncoding", "MacExpertEncoding",
                                  "WinAnsiEncoding"):
        base_encoding_name = None

    if base_encoding_name:
        default_name = base_encoding_name
    else:
        symbolic = bool(flags & 4)
        nonsymbolic = bool(flags & 32)
        default_name = "StandardEncoding"
        if subtype == "TrueType" and not nonsymbolic:
            default_name = "WinAnsiEncoding"
        if symbolic:
            default_name = "MacRomanEncoding"
            if file_key is None:
                if re.search(r"Symbol", base_font, re.IGNORECASE):
                    default_name = "SymbolSetEncoding"
                elif re.search(r"Dingbats|Wingdings", base_font, re.IGNORECASE):
                    default_name = "ZapfDingbatsEncoding"
    default_enc = encs[default_name]
    has_encoding = bool(base_encoding_name) or bool(differences)

    included: dict[int, str] | None = None
    pdf_value_type, pdf_value = _xref_key(xref, "ToUnicode")
    if pdf_value_type == "xref":
        try:
            included = _parse_tounicode_cmap(pdf_doc.xref_stream(int(pdf_value.split()[0])))
        except Exception:
            included = None  # ToUnicode parsing error path: treat as absent

    # Detail: included ToUnicode-map flag = !!toUnicode  and  toUnicode.length > 0. An

    # empty-but-valid ToUnicode (parsed to {}) is treated as ABSENT, so fall
    # through to _simple_font_to_unicode + the Type1 builtin amend below
    # (Type 1 Unicode-map repair), while preserving the existing item-boundary semantics.
    if included:
        final = dict(included)
        if has_encoding:  # predefined collection Unicode-map construction -> fallback Unicode map gap fill
            for font, glyph_name in _simple_font_to_unicode(
                    default_enc, base_encoding_name, differences).items():
                if font not in final:
                    final[font] = glyph_name
        return 1, final

    final = _simple_font_to_unicode(default_enc, base_encoding_name, differences)
    # Type 1 Unicode-map repair: amend from the embedded Type1 program's builtin
    # encoding (codes not already fixed by the dict's Encoding entry).
    if file_key is not None and file_key[0] == "FontFile" and subtype in (
            "Type1", "MMType1"):
        try:
            builtin = _type1_builtin_encoding(pdf_doc.xref_stream(file_key[1]))
        except Exception:
            builtin = None
        if builtin is not None:
            kind, payload = builtin
            # `built-in encoding == properties.defaultEncoding` (same module

            # array object) -- true iff both name the same predefined encoding.
            if not (kind == "named" and payload == default_name):
                items: list[tuple[int, str]] = (
                    list(enumerate(encs[payload])) if isinstance(payload, str)
                    else sorted(payload.items()))
                for font, name_value in items:
                    if has_encoding and (base_encoding_name or font in differences):
                        continue
                    if not name_value:
                        continue
                    codepoint = _get_unicode_for_glyph(name_value, glyphs)
                    if codepoint != -1:
                        final[font] = _from_char_code(codepoint)  # amend overwrites
    return 1, final
