"""Content-stream show-operator tokenization and per-page operator tagging."""

from __future__ import annotations

from .pdf_objects import (
    _PDF_WHITESPACE_BYTES,
    _PDF_DELIMITER_BYTES,
    _PDF_STRING_ESCAPE_BYTES,
    _decode_pdf_name,
)


# Text items start at font/size changes, positional line breaks or gaps, and
# content-stream flush operators (q/Q, Do, gs-/Font, marked content). PDFium's
# flattened FPDF_PAGEOBJ_TEXT objects can be one-per-glyph for per-glyph Tj
# streams, so this merger keeps a strict per-object split unless a later
# whitespace-aware rule proves a prose continuation.
#
# q/Q flush grouping is intentionally disabled. It requires fragile ordinal
# alignment between flattened PDFium objects and content-stream show operators,
# while the merger only needs the content stream for per-show-op font names
# (vertical-font flags) and Unicode-map reconstruction. setFont/gs-font are not
# flush scopes here; font_key/fs changes carry the style-boundary split.
_FLUSH_OPS = frozenset({b"q", b"Q", b"Do", b"BDC", b"BMC", b"EMC"})
_SHOW_OPS = frozenset({b"Tj", b"TJ", b"'", b'"'})
# content stream tokenizer content operator table: {op: (operand count, variable operand count)}.
# Commands NOT in this table are span merger "Unknown command" -- warned and skipped
# with the accumulated args PRESERVED (not cleared).
# content stream tokenizer operator table's null-value entries: pure lexer aids so object parser's
# longest-known-command walk can pass through prefixes of longer commands
# (B -> BM -> BMC, f -> false, n -> null). Not operators.
_OP_LEX_PREFIX = frozenset({
    b"BM", b"BD", b"true", b"fa", b"fal", b"fals", b"false",
    b"nu", b"nul", b"null",
})
_OP_OPERAND_COUNTS: dict[bytes, tuple[int, bool]] = {
    b"w": (1, False), b"J": (1, False), b"j": (1, False), b"M": (1, False),
    b"d": (2, False), b"ri": (1, False), b"i": (1, False), b"gs": (1, False),
    b"q": (0, False), b"Q": (0, False), b"cm": (6, False), b"m": (2, False),
    b"l": (2, False), b"c": (6, False), b"v": (4, False), b"y": (4, False),
    b"h": (0, False), b"re": (4, False), b"S": (0, False), b"s": (0, False),
    b"f": (0, False), b"F": (0, False), b"f*": (0, False), b"B": (0, False),
    b"B*": (0, False), b"b": (0, False), b"b*": (0, False), b"n": (0, False),
    b"W": (0, False), b"W*": (0, False), b"BT": (0, False), b"ET": (0, False),
    b"Tc": (1, False), b"Tw": (1, False), b"Tz": (1, False), b"TL": (1, False),
    b"Tf": (2, False), b"Tr": (1, False), b"Ts": (1, False), b"Td": (2, False),
    b"TD": (2, False), b"Tm": (6, False), b"T*": (0, False), b"Tj": (1, False),
    b"TJ": (1, False), b"'": (1, False), b'"': (3, False), b"d0": (2, False),
    b"d1": (6, False), b"CS": (1, False), b"cs": (1, False), b"SC": (4, True),
    b"SCN": (33, True), b"sc": (4, True), b"scn": (33, True), b"G": (1, False),
    b"g": (1, False), b"RG": (3, False), b"rg": (3, False), b"K": (4, False),
    b"k": (4, False), b"sh": (1, False), b"BI": (0, False), b"ID": (0, False),
    b"EI": (1, False), b"Do": (1, False), b"MP": (1, False), b"DP": (2, False),
    b"BMC": (1, False), b"BDC": (2, False), b"EMC": (0, False),
    b"BX": (0, False), b"EX": (0, False),
}


def _tokenize_show_operators(
    content_bytes: bytes, init_tz: float = 1.0,
) -> tuple[list[int], list[bytes | None], list[tuple[int, ...]], list[float],
           list[tuple[int, bytes, bytes | None, float]]]:
    """Tokenize a PDF page content stream. For each text-showing operator, records the active flush scope, font redefinition name, raw charcode units, and horizontal scaling (starting at ``init_tz`` on stream entry, saved and restored by q/Q), plus every Form XObject paint position with the font and horizontal scaling live at that paint. The tokenizer is deliberately tolerant of malformed operators: it skips bad or short operands, preserves unknown-command operands, and emits an empty string for a show operator with the wrong string operand type."""
    flush_ids: list[int] = []
    fonts: list[bytes | None] = []
    show_text_units: list[tuple[int, ...]] = []
    xobject_paints: list[tuple[int, bytes, bytes | None, float]] = []
    horizontal_scales: list[float] = []
    flush_id = 0
    cur_font: bytes | None = None
    font_stack: list[bytes | None] = []
    cur_tz = init_tz
    tz_stack: list[float] = []
    opnds: list[tuple[str, object]] = []
    frames: list[tuple[str, list]] = []      # open [ / << collectors
    non_processed: list[tuple[str, object]] = []
    bi_mark: int | None = None

    def push(kind: str, val: object) -> None:
        (frames[-1][1] if frames else opnds).append((kind, val))

    index_value = 0
    count_item = len(content_bytes)
    while index_value < count_item:
        byte_value = content_bytes[index_value]
        if byte_value in _PDF_WHITESPACE_BYTES:
            index_value += 1
        elif byte_value == 0x25:  # % comment -> end of line
            while index_value < count_item and content_bytes[index_value] not in b"\r\n":
                index_value += 1
        elif byte_value == 0x28:  # ( literal string: decode per PDF 7.3.4.2
            depth = 0
            out: list[int] = []
            while index_value < count_item:
                literal_byte = content_bytes[index_value]
                if literal_byte == 0x5c:  # backslash escape
                    if index_value + 1 >= count_item:
                        index_value += 1
                        break
                    escape_byte = content_bytes[index_value + 1]
                    if escape_byte in _PDF_STRING_ESCAPE_BYTES:
                        out.append(_PDF_STRING_ESCAPE_BYTES[escape_byte])
                        index_value += 2
                    elif 0x30 <= escape_byte <= 0x37:  # \ddd octal, 1-3 digits
                        token_end = index_value + 1
                        val = 0
                        while token_end < count_item and token_end - index_value <= 3 and 0x30 <= content_bytes[token_end] <= 0x37:
                            val = (val << 3) | (content_bytes[token_end] - 0x30)
                            token_end += 1
                        # text extraction literal-string lexer pushes byte-to-character conversion with NO
                        # byte mask: \400..\777 stay 256..511 (the PDF-spec
                        # high-order-overflow mask is deliberately absent).
                        out.append(val)
                        index_value = token_end
                    elif escape_byte in (0x0D, 0x0A):  # \<EOL> line continuation
                        index_value += 2
                        if escape_byte == 0x0D and index_value < count_item and content_bytes[index_value] == 0x0A:
                            index_value += 1
                    else:  # \x -> x
                        out.append(escape_byte)
                        index_value += 2
                    continue
                # NOTE: bare CR/LF inside a literal string fall through to the
                # raw push below: literal strings keep bare CR/LF as-is here
                # rather than applying PDF-spec "treat as 0x0A" normalization
                # is deliberately absent there; no CRLF collapsing either).
                if literal_byte == 0x28:
                    if depth:
                        out.append(literal_byte)
                    depth += 1
                elif literal_byte == 0x29:
                    depth -= 1
                    if depth == 0:
                        index_value += 1
                        break
                    out.append(literal_byte)
                else:
                    out.append(literal_byte)
                index_value += 1
            push("str", tuple(out))
        elif byte_value == 0x3c:  # < : << dict-open, else <hex>
            if index_value + 1 < count_item and content_bytes[index_value + 1] == 0x3c:
                frames.append(("dict", []))
                index_value += 2
            else:
                index_value += 1
                nib: list[int] = []
                while index_value < count_item and content_bytes[index_value] != 0x3e:
                    hex_byte = content_bytes[index_value]
                    if 0x30 <= hex_byte <= 0x39:
                        nib.append(hex_byte - 0x30)
                    elif 0x41 <= hex_byte <= 0x46:
                        nib.append(hex_byte - 0x37)
                    elif 0x61 <= hex_byte <= 0x66:
                        nib.append(hex_byte - 0x57)
                    index_value += 1
                index_value += 1
                if len(nib) % 2:
                    nib.pop()  # drop a lone trailing hex digit
                push("str", tuple(
                    (nib[key_value] << 4) | nib[key_value + 1] for key_value in range(0, len(nib), 2)
                ))
        elif byte_value == 0x3e:  # >> dict-close (or stray >)
            if index_value + 1 < count_item and content_bytes[index_value + 1] == 0x3e:
                index_value += 2
                if frames and frames[-1][0] == "dict":
                    frames.pop()
                    push("dict", None)
                # stray >> : text extraction command token -> unknown command -> tally preserved
            else:
                index_value += 1
        elif byte_value == 0x5b:  # [ -- one array operand (text extraction parser builds an array)
            frames.append(("arr", []))
            index_value += 1
        elif byte_value == 0x5d:  # ]
            index_value += 1
            if frames and frames[-1][0] == "arr":
                items = frames.pop()[1]
                # TJ semantics: only direct string elements show; numbers are
                # kern adjustments and nested non-strings are ignored.
                push("arr", tuple(codepoint for kerning_delta, vertical_value in items if kerning_delta == "str"
                                  for codepoint in vertical_value))  # type: ignore[union-attr]
            # stray ] : text extraction command token(']') -> unknown command -> tally preserved
        elif byte_value in b"{}":
            index_value += 1  # text extraction command token -> not in operator table -> "Unknown command", preserved
        elif byte_value == 0x2f:  # /name operand
            index_value += 1
            token_end = index_value
            while token_end < count_item and content_bytes[token_end] not in _PDF_WHITESPACE_BYTES and content_bytes[token_end] not in _PDF_DELIMITER_BYTES:
                token_end += 1
            # Decode #XX escapes while lexing names so consumers receive the
            # canonical name and do not re-decode downstream.
            push("name", _decode_pdf_name(content_bytes[index_value:token_end]))
            index_value = token_end
        else:  # number, keyword operand, or operator
            first_char = content_bytes[index_value]
            if (0x30 <= first_char <= 0x39) or first_char in b"+-.":
                # Number token: consume the whole run to whitespace/delimiter.
                # Malformed numeric runs are zeroed by the downstream parser.
                token_end = index_value
                while token_end < count_item and content_bytes[token_end] not in _PDF_WHITESPACE_BYTES and content_bytes[token_end] not in _PDF_DELIMITER_BYTES:
                    token_end += 1
            else:
                # text extraction command lexing: once the accumulated run IS a
                # known command, stop extending as soon as the next char
                # would break that -- 'q1' lexes as command token 'q' + number 1
                # (real-world PDFs; text extraction built known commands for them).
                token_end = index_value
                known = False
                while token_end < count_item and content_bytes[token_end] not in _PDF_WHITESPACE_BYTES and content_bytes[token_end] not in _PDF_DELIMITER_BYTES:
                    cand = content_bytes[index_value:token_end + 1]
                    if (known and cand not in _OP_OPERAND_COUNTS
                            and cand not in _OP_LEX_PREFIX):
                        break
                    token_end += 1
                    known = (content_bytes[index_value:token_end] in _OP_OPERAND_COUNTS
                             or content_bytes[index_value:token_end] in _OP_LEX_PREFIX)
            operator_token = content_bytes[index_value:token_end]
            index_value = token_end
            if not operator_token:
                index_value += 1
                continue
            if (0x30 <= first_char <= 0x39) or first_char in b"+-.":
                # text extraction number lexer accepts only digit/sign/dot/exponent
                # runs; Python float would also take "-inf"/"nan" tokens,
                # which must not poison the operand (or the Tz state).
                try:
                    num_val = float(operator_token)
                except ValueError:
                    num_val = 0.0
                if num_val != num_val or num_val in (float("inf"), -float("inf")):
                    num_val = 0.0
                push("num", num_val)
                continue
            if operator_token in (b"true", b"false"):
                push("other", None)  # text extraction booleans -> operands
                continue
            if operator_token == b"null":
                continue  # content operator evaluator: `if (obj != null) args append`

            if frames:
                # text extraction builds arrays/dicts by recursive object parser: a command
                # token inside an open [ / << becomes an ELEMENT, never an op.
                frames[-1][1].append(("other", None))
                continue
            if operator_token == b"BI":
                # text extraction object parser intercepts BI (inline-image parser): the
                # image never reaches the operator table protocol; it becomes ONE arg.
                bi_mark = len(opnds)
                continue
            spec = _OP_OPERAND_COUNTS.get(operator_token)
            if spec is None:
                continue  # span merger: warn "Unknown command", tally PRESERVED
            if operator_token == b"ID":  # inline image data (text extraction inline-image parser)
                # Filter-specific ender first (content stream tokenizer dispatch): DCT scans
                # for the FFD9 EOI, ASCII85 for '~>', ASCIIHex for '>'; then
                # the 'EI' marker whose FOLLOWING byte is SPACE/LF/CR
                # (inline-image end search -- there is NO whitespace
                # requirement BEFORE the marker: inline-image data can touch it).
                # span merger extra 10-byte-lookahead / lookahead false-EI checks
                # are not reproduced (light version).
                filt = b""
                if bi_mark is not None:
                    for kerning_delta, vertical_value in opnds[bi_mark:]:
                        if kerning_delta == "name" and vertical_value in (
                                b"DCTDecode", b"DCT", b"ASCII85Decode",
                                b"A85", b"ASCIIHexDecode", b"AHx"):
                            filt = vertical_value
                            break
                key_value = index_value + 1
                if filt in (b"DCTDecode", b"DCT"):
                    measure_item = content_bytes.find(b"\xff\xd9", key_value)
                    if measure_item >= 0:
                        key_value = measure_item + 2
                elif filt in (b"ASCII85Decode", b"A85"):
                    measure_item = content_bytes.find(b"~>", key_value)
                    if measure_item >= 0:
                        key_value = measure_item + 2
                elif filt in (b"ASCIIHexDecode", b"AHx"):
                    measure_item = content_bytes.find(b">", key_value)
                    if measure_item >= 0:
                        key_value = measure_item + 1
                while key_value < count_item - 1:
                    if (content_bytes[key_value] == 0x45 and content_bytes[key_value + 1] == 0x49
                            and (key_value + 2 >= count_item or content_bytes[key_value + 2] in b" \n\r")):
                        index_value = key_value + 2
                        break
                    key_value += 1
                else:
                    index_value = count_item  # EOF recovery (text extraction inline-image end recovery)
                if bi_mark is not None:
                    del opnds[bi_mark:]           # the BI..ID dict guts
                    bi_mark = None
                    opnds.append(("other", None))  # the InlineImage operand
                    # text extraction then executes a synthetic command token EI (operand count 1):
                    # pre-BI dangles shift into deferred-operand, the image
                    # operand is consumed, args end empty.
                    while len(opnds) > 1:
                        non_processed.append(opnds.pop(0))
                    opnds.clear()
                else:
                    # Stray ID without BI: text extraction dispatches it via operator table
                    # (operand count 0), shifting every pending arg into the
                    # deferred-operand stack before the no-op executes.
                    non_processed.extend(opnds)
                    opnds.clear()
                continue
            need, variable = spec
            if not variable and len(opnds) != need:
                while len(opnds) > need:
                    non_processed.append(opnds.pop(0))
                while len(opnds) < need and non_processed:
                    opnds.insert(0, non_processed.pop())
                if len(opnds) < need:
                    # Detail: "Skipping command ...: expected N args" + args

                    # cleared; the op has NO side effect (no flush, no Tf).
                    opnds.clear()
                    continue
            if operator_token in _FLUSH_OPS:
                flush_id += 1
                if operator_token == b"q":
                    font_stack.append(cur_font)
                    tz_stack.append(cur_tz)
                elif operator_token == b"Q":
                    if font_stack:
                        cur_font = font_stack.pop()
                    if tz_stack:
                        cur_tz = tz_stack.pop()
                elif operator_token == b"Do" and opnds[0][0] == "name":
                    xobject_paints.append((len(flush_ids), opnds[0][1], cur_font, cur_tz))  # type: ignore[arg-type]
            elif operator_token == b"Tf":
                if opnds[0][0] == "name":
                    cur_font = opnds[0][1]  # type: ignore[assignment]
                else:
                    # text extraction REPLACES the font either way: a non-name slot
                    # loads undefined -> fallback/fallback font, so the previous
                    # font is gone. None = "no usable resname" here.
                    cur_font = None
            elif operator_token == b"Tz":
                # content stream tokenizer horizontal-scale operator: the text state's horizontal scale = args[0]/100
                # (any type, ToNumber-coerced). We track only numeric operands:
                # the divisor must account for what PDFIUM folded into the object
                # matrix, and PDFium's own parser rejects non-numeric Tz --
                # following span merger coercion here would break consistency with the horizontal font scale.
                if opnds[0][0] == "num":
                    cur_tz = opnds[0][1] / 100.0  # type: ignore[operator]
            elif operator_token in _SHOW_OPS:
                if operator_token == b"TJ":
                    primary_item = opnds[0]
                    # the spaced-text show operator iterates elements by .length/.at, which a
                    # plain STRING also satisfies -- its chars all show.
                    units = primary_item[1] if primary_item[0] in ("arr", "str") else ()
                elif operator_token == b'"':
                    primary_item = opnds[2]
                    units = primary_item[1] if primary_item[0] == "str" else ()
                else:  # Tj, '
                    primary_item = opnds[0]
                    units = primary_item[1] if primary_item[0] == "str" else ()
                # A wrong-typed slot or an empty string yields ZERO glyphs in
                # span merger (glyph conversion -> no item pushed) and no PDFium text
                # object either -- emit no show entry, so both ordinal
                # alignments (objects <-> show ops) stay tight.
                if units:
                    flush_ids.append(flush_id)
                    fonts.append(cur_font)
                    horizontal_scales.append(cur_tz)
                    show_text_units.append(units)  # type: ignore[arg-type]
            opnds.clear()  # executed op consumes its args (caller resets)
    return flush_ids, fonts, show_text_units, horizontal_scales, xobject_paints


def _assign_vertical_tags(
    objects: list[dict],
    show_fonts: list[bytes | None] | None = None,
    vertical_resnames: set[bytes] | None = None,
) -> None:
    """Tag each text object (paint order) with the vertical-font flag from its matching show-text operator. Ordinal alignment is valid when object and show-op counts agree, such as ligature-free pages and per-glyph CJK Tj streams. On a count mismatch the tag stays False and vertical runs fall back to per-glyph handling. Objects keep ``flush_id=None`` so the merger always splits per object."""
    if not objects or not vertical_resnames or not show_fonts:
        return
    if len(show_fonts) != len(objects):
        return
    for item_value, font_name_value in zip(objects, show_fonts):
        if font_name_value is not None and font_name_value in vertical_resnames:
            item_value["vertical"] = True


def _assign_show_tz(objects: list[dict], show_tzs: list[float]) -> None:
    """Tag each text object with its show-op's text horizontal scale (Tz/100) by the same ordinal alignment as ``_assign_vertical_tags``; on a count mismatch every object keeps tz=1.0 (thresholds behave as before)."""
    if not objects or not show_tzs or len(show_tzs) != len(objects):
        return
    for item_value, timezone_value in zip(objects, show_tzs):
        item_value["tz"] = timezone_value


def _page_vertical_resource_names(pdf_doc, page_idx: int) -> set[bytes]:
    """Font resource names (``F4`` of ``/F4 14 Tf``) on this page whose encoding is a vertical CMap: a predefined ``*-V`` name (Identity-V, UniJIS-UCS2-V, ...) or an embedded CMap stream with ``/WMode 1``. This derives the vertical-font flag used by the item merger, read from the same PyPDF2 document already opened for content streams. Returns an empty set on any failure, which leaves vertical handling disabled for that page."""
    names: set[bytes] = set()
    try:
        for rec in pdf_doc[page_idx].get_fonts(full=True):
            xref, font_extension, font_type, _basefont, resname, enc = rec[:6]
            # Predefined vertical CMaps: every shipped vertical bcmap ends in
            # "-V" EXCEPT the bare Adobe-Japan1 "V" (bcmaps/V.bcmap, header
            # bit 1 set -- content stream tokenizer reads verticality from that bit).
            if isinstance(enc, str) and (enc == "V" or enc.endswith("-V")):
                names.add(resname.encode("latin-1", "replace"))
                continue
            # Embedded CMap: /Encoding is an indirect stream; vertical iff its
            # dict carries /WMode 1.
            page, resource_names = pdf_doc.xref_get_key(xref, "Encoding")
            if page == "xref":
                width_type, width_value_local = pdf_doc.xref_get_key(int(resource_names.split()[0]), "WMode")
                if width_type in ("int", "real"):
                    try:
                        wmode_number = float(width_value_local.split()[0])
                    except ValueError:
                        wmode_number = float("nan")
                    # Only integer-valued nonzero numbers enable the vertical
                    # font flag, so ``/WMode 1.0`` still counts.
                    if wmode_number.is_integer() and wmode_number != 0:
                        names.add(resname.encode("latin-1", "replace"))
    except Exception:
        return set()
    return names
