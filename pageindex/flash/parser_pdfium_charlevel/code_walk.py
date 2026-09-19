"""Resource-dictionary xref walking and per-page show-code enumeration."""

from __future__ import annotations

import re
import unicodedata
from PyPDF2.generic import (
    IndirectObject as PdfIndirectRef, NameObject as PdfName, NumberObject as PdfNumber,
    FloatObject as PdfFloat, BooleanObject as PdfBoolean,
    DictionaryObject as PdfDictionary, ArrayObject as PdfArray,
)

from .pdf_objects import _decode_pdf_name
from .text_normalize import (
    _normalize_unicodes,
    _WHITESPACE_CODEPOINTS,
    _is_whitespace,
)
from .content_stream import _tokenize_show_operators


def _resource_dict_xrefs(pdf_doc, owner_xref: int, sub: str) -> dict[bytes, int]:
    """{canonical resname bytes: xref} for /Redefinitions/<sub> of a page or Form XObject dict, following indirection; for pages, /Redefinitions may be inherited through the /Parent chain."""
    val = ("null", "null")
    xref_cursor = owner_xref
    for _ in range(32):  # /Parent chain (pages); XObjects never recurse here
        val = pdf_doc.xref_get_key(xref_cursor, f"Resources/{sub}")
        if val[0] != "null":
            break
        if pdf_doc.xref_get_key(xref_cursor, "Resources")[0] != "null":
            break  # Redefinitions exists but lacks <sub>
        parent_key_type, parent_xref_value = pdf_doc.xref_get_key(xref_cursor, "Parent")
        if parent_key_type != "xref":
            break
        xref_cursor = int(parent_xref_value.split()[0])
    if val[0] == "xref":
        body = pdf_doc.xref_object(int(val[1].split()[0]), compressed=True)
    elif val[0] == "dict":
        body = val[1]
    else:
        return {}
    out: dict[bytes, int] = {}
    for measure_item in re.finditer(r"/([^\s/\[\]<>()]+)\s+(\d+)\s+\d+\s+R", body):
        out[_decode_pdf_name(measure_item.group(1).encode("latin-1"))] = int(measure_item.group(2))
    # DIRECT (inline) sub-dict entries carry no `N G R` for the regex; span merger
    # reference resolution resolves them all the same, so register each as a virtual
    # pseudo-xref and the normal integer-keyed pipeline address it.
    try:
        node = pdf_doc._resolve_object(xref_cursor)
        for part in ("Resources", sub):
            if isinstance(node, PdfIndirectRef):
                node = node.get_object()
            node = node["/" + part] if (node is not None and "/" + part in node) else None
        if node is not None:
            if isinstance(node, PdfIndirectRef):
                node = node.get_object()
            for key_value in node.keys():
                raw = node.raw_get(key_value)
                if isinstance(raw, PdfIndirectRef):
                    continue  # indirect: the regex pass covered it
                if not hasattr(raw, "raw_get"):
                    continue  # not a dict (malformed entry)
                name = _decode_pdf_name(key_value.lstrip("/").encode("latin-1"))
                if name not in out:
                    out[name] = pdf_doc.register_virtual(raw)
    except Exception:
        pass
    return out


def _page_show_codes(
    pdf_doc, page_idx: int,
) -> list[tuple[int | None, tuple[int, ...], float]] | None:
    """Every show op the page paints, in paint order, as ``(font_xref | None, charcode units, horizontal scale)-- including text inside Form XObjects, spliced at their ``Do`` position with the XObject's own font redefinitions (span merger text-content extraction recurses the same way; PDFium's textpage flattens them inline). The recursion runs on a CLONE of the live text state, so a form inherits both the active font and the horizontal scale; a Tz inside the form REPLACES it and never leaks back out. None when the page can't be read."""
    page_xref = pdf_doc.page_xref(page_idx)

    def walk(stream: bytes, fonts_res: dict[bytes, int],
             xobjs_res: dict[bytes, int], cur_font: int | None, cur_tz: float,
             visited: frozenset, depth: int,
             out: list[tuple[int | None, tuple[int, ...], float]]) -> None:
        if depth > 8:
            return
        flush_ids, fonts, show_text_units, horizontal_scales, xobject_paints = _tokenize_show_operators(stream, cur_tz)
        dict_index = 0
        for key_value in range(len(show_text_units) + 1):
            while dict_index < len(xobject_paints) and xobject_paints[dict_index][0] == key_value:
                paint_position, xname, font_at_do, tz_at_do = xobject_paints[dict_index]
                dict_index += 1
                xobject_ref = xobjs_res.get(xname)  # lexer names arrive #XX-parsed
                if xobject_ref is None or xobject_ref in visited:
                    continue
                state_values, string_value = pdf_doc.xref_get_key(xobject_ref, "Subtype")
                if state_values != "name" or string_value.lstrip("/") != "Form":
                    continue
                sub_fonts = _resource_dict_xrefs(pdf_doc, xobject_ref, "Font") or fonts_res
                sub_xobjs = _resource_dict_xrefs(pdf_doc, xobject_ref, "XObject") or xobjs_res
                inherited = (fonts_res.get(font_at_do)
                             if font_at_do is not None else None)
                try:
                    sub_stream = pdf_doc.xref_stream(xobject_ref)
                except Exception:
                    # span merger: "XObject should be a stream" -> recovery mode skips
                    # THIS Do and keeps walking the page (a direct dict posing
                    # as /Form has no stream; must not kill the whole page).
                    continue
                walk(sub_stream, sub_fonts, sub_xobjs,
                     inherited, tz_at_do, visited | {xobject_ref}, depth + 1, out)
            if key_value < len(show_text_units):
                resource_font_name = fonts[key_value]
                resource_font_index = (fonts_res.get(resource_font_name)
                      if resource_font_name is not None else cur_font)
                out.append((resource_font_index, show_text_units[key_value], horizontal_scales[key_value]))

    try:
        out: list[tuple[int | None, tuple[int, ...], float]] = []
        walk(
            pdf_doc[page_idx].read_contents(),
            _resource_dict_xrefs(pdf_doc, page_xref, "Font"),
            _resource_dict_xrefs(pdf_doc, page_xref, "XObject"),
            None, 1.0, frozenset(), 0, out,   # the initial text state starts at scale 1
        )
        return out
    except Exception:
        return None


def _char_category(text: str) -> tuple[bool, bool, bool]:
    """text extraction glyph Unicode category classification over a (possibly multi-char) glyph Unicode string: first match of /^(\\s)|(\\p{Mn})|(\\p{Cf})$/u decides (isWhitespace, zero-width diacritic classification, invisible format-mark classification)."""
    for pos, char in enumerate(text):
        codepoint = ord(char)
        if pos == 0 and codepoint in _WHITESPACE_CODEPOINTS:
            return True, False, False
        cat = unicodedata.category(char)
        if cat == "Mn":
            return False, True, False
        if cat == "Cf" and pos == len(text) - 1:
            return False, False, True
    return False, False, False


def _walk_codes(
    chars: list[tuple[int, str]],
    targets: list[str],
    allow_skips: bool = False,
) -> tuple[list[tuple[int, str]], list[int], list[tuple[int, int]],
           list[tuple[int, int]]] | None:
    """Walk one run of font-resolved per-code targets against the PDFium chars emitted for the same run; return (patches, drops, consumed, skips) or None on desync. ``consumed`` maps each consumed char's textpage index to the target index that consumed it, which lets the group re-walk repair char-to-object attribution. ``skips`` records each skipped target as (target index, char position it belongs before) for glyph re-synthesis. PDFium's emission per code is unknowable a priori: it may match the font target, fall back to the raw code, or expand a glyph into several chars. Consumption is resolved per code by candidate match: the font target, then its normalized expansions (fixed unicode substitution table, NFKC, NFKD, NFD). On an expansion match where the final span text still converges, the chars are left alone; otherwise the first char is patched to the target unicode and the rest of the run is dropped so the single glyph still carries the advance. The run is valid only if both streams end in sync. """
    text = "".join(target_char for _, target_char in chars)
    line_value = len(text)
    pos = 0
    patches: list[tuple[int, str]] = []
    drops: list[int] = []
    consumed: list[tuple[int, int]] = []  # (char textpage index, target index)
    skips: list[tuple[int, int]] = []     # (target index, char position)
    skip_until = -1
    shift_run = 0
    for index_value, token_value in enumerate(targets):
        if index_value <= skip_until:
            continue  # part of an anchored skip run recorded below
        if pos >= line_value:
            if allow_skips:
                # Chars exhausted with targets left: the walk arrived here in
                # sync, so every remaining target is a glyph PDFium never
                # emitted (the both-exhaust gate in reverse).
                skips.append((index_value, pos))
                continue
            return None  # codes left over: desync
        target_len = len(token_value)
        if text[pos:pos + target_len] == token_value:
            consumed.extend((chars[query_value][0], index_value) for query_value in range(pos, pos + target_len))
            pos += target_len
            shift_run = 0
            continue
        matched = False
        for normalized_text in (_normalize_unicodes(token_value),
                   unicodedata.normalize("NFKC", token_value),
                   unicodedata.normalize("NFKD", token_value),
                   unicodedata.normalize("NFD", token_value)):
            if normalized_text != token_value and text[pos:pos + len(normalized_text)] == normalized_text:
                if _normalize_unicodes(token_value) != normalized_text:
                    patches.append((chars[pos][0], token_value))
                    drops.extend(chars[query_value][0] for query_value in range(pos + 1, pos + len(normalized_text)))
                consumed.extend((chars[query_value][0], index_value) for query_value in range(pos, pos + len(normalized_text)))
                pos += len(normalized_text)
                matched = True
                break
        if matched:
            shift_run = 0
            continue
        # Anchored drop-skip (LAST-RESORT mode only: the window re-walk has
        # already ruled out the stolen-edge-glyph hypothesis): when PDFium
        # genuinely never emitted the glyph (font-layer drop -- dense math-heavy page's
        # 4 α, math-heavy page's scanned-page '~', both absent from the textpage AND
        # FPDFTextObj_GetText), the target has no char anywhere. Skip it
        # WITHOUT consuming, but only when the next two unskipped targets
        # literally anchor on the upcoming chars, so a mis-decode (which
        # needs the 1-char patch below instead) can't be eaten as a skip.
        # Drops can be CONSECUTIVE (OCR pages drop runs of glyphs), so scan
        # forward for the smallest run i..i+m-1 whose following pair
        # anchors; a wrong run leaves chars unconsumed and the exhaust gate
        # below still rolls everything back.
        if allow_skips:
            skip_run_length = 0
            for skip_len in range(1, len(targets) - index_value + 1):
                anchor = targets[index_value + skip_len:index_value + skip_len + 2]
                if not anchor or not all(len(primary_item) == 1 for primary_item in anchor):
                    break
                str_value = "".join(anchor)
                if text[pos:pos + len(str_value)] == str_value:
                    skip_run_length = skip_len
                    break
            if skip_run_length:
                skips.extend((query_value, pos) for query_value in range(index_value, index_value + skip_run_length))
                skip_until = index_value + skip_run_length - 1
                shift_run = 0
                continue
        # PDFium's textpage COLLAPSES space runs: a whitespace target facing
        # a non-whitespace char means the space's char simply does not exist
        # in the textpage (it can never be a re-decode of the current char).
        # Desync rather than mis-patch the neighbouring glyph into a space;
        # table rows can contain real star glyphs adjacent to synthetic spaces.
        if (all(_is_whitespace(ord(unit_char)) for unit_char in token_value)
                and not _is_whitespace(ord(text[pos]))):
            return None
        # Off-by-one guard for the 1-char assumption below: when an edge
        # glyph was mis-attributed to a neighbouring object, every pair
        # mismatches with the streams shifted by one, and a ligature
        # expansion elsewhere can re-balance the counts so the exhaust gate
        # alone would COMMIT the shifted alignment and attach punctuation to
        # the wrong run. The shift has a literal signature --
        # the NEXT target equals the current char(s), or the current target
        # equals the NEXT char(s) -- which legitimate decode mismatches
        # (text extraction symbol vs PDFium control char) never produce. Two
        # consecutive hits = systematic shift -> desync, letting the
        # adjacent-run group re-walk re-align both objects cleanly.
        nxt = targets[index_value + 1] if index_value + 1 < len(targets) else None
        if ((nxt is not None and text[pos:pos + len(nxt)] == nxt)
                or text[pos + 1:pos + 1 + target_len] == token_value):
            shift_run += 1
            if shift_run >= 2:
                return None
        else:
            shift_run = 0
        patches.append((chars[pos][0], token_value))
        consumed.append((chars[pos][0], index_value))
        pos += 1
    if pos != line_value:
        return None  # chars left over: desync
    return patches, drops, consumed, skips
