"""Raw textpage char extraction, Type3 sizing, and page viewport handling."""

from __future__ import annotations

import ctypes
import pypdfium2.raw as pdfium_c

from .text_normalize import (
    _is_whitespace,
    _is_zero_width_diacritic,
    _is_invisible_format_mark,
)
from .geometry import (
    _collect_text_objs,
    _build_obj_index,
    _find_obj_for_char,
)


def _extract_raw_chars(page, text_page) -> tuple[list[dict], list[dict]]:
    """First pass: walk textpage chars and attach font info via the bbox-containing text-object lookup. Returns ``(raw_chars, objects)``; glyph widths and the identity-matrix Type-3 size override are applied later, after document-wide Type-3 extents are known."""
    objects = _collect_text_objs(page, text_page)
    if not objects:
        return [], []
    obj_index = _build_obj_index(objects)

    # First pass: collect raw textpage chars with their host obj.
    count_item = pdfium_c.FPDFText_CountChars(text_page)
    font_name_buffer = (ctypes.c_char * 256)()
    flags = ctypes.c_int(0)
    field = ctypes.c_float(0)
    # Per-char FFI out-buffers and entry points, hoisted: each is overwritten
    # by its call (buffers whose call result is unchecked are re-zeroed below,
    # so a failed call reads back 0 exactly as a fresh buffer would).
    char_origin_x = ctypes.c_double(0); char_origin_y = ctypes.c_double(0)
    char_left_box = ctypes.c_double(0); char_right_box = ctypes.c_double(0)
    value = ctypes.c_double(0); char_top_box = ctypes.c_double(0)
    loose_box = pdfium_c.FS_RECTF(0, 0, 0, 0)
    u32 = ctypes.c_uint32(0)
    fs32 = ctypes.c_float(0)
    byref = ctypes.byref
    ox_ref = byref(char_origin_x); oy_ref = byref(char_origin_y)
    l_ref = byref(char_left_box); r_ref = byref(char_right_box)
    b_ref = byref(value); t_ref = byref(char_top_box)
    loose_ref = byref(loose_box)
    w_ref = byref(field)
    flags_ref = byref(flags)
    get_unicode = pdfium_c.FPDFText_GetUnicode
    is_generated = pdfium_c.FPDFText_IsGenerated
    get_char_origin = pdfium_c.FPDFText_GetCharOrigin
    get_char_box = pdfium_c.FPDFText_GetCharBox
    get_loose_box = pdfium_c.FPDFText_GetLooseCharBox
    get_font_info = pdfium_c.FPDFText_GetFontInfo
    get_glyph_width = pdfium_c.FPDFFont_GetGlyphWidth
    js_is_ws = _is_whitespace
    name_cache: dict[bytes, str] = {}
    raw_chars: list[dict] = []
    last_obj: dict | None = None
    for index_value in range(count_item):
        codepoint = get_unicode(text_page, index_value)
        if codepoint < 0:
            continue
        # u == 0 (PDFium found no unicode for the glyph) is KEPT as '\x00':
        # text extraction emits the raw charcode for unmapped codes, so its items
        # really contain chr(0) for extension-font pieces at code 0, and the
        # textpage char carries normal geometry. Skipping it lost the char AND desynced
        # the unicode walk's object pairing around it.
        ch_str = chr(codepoint)
        is_ws = js_is_ws(codepoint)
        # FPDFText_IsGenerated returns a c_int: 1 generated, 0 real, -1 error.
        # Only a POSITIVE 1 may mark a char generated -- the -1 has to read the
        # same way here as it does in the page-mode unicode walk, or the two
        # char sets disagree and that walk desyncs.
        is_gen = is_generated(text_page, index_value) == 1
        # PDFium inserts is_generated chars as layout placeholders for
        # Td/Tm jumps with no literal content-stream char (typically
        # " ", "\r", "\n"). Dropping them outright leaves an
        # unexplained advance gap that the merger then turns into a
        # fake-space chunk, splitting e.g. "2.1 Computing the EMD"
        # into three spans (2.1, " ", Computing the EMD) that pipeline
        # treats as a numeric prefix alone (not a heading). Keep
        # generated whitespace so the merger's whitespace branch fires
        # save_last_char without emitting, letting the next visible
        # glyph compute a tracking-size in-flow advance. Drop only
        # non-whitespace generated chars (very rare).
        if is_gen and not is_ws:
            continue
        char_origin_x.value = 0.0; char_origin_y.value = 0.0
        get_char_origin(text_page, index_value, ox_ref, oy_ref)
        ox_v = char_origin_x.value; oy_v = char_origin_y.value

        # Fetch char bbox first so we can use its center for the obj
        # lookup — origin alone fails when adjacent obj bboxes nearly
        # touch (e.g. math-heavy page "(", math italic font \x01, ")" all on the same line
        # with sub-pt gaps, where origin x falls inside the wrong obj's
        # tolerance window). Using bbox center gives unambiguous
        # containment.
        char_left_box.value = 0.0; char_right_box.value = 0.0; value.value = 0.0; char_top_box.value = 0.0
        get_char_box(text_page, index_value, l_ref, r_ref, b_ref, t_ref)
        char_left, char_right, char_top, char_bottom = char_left_box.value, char_right_box.value, char_top_box.value, value.value
        # Tight (ink) box center -> font-object disambiguation only.
        center_x = (char_left + char_right) / 2 if char_right > char_left else ox_v
        center_y = (char_top + char_bottom) / 2 if char_top > char_bottom else oy_v
        # Horizontal extent for the SPAN comes from the LOOSE char box (the
        # glyph's full advance cell), not the tight ink box. the PDF text-item
        # widths are advance-based; the ink box undershoots each glyph's right
        # edge by its side bearing (e.g. "]" ink-right 274.0 vs advance 275.2,
        # as expected for advance-based text items). Using the ink box cumulatively under-fills
        # display-math gaps so the column detector mis-reads them as gutters
        # and splits a line ("E[x] = μ" -> "E[x]" fragment). Fall back to the
        # ink box if the loose box is unavailable/degenerate.
        # (_loose is only READ when the call succeeded, so the hoisted struct
        # never leaks a previous char's values.)
        if (get_loose_box(text_page, index_value, loose_ref)
                and loose_box.right > loose_box.left):
            loose_left, loose_right = loose_box.left, loose_box.right
            # Vertical edges of the loose (advance-cell) box. For vertical-
            # writing (Identity-V / WMode 1) text PDFium builds this cell by
            # advancing -y from the PEN, so its upper edge IS the pen y and
            # its extent IS the per-char vertical advance (W2/DW2 applied by
            # PDFium itself). PDFium fills top/bottom in flow order here, so
            # they arrive inverted (top < bottom); keep both raw edges.
            cell_top, cell_bottom = loose_box.top, loose_box.bottom
        else:
            loose_left, loose_right = char_left, char_right
            cell_top, cell_bottom = char_top, char_bottom

        # Character font size disambiguates overlapping objects, such as large
        # figure labels sharing a y range with smaller heading text.
        # text-page and character-index lookup read the true per-char rendered size
        # (FPDFText_GetMatrix) and the reported font size (FPDFText_GetFontSize)
        # lazily, only to break a multi-object containment tie — see
        # _find_obj_for_char.
        obj = _find_obj_for_char(
            obj_index, center_x, center_y, tol=1.0, char_fs=None, text_page=text_page, char_idx=index_value
        )
        if obj is None:
            obj = (
                _find_obj_for_char(obj_index, ox_v, oy_v, tol=1.0,
                                   char_fs=None, text_page=text_page, char_idx=index_value)
                or _find_obj_for_char(obj_index, ox_v, oy_v, tol=5.0,
                                      char_fs=None, text_page=text_page, char_idx=index_value)
                or last_obj
            )
        if obj is None:
            continue
        last_obj = obj

        name = get_font_info(text_page, index_value, font_name_buffer, 256, flags_ref)
        if name > 1:
            raw_name = font_name_buffer[:name]
            char_font_name = name_cache.get(raw_name)
            if char_font_name is None:
                char_font_name = raw_name.decode(
                    "latin-1", errors="replace").rstrip("\x00")
                name_cache[raw_name] = char_font_name
        else:
            char_font_name = obj["font_name"]

        # Use baseline (oy) as bbox bottom and baseline + fs_eff as top.
        # the span anchoring rule uses matrix.f (= baseline y) for both top/
        # bottom anchors of its span, so chars of the same line all
        # land at the same bottom even when their ink extends below
        # baseline ("(", "g", "y" with descenders) or above ("\x01"
        # math glyphs). This is what the heading heuristics' tokenizer assumes when
        # checking |c1.C - c2.C| < 1 to decide whether two spans are on
        # the same line.
        baseline_y = oy_v
        char_top = baseline_y + obj["fs_eff"]
        # Capture the raw glyph advance now, while this page (and thus the
        # font handle) is alive. The fs_eff-dependent scaling happens later
        # in _finalize_chars, after the document-wide Type-3 size is known,
        # so deferring the call would require keeping every page open just to
        # keep font handles valid (PDFium frees the font when the page is
        # closed -> dangling handle).
        u32.value = codepoint
        fs32.value = obj["fs_raw"]
        get_glyph_width(obj["font"], u32, fs32, w_ref)
        raw_chars.append({
            "i": index_value, "ch": ch_str, "u": codepoint,
            "is_gen": is_gen,
            "is_ws": is_ws,
            "is_mn": _is_zero_width_diacritic(codepoint),
            "is_cf": _is_invisible_format_mark(codepoint),
            "ox": ox_v, "oy": oy_v,
            "left": loose_left, "right": loose_right,
            "top": char_top, "bottom": baseline_y,
            "box_top": char_top,
            "box_bottom": char_bottom,
            "cell_top": cell_top, "cell_bot": cell_bottom,
            "w_raw": field.value,
            "obj": obj, "font_name": char_font_name,
        })

    return raw_chars, objects


def _accumulate_type3_extents(raw_chars: list[dict], acc: dict) -> None:
    """Accumulate document-wide per-font glyph-bbox extents for identity-matrix Type-3 fonts. These fonts use a synthesized font bbox from the union of CharProc glyph boxes and render every glyph at that uniform height. PDFium reports a constant font size and identity CTM for these fonts, but its char box returns each glyph's declared bounds exactly, so box-top/bottom relative to the baseline reveal the rendered glyph extents. Aggregating across the whole document makes the font sizing coverage-independent; a per-page union would drift with sparse page content. Scoped to the identity-matrix Type-3 branch so normal and scaled-matrix fonts are untouched."""
    for candidate_item in raw_chars:
        item_value = candidate_item["obj"]
        if item_value["fs_raw"] >= 1.5 or item_value["scale_y"] >= 1.5 or candidate_item["is_ws"]:
            continue
        top = candidate_item["box_top"] - candidate_item["oy"]
        bot = candidate_item["box_bottom"] - candidate_item["oy"]
        if top <= bot:  # degenerate glyph box (text extraction skips d1 i==0)
            continue
        _xref_key = item_value["font_key"]
        entry_item = acc.get(_xref_key)
        if entry_item is None:
            acc[_xref_key] = [top, bot]
        else:
            if top > entry_item[0]:
                entry_item[0] = top
            if bot < entry_item[1]:
                entry_item[1] = bot


def _type3_size_by_font(acc: dict) -> dict:
    """font handle -> rendered font.bbox height = max ascent - min descent, i.e. span merger ``a = font.bbox[3] - font.bbox[1]`` in page units. Snap to the shortest decimal (PDFium float32 vs span merger float64) for clean knife-edge size comparisons downstream (the page-median gate)."""
    out: dict = {}
    for _xref_key, (top, bot) in acc.items():
        if top > bot:
            out[_xref_key] = float(f"{top - bot:.6g}")
    return out


def _apply_type3_sizes(raw_chars: list[dict], size_by_font: dict) -> None:
    """Override fs_eff with the document-wide Type-3 size and reset each char's span top to baseline + that size."""
    if not size_by_font:
        return
    for candidate_item in raw_chars:
        item_value = candidate_item["obj"]
        if item_value["fs_raw"] >= 1.5 or item_value["scale_y"] >= 1.5:
            continue
        font_size_value = size_by_font.get(item_value["font_key"])
        if font_size_value:
            item_value["fs_eff"] = font_size_value
            candidate_item["top"] = candidate_item["oy"] + font_size_value


def _finalize_chars(raw_chars: list[dict]) -> list[dict]:
    """Second pass: compute glyph_w per char and emit the merged-ready dicts. The right glyph width definition depends on how PDFium reports the font's metrics: (a) Normal Type 1 fonts (fs_raw >= 1.5, scale.a ~= 1): FPDFFont_GetGlyphWidth(font, code, fs_raw) returns the advance in page units. Use as-is x matrix.a. (b) Scaled-matrix Type 3 (fs_raw < 1.5 but matrix scale >= 1.5, e.g. vector-heavy page's a scaled Type-3 subset with scale=36.49): GetGlyphWidth at fs_raw=0.19 gives font-natural-unit width; x matrix scale recovers page units. (c) Identity-matrix Type 3 (fs_raw < 1.5, matrix.a ~= 1, e.g. identity-matrix Type-3 sample an identity-matrix Type-3 font): GetGlyphWidth's output is wrong by an unknown FontMatrix factor (PDFium doesn't fold this for these fonts). Fall back to neighbor-step fallback (next_char.ox - this_char.ox within same obj). """
    out: list[dict] = []
    for key_value, candidate_item in enumerate(raw_chars):
        if candidate_item.get("drop"):
            # Folded into the previous char by _apply_font_unicode (PDFium's
            # decomposition of a glyph text extraction emits as ONE precomposed char).
            continue
        obj = candidate_item["obj"]
        # w_raw = FPDFFont_GetGlyphWidth(font, code, fs_raw), captured in the
        # first pass while the page/font handle was alive.
        raw = candidate_item["w_raw"]
        if "w_synth" in candidate_item:
            # Synthesized glyph (PDFium font-layer drop): the advance was
            # computed from the surviving neighbors' pen gap.
            glyph_w = candidate_item["w_synth"]
        elif obj["fs_raw"] >= 1.5 or obj["scale_y"] >= 1.5:
            # Cases (a) and (b): GetGlyphWidth + matrix scaling works.
            glyph_w = raw * obj["scale_x"]
        else:
            # Case (c): Identity-matrix Type 3 — derive from neighbor.
            nxt = raw_chars[key_value + 1] if key_value + 1 < len(raw_chars) else None
            if (
                nxt is not None
                and nxt["obj"] is obj
                and abs(nxt["oy"] - candidate_item["oy"]) < 0.5
                and nxt["ox"] > candidate_item["ox"]
            ):
                glyph_w = nxt["ox"] - candidate_item["ox"]
            else:
                # Last char in obj or new line — scale by fs_eff/fs_raw.
                scale = (obj["fs_eff"] / obj["fs_raw"]) if obj["fs_raw"] > 0 else 1.0
                glyph_w = raw * scale

        # NOTE: glyph_w is PDFium's FPDFFont_GetGlyphWidth, used by the extraction advance model
        # the font's glyph width; this is the advance model. For some RTL
        # (Hebrew/Arabic) fonts PDFium's GetGlyphWidth does not match the actual
        # rendered char spacing, which leaves spurious intra-word spaces; that is
        # a PDF backend DATA LIMITATION (PDFium's hmtx/advance reporting),
        # not a condition to compensate for here (any positional override conflates
        # glyph advance with TJ word-gaps and breaks shaped Arabic). Left as-is.

        reference_item = {
            "ch": candidate_item["ch"],
            "is_ws": candidate_item["is_ws"],
            "is_mn": candidate_item["is_mn"],
            "is_cf": candidate_item["is_cf"],
            "ox": candidate_item["ox"], "oy": candidate_item["oy"],
            "glyph_w": glyph_w,
            "fs": obj["fs_eff"],
            "fs_x": obj["fs_raw"] * obj["scale_x"] if obj["scale_x"] > 0 else obj["fs_eff"],
            # Left edge from the text-positioning pen origin (ox), matching
            # span merger, not the glyph ink box: the ink-box left drifts ~0.1pt by
            # first-glyph side bearing, which trips the column-alignment gate
            # gate (tol 0.1) and over-splits double-spaced blocks. Right stays
            # ink-box (pen-right via glyph_w is unreliable for Type-3 fonts).
            "left": candidate_item["ox"], "right": candidate_item["right"],
            "top": candidate_item["top"], "bottom": candidate_item["bottom"],
            "font_name": candidate_item["font_name"],
            # Unique per-font identity (the PDFium font handle, == span merger'
            # loaded font identity). The merger splits chunks on this, not on font_name:
            # identity-matrix Type-3 fonts (an identity-matrix Type-3 font) all report an
            # empty name, so a name-based split can't separate a 12pt body run
            # from an inline 11pt code word ("...of expressions..."). span merger
            # emits a separate text item per font, so the body keeps fs=12 and
            # the code word fs=11.16 instead of the whole run collapsing to the
            # smaller fs_min.
            "font_key": obj["font_key"],
            "weight": obj["weight"],
            "obj": obj,                    # host text object (Tj/show-text)
        }
        if obj.get("vertical"):
            # Vertical-writing pen model, from the loose advance cell (probe-
            # for Identity-V: cell upper edge == pen y, cell extent ==
            # the per-char vertical advance with W2/DW2 applied by PDFium, and
            # the cell is horizontally centred on the pen x because the default
            # vertical origin vx is w/2 -- the default vertical-origin convention when the
            # font has no per-char vmetric).
            pen_y = max(candidate_item["cell_top"], candidate_item["cell_bot"])
            reference_item["v_pen_x"] = (candidate_item["left"] + candidate_item["right"]) / 2.0
            reference_item["v_pen_y"] = pen_y
            # pen y after this glyph's advance (text extraction previous glyph transform[5])
            reference_item["v_after"] = min(candidate_item["cell_top"], candidate_item["cell_bot"])
        out.append(reference_item)
    return out


def _inherited_box(pdf_doc, page_idx: int, name: str):
    """span merger ``inherited page-box lookup`` definition: MediaBox/CropBox resolved through the page-tree ``/Parent`` chain (page-tree inheritance lookup). PDFium's FPDFPage_Get*Box does NOT inherit (pdfium bug 1786), so inherited boxes must come from the PyPDF2 channel. Returns a raw 4-tuple or None (absent / not a 4-number array, matching span merger length gate)."""
    try:
        xref_cursor = pdf_doc.page_xref(page_idx)
        for _ in range(32):
            token_value, value = pdf_doc.xref_get_key(xref_cursor, name)
            if token_value != "null":
                if token_value != "array":
                    return None
                box_tokens = value.strip().lstrip("[").rstrip("]").split()
                if len(box_tokens) != 4:
                    return None  # span merger: array check  and  length == 4

                try:
                    return tuple(float(box_token) for box_token in box_tokens)
                except ValueError:
                    return None
            parent_key_type, position_value = pdf_doc.xref_get_key(xref_cursor, "Parent")
            if parent_key_type != "xref":
                return None
            xref_cursor = int(position_value.split()[0])
    except Exception:
        return None
    return None


def _page_view_rect(page, med_raw=None, crop_raw=None) -> tuple[float, float, float, float] | None:
    """span merger ``normalized page view`` : rectangle normalization'd CropBox clamped to the rectangle normalization'd MediaBox. Differing boxes are intersected (rectangle intersection); an empty or zero-area intersection, and a degenerate CropBox, fall back to the MediaBox (a degenerate MediaBox falls back to US-Letter, text extraction US-Letter fallback media box). ``med_raw``/``crop_raw`` are the INHERITED boxes from ``_inherited_box`` (None = absent/no reader); the PDFium getters below are the non-inheriting fallback."""
    def norm(secondary_item):
        if secondary_item is None:
            return None
        box_x_min, box_y_min, box_x_max, box_y_max = secondary_item
        count_item = (min(box_x_min, box_x_max), min(box_y_min, box_y_max), max(box_x_min, box_x_max), max(box_y_min, box_y_max))
        return count_item if (count_item[2] - count_item[0] > 0 and count_item[3] - count_item[1] > 0) else None

    med = norm(med_raw)
    if med is None:
        try:
            med = norm(tuple(page.get_mediabox()))
        except Exception:
            med = None
    if med is None:
        med = (0.0, 0.0, 612.0, 792.0)
    crop = norm(crop_raw)
    if crop is None:
        try:
            crop = norm(tuple(page.get_cropbox()))
        except Exception:
            crop = None
    if crop is None or crop == med:
        return med
    x_min, y_min = max(crop[0], med[0]), max(crop[1], med[1])
    x_max, y_max = min(crop[2], med[2]), min(crop[3], med[3])
    if x_max - x_min <= 0 or y_max - y_min <= 0:
        return med
    return (x_min, y_min, x_max, y_max)


def _off_page(mapping: dict, view_box) -> bool:
    """Position-comparison view box test: a non-diacritic glyph whose text origin is outside the page view box is dropped. The check compares ``pos - view box origin`` against the raw x1/y1 upper bounds, not width/height. ``view_box`` is the normalized page view as ``(x0, y0, x1, y1)``; None disables the test."""
    if view_box is None:
        return False
    origin_offset_x = mapping["ox"] - view_box[0]
    origin_offset_y = mapping["oy"] - view_box[1]
    return origin_offset_x < 0 or origin_offset_x > view_box[2] or origin_offset_y < 0 or origin_offset_y > view_box[3]
