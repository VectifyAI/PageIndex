"""Joins page glyphs into text runs with spacing and style thresholds."""

from __future__ import annotations

from .text_normalize import (
    TRACKING_SPACE_FACTOR,
    NON_SPACE_GAP_FACTOR,
    NEGATIVE_SPACE_FACTOR,
    SPACE_IN_FLOW_MIN_FACTOR,
    SPACE_IN_FLOW_MAX_FACTOR,
    _rtl_sign,
    _read_end,
    _read_gap,
)
from .char_extract import _off_page


def _merge_text_items(chars: list[dict], view_box=None) -> list[dict]:
    """exact text extraction position comparison + synthetic-space insertion + last-character buffer."""
    items: list[dict] = []
    chunk: dict | None = None
    two_last = [" ", " "]
    two_last_pos = [0]
    # text extraction active text item.previous glyph transform: set only by a glyph with a real
    # advance (`if (scaled advance)`), NEVER reset by text-item flush/setFont
    # -- it survives across item flushes for the whole page. (None, None)
    # until the first real glyph.
    last_ref: tuple = (None, None)

    def _object_merge_id(mapping: dict):
        # Per-object merge id: the boundary test below hard-splits between
        # different objects (the per-object split). q/Q grouping would require
        # fragile object/show-op ordinal alignment, so this is just the object's
        # identity.
        return id(mapping["obj"])

    def reset_last_chars() -> None:
        two_last[0] = " "
        two_last[1] = " "
        two_last_pos[0] = 0

    def save_last_char(char: str) -> bool:
        next_pos = (two_last_pos[0] + 1) % 2
        ret = (two_last[two_last_pos[0]] != " " and two_last[next_pos] == " ")
        two_last[two_last_pos[0]] = char
        two_last_pos[0] = next_pos
        return ret

    def flush() -> None:
        nonlocal chunk
        if chunk is not None and chunk["str"]:
            items.append(chunk)
        chunk = None

    def open_chunk(mapping: dict) -> None:
        nonlocal chunk, last_ref
        sign = _rtl_sign(mapping["ch"])
        # |text horizontal scale|: fs_x = |matrix scale| carries |Tz|, so the divisor is
        # the magnitude (a negative Tz uses the same text but scales it by |Tz|).
        # Tz == 0 keeps 1.0 (moot: PDFium emits no textpage chars for a
        # degenerate x-column). Boundary conditions:
        # PDFium's SYNTHESIZED layout spaces derive from the unscaled
        # text-space gap (~0.135em), so compressed Tz < ~75 can inject
        # spaces text extraction would not; negative-Tz runs re-merge via the
        # 180-degree pass with different item structure than span merger'
        # text orientation=-1 model; anisotropic CTM x rotated Tm differs
        # (norm-of-product vs span merger product-of-norms text advance scale).
        horizontal_scale_factor = abs(mapping["obj"].get("tz", 1.0))
        if not (horizontal_scale_factor > 0):
            horizontal_scale_factor = 1.0
        fs_x_tz = mapping["fs_x"] / horizontal_scale_factor
        chunk = {
            "str": [],
            "sign": sign,                       # +1 LTR, -1 RTL (signed x-axis)
            "obj": mapping["obj"],                    # host text object (Tj/show-text)
            # text extraction fixes item transform at the item's FIRST glyph
            # (item initialization) and never updates it mid-item, while
            # chunk["obj"] re-points to the LAST appended glyph's object (the
            # flush/prose bookkeeping needs that). Snapshot the opening
            # object's matrix so the emitted skew reads first-glyph geometry.
            "mtx0": mapping["obj"]["mtx"],
            "flush_id": _object_merge_id(mapping),                # per-object merge id (was q/Q flush scope)
            "left": mapping["left"], "right": mapping["right"],
            "top": mapping["top"], "bottom": mapping["bottom"],
            "fs": mapping["fs"],
            "fs_min": mapping["fs"],
            # Glyph advance (FPDFFont_GetGlyphWidth*scale). Unused by the
            # horizontal merger (it reads prev_text_x); carried only so
            # _remerge_rotated can run a direct 1-D position comparison
            # (gap = next_origin - (cur_origin + glyph_w)) along the rotation axis.
            "glyph_w": mapping.get("glyph_w", 0.0),
            "font_name": mapping["font_name"],
            "font_key": mapping["font_key"],
            "weight": mapping["weight"],
            # Per-char style tallies for majority-vote at span emission.
            # span merger text item records only the first char's font name;
            # the heading heuristics' heading detection ends up marking paragraph
            # lead-ins like **Bold prefix.** Regular continuation as
            # "bold lines" because of that. Tally per-char so we can
            # emit the dominant font/weight instead.
            "font_tally": {mapping["font_name"]: 1},
            "weight_tally": {mapping["weight"]: 1},
            # ``prev_text_x`` tracks where the next glyph would land if
            # charSpacing=0 — i.e. text matrix.e after this glyph's emit.
            # For ligature components, PDFium reports them at the same
            # origin but with bbox spanning the full ligature, so taking
            # max(ox+glyph_w, bbox.right) makes the next non-ligature
            # char see a small positive advance instead of a big gap.
            # (_read_end uses the same this for an RTL chunk.)
            "prev_text_x": _read_end(mapping, sign),
            "prev_oy": mapping["oy"],
            # text extraction threshold base is text state.font size WITHOUT Tz
            # (item initialization: Tz enters only the pen advance, not
            # text advance scale). PDFium folds Tz into the object matrix, so
            # fs_x carries it; divide the show-op's text horizontal scale back out.
            "tracking": fs_x_tz * TRACKING_SPACE_FACTOR,
            "not_a_space": fs_x_tz * NON_SPACE_GAP_FACTOR,
            "negative": fs_x_tz * NEGATIVE_SPACE_FACTOR,
            "flow_min": fs_x_tz * SPACE_IN_FLOW_MIN_FACTOR,
            "flow_max": fs_x_tz * SPACE_IN_FLOW_MAX_FACTOR,
            "height": mapping["fs"],
            # True once a real whitespace glyph follows the last visible glyph in
            # this chunk; gates whether an object boundary is a prose word-break
            # (merge) or a layout jump (hard split). See the is_ws handler.
            "ws_pending": False,
        }
        if "v_pen_y" in mapping:
            # Vertical-writing pen state for _remerge_vertical (set only for
            # vertical-CMap objects).
            chunk["v_pen_x"] = mapping["v_pen_x"]
            chunk["v_pen_y"] = mapping["v_pen_y"]
            chunk["v_after"] = mapping["v_after"]
            chunk["v_last_x"] = mapping["v_pen_x"]
        if mapping["is_mn"]:
            # A zero-width diacritic has scaled advance == 0, so it does NOT
            # establish the advance reference; the chunk INHERITS the
            # page-surviving one (text extraction previous glyph transform persists across
            # flushes; (None, None) only until the page's first real glyph).
            chunk["prev_text_x"], chunk["prev_oy"] = last_ref
        else:
            last_ref = (chunk["prev_text_x"], chunk["prev_oy"])

    def emit_fake_space(gap: float) -> None:
        """Emit an out-of-flow synthetic space after the current chunk. The synthetic item uses the previous glyph transform, not the next glyph, so the space remains attached to the line it trails. Its height stays zero; otherwise vertical-alignment checks can attach the space to a neighboring line and create a spurious line merge. """
        assert chunk is not None
        reset_last_chars()  # text extraction synthetic-space insertion standalone path resets first
        page_x, baseline = chunk["prev_text_x"], chunk["prev_oy"]
        # WIDTH = abs(gap). Synthetic out-of-flow spaces use ``width: abs(e)``,
        # where e is the out-of-flow advance (the gap it
        # spans), height 0 for horizontal text. The box therefore runs from the
        # previous glyph's pen end (px) forward by the gap: [px, px+gap] LTR,
        # [px-gap, px] RTL. In-flow spaces are handled by pushing " " into the
        # current item; standalone spaces use this separate geometry. Height
        # stays 0, so vertical-alignment guards are
        # untouched and the outline is unaffected.
        width_value = abs(gap)
        if chunk["sign"] >= 0:
            sp_left, sp_right = page_x, page_x + width_value
        else:
            sp_left, sp_right = page_x - width_value, page_x
        meta = (chunk["obj"], chunk["fs"], chunk["font_name"],
                chunk["font_key"], chunk["weight"])
        flush()
        items.append({
            "str": [" "], "sign": 1, "obj": meta[0],
            "left": sp_left, "right": sp_right,
            "top": baseline, "bottom": baseline,        # HEIGHT 0
            "fs": meta[1], "fs_min": meta[1],
            "font_name": meta[2], "font_key": meta[3], "weight": meta[4],
            "font_tally": {meta[2]: 1}, "weight_tally": {meta[4]: 1},
        })

    def extend_chunk(mapping: dict, leading_space: bool) -> None:
        nonlocal last_ref
        assert chunk is not None
        if leading_space:
            chunk["str"].append(" ")
        chunk["str"].append(mapping["ch"])
        chunk["left"] = min(chunk["left"], mapping["left"])
        chunk["right"] = max(chunk["right"], mapping["right"])
        # Text-item box accumulation: appending a glyph only grows the item's
        # width. The item's vertical box is
        # fixed at item creation -- transform[5] = first-glyph baseline, and
        # height = font size (== the item's em). A per-glyph baseline offset
        # within the item (e.g. a lowered character inside a mixed-baseline
        # logo, or any sub/superscript not split into its own item) is therefore
        # absorbed: it does NOT extend the item box. Do not expand top/bottom
        # here; they stay at the open glyph's [oy, oy+fs]. Expanding them would
        # let inline baseline offsets distort downstream line-height gates.
        chunk["prev_text_x"] = _read_end(mapping, chunk["sign"])
        chunk["prev_oy"] = mapping["oy"]
        last_ref = (chunk["prev_text_x"], chunk["prev_oy"])
        chunk["glyph_w"] = mapping.get("glyph_w", 0.0)   # last glyph's advance (for _remerge_rotated)
        # When a real-space word-break us merge across a text-object boundary
        # (prose case), the chunk must adopt the new object so the rest of that
        # word's glyphs (same object, no space before them) don't re-trigger the
        # object hard-split mid-word. span merger line item has no per-glyph object.
        chunk["obj"] = mapping["obj"]
        chunk["flush_id"] = _object_merge_id(mapping)
        chunk["font_tally"][mapping["font_name"]] = chunk["font_tally"].get(mapping["font_name"], 0) + 1
        chunk["weight_tally"][mapping["weight"]] = chunk["weight_tally"].get(mapping["weight"], 0) + 1
        # Track min fs within the chunk so small-caps headings ("A"+
        # "BSTRACT") expose the body-text fs of the small-cap part
        # rather than the leading full-cap fs. The downstream big-font
        # check then doesn't false-positive on inline math labels like
        # "LEMMA 1" whose small-cap fs is below body size.
        if mapping["fs"] > 0:
            chunk["fs_min"] = min(chunk["fs_min"], mapping["fs"])
        if "v_pen_y" in mapping and "v_pen_y" in chunk:
            chunk["v_after"] = mapping["v_after"]
            chunk["v_last_x"] = mapping["v_pen_x"]

    for text in chars:
        # text extraction text-item box accumulation char loop order (span merger+):
        # invisible format-mark classification is skipped entirely BEFORE the whitespace test.
        if text["is_cf"]:
            # The format-mark skip sits ahead of the scaled-advance and
            # char-spacing block, so the mark moves neither the text matrix nor
            # the previous-position reference: the reference pen never sees it.
            # PDFium's char origins DO include its advance, so carry the
            # reading-direction reference past it; otherwise that advance
            # reappears as a gap and the next glyph gets an in-flow or
            # standalone " " with no counterpart. (Char spacing, also skipped
            # here, is not separable from PDFium's origins.) With no chunk open
            # the page's previous-position reference is still unset, so the
            # position comparison is unconditionally true and there is no gap
            # to correct.
            if chunk is not None and chunk["prev_text_x"] is not None:
                chunk["prev_text_x"] += chunk["sign"] * text.get("glyph_w", 0.0)
                last_ref = (chunk["prev_text_x"], chunk["prev_oy"])
            continue
        if text["is_ws"]:
            save_last_char(" ")
            # Remember a real whitespace glyph bridged the gap. PDFium fragments a
            # flowing prose line into per-word text-objects (each with a trailing
            # space glyph); text extraction keeps the whole line as one Tj item. A real space
            # at an object boundary marks a prose word-break -> merge across it.
            # A positional (spaceless) object change marks a layout jump (table
            # cell, separate Tj) -> keep the hard object split.
            if chunk is not None:
                chunk["ws_pending"] = True
            continue
        # Zero-width diacritics append without a position-based flush: they do NOT call
        # position comparison (no position-based flush) and uses
        # scaled advance=0 (no advance) -- it just appends the mark to the
        # current item. Append it without touching prev_text_x.
        # BUT a Tf style flush is an operator-level split that already closed
        # the previous item before the glyph loop ran, so a mark arriving in a
        # different font/size (for example, a math accent over an italic letter,
        # letter, each its own Tf'd show op) opens its OWN item, with its own
        # raised origin and em height. Only
        # the position-based flush is skipped for diacritics, never the style
        # flush, so the mark passes the same font_key/fs/object boundary test
        # as any visible glyph.
        if text["is_mn"]:
            if chunk is not None and (
                chunk["font_key"] != text["font_key"]
                or abs(text["fs"] - chunk["fs"]) > 1e-6
                or (_object_merge_id(text) != chunk["flush_id"] and not chunk["ws_pending"])
            ):
                flush()
            if chunk is None:
                # open_chunk inherits the page-surviving advance reference
                # (text extraction previous glyph transform persists across flushes; None only at
                # page start -- see the prev_text_x-is-None guard below).
                open_chunk(text)
                assert chunk is not None
                lead = save_last_char(text["ch"])
                # the heading heuristics pushes the last-character buffer lead into the fresh mark item
                # (a Tf flush does not reset the last-character buffer).
                if lead:
                    chunk["str"].append(" ")
                chunk["str"].append(text["ch"])
            else:
                lead = save_last_char(text["ch"])
                if lead:
                    chunk["str"].append(" ")
                chunk["str"].append(text["ch"])
                # span merger: a zero-width diacritic has scaled advance=0, so it neither
                # moves the text matrix NOR updates previous glyph transform (span merger
                # `if (scaled advance)` is false). The next glyph's line-break /
                # dy test therefore compares against the last VISIBLE glyph's
                # baseline -> leave BOTH prev_text_x and prev_oy untouched here
                # (the box also stays the open glyph's -- see extend_chunk note).
            continue

        # span merger: a non-diacritic glyph whose origin is off the page view box is
        # skipped (position comparison returns false only off-page). cf/ws
        # were handled above and diacritics (is_mn) never reach here, matching
        # span merger `!zero-width diacritic classification  and  !position comparison`.

        if _off_page(text, view_box):
            continue

        if chunk is None:
            open_chunk(text)
            save_last_char(text["ch"])
            assert chunk is not None
            chunk["str"].append(text["ch"])
            continue

        # Style boundary: split on font-identity change (font_key, the PDFium
        # font handle == span merger per-font loaded font identity) OR ANY font size change.
        # span merger emit a separate text item on every setFont
        # (Tf) operator -- i.e. on any font OR size change. Represent

        # that with an exact effective-fs compare (the 1e-6 is only to absorb
        # float noise in the snapped fs). An earlier 10% tolerance under-split
        # small-caps runs; exact is intentional here.

        # font_key/fs is a proxy for the Tf flush, not a literal replay of every
        # content-stream flush boundary. Text-item boundaries are driven mostly
        # by position comparison; PDFium exposes final glyph coordinates, so the
        # per-object + font_key/fs proxy gives the heading pipeline the intended
        # span structure without overfitting to partial operator state. The
        # remaining boundary cases, such as missing-glyph fallback handles or
        # rendered-size jitter under scaled Type-3 matrices, are limited to span
        # boundaries.
        # The heading heuristics' downstream heading detector then
        # treats a chunk's first-char style as the whole chunk's style:
        #
        # * font split: a paragraph lead-in like "**Inflation.**
        # Consumer price..." would otherwise be a single bold chunk
        # and false-detect as a heading on every paragraph. (font_name
        # alone can't separate identity-matrix Type-3 fonts, whose names
        # are all empty, so a 12pt body run and an inline 11pt code word
        # would merge and collapse to the smaller fs_min.)
        # * fs split: inline math labels like "LEMMA 1 (...) ..."
        # (first-cap large + small-cap rest + body) would otherwise
        # merge into a single chunk that pipeline accepts as a
        # heading; splitting forces the small-cap rest into its own
        # chunk where the heading heuristics' short-text/type checks reject it. Same
        # guard also helps math-heavy page/identity-matrix Type-3 sample exercise items
        # ("X.Y www") and section headings stay detectable —
        # without it they collapse into the surrounding body chunk.
        #
        # Trade-off: small-caps "ABSTRACT" / "ECONOMIC ANALYSIS"
        # don't merge across the cap-to-small-cap fs step. The
        # downstream tokenizer relaxation (LineTokenizer.add_line below)
        # joins them at token level instead.
        ws_bridge = chunk["ws_pending"]
        chunk["ws_pending"] = False
        if chunk["prev_text_x"] is None:
            # The item was opened by a zero-width diacritic AT PAGE START (no
            # real glyph has set the page's advance reference yet, so span merger'
            # previous glyph transform is still null): position comparison returns
            # true unconditionally -- no positional boundary, no fake space,
            # no line break. Only the style flush (the Tf proxy) still
            # applies; otherwise the glyph appends plainly and, being a real
            # advance, establishes the reference via extend_chunk.
            if (chunk["font_key"] != text["font_key"]
                    or abs(text["fs"] - chunk["fs"]) > 1e-6):
                flush()
                open_chunk(text)
                lead = save_last_char(text["ch"])
                assert chunk is not None
                if lead:  # Detail: no reset on this path, the lead survives

                    chunk["str"].append(" ")
                chunk["str"].append(text["ch"])
            else:
                extend_chunk(text, save_last_char(text["ch"]))
            continue
        if (
            chunk["font_key"] != text["font_key"]
            or abs(text["fs"] - chunk["fs"]) > 1e-6
            # Hard-split at a text-object boundary -- but ONLY when no real
            # whitespace glyph bridged it. the object merge id groups consecutive per-glyph show operators
            # objects (PDFium emits one FPDF_PAGEOBJ_TEXT per glyph when the PDF
            # draws glyphs individually) into one id, so a CJK title set as N
            # per-glyph Tj does NOT shatter into N single-glyph items -- the
            # positional logic below merges it / line-breaks it like the item merger.
            # Every normal (multi-glyph) object keeps its own id, so this stays
            # the per-object split for Latin text: on dense justified tables (2023
            # dense table document) each fragment is its own object -> hard split, exact
            # with the item merger. On flowing prose PDFium may split
            # per word with a real space glyph between words, where text extraction keeps
            # the whole line as one item; a real space at the boundary (ws_bridge)
            # marks the prose case -> fall through to the in-flow/out-of-flow gap
            # logic, which merges the word-objects into one line item like span merger.
            or (_object_merge_id(text) != chunk["flush_id"] and not ws_bridge)
        ):
            # span merger position comparison runs synthetic-space insertion for EVERY glyph,
            # including the first glyph of a new item/Tj. So an out-of-flow gap
            # across an item boundary still gets a standalone height-0 " "
            # (this is the trailing space after math-heavy page's "...y)" before the next
            # equation-number object). An in-flow / adjacent boundary does not.
            # text extraction position comparison order: a line break (|advance-y| >
            # height -> line-break emission) or backward jump takes precedence over the
            # space logic; only a same-line gap past tracking-space threshold emits the
            # standalone " " (covering BOTH the in-flow empty-item case and
            # the out-of-flow synthetic-space insertion case, which are identical here).
            boundary_gap = _read_gap(chunk["prev_text_x"], text, chunk["sign"])
            _same_line = abs(text["oy"] - chunk["prev_oy"]) <= chunk["height"]
            if _same_line and boundary_gap > chunk["tracking"]:
                emit_fake_space(boundary_gap)
                keep_lead = False   # the heading heuristics synthetic-space insertion reset the last-character buffer
            else:
                # the heading heuristics resets the two-char buffer on every positional branch
                # (line-break emission / negative / non-space) but NOT in the tracking
                # window (non-space, tracking-space threshold] -- a thin real space
                # just before a Tf-style flush survives into the new item.
                keep_lead = (_same_line
                             and chunk["not_a_space"] < boundary_gap <= chunk["tracking"])
                flush()
            open_chunk(text)
            lead = save_last_char(text["ch"])
            assert chunk is not None
            if lead and keep_lead:
                chunk["str"].append(" ")
            chunk["str"].append(text["ch"])
            continue

        advance = _read_gap(chunk["prev_text_x"], text, chunk["sign"])
        line_delta_y = text["oy"] - chunk["prev_oy"]
        height = chunk["height"]

        # Ligature decomposition: PDFium reports consecutive ligature
        # components at the same x origin (e.g. "fi" -> 'f' and 'i' at the
        # identical origin). prev_text_x was set to prev.ox +
        # prev.glyph_w, so we see advance ≈ -prev.glyph_w. Glyph widths
        # of typical Latin chars are in [0.2*fs, 0.9*fs]. Detect this
        # case (negative advance whose magnitude is in that range) and
        # silently merge — matches span merger on same-origin
        # ligature components. ONLY within one text object: decomposition
        # is per-glyph, so both components always share the show op. A
        # cross-object negative advance is a real content-stream back-jump
        # that text extraction itself sees and breaks on (TeX standalone accents:
        # math-heavy page 'accented name stem'+'¨'+'lkopf' is three show ops, '¨' jumps back -0.41fs;
        # text extraction raw-categorizes U+00A8 as a normal glyph -- category comes
        # from glyph Unicode BEFORE the normalized Unicode expansion -- so
        # position comparison flushes and ' ̈lkopf' opens a new item).
        if (text["obj"] is chunk["obj"] and abs(line_delta_y) < 0.1 * height
                and -0.9 * chunk["fs"] <= advance < -0.2 * chunk["fs"]):
            lead = save_last_char(text["ch"])
            # Don't extend prev_text_x backwards; ligature component
            # shares position with prev, so prev_text_x stays the same.
            if lead:
                chunk["str"].append(" ")
            chunk["str"].append(text["ch"])
            chunk["left"] = min(chunk["left"], text["left"])
            chunk["right"] = max(chunk["right"], text["right"])
            # Ligature component shares the open glyph's item box; only width
            # grows (see extend_chunk note -- text extraction never expands the item's
            # vertical extent on append).
            chunk["prev_oy"] = text["oy"]
            last_ref = (chunk["prev_text_x"], chunk["prev_oy"])
            continue

        # text extraction position comparison compares advance-x against
        # ``text orientation * threshold`` (text orientation = sign(item.width)).
        # We get the same result for HORIZONTAL text by normalising the gap into
        # READING DIRECTION up front: ``advance`` (= _read_gap with chunk["sign"]
        # from _rtl_sign) is already signed so that "forward" is positive for BOTH
        # LTR and RTL, hence the thresholds below are compared UNMULTIPLIED.
        # * LTR (sign=+1): intentional to the literal layout-classifier form.

        # * horizontal RTL (Hebrew/Arabic, sign=-1): handled via the x-axis
        # paired logic in _read_end/_read_gap (added in 528f958).
        # VERTICAL text (span merger vertical-font flag / advance-y branch) is NOT handled
        # here: a vertical column shatters per-glyph below and is re-merged by
        # the gated _remerge_vertical post-pass (detection: vertical-CMap fonts
        # via _page_vertical_resnames; matched against the item merger's
        # vertical-text rules.
        if advance < chunk["negative"]:
            if abs(line_delta_y) > 0.5 * height:
                # the heading heuristics line-break emission calls reset the last-character buffer before flushing.
                reset_last_chars()
                flush()
                open_chunk(text)
                save_last_char(text["ch"])
                assert chunk is not None
                chunk["str"].append(text["ch"])
            else:
                reset_last_chars()
                flush()
                open_chunk(text)
                save_last_char(text["ch"])
                assert chunk is not None
                chunk["str"].append(text["ch"])
            continue

        if abs(line_delta_y) > height:
            # the heading heuristics line-break emission calls reset the last-character buffer before flushing.
            reset_last_chars()
            flush()
            open_chunk(text)
            save_last_char(text["ch"])
            assert chunk is not None
            chunk["str"].append(text["ch"])
            continue

        if advance <= chunk["not_a_space"]:
            reset_last_chars()

        if advance <= chunk["tracking"]:
            lead = save_last_char(text["ch"])
            extend_chunk(text, lead)
            continue

        if chunk["flow_min"] <= advance <= chunk["flow_max"]:
            reset_last_chars()
            chunk["str"].append(" ")
            lead = save_last_char(text["ch"])
            extend_chunk(text, lead)
            continue

        # OUT-OF-FLOW gap (advance > in-flow space threshold): text extraction synthetic-space insertion
        # flushes the current item and pushes a
        # STANDALONE " " item with height 0, then a new item begins at this glyph.
        # The zero height is load-bearing: the heading heuristics' vertical-alignment test
        # can't align this inter-run space with a neighbouring line, so it doesn't
        # cause a spurious line merge (the math-heavy page inline math heading heading drop). The
        # standalone " " also keeps the word separator in the joined line text Yf
        # so a positionally-spaced title like "3 The section heading"
        # (Type-3 fonts, no real space glyphs) does not collapse to
        # "3TheStaticSemantics" and lose its section number to the _Tf regex.
        reset_last_chars()
        emit_fake_space(advance)     # standalone height-0 " ", width=abs(gap) (exact span merger)
        open_chunk(text)
        save_last_char(text["ch"])
        assert chunk is not None
        chunk["str"].append(text["ch"])

    flush()
    return items
