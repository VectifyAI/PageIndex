"""Applies per-font Unicode maps to page chars and synthesizes dropped glyphs."""

from __future__ import annotations

import bisect
import difflib
from collections import Counter
import pypdfium2.raw as pdfium_c

from .text_normalize import _is_whitespace
from .font_unicode import _font_unicode_map
from .code_walk import (
    _char_category,
    _walk_codes,
)


def _apply_font_unicode(
    text_page,
    raw_chars: list[dict],
    objects: list[dict],
    show_codes: list[tuple[int | None, tuple[int, ...], float]],
    pdf_doc,
    map_cache: dict,
) -> None:
    """Patch each char's unicode to span merger glyph Unicode (`map.get(code)  or  chr(code)`, content stream tokenizer glyph mapping) where PDFium's decode disagrees. Two granularities, both gated by _walk_codes' both-streams-exhaust rule: - object mode (when PDFium's text objects pair consistent with the page's show ops, the _assign_flush_ids precondition): each object's chars are walked against its own show op's codes. This is immune to PDFium's textpage segment reordering (e.g. math-heavy page margin labels emitted at a different page position than paint order) because chars keep stream order WITHIN an object; a desync rolls back only that object. - page mode (counts differ, e.g. PDFium splitting a TJ into several objects): all non-generated textpage chars are walked against all show ops' codes in paint order; any desync rolls back the whole page. """
    if not show_codes:
        return

    def targets_for(font_xref: int | None, other_numbers: tuple[int, ...]) -> list[str] | None:
        if font_xref is None:
            return None
        if font_xref not in map_cache:
            try:
                map_cache[font_xref] = _font_unicode_map(pdf_doc, font_xref)
            except Exception:
                map_cache[font_xref] = None
        entry = map_cache[font_xref]
        if entry is None:
            return None
        next_block, measure_item = entry
        if next_block == 1:
            return [measure_item.get(code) or chr(code) for code in other_numbers]
        return [measure_item.get((other_numbers[key_value] << 8) | other_numbers[key_value + 1]) or chr((other_numbers[key_value] << 8) | other_numbers[key_value + 1])
                for key_value in range(0, len(other_numbers) - 1, 2)]

    def apply(patches: list[tuple[int, str]], drops: list[int],
              chars_by_index: dict[int, dict]) -> None:
        for index_value, token_value in patches:
            candidate_item = chars_by_index.get(index_value)
            if candidate_item is None:
                continue  # char was dropped at extraction; nothing to patch
            candidate_item["ch"] = token_value
            candidate_item["is_ws"], candidate_item["is_mn"], candidate_item["is_cf"] = _char_category(token_value)
        for index_value in drops:
            candidate_item = chars_by_index.get(index_value)
            if candidate_item is not None:
                candidate_item["drop"] = True

    chars_by_index = {raw_char["i"]: raw_char for raw_char in raw_chars}

    if len(objects) == len(show_codes):
        # Object mode: pair text objects with show ops ordinally (both are in
        # content-stream paint order) and walk each pair independently.
        chars_by_obj: dict[int, list[tuple[int, str]]] = {}
        for raw_char in raw_chars:
            if raw_char["is_gen"]:
                continue
            chars_by_obj.setdefault(id(raw_char["obj"]), []).append((raw_char["i"], raw_char["ch"]))
        desynced: list[int] = []
        failed_windows: list[list[int]] = []
        synth_sites: list[dict] = []
        targets_by_object_index: dict[int, list[str] | None] = {}
        for object_index, (obj, (font_index, encoded_text, _tz)) in enumerate(zip(objects, show_codes)):
            target_text_items = targets_for(font_index, encoded_text)
            targets_by_object_index[object_index] = target_text_items
            if target_text_items is None:
                continue  # uncovered font: this object keeps PDFium's output
            res = _walk_codes(chars_by_obj.get(id(obj), []), target_text_items)
            if res is None:
                # Desync: often a boundary-attribution error (the geometric
                # char->object lookup parks a show op's edge glyph in the
                # NEIGHBOURING object's list: punctuation at a run boundary can
                # land in the previous object, and heavily overlapped chart
                # labels can park a leading glyph in the wrong object. Record for the
                # window re-walk below; a genuine mismatch stays rolled back
                # there too.
                desynced.append(object_index)
                continue
            apply(res[0], res[1], chars_by_index)
        # Re-walk each window of desynced objects (bridging up to 2 covered,
        # successfully-walked objects between them) as one unit: boundary-
        # attribution errors cancel inside the window (the page-mode walk
        # scoped to the ambiguous region) and the exhaust-in-sync gate still
        # rejects anything else. On commit, REASSIGN each consumed char to
        # the object whose show op consumed it -- the stream-side ownership --
        # repairing the geometric attribution for the paint-order sort, the
        # merger's font/fs identity and the Type-3 sizing alike.
        def _rewalk_window(window: list[int]) -> bool:
            char_value = sorted(
                (pair for state_item in window for pair in chars_by_obj.get(id(objects[state_item]), [])))
            text_transform: list[str] = []
            owner: list[int] = []
            for state_item in window:
                target_text_items = targets_by_object_index[state_item]
                assert target_text_items is not None
                text_transform.extend(target_text_items)
                owner.extend([state_item] * len(target_text_items))
            def _commit(res) -> bool:
                if res is None:
                    return False
                apply(res[0], res[1], chars_by_index)
                for char_index, text_index in res[2]:
                    candidate_item = chars_by_index.get(char_index)
                    if candidate_item is not None and candidate_item["obj"] is not objects[owner[text_index]]:
                        candidate_item["obj"] = objects[owner[text_index]]
                # Skipped targets are glyphs PDFium never emitted; record
                # each with its show op and surviving stream neighbours so
                # _synthesize_dropped_glyphs can re-emit it (text extraction does).
                for text_index, pos in res[3]:
                    synth_sites.append({
                        "t": text_transform[text_index], "owner": objects[owner[text_index]],
                        "prev_i": char_value[pos - 1][0] if pos > 0 else None,
                        "next_i": char_value[pos][0] if pos < len(char_value) else None,
                    })
                return True
            if len(window) >= 2 and _commit(_walk_codes(char_value, text_transform)):
                return True
            # Pure-displacement fallback: PDFium's textpage can also REORDER a
            # char across the window (TeX accents again: 'accented word stem'+'´'+'es'
            # arrives as '...ilites´', and the 't' sits in the 'es' object),
            # which the linear walk above can never align. When the chars are
            # EXACTLY the targets as a multiset (no decode work left -- only
            # placement is wrong), align via SequenceMatcher and repair
            # OWNERSHIP alone: equal blocks map positionally, the few
            # displaced chars (<=4) map by literal value. Single-char targets
            # only, so target index == string position.
            def _displacement_repair() -> bool:
                if any(len(token_value) != 1 for token_value in text_transform):
                    return False
                chs = "".join(candidate_item for _, candidate_item in char_value)
                tts = "".join(text_transform)
                deficit = len(tts) - len(chs)
                if (chs == tts or deficit < 0 or deficit > 8
                        or (Counter(chs) - Counter(tts))):
                    return False
                state_map = difflib.SequenceMatcher(None, tts, chs, autojunk=False)
                char_to_tgt: dict[int, int] = {}
                loose_target_indexes: list[int] = []
                loose_char_indexes: list[int] = []
                for tag, index_one, index_two, char_start, char_end in state_map.get_opcodes():
                    if tag == "equal":
                        for reference_item in range(index_two - index_one):
                            char_to_tgt[char_start + reference_item] = index_one + reference_item
                    else:
                        loose_target_indexes.extend(range(index_one, index_two))
                        loose_char_indexes.extend(range(char_start, char_end))
                if len(loose_char_indexes) > 24:
                    return False
                used_targets: set[int] = set()
                for char_index in loose_char_indexes:
                    cdict = chars_by_index.get(char_value[char_index][0])
                    cands = [target_index for target_index in loose_target_indexes
                             if target_index not in used_targets and tts[target_index] == chs[char_index]]
                    if not cands:
                        return False  # a displaced char with no equal target
                    if cdict is not None and len(cands) > 1:
                        # Identical glyphs (the 21 scattered 'α' labels):
                        # pick the candidate whose OBJECT box sits closest
                        # to the char -- the one signal that distinguishes
                        # equal-valued slots.
                        origin_x, origin_y = cdict["ox"], cdict["oy"]
                        def _object_distance_sq(target_index: int) -> float:
                            item_value = objects[owner[target_index]]
                            delta_x = max(item_value["l"] - origin_x, 0.0, origin_x - item_value["r"])
                            delta_y = max(item_value["b"] - origin_y, 0.0, origin_y - item_value["t"])
                            return delta_x * delta_x + delta_y * delta_y
                        cands.sort(key=_object_distance_sq)
                    char_to_tgt[char_index] = cands[0]
                    used_targets.add(cands[0])
                # Leftover loose TARGETS = glyphs PDFium never emitted (the
                # font-layer drop class). Record each between its
                # nearest MAPPED neighbours for re-synthesis.
                leftover = [target_index for target_index in loose_target_indexes if target_index not in used_targets]
                if leftover:
                    tgt_to_char = {target_index: char_index for char_index, target_index in char_to_tgt.items()}
                    mapped_tis = sorted(tgt_to_char)
                    for target_index in leftover:
                        page_value = bisect.bisect_left(mapped_tis, target_index)
                        point_value = mapped_tis[page_value - 1] if page_value > 0 else None
                        normalized_token = mapped_tis[page_value] if page_value < len(mapped_tis) else None
                        synth_sites.append({
                            "t": tts[target_index], "owner": objects[owner[target_index]],
                            "prev_i": char_value[tgt_to_char[point_value]][0] if point_value is not None else None,
                            "next_i": char_value[tgt_to_char[normalized_token]][0] if normalized_token is not None else None,
                        })
                for char_index, target_index in char_to_tgt.items():
                    candidate_item = chars_by_index.get(char_value[char_index][0])
                    if candidate_item is not None and candidate_item["obj"] is not objects[owner[target_index]]:
                        candidate_item["obj"] = objects[owner[target_index]]
                return True
            if _displacement_repair():
                return True
            # Final resort: the same walk with anchored drop-skips, for
            # windows containing glyphs PDFium never emitted (font-layer
            # drops). The rest of the window still gets its patches and
            # stream-side ownership; the dropped glyphs are recorded for
            # synthesis.
            if not _commit(_walk_codes(char_value, text_transform, allow_skips=True)):
                failed_windows.append(list(window))
                return False
            return True
        # A window that resolves only by DECLARING drops (recording synth
        # sites) has trusted its local char census; when chars were stolen
        # ACROSS window boundaries that census lies (a starved window
        # "drops" a glyph whose char sits, surplus, in another failed
        # window). Track those windows so the mega pass below can supersede
        # their local verdicts.
        synth_windows: list[tuple[list[int], int, int]] = []
        def _run_window(window: list[int]) -> None:
            before = len(synth_sites)
            if _rewalk_window(window) and len(synth_sites) > before:
                synth_windows.append((list(window), before, len(synth_sites)))
        window: list[int] = []
        for object_index in desynced:
            if window:
                gap = range(window[-1] + 1, object_index)
                if (len(gap) <= 2
                        and all(targets_by_object_index.get(bridge_index) is not None for bridge_index in gap)):
                    window.extend(gap)
                    window.append(object_index)
                    continue
                _run_window(window)
            window = [object_index]
        if window:
            _run_window(window)
        # Page-scope last resort: scattered same-glyph labels (dense math-heavy page's
        # 21 'α' show ops over a vector figure) defeat per-window walks --
        # the geometric attribution piles several chars on some ops and
        # leaves others empty ACROSS window boundaries (donor ops hold a
        # stolen surplus char, starved ops none). Merge every failed AND
        # every drop-declaring window into one final window so the
        # displacement/skip repairs see the whole cluster at once: the
        # surplus cancels the deficit, stolen chars are reassigned to their
        # true ops, and only the genuine font-layer drops remain as synth
        # sites. The locally-recorded sites are dropped first (the mega
        # re-records with full context) and restored if the mega fails.
        cand = failed_windows + [window for window, _, _ in synth_windows]
        if len(cand) >= 2:
            stash = synth_sites[:]
            for _, font, window_end in reversed(synth_windows):
                del synth_sites[font:window_end]
            failed_windows = []
            mega = sorted({mega_index for window in cand for mega_index in window})
            if not _rewalk_window(mega):
                synth_sites[:] = stash  # mega failed: keep local verdicts
        if synth_sites:
            # Census gate: a recorded site is a REAL font-layer drop only if
            # the PAGE-WIDE multiset still misses that value (covered ops'
            # target codepoints minus PDFium's final chars). A window-local
            # repair can otherwise declare a glyph dropped whose char simply
            # sits, mis-attributed, in an op that walked clean -- the a clipped-cell table
            # with star glyphs: 7 star codes, 7 star chars page-wide, but the
            # clip-overlapped cells starve two ops, and the donors never
            # fail so the mega pass can't see them. WHITESPACE is never
            # synthesized: a missing space char is PDFium's textpage
            # space-run normalization (text extraction runs its own space
            # normalization, already implemented in the merger), not a font-layer
            # drop.
            census: Counter = Counter()
            for text_adjustment in targets_by_object_index.values():
                if text_adjustment is not None:
                    for target_text in text_adjustment:
                        census.update(target_text)
            for raw_char in raw_chars:
                if not raw_char["is_gen"] and not raw_char.get("drop"):
                    census.subtract(raw_char["ch"])
            kept: list[dict] = []
            for encoded_text in synth_sites:
                if all(_is_whitespace(ord(ch_)) for ch_ in encoded_text["t"]):
                    continue
                if all(census[ch_] > 0 for ch_ in encoded_text["t"]):
                    for ch_ in encoded_text["t"]:
                        census[ch_] -= 1
                    kept.append(encoded_text)
            if kept:
                _synthesize_dropped_glyphs(kept, raw_chars, chars_by_index)
        return

    # Page mode.
    seq: list[tuple[int, str]] = []
    char_count = pdfium_c.FPDFText_CountChars(text_page)
    for char_index in range(char_count):
        if pdfium_c.FPDFText_IsGenerated(text_page, char_index) == 1:
            continue
        codepoint = pdfium_c.FPDFText_GetUnicode(text_page, char_index)
        seq.append((char_index, chr(codepoint) if codepoint > 0 else "\x00"))
    targets: list[str] = []
    for font_index, encoded_text, _tz in show_codes:
        if not encoded_text:
            continue
        text_state = targets_for(font_index, encoded_text)
        if text_state is None:
            return  # uncovered font used on this page: no patch
        targets.extend(text_state)
    res = _walk_codes(seq, targets)
    if res is None:
        return
    apply(res[0], res[1], chars_by_index)


def _synthesize_dropped_glyphs(
    sites: list[dict], raw_chars: list[dict], chars_by_index: dict[int, dict],
) -> None:
    """Re-emit glyphs PDFium's font layer never produced, even though the content stream contains them. Geometry comes from the pen model rather than a guess: PDFium still advances the pen over the missing glyph when placing surviving neighbours, so a dropped glyph starts at the previous survivor's advance-cell right edge and its advance is the gap to the next survivor's origin. With no surviving neighbour on a side, the advance is unknowable; emit zero-width there so presence and stream order are preserved without inserting a synthetic gap."""
    groups: list[list[dict]] = []
    for site in sites:
        if (groups and groups[-1][0]["prev_i"] == site["prev_i"]
                and groups[-1][0]["next_i"] == site["next_i"]
                and groups[-1][0]["owner"] is site["owner"]):
            groups[-1].append(site)
        else:
            groups.append([site])
    for group_value in groups:
        owner = group_value[0]["owner"]
        prev = chars_by_index.get(group_value[0]["prev_i"]) if group_value[0]["prev_i"] is not None else None
        nxt = chars_by_index.get(group_value[0]["next_i"]) if group_value[0]["next_i"] is not None else None
        text = "".join(site["t"] for site in group_value)  # one char per target codepoint
        count_item = len(text)
        if not count_item:
            continue
        if prev is not None:
            pen, baseline_y = prev["right"], prev["oy"]
        elif nxt is not None:
            pen, baseline_y = nxt["ox"], nxt["oy"]
        else:
            # Whole show op dropped: park at the object box's pen start.
            pen, baseline_y = owner["l"], owner["b"]
        total = 0.0
        if (prev is not None and nxt is not None
                and abs(nxt["oy"] - baseline_y) < 0.5 and nxt["ox"] > pen):
            total = nxt["ox"] - pen
        adv = total / count_item
        # Textpage index: fractional, slotted against the owner's own chars
        # so the paint-order sort keys (page_order, i) place the run in
        # stream position; only order WITHIN the owner object matters.
        if prev is not None and prev["obj"] is owner:
            base, sgn = prev["i"], 1.0
        elif nxt is not None and nxt["obj"] is owner:
            base, sgn = nxt["i"], -1.0
        elif prev is not None:
            base, sgn = prev["i"], 1.0
        elif nxt is not None:
            base, sgn = nxt["i"], -1.0
        else:
            base, sgn = -1.0, 1.0
        for key_value, char in enumerate(text):
            is_ws, is_mn, is_cf = _char_category(char)
            glyph_left = pen + adv * key_value
            step = (key_value + 1) if sgn > 0 else (count_item - key_value)
            raw_chars.append({
                "i": base + sgn * step * 1e-3,
                "ch": char, "u": ord(char),
                "is_gen": False, "synth": True,
                "is_ws": is_ws, "is_mn": is_mn, "is_cf": is_cf,
                "ox": glyph_left, "oy": baseline_y,
                "left": glyph_left, "right": glyph_left + adv,
                "top": baseline_y + owner["fs_eff"], "bottom": baseline_y,
                # Degenerate ink box: PDFium reports no ink box for the glyph
                # (this also keeps it out of the Type-3 extent union).
                "box_top": baseline_y, "box_bottom": baseline_y,
                "cell_top": baseline_y, "cell_bot": baseline_y,
                "w_raw": 0.0, "w_synth": adv,
                "obj": owner, "font_name": owner["font_name"],
            })
