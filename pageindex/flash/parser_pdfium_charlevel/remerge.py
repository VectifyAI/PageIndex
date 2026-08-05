"""Re-merges rotated, oblique, and vertical spans after the first join pass."""

from __future__ import annotations

import math

from .text_normalize import (
    TRACKING_SPACE_FACTOR,
    NEGATIVE_SPACE_FACTOR,
    SPACE_IN_FLOW_MIN_FACTOR,
    SPACE_IN_FLOW_MAX_FACTOR,
)


def _start_rot_span(chunk: dict) -> dict:
    """A fresh single-glyph rotated span = a deep-enough copy of the merger chunk (keeps fs/font/obj/sign so the downstream span conversion is unchanged)."""
    span = dict(chunk)
    span["str"] = list(chunk["str"])
    span["font_tally"] = dict(chunk.get("font_tally", {}))
    span["weight_tally"] = dict(chunk.get("weight_tally", {}))
    return span


def _grow_rot_span(cur: dict, chunk: dict) -> None:
    """Extend a rotated span with the next glyph: append text, union the page box (left/right/top/bottom stay in page coords -> output box is exact), merge the per-char style tallies."""
    cur["str"].extend(chunk["str"])
    cur["left"] = min(cur["left"], chunk["left"])
    cur["right"] = max(cur["right"], chunk["right"])
    cur["top"] = max(cur["top"], chunk["top"])
    cur["bottom"] = min(cur["bottom"], chunk["bottom"])
    for span, count in chunk.get("font_tally", {}).items():
        cur["font_tally"][span] = cur["font_tally"].get(span, 0) + count
    for span, count in chunk.get("weight_tally", {}).items():
        cur["weight_tally"][span] = cur["weight_tally"].get(span, 0) + count


def _merge_rotated_one(group: list[dict], rot: int) -> list[dict]:
    """1-D position comparison along the rotation axis for one cardinally rotated text object. ``read_origin`` is the glyph origin in reading order (90 reads up +y, 270 down -y, 180 left -x); the pen advances by glyph_w, so the inter-glyph gap is ``next_origin - (cur_origin + glyph_w)``. In-flow gaps join, larger gaps start a new item, and the box remains the page-space AABB required by downstream layout. Cardinal rotation intentionally does less than the oblique path: its box convention cannot match the oblique item-box convention, and the extra out-of-flow/cross-axis branches are not useful for these short rotated labels."""
    def read_origin(chunk: dict) -> float:
        if rot == 90:
            return chunk["bottom"]
        if rot == 270:
            return -chunk["top"]
        if rot == 180:
            return -chunk["right"]
        return chunk["left"]

    ordered = sorted(group, key=read_origin)
    spans: list[dict] = []
    cur: dict | None = None
    pen = 0.0
    for chunk in ordered:
        font_size = chunk.get("fs", 0.0) or 0.0
        glyph_width = chunk.get("glyph_w", 0.0) or 0.0
        origin = read_origin(chunk)
        if cur is None:
            cur = _start_rot_span(chunk)
            pen = origin + glyph_width
            continue
        gap = origin - pen
        if gap <= font_size * SPACE_IN_FLOW_MAX_FACTOR:
            if gap > font_size * TRACKING_SPACE_FACTOR:
                cur["str"].append(" ")
            _grow_rot_span(cur, chunk)
        else:
            spans.append(cur)
            cur = _start_rot_span(chunk)
        pen = origin + glyph_width
    if cur is not None:
        spans.append(cur)
    return spans


def _remerge_rotated(items: list[dict]) -> list[dict]:
    """Re-merge the per-glyph chunks of each rotated text object into text items along the rotation axis. Upright text is untouched; merged spans keep the first chunk position for reading order."""
    rot_groups: dict[int, list[dict]] = {}
    for item in items:
        obj = item.get("obj")
        if isinstance(obj, dict) and obj.get("rot") in (90, 180, 270):
            rot_groups.setdefault(id(obj), []).append(item)
    if not rot_groups:
        return items

    merged_for = {
        oid: _merge_rotated_one(group, group[0]["obj"]["rot"])
        for oid, group in rot_groups.items()
    }
    out: list[dict] = []
    emitted: set[int] = set()
    for item in items:
        obj = item.get("obj")
        if isinstance(obj, dict) and obj.get("rot") in (90, 180, 270):
            oid = id(obj)
            if oid not in emitted:
                emitted.add(oid)
                out.extend(merged_for[oid])
        else:
            out.append(item)
    return out


def _new_oblique_span(glyph: dict, baseline_pos: float, cross_pos: float, glyph_width: float) -> dict:
    """Open an oblique item at its first reading-order glyph. Records the glyph's page-space pen origin, along-baseline start, cross-axis position, and running pen so the gap logic can compare the next glyph."""
    return {
        "str": [glyph["ch"]],
        "obj": glyph["obj"],
        "fs": glyph["fs"],
        "font_name": glyph["font_name"],
        "_ox0": glyph["ox"], "_oy0": glyph["oy"],
        "_u0": baseline_pos, "_uend": baseline_pos + glyph_width, "_pen": baseline_pos + glyph_width, "_vlast": cross_pos,
        "_lox": glyph["ox"], "_loy": glyph["oy"], "_lgw": glyph_width,
    }


def _close_oblique(cur: dict) -> dict:
    """Finalize an oblique item's box. The item merger is rotation-agnostic -- it turns ANY text extraction item into a span via left=transform[4], right=+width, bottom=transform[5], top=+height -- so an oblique item's box is upright at its pen origin, with width = the along-baseline advance (text extraction item.width, NOT the diagonal x-extent the horizontal merger would compute) and height = font size."""
    width = cur["_uend"] - cur["_u0"]
    cur["left"] = cur["_ox0"]
    cur["right"] = cur["_ox0"] + width
    cur["bottom"] = cur["_oy0"]
    cur["top"] = cur["_oy0"] + cur["fs"]
    return cur


def _oblique_space(cur: dict, adv: float, baseline_unit_x: float, baseline_unit_y: float, scale: float) -> dict:
    """span merger ``synthetic-space insertion`` out-of-flow item: a STANDALONE " " at the previous glyph's pen (previous glyph transform), width=|advance-x|, height 0 (horizontal). The pen sits at the last glyph's origin advanced by its width along the baseline unit direction ``(ux,uy)``. span merger ``advance-x`` is ``(posX-lastPosX)/text advance scale``, so the width is normalised by the matrix scale (== text advance scale here); on identity CTM scale==1 so this is a no-op, but under a scaled CTM it matters. Output box = left=pen_x, right=+width, bottom=top=pen_y."""
    pen_x = cur["_lox"] + cur["_lgw"] * baseline_unit_x
    pen_y = cur["_loy"] + cur["_lgw"] * baseline_unit_y
    width_value = abs(adv) / scale
    return {
        "str": [" "], "obj": cur["obj"], "fs": cur["fs"], "font_name": cur["font_name"],
        "left": pen_x, "right": pen_x + width_value, "bottom": pen_y, "top": pen_y,
    }


def _merge_oblique_one(chs: list[dict]) -> list[dict]:
    """text extraction position comparison (inverse-rotation projection path) for ONE oblique text object's glyphs -- the explicit horizontal-branch implementation. ``inverse-rotation projection(x,y,m) = [(m0*x+m1*y)/s, (m2*x+m3*y)/s]`` (s=hypot(m0,m1)); component 0 is the reading-order (baseline) coordinate, component 1 the cross axis. Projecting each glyph's pen origin onto these gives advance-x (along, the gap beyond the prev glyph's advance) and advance-y (cross). Then apply the item split thresholds: advance-x<backward-jump threshold (back-jump) or |advance-y|>height -> split; advance-x<=tracking-space threshold -> join no space; <=in-flow space threshold -> in-flow space in str; else synthetic-space insertion -> a STANDALONE " " item then split. Items carry the item-box convention box (see _close_oblique)."""
    matrix_a, matrix_b, matrix_c, matrix_d = chs[0]["obj"]["mtx"]
    scale = math.hypot(matrix_a, matrix_b) or 1.0
    baseline_unit_x, baseline_unit_y = matrix_a / scale, matrix_b / scale          # baseline unit direction (page space)

    def along(glyph: dict) -> float:
        return (matrix_a * glyph["ox"] + matrix_b * glyph["oy"]) / scale

    def cross(glyph: dict) -> float:
        return (matrix_c * glyph["ox"] + matrix_d * glyph["oy"]) / scale

    ordered = sorted(chs, key=along)
    spans: list[dict] = []
    cur: dict | None = None
    for glyph in ordered:
        if glyph.get("is_ws"):
            # Skip whitespace glyphs entirely (== main span merger skips whitespace,
            # no pen update): text extraction never pushes a raw space glyph to str; the gap
            # they leave is re-synthesised by the in-flow/out-of-flow logic below
            # for the next visible glyph. This collapses runs of spaces to one and
            # trims trailing/leading spaces using the last-character buffer.
            continue
        font_size = glyph.get("fs", 0.0) or 0.0
        glyph_width = glyph.get("glyph_w", 0.0) or 0.0
        baseline_pos = along(glyph)
        cross_pos = cross(glyph)
        if cur is None:
            cur = _new_oblique_span(glyph, baseline_pos, cross_pos, glyph_width)
            continue
        baseline_gap = baseline_pos - cur["_pen"]             # along-baseline gap beyond prev advance
        cross_shift = cross_pos - cur["_vlast"]           # cross-axis shift
        if baseline_gap < font_size * NEGATIVE_SPACE_FACTOR or abs(cross_shift) > font_size:
            # back-jump (backward-jump threshold) or cross-axis line break: span merger
            # flush/line-break emission -- either way the item merger just starts a new item.
            spans.append(_close_oblique(cur))
            cur = _new_oblique_span(glyph, baseline_pos, cross_pos, glyph_width)
            continue
        if baseline_gap <= font_size * TRACKING_SPACE_FACTOR:
            cur["str"].append(glyph["ch"])                        # join, no space
        elif baseline_gap <= font_size * SPACE_IN_FLOW_MAX_FACTOR:
            cur["str"].append(" ")                             # in-flow space
            cur["str"].append(glyph["ch"])
        else:
            spans.append(_close_oblique(cur))                  # out-of-flow:
            spans.append(_oblique_space(cur, baseline_gap, baseline_unit_x, baseline_unit_y, scale))  # standalone " "
            cur = _new_oblique_span(glyph, baseline_pos, cross_pos, glyph_width)
            continue
        cur["_uend"] = baseline_pos + glyph_width
        cur["_pen"] = baseline_pos + glyph_width
        cur["_vlast"] = cross_pos
        cur["_lox"], cur["_loy"], cur["_lgw"] = glyph["ox"], glyph["oy"], glyph_width
    if cur is not None:
        spans.append(_close_oblique(cur))
    return spans


def _remerge_oblique(items: list[dict], fin_chars: list[dict]) -> list[dict]:
    """Rebuild oblique text objects by re-merging per-glyph chunks along the baseline and emitting item-box-convention boxes. Upright and cardinal text are untouched."""
    groups: dict[int, list[dict]] = {}
    for glyph in fin_chars:
        obj = glyph.get("obj")
        if isinstance(obj, dict) and obj.get("rot") == -1:
            groups.setdefault(id(obj), []).append(glyph)
    if not groups:
        return items

    merged_for = {oid: _merge_oblique_one(chs) for oid, chs in groups.items()}
    out: list[dict] = []
    emitted: set[int] = set()
    for item in items:
        obj = item.get("obj")
        if isinstance(obj, dict) and obj.get("rot") == -1:
            oid = id(obj)
            if oid not in emitted:
                emitted.add(oid)
                out.extend(merged_for[oid])
        else:
            out.append(item)
    return out


def _start_vert_span(chunk: dict) -> dict:
    """Create a vertical item from its first chunk. Vertical items use the rendered font size as width, accumulate height per glyph, and keep the first glyph's pen as the item transform. The item-to-span conversion reads the style's vertical flag and flips the sign of the height offset, so a vertical item's box runs DOWN from the pen where a horizontal one runs up. The span box reproduces that convention rather than the ink AABB."""
    span = dict(chunk)
    span["str"] = list(chunk["str"])
    span["font_tally"] = dict(chunk.get("font_tally", {}))
    span["weight_tally"] = dict(chunk.get("weight_tally", {}))
    span["v_height"] = chunk["v_pen_y"] - chunk["v_after"]   # first glyph's advance
    return span


def _close_vert_span(mapping: dict) -> dict:
    """Finalize the item merger-convention box of a vertical item."""
    mapping["left"] = mapping["v_pen_x"]
    mapping["right"] = mapping["v_pen_x"] + mapping["fs"]
    mapping["top"] = mapping["v_pen_y"]
    mapping["bottom"] = mapping["v_pen_y"] - abs(mapping["v_height"])
    return mapping


def _merge_vertical_one(group: list[dict]) -> list[dict]:
    """Apply vertical-writing position comparison over one text object's per-glyph chunks in stream order. The previous pen-after-advance and current pen define the along-axis gap; x shift is the cross-axis break signal. Small gaps join, in-flow gaps insert a space, out-of-flow gaps emit a standalone zero-width space item, and backward or cross-axis jumps start a new item. Whitespace glyphs are consumed by the span merger, so their advance arrives here as an in-flow gap."""
    spans: list[dict] = []
    cur: dict | None = None
    after = 0.0   # text extraction previous glyph transform[5]: pen y after the previous glyph
    last_x = 0.0  # text extraction previous glyph transform[4]
    for chunk in group:
        font_size = chunk.get("fs", 0.0) or 0.0
        if cur is None:
            cur = _start_vert_span(chunk)
            after, last_x = chunk["v_after"], chunk["v_pen_x"]
            continue
        vertical_gap = after - chunk["v_pen_y"]
        x_shift = chunk["v_pen_x"] - last_x
        direction_sign = 1.0 if cur["v_height"] >= 0 else -1.0
        width = cur["fs"]
        if vertical_gap < direction_sign * NEGATIVE_SPACE_FACTOR * font_size or abs(x_shift) > width:
            # backward jump or cross-axis break: text extraction line-break emission/flush -- both
            # end the item (we don't model line-break marker, and the item merger ignores it).
            spans.append(_close_vert_span(cur))
            cur = _start_vert_span(chunk)
        elif vertical_gap <= direction_sign * TRACKING_SPACE_FACTOR * font_size:
            cur["v_height"] += vertical_gap + (chunk["v_pen_y"] - chunk["v_after"])
            _grow_vert_span(cur, chunk)
        elif direction_sign * SPACE_IN_FLOW_MIN_FACTOR * font_size <= vertical_gap <= direction_sign * SPACE_IN_FLOW_MAX_FACTOR * font_size:
            cur["str"].append(" ")
            cur["v_height"] += vertical_gap + (chunk["v_pen_y"] - chunk["v_after"])
            _grow_vert_span(cur, chunk)
        else:
            # out-of-flow: standalone " " at previous glyph transform, width 0, height |e|
            # (vertical synthetic spaces store the gap as height and leave width at zero).
            meta = cur
            spans.append(_close_vert_span(cur))
            spans.append({
                "str": [" "], "sign": 1, "obj": meta["obj"],
                "left": last_x, "right": last_x,            # WIDTH 0
                # A vertical style flips the height offset: the box runs DOWN
                # from the previous pen, like _close_vert_span's.
                "top": after, "bottom": after - abs(vertical_gap),
                "fs": meta["fs"], "fs_min": meta["fs"],
                "font_name": meta["font_name"], "font_key": meta["font_key"],
                "weight": meta["weight"],
                "font_tally": {meta["font_name"]: 1},
                "weight_tally": {meta["weight"]: 1},
            })
            cur = _start_vert_span(chunk)
        after, last_x = chunk["v_after"], chunk["v_pen_x"]
    if cur is not None:
        spans.append(_close_vert_span(cur))
    return spans


def _grow_vert_span(cur: dict, chunk: dict) -> None:
    """Append a glyph to a vertical item: text + style tallies. The box is NOT unioned here -- it is derived from the first pen + accumulated v_height in _close_vert_span, with transform fixed at the first glyph and height accumulated."""
    cur["str"].extend(chunk["str"])
    for span, count in chunk.get("font_tally", {}).items():
        cur["font_tally"][span] = cur["font_tally"].get(span, 0) + count
    for span, count in chunk.get("weight_tally", {}).items():
        cur["weight_tally"][span] = cur["weight_tally"].get(span, 0) + count


def _remerge_vertical(items: list[dict]) -> list[dict]:
    """Re-merge the per-glyph chunks of each vertical-writing (Identity-V / WMode 1) text object into PDF content tokenizer style items. Uses the same _remerge_rotated: the horizontal merger is untouched (it shatters a vertical column because the glyphs stack along its line-break axis) and this gated post-pass rewrites only vertical-object chunks. One extra wrinkle vs the rotated pass: text extraction emits items in content-stream order, but PDFium's textpage reorders vertical chars page-wide (its own column heuristic), so the merged groups are reassigned to the vertical slot positions in object paint order."""
    groups: dict[int, list[dict]] = {}
    obj_of: dict[int, dict] = {}
    for item in items:
        obj = item.get("obj")
        if (isinstance(obj, dict) and obj.get("vertical") and not obj.get("rot")
                and "v_pen_y" in item):
            oid = id(obj)
            groups.setdefault(oid, []).append(item)
            obj_of[oid] = obj
    if not groups:
        return items

    merged_for = {oid: _merge_vertical_one(group_value) for oid, group_value in groups.items()}
    paint_order = sorted(groups, key=lambda oid: obj_of[oid]["page_order"])
    out: list[dict] = []
    slot = 0  # next paint-order group to emit at the next vertical slot
    seen: set[int] = set()
    for item in items:
        obj = item.get("obj")
        oid = id(obj) if isinstance(obj, dict) else None
        if oid in groups:
            if oid not in seen:
                seen.add(oid)
                out.extend(merged_for[paint_order[slot]])
                slot += 1
        else:
            out.append(item)
    return out
