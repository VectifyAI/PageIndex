"""Transform matrices, text-object collection, and char-to-object mapping."""

from __future__ import annotations

import ctypes
import math
import pypdfium2.raw as pdfium_c


def _obj_rotation(value: float, other_item: float, candidate_item: float, reference_item: float) -> int:
    """Classify a text-object matrix as upright, cardinal rotation, or oblique. Near-cardinal matrices snap to the cardinal bucket; genuinely oblique matrices use the baseline remerge path."""
    x_scale = math.hypot(value, other_item)
    y_scale = math.hypot(candidate_item, reference_item)
    if x_scale < 1e-9 or y_scale < 1e-9:
        return 0
    eps = 1e-3
    if abs(other_item) < eps * x_scale and abs(candidate_item) < eps * y_scale:
        return 0 if value >= 0 else 180
    if abs(value) < eps * x_scale and abs(reference_item) < eps * y_scale:
        return 90 if other_item > 0 else 270
    return -1


def _xf_point(items: tuple, other_item: float, candidate_item: float) -> tuple[float, float]:
    """Apply an (a,b,c,d,e,f) PDF matrix to a point (row-vector convention)."""
    return (items[0] * other_item + items[2] * candidate_item + items[4], items[1] * other_item + items[3] * candidate_item + items[5])


def _compose_mtx(first_matrix: tuple, second_matrix: tuple) -> tuple:
    """Matrix product applying ``m1`` first, then ``m2``."""
    return (
        first_matrix[0] * second_matrix[0] + first_matrix[1] * second_matrix[2],
        first_matrix[0] * second_matrix[1] + first_matrix[1] * second_matrix[3],
        first_matrix[2] * second_matrix[0] + first_matrix[3] * second_matrix[2],
        first_matrix[2] * second_matrix[1] + first_matrix[3] * second_matrix[3],
        first_matrix[4] * second_matrix[0] + first_matrix[5] * second_matrix[2] + second_matrix[4],
        first_matrix[4] * second_matrix[1] + first_matrix[5] * second_matrix[3] + second_matrix[5],
    )


_IDENT_MTX = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _collect_text_objs(page, text_page) -> list[dict]:
    """Per-page list of (font_handle, fs_raw, matrix_scale_*, bbox, ...) for each text object. Used for bbox-containment lookup. Walks Form XObjects manually in stream order, composing each ancestor form's matrix. Without the composition a scaled or shifted chart's text objects land at the wrong page position and every chart glyph fails the bbox-containment lookup."""
    objects: list[dict] = []
    sz_field = ctypes.c_float(0)
    matrix = pdfium_c.FS_MATRIX()
    font_name_buffer = (ctypes.c_char * 256)()
    bounds_left = ctypes.c_float(0)
    value = ctypes.c_float(0)
    bounds_right = ctypes.c_float(0)
    bounds_top = ctypes.c_float(0)

    def iter_text_objs(parent, anc_mtx, depth):
        """Yield (raw_text_obj, ancestor_matrix) in stream order."""
        object_count = (pdfium_c.FPDFFormObj_CountObjects(parent) if parent is not None
             else pdfium_c.FPDFPage_CountObjects(page.raw))
        for text in range(object_count):
            raw = (pdfium_c.FPDFFormObj_GetObject(parent, text) if parent is not None
                   else pdfium_c.FPDFPage_GetObject(page.raw, text))
            if not raw:
                continue
            typ = pdfium_c.FPDFPageObj_GetType(raw)
            if typ == pdfium_c.FPDF_PAGEOBJ_TEXT:
                yield raw, anc_mtx
            elif typ == pdfium_c.FPDF_PAGEOBJ_FORM and depth < 10:
                pdfium_c.FPDFPageObj_GetMatrix(raw, matrix)
                font_matrix = (matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f)
                yield from iter_text_objs(raw, _compose_mtx(font_matrix, anc_mtx), depth + 1)

    for raw_obj, anc_mtx in iter_text_objs(None, _IDENT_MTX, 0):
        font = pdfium_c.FPDFTextObj_GetFont(raw_obj)
        if not font:
            continue
        pdfium_c.FPDFTextObj_GetFontSize(raw_obj, ctypes.byref(sz_field))
        fs_raw = sz_field.value
        pdfium_c.FPDFPageObj_GetMatrix(raw_obj, matrix)
        # Effective (page-space) matrix: the object's own matrix composed with
        # its ancestor forms' -- text extraction folds that ancestor chain into the text matrix.
        matrix_a, matrix_b, matrix_c, matrix_d, _, _ = _compose_mtx(
            (matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f), anc_mtx)
        scale_x = math.sqrt(matrix_a * matrix_a + matrix_b * matrix_b) or 1.0
        scale_y = math.sqrt(matrix_c * matrix_c + matrix_d * matrix_d) or 1.0

        if not pdfium_c.FPDFPageObj_GetBounds(
                raw_obj, ctypes.byref(bounds_left), ctypes.byref(value),
                ctypes.byref(bounds_right), ctypes.byref(bounds_top)):
            continue
        # Bounds include the object's own matrix but not its ancestors'; map
        # the four corners into page space.
        x00, y00 = _xf_point(anc_mtx, bounds_left.value, value.value)
        x01, y01 = _xf_point(anc_mtx, bounds_left.value, bounds_top.value)
        x10, y10 = _xf_point(anc_mtx, bounds_right.value, value.value)
        x11, y11 = _xf_point(anc_mtx, bounds_right.value, bounds_top.value)
        object_left = min(x00, x01, x10, x11)
        object_right = max(x00, x01, x10, x11)
        text = min(y00, y01, y10, y11)
        object_top = max(y00, y01, y10, y11)
        ink_height = max(0.0, object_top - text)
        # text extraction folds Tfs (text font size) + FontMatrix into the text transform
        # so ``hypot(transform[2], transform[3])`` always gives the
        # rendered font size. PDFium splits these and doesn't fold non-identity
        # FontMatrix back. Rendered-font-size fallback chain:
        # raw >= 1.5 and scale > 0 -> raw * scale (normal text)
        # scale >= 1.5 -> scale (Type 3: raw=0.1, ctm=N)
        # raw >= 1.5 -> raw (no scale info)
        # else -> ink_h (Type 3 inside identity ctm)
        if anc_mtx is not _IDENT_MTX and fs_raw > 0 and scale_y > 0:
            # Inside a Form XObject, span merger font size = hypot(trm[2],trm[3])
            # with the form CTM folded in = Tfs * composed scale, exactly
            # (scaled vector-figure case: Tf 0.167 * 72 * form 0.5722 =
            # 6.88 == the heading heuristics' item height; the placeholder chain below
            # would misread it as Type-3-with-fs-in-ctm and emit 41pt boxes
            # that swallow the neighbouring "2.2" heading). The chain stays
            # for top-level objects, for top-level objects.
            fs_eff = fs_raw * scale_y
        elif fs_raw >= 1.5 and scale_y > 0:
            fs_eff = fs_raw * scale_y
        elif scale_y >= 1.5:
            fs_eff = scale_y
        elif fs_raw >= 1.5:
            fs_eff = fs_raw
        else:
            fs_eff = max(1.0, ink_height)
        # PDFium's FS_MATRIX is float32, so a size authored as 9.9pt arrives as
        # 9.89999962; text extraction parses the content stream in float64 and keeps 9.9.
        # Snap back to the shortest decimal so knife-edge font-size comparisons
        # match the content-stream value.
        fs_eff = float(f"{fs_eff:.6g}")
        name = pdfium_c.FPDFFont_GetFontName(font, font_name_buffer, 256)
        font_name = (
            bytes(font_name_buffer[:name]).decode("latin-1", errors="replace").rstrip("\x00")
            if name > 1 else ""
        )
        weight = int(pdfium_c.FPDFFont_GetWeight(font))

        objects.append({
            "font": font,
            # Handle address as a hashable per-document font identity; computed
            # once here so per-char consumers never re-cast.
            "font_key": ctypes.cast(font, ctypes.c_void_p).value,
            "fs_raw": fs_raw,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "fs_eff": fs_eff,
            "l": object_left, "r": object_right, "b": text, "t": object_top,
            "area": max(0.0, (object_right - object_left) * (object_top - text)),
            "font_name": font_name,
            "weight": weight,
            # Rotation class of this text object (0/90/180/270, or -1 oblique).
            # text extraction normalises it inside position comparison; the charlevel
            # merger is horizontal-only, so cardinal runs (rotated-sidebar sidebar stamp,
            # chart axis labels) shatter per-glyph and are re-merged by
            # _remerge_rotated; oblique objects go to _remerge_oblique (needs the
            # matrix below for the inverse-rotation projection baseline projection).
            "rot": _obj_rotation(matrix_a, matrix_b, matrix_c, matrix_d),
            "mtx": (matrix_a, matrix_b, matrix_c, matrix_d),
            # Paint (content-stream) order. text extraction emits items in stream order but
            # PDFium's textpage reorders vertical-writing chars page-wide, so
            # _remerge_vertical needs this to restore text extraction item order.
            "page_order": len(objects),
            # True iff this object's show-op used a vertical-CMap (-V / WMode 1)
            # font -- span merger vertical-font flag. Set by _assign_vertical_tags.
            "vertical": False,
            # Show-op text horizontal scale (Tz/100). text extraction keeps Tz OUT of the space
            # thresholds (base = raw font size) while PDFium folds it into the
            # object matrix (hence into fs_x); open_chunk divides it back out.
            # Set by _assign_show_tz via the same ordinal alignment as
            # ``vertical``; stays 1.0 on a count mismatch.
            "tz": 1.0,
        })
    return objects


def _build_obj_index(objects: list[dict]) -> dict[int, list[dict]]:
    """Bucket text objects by integer y so per-character lookup scans only nearby baselines. Each object is inserted into padded y-buckets that form a superset for the exact containment check."""
    index: dict[int, list[dict]] = {}
    for item_value in objects:
        lower_bound = int(math.floor(item_value["b"])) - 6
        upper_bound = int(math.ceil(item_value["t"])) + 6
        for text_key in range(lower_bound, upper_bound + 1):
            index.setdefault(text_key, []).append(item_value)
    return index


def _char_render_fs(text_page, char_idx: int) -> float:
    """True per-char rendered size: ``FPDFText_GetMatrix`` folds Tfs and FontMatrix into the rendered text matrix, so ``sqrt(c^2+d^2)`` is the text-item height. Returns 0.0 when the call is unavailable. Read lazily, only when a char is contained by more than one object, since the FFI call is expensive and most chars have a single, unambiguous host object."""
    current_matrix = pdfium_c.FS_MATRIX()
    if pdfium_c.FPDFText_GetMatrix(text_page, char_idx, ctypes.byref(current_matrix)):
        return math.sqrt(current_matrix.c * current_matrix.c + current_matrix.d * current_matrix.d)
    return 0.0


def _find_obj_for_char(
    obj_index: dict[int, list[dict]], query_origin_x: float, query_origin_y: float, tol: float = 1.0,
    char_fs: float | None = None, text_page=None, char_idx: int | None = None,
) -> dict | None:
    """Bbox containment lookup. When a char falls inside more than one text object, pick the candidate whose effective rendered size matches the char's true per-char matrix size from ``FPDFText_GetMatrix``. That folds Tfs and FontMatrix into the same glyph-to-font attribution used by the text-item reconstruction. This disambiguates overlapping objects such as a large figure-axis label drawn over a smaller heading, and avoids selecting tiny ghost objects that share the same raw textpage font size. Falls back to the PDFium ``fs_raw`` textpage font size and finally to smallest area."""
    first: dict | None = None
    cands: list[dict] | None = None
    for item_value in obj_index.get(int(round(query_origin_y)), ()):
        if (item_value["l"] - tol) <= query_origin_x <= (item_value["r"] + tol) and\
           (item_value["b"] - tol) <= query_origin_y <= (item_value["t"] + tol):
            if first is None:
                first = item_value
            elif cands is None:
                cands = [first, item_value]
            else:
                cands.append(item_value)
    if first is None:
        return None
    if cands is None:
        return first
    char_render = (
        _char_render_fs(text_page, char_idx) if text_page is not None and char_idx is not None
        else 0.0
    )
    if char_render > 0:
        # Match the per-char rendered size (== text extraction font size); area tiebreak.
        return min(
            cands,
            key=lambda item_value: (abs(item_value["fs_eff"] - char_render), item_value["area"]),
        )
    if char_fs is None and text_page is not None and char_idx is not None:
        # Deferred FPDFText_GetFontSize: only this rare branch (multi-candidate
        # AND no per-char matrix) consumes it, so the caller no longer pays the
        # FFI call on every char.
        char_fs = pdfium_c.FPDFText_GetFontSize(text_page, char_idx)
    if char_fs is not None and char_fs > 0:
        # Sort by absolute fs diff first, then smallest area as tiebreak.
        return min(
            cands,
            key=lambda item_value: (abs(item_value["fs_raw"] - char_fs) / max(char_fs, 0.01), item_value["area"]),
        )
    return min(cands, key=lambda item_value: item_value["area"])
