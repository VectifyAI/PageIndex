"""Whole-document parse drivers assembling per-page charlevel metadata."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Union

import pypdfium2 as pdfium

# Raw PDF object access (ToUnicode CMaps, content streams, font dicts, /WMode)
# that PDFium does not expose, read via PyPDF2 -- already a project dependency and
# permissively licensed. A thin adapter exposes the small raw-object API the
# helpers below need, so their calibrated logic stays unchanged.
import PyPDF2 as _pypdf2  # declared dependency (also imported by pageindex.utils/client)

from ..model import Span, Rect

from .pdf_objects import _PdfDoc
from .text_normalize import (
    _DROP_CHARS,
    _NORMALIZED_UNICODES,
    _apply_bidi_reordering,
    _reverse_if_rtl,
)
from .content_stream import (
    _tokenize_show_operators,
    _assign_vertical_tags,
    _assign_show_tz,
    _page_vertical_resource_names,
)
from .cmap_parse import _compute_skew
from .code_walk import _page_show_codes
from .unicode_apply import _apply_font_unicode
from .char_extract import (
    _extract_raw_chars,
    _accumulate_type3_extents,
    _type3_size_by_font,
    _apply_type3_sizes,
    _finalize_chars,
    _inherited_box,
    _page_view_rect,
)
from .merge import _merge_text_items
from .remerge import (
    _remerge_rotated,
    _remerge_oblique,
    _remerge_vertical,
)


def _page_pass1(pdf, pdf_doc, page_idx: int, type3_ext: dict, font_map_cache: dict):
    """Pass-1 body for ONE page: extract raw chars, tag objects, accumulate
    Type-3 extents into ``type3_ext``. Returns ``(page, raw_chars, page_vb,
    page_rot)``; the PAGE is returned still open — the caller owns closing it
    (the sequential driver must keep every page open until pass 2's Type-3
    size lookups are done; see keep_pages in ``parse_charlevel_meta``)."""
    page = pdf[page_idx]
    text_page = page.get_textpage()
    raw_chars, objects = _extract_raw_chars(page, text_page.raw)
    try:
        media_box_raw = _inherited_box(pdf_doc, page_idx, "MediaBox") if pdf_doc is not None else None
        crop_box_raw = _inherited_box(pdf_doc, page_idx, "CropBox") if pdf_doc is not None else None
        page_vb = _page_view_rect(page, media_box_raw, crop_box_raw)  # (x0, y0, x1, y1) page space
    except Exception:
        page_vb = None                      # no box -> off-page test disabled
    try:
        page_rot = int(page.get_rotation())  # PDFium /Rotate (0/90/180/270)
    except Exception:
        page_rot = 0
    show_fonts: list[bytes | None] = []
    show_tzs: list[float] = []
    vert_names: set[bytes] = set()
    if pdf_doc is not None and page_idx < pdf_doc.page_count:
        try:
            # show-op flush ids (q/Q flush scope) are no longer used -- the merge-id
            # grouping was removed; only show_fonts (per-op font resname)
            # feeds vertical tagging.
            show_flush_ids, show_fonts, show_text_units, horizontal_scales, xobject_paints = _tokenize_show_operators(
                pdf_doc[page_idx].read_contents())
        except Exception:
            show_fonts = []
        vert_names = _page_vertical_resource_names(pdf_doc, page_idx)
        # Tz follows the text into Form XObjects (the whole text state is
        # cloned for the recursion), so the per-show-op horizontal-scale
        # list has to come from the SAME form-descending walk as the codes:
        # the page's own stream alone under-counts every form page and the
        # ordinal gate below would then drop the tag for the whole page.
        try:
            show_codes = _page_show_codes(pdf_doc, page_idx)
            if show_codes:
                show_tzs = [horizontal_scale for _fx, _s, horizontal_scale in show_codes]
                # Patch per-char unicode to span merger glyph Unicode where
                # PDFium's decode differs (guarded: any failure keeps
                # PDFium's output).
                if raw_chars:
                    _apply_font_unicode(
                        text_page.raw, raw_chars, objects, show_codes, pdf_doc,
                        font_map_cache)
        except Exception:
            pass
    _assign_vertical_tags(objects, show_fonts, vert_names)
    _assign_show_tz(objects, show_tzs)
    _accumulate_type3_extents(raw_chars, type3_ext)
    text_page.close()
    return page, raw_chars, page_vb, page_rot


def _page_pass2(raw_chars: list[dict], page_vb, size_by_font: dict) -> list[dict]:
    """Pass-2 body for ONE page: apply the document-wide Type-3 sizes,
    restore paint order, finalize glyph widths, run the text merger."""
    _apply_type3_sizes(raw_chars, size_by_font)
    # text extraction emits glyphs in CONTENT-STREAM (paint) order; PDFium's textpage
    # reorders whole segments page-wide (math-heavy page margin labels 'margin label' /
    # 'Section N' arrive at a different point of the char stream than their
    # show ops). obj["page_order"] is the object's stream position (objects
    # parse sequentially, incl. the Form XObject walk), so sorting real
    # glyphs by it restores span merger processing order for the merger.
    # GENERATED chars (PDFium's synthetic layout whitespace -- no span merger
    # counterpart, pure merger bookkeeping) keep no position of their own:
    # their geometric obj lookup can land on the WRONG object (the
    # multi-column "4 | Super | vision" heading puts the '4'->'S' gap
    # space inside the 'vision' object, which would re-emit it mid-word as
    # "Super vision"), so each one stays glued behind the real glyph that
    # precedes it in textpage order. Character-level ordering's
    # own items on the reordered pages.
    keys: list[tuple] = [()] * len(raw_chars)
    last_key = None
    lead_gens: list[int] = []
    for key_value, candidate_item in enumerate(raw_chars):
        if candidate_item["is_gen"]:
            if last_key is None:
                lead_gens.append(key_value)
            else:
                keys[key_value] = (last_key[0], last_key[1], 1, key_value)
        else:
            last_key = (candidate_item["obj"]["page_order"], candidate_item["i"])
            keys[key_value] = (last_key[0], last_key[1], 0, key_value)
    for key_value in lead_gens:
        keys[key_value] = (-1, -1, 1, key_value)
    raw_chars[:] = [raw_chars[key_value] for key_value in sorted(range(len(raw_chars)),
                                                 key=keys.__getitem__)]
    fin = _finalize_chars(raw_chars)
    merged = _merge_text_items(fin, page_vb)
    merged = _remerge_rotated(merged)   # collapse cardinal-rotated per-glyph shards
    merged = _remerge_vertical(merged)  # collapse vertical-writing per-glyph shards
    return _remerge_oblique(merged, fin)  # oblique objects: inverse-rotation projection re-merge


def _page_spans(raw: list[dict]) -> list[Span]:
    """Final emission for ONE page: merged chunks -> ``Span`` objects."""
    spans: list[Span] = []
    for item in raw:
        # the heading heuristics pushes normalized glyph Unicode = the normalized-Unicode table[u]  or  u

        # per glyph, a WHOLE-string lookup. Each r["str"] piece is one glyph's
        # unicode (or a synthesized space), so look up per piece -- a
        # multi-codepoint ToUnicode value is left intact when the whole-string
        # lookup misses, instead of decomposing a table-key char inside it.
        # span merger: normalized glyph Unicode = RTL ligature reversal(the normalized-Unicode table
        # [u]  or  u) -- the table lookup is then wrapped in RTL ligature reversal, which

        # reverses a multi-char Arabic/Hebrew ligature value (span merger
        #). Apply per piece (each r["str"] piece is one glyph's unicode).
        joined = "".join(
            _reverse_if_rtl(_NORMALIZED_UNICODES.get(page_value, page_value)) for page_value in item["str"]  # type: ignore[arg-type]
        )
        # text extraction text-item flush -> bidirectional transform: the joined item
        # text runs the bidi pass ON TOP of the per-glyph RTL ligature reversal
        # above (both layers exist in span merger). Pass-through for LTR text
        # and vertical items (dir 'ttb').
        joined = _apply_bidi_reordering(joined, -1, bool(item["obj"].get("vertical")))
        text = joined.translate(_DROP_CHARS)
        if not text:
            continue
        # font_size = hypot(text matrix[2], text matrix[3])
        # taken once at the item's open glyph, i.e. the chunk's first-char
        # fs. The merger breaks a chunk on any fs change (exact compare;
        # see the font_key/fs guard above) and never lowers fs mid-chunk, so
        # chunk["fs"] (set in open_chunk from the first char) is exactly
        # that value. Emit it rather than the per-chunk minimum.
        fs_emit = item["fs"]
        spans.append(
            Span(
                bbox=Rect(item["left"], item["right"], item["top"], item["bottom"]),
                text=text,
                font_name_raw=item["font_name"],
                font_size=fs_emit,
                # the heading heuristics bold is name-regex only (the font-name bold regex,
                # OR'd into the emitted span). span merger bold detector ignores the descriptor
                # ForceBold flag and numeric weight, so we must NOT inject a
                # weight-based bold here — that over-bolds Demi/Medium/bold math font
                # faces (weight 665-675) text extraction treats as regular.
                bold=False,
                italic=False,
                # Span skew score: P = (f[1]/f[0])² + (f[2]/f[3])² from the item
                # transform (IEEE: cardinal rotation -> Inf, upright -> 0).
                # The owning object's PDFium matrix has the same
                # rotation/shear structure as span merger item transform.
                # mtx0 = the FIRST glyph's object matrix (text extraction fixes the
                # item transform at open); standalone fake-space items
                # carry no mtx0 and fall back to their obj (= the previous
                # glyph's object == text extraction previous glyph transform for that space).
                skew=_compute_skew(item.get("mtx0") or item["obj"]["mtx"]),
            )
        )
    return spans


def parse_charlevel_meta(doc_handle: Union[str, Path, BytesIO]) -> tuple[list[list[Span]], list]:
    if isinstance(doc_handle, (str, Path)):
        pdf = pdfium.PdfDocument(str(doc_handle))
    elif isinstance(doc_handle, BytesIO):
        pdf = pdfium.PdfDocument(doc_handle)
    else:
        pdf = doc_handle

    # Open the same document in PyPDF2 (already a project dependency) to read the
    # page content streams: span merger item-flush operators (q/Q save/restore, marked
    # content, XObject) live there and PDFium's flattened object model cannot expose
    # them. Optional/guarded -- any failure leaves flush_id unset so the merger
    # keeps its per-object split (the fallback behavior). A separate bytes copy
    # avoids racing pypdfium2's read of the same BytesIO.
    pdf_doc = None
    if _pypdf2 is not None:
        try:
            if isinstance(doc_handle, (str, Path)):
                pdf_doc = _PdfDoc(_pypdf2.PdfReader(str(doc_handle)))
            elif isinstance(doc_handle, BytesIO):
                # Read a copy so we never race pdfium's read of the same buffer.
                pdf_doc = _PdfDoc(_pypdf2.PdfReader(BytesIO(doc_handle.getvalue())))
        except Exception:
            pdf_doc = None

    # Pass 1: extract raw chars for every page (including each glyph's raw
    # advance) and accumulate per-font identity-matrix Type-3 glyph-bbox
    # extents document-wide, so each Type-3 font is sized once over every
    # glyph it renders anywhere (coverage-independent), matching span merger
    # synthesizing font.bbox once from the CharProcs. Font handles are only
    # stable per document while their pages stay open (see keep_pages below).
    per_page: list[list[dict]] = []
    page_view_boxes: list = []   # parallel to per_page: text extraction page view box per page
    page_rotations: list = []   # parallel: PDFium page /Rotate in degrees per page
    type3_ext: dict = {}
    font_map_cache: dict = {}
    # Hold every page open until pass 2's Type-3 size lookups are done.
    # type3_ext / size_by_font key on the raw FPDF_FONT pointer VALUE, and
    # PDFium frees a font once the last page using it closes -- a later
    # page's (different) font can then be allocated at the same address,
    # silently merging two fonts' extent bins. Which addresses get reused
    # depends on the process's prior malloc state, so the output could vary
    # with whatever ran earlier in the process. Keeping the pages alive makes
    # the handle a true per-document
    # font identity (PDFium's document-level font cache returns one handle
    # per font redefinition).
    keep_pages = []
    for page_idx in range(len(pdf)):
        page, raw_chars, page_vb, page_rot = _page_pass1(
            pdf, pdf_doc, page_idx, type3_ext, font_map_cache)
        keep_pages.append(page)
        per_page.append(raw_chars)
        page_view_boxes.append(page_vb)
        page_rotations.append(page_rot)
    if pdf_doc is not None and pdf_doc is not doc_handle:
        try:
            pdf_doc.close()
        except Exception:
            pass
    size_by_font = _type3_size_by_font(type3_ext)

    # Pass 2: apply the document-wide Type-3 sizes, finalize glyph widths,
    # then run text extraction text merger.
    raw_pages: list[list[dict]] = []
    for page_view_index, raw_chars in enumerate(per_page):
        raw_pages.append(_page_pass2(raw_chars, page_view_boxes[page_view_index], size_by_font))
    for page_handle in keep_pages:
        try:
            page_handle.close()
        except Exception:
            pass
    keep_pages.clear()

    out: list[list[Span]] = []
    for raw in raw_pages:
        out.append(_page_spans(raw))
    pdf.close()
    # Per-page viewport metadata (text extraction normalized page view = cropbox clamped to the
    # mediabox, via _page_view_rect, + /Rotate) parallel to out, so heading
    # coordinates can apply span merger viewport-coordinate transform.
    return out, list(zip(page_view_boxes, page_rotations))


def parse_charlevel(doc_handle: Union[str, Path, BytesIO]) -> list[list[Span]]:
    """Per-page span entry: per-page spans only (drops viewport meta). the high-level TOC pipeline uses ``parse_charlevel_meta`` to also get the per-page (view box, /Rotate) for heading coordinates; every other caller just wants the spans. """
    return parse_charlevel_meta(doc_handle)[0]
