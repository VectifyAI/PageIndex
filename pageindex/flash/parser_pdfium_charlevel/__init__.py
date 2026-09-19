"""PDFium-backed text-item reconstruction via textpage chars and bbox-mapped font handles.

The parser reconstructs content-stream text items from rendered characters while
preserving the geometry needed by downstream line clustering and heading
detection. The merge thresholds operate on glyph advance, font size, text
matrix scale, and spacing introduced by char spacing, text-position operators,
and ``TJ`` adjustments.

Per page, the reconstruction uses rendered character origins, glyph widths,
font bbox containment, effective font size, text-item merging, baseline-anchored
character boxes, and the minimum font size derived in each emitted chunk. Those
calibrations keep small caps, math glyphs, ligatures, Type 3 fonts, rotated
text, and vertical writing stable enough for layout statistics.
"""

import bisect
import ctypes
import difflib
import json
import math
import re
import unicodedata
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Union

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

# Raw PDF object access (ToUnicode CMaps, content streams, font dicts, /WMode)
# that PDFium does not expose, read via PyPDF2 -- already a project dependency and
# permissively licensed. A thin adapter exposes the small raw-object API the
# helpers below need, so their calibrated logic stays unchanged.
import PyPDF2 as _pypdf2  # declared dependency (also imported by pageindex.utils/client)
from PyPDF2.generic import (
    IndirectObject as PdfIndirectRef, NameObject as PdfName, NumberObject as PdfNumber,
    FloatObject as PdfFloat, BooleanObject as PdfBoolean,
    DictionaryObject as PdfDictionary, ArrayObject as PdfArray,
)

from ..model import Span, Rect

from .pdf_objects import (
    _pdf_tok,
    _pdf_obj_str,
    _pdf_typed,
    _PdfPage,
    _PdfDoc,
    _PDF_WHITESPACE_BYTES,
    _PDF_DELIMITER_BYTES,
    _PDF_STRING_ESCAPE_BYTES,
    _decode_pdf_name,
)
from .text_normalize import (
    _DROP_CHARS,
    _NORMALIZED_UNICODES,
    _normalize_unicodes,
    TRACKING_SPACE_FACTOR,
    NON_SPACE_GAP_FACTOR,
    NEGATIVE_SPACE_FACTOR,
    SPACE_IN_FLOW_MIN_FACTOR,
    SPACE_IN_FLOW_MAX_FACTOR,
    _WHITESPACE_CODEPOINTS,
    _is_whitespace,
    _is_zero_width_diacritic,
    _is_invisible_format_mark,
    _BIDI_BASE_TYPES,
    _BIDI_ARABIC_TYPES,
    _apply_bidi_reordering,
    _rtl_sign,
    _reverse_if_rtl,
    _read_end,
    _read_gap,
)
from .content_stream import (
    _FLUSH_OPS,
    _SHOW_OPS,
    _OP_LEX_PREFIX,
    _OP_OPERAND_COUNTS,
    _tokenize_show_operators,
    _assign_vertical_tags,
    _assign_show_tz,
    _page_vertical_resource_names,
)
from .glyph_tables import (
    _GLYPHLIST_PATH,
    _cached_glyphs,
    _cached_encodings,
    _load_glyph_tables,
    _get_unicode_for_glyph,
    _from_char_code,
)
from .cmap_parse import (
    _utf16be_units_to_str,
    _NUM_DECIMAL_RE,
    _NUM_INFINITY_RE,
    _NUM_HEX_RE,
    _NUM_OCTAL_RE,
    _NUM_BINARY_RE,
    _WHITESPACE_STRIP,
    _ieee_div,
    _compute_skew,
    _to_number,
    _parse_int,
    _cmap_str_to_int,
    _parse_tounicode_cmap,
)
from .font_unicode import (
    _TYPE1_SPECIAL_BYTES,
    _TYPE1_WHITESPACE_BYTES,
    _type1_builtin_encoding,
    _simple_font_to_unicode,
    _font_unicode_map,
)
from .code_walk import (
    _resource_dict_xrefs,
    _page_show_codes,
    _char_category,
    _walk_codes,
)
from .unicode_apply import (
    _apply_font_unicode,
    _synthesize_dropped_glyphs,
)
from .geometry import (
    _obj_rotation,
    _xf_point,
    _compose_mtx,
    _IDENT_MTX,
    _collect_text_objs,
    _build_obj_index,
    _char_render_fs,
    _find_obj_for_char,
)
from .char_extract import (
    _extract_raw_chars,
    _accumulate_type3_extents,
    _type3_size_by_font,
    _apply_type3_sizes,
    _finalize_chars,
    _inherited_box,
    _page_view_rect,
    _off_page,
)
from .merge import _merge_text_items
from .remerge import (
    _start_rot_span,
    _grow_rot_span,
    _merge_rotated_one,
    _remerge_rotated,
    _new_oblique_span,
    _close_oblique,
    _oblique_space,
    _merge_oblique_one,
    _remerge_oblique,
    _start_vert_span,
    _close_vert_span,
    _merge_vertical_one,
    _grow_vert_span,
    _remerge_vertical,
)
from .pipeline import (
    _page_pass1,
    _page_pass2,
    _page_spans,
    parse_charlevel_meta,
    parse_charlevel,
)

__all__ = ["parse_charlevel", "parse_charlevel_meta"]
