"""Bundled glyph-name and encoding tables with cached lazy loading."""

from __future__ import annotations

import json
from pathlib import Path

from .cmap_parse import _parse_int


# ---------------------------------------------------------------------------
# Font Unicode-map construction.
#
# PDFium's per-character Unicode can diverge when a simple font's ToUnicode CMap
# is missing or incomplete. The repair path resolves the charcode through the
# font's encoding (dictionary /Encoding BaseEncoding+Differences, or an embedded
# Type1 program's builtin encoding) to a glyph name, maps that name through the
# bundled glyph table, and otherwise falls back to the raw charcode. The map is
# rebuilt from the PDF's own font dictionaries via the PyPDF2 xref channel
# (font metadata only, no text decode), then applied where PDFium's output
# disagrees.
#
# Covered rules: encoding and Differences extraction, simple-font Unicode-map
# construction, predefined collection Unicode-map construction, ToUnicode
# parsing, fallback Unicode-map repair, Type 1 Unicode-map repair, and glyph
# mapping as the included ToUnicode value when present, otherwise the raw charcode.

# /Encoding extraction from an embedded Type1 file
# Glyph-name Unicode lookup
# glyph names and standard encodings are bundled in data/glyph_name_table.json
# (kept deliberately conservative
#
# Boundaries (documented, all conservative -- no map entry means no patch):
# - composite (Type0) fonts: separate path, never patched here;
# - CFF (FontFile3) builtin encodings: not parsed here; dict-encoding-based
# mapping still applies;
# - symbolic-TrueType WinAnsi inference (content stream tokenizer TrueType Unicode-map repair):
# needs the TTF name records, not implemented.
# ---------------------------------------------------------------------------

_GLYPHLIST_PATH = Path(__file__).parent.parent / "data" / "glyph_name_table.json"
_cached_glyphs: dict[str, int] | None = None
_cached_encodings: dict[str, list[str]] | None = None


def _load_glyph_tables() -> tuple[dict[str, int], dict[str, list[str]]]:
    global _cached_glyphs, _cached_encodings
    glyphs, encodings = _cached_glyphs, _cached_encodings
    if glyphs is None or encodings is None:
        data = json.loads(_GLYPHLIST_PATH.read_text(encoding="utf-8"))
        glyphs = _cached_glyphs = data["glyphs"]
        encodings = _cached_encodings = data["encodings"]
    return glyphs, encodings


def _get_unicode_for_glyph(name: str, glyphs: dict[str, int]) -> int:
    """Resolve a glyph name through glyphlist lookup and uppercase-hex recovery patterns."""
    codepoint = glyphs.get(name)
    if codepoint is not None:
        return codepoint
    if not name:
        return -1
    if name[0] == "u":
        glyph_name_length = len(name)
        if glyph_name_length == 7 and name[1] == "n" and name[2] == "i":
            hex_str = name[3:]
        elif 5 <= glyph_name_length <= 7:
            hex_str = name[1:]
        else:
            return -1
        if hex_str == hex_str.upper():
            # Tolerant base-16 parsing trims Unicode whitespace and accepts an
            # optional sign / 0X prefix. NaN fails the >= 0 gate; "-0" passes it.
            u16 = _parse_int(hex_str, 16)
            if u16 >= 0:
                return int(u16)
    return -1


def _from_char_code(number: int) -> str:
    """Return the UTF-16 code unit after ToUint16 truncation."""
    return chr(number & 0xFFFF)
