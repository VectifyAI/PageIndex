"""Raw PDF object access (PyPDF2-backed) and PDF lexical primitives."""

from __future__ import annotations

from PyPDF2.generic import (
    IndirectObject as PdfIndirectRef, NameObject as PdfName, NumberObject as PdfNumber,
    FloatObject as PdfFloat, BooleanObject as PdfBoolean,
    DictionaryObject as PdfDictionary, ArrayObject as PdfArray,
)


def _pdf_tok(value) -> str:
    """Serialise one PDF value back to content-syntax (for xref_object's regex)."""
    if isinstance(value, PdfIndirectRef):
        return f"{value.idnum} {value.generation} R"
    if isinstance(value, PdfName):
        return str(value)
    if isinstance(value, PdfBoolean):
        return "true" if value.value else "false"
    if isinstance(value, PdfDictionary):
        return _pdf_obj_str(value)
    if isinstance(value, PdfArray):
        return "[ " + " ".join(_pdf_tok(array_item) for array_item in value) + " ]"
    return str(value)


def _pdf_obj_str(obj) -> str:
    """Serialize an object body as a PDF-syntax string."""
    if isinstance(obj, PdfIndirectRef):
        obj = obj.get_object()
    if isinstance(obj, PdfDictionary):
        parts = ["<<"]
        for key_value, val in obj.items():
            parts.append(str(key_value))
            parts.append(_pdf_tok(val))
        parts.append(">>")
        return " ".join(parts)
    if isinstance(obj, PdfArray):
        return "[ " + " ".join(_pdf_tok(array_item) for array_item in obj) + " ]"
    return _pdf_tok(obj)


def _pdf_typed(value):
    """Return ``(type, value-string)`` for a raw, unresolved PDF value."""
    if value is None:
        return ("null", "null")
    if isinstance(value, PdfIndirectRef):
        return ("xref", f"{value.idnum} {value.generation} R")
    if isinstance(value, PdfName):
        return ("name", str(value))
    if isinstance(value, PdfBoolean):
        return ("bool", "true" if value.value else "false")
    if isinstance(value, PdfFloat):
        return ("real", str(value))
    if isinstance(value, PdfNumber):
        return ("int", str(int(value)))
    if isinstance(value, PdfDictionary):
        return ("dict", _pdf_obj_str(value))
    if isinstance(value, PdfArray):
        return ("array", _pdf_obj_str(value))
    try:
        return ("string", str(value))
    except Exception:
        return ("null", "null")


class _PdfPage:
    __slots__ = ("_page_object",)

    def __init__(self, page):
        self._page_object = page

    def read_contents(self) -> bytes:
        candidate_item = self._page_object.get_contents()
        if candidate_item is None:
            return b""
        if isinstance(candidate_item, PdfIndirectRef):
            candidate_item = candidate_item.get_object()
        if hasattr(candidate_item, "get_data"):
            return candidate_item.get_data()
        # /Contents is an array of streams; concatenate them with a single
        # space (intentional); join the raw decompressed data the same.

        return b" ".join(text.get_object().get_data() for text in candidate_item)

    def get_fonts(self, full: bool = True):
        out: list = []
        res = self._page_object.get("/Resources")
        if res is None:
            return out
        fonts = res.get_object().get("/Font")
        if fonts is None:
            return out
        for _xref_key, ref in fonts.get_object().items():
            idnum = ref.idnum if isinstance(ref, PdfIndirectRef) else 0
            filter_context = ref.get_object()
            subtype = str(filter_context.get("/Subtype", "")).lstrip("/")
            basefont = str(filter_context.get("/BaseFont", "")).lstrip("/")
            enc_raw = filter_context.raw_get("/Encoding") if "/Encoding" in filter_context else None
            enc = str(enc_raw).lstrip("/") if isinstance(enc_raw, PdfName) else ""
            out.append((idnum, "", subtype, basefont, str(_xref_key).lstrip("/"), enc))
        return out


class _PdfDoc:
    """PyPDF2-backed adapter for raw object and stream access PDFium cannot expose."""

    __slots__ = ("_reader", "_virtual")

    def __init__(self, reader):
        self._reader = reader
        # Negative pseudo-xrefs for DIRECT (inline) dicts that have no object
        # number -- text extraction reference resolution treats direct and indirect values alike,
        # so inline font dicts must be addressable by the same integer-keyed
        # pipeline (_redefinition_dict_xrefs registers them).
        self._virtual: dict[int, object] = {}

    def register_virtual(self, obj) -> int:
        vid = -(len(self._virtual) + 1)
        self._virtual[vid] = obj
        return vid

    @property
    def page_count(self) -> int:
        return len(self._reader.pages)

    def __getitem__(self, idx):
        return _PdfPage(self._reader.pages[idx])

    def page_xref(self, idx: int) -> int:
        return self._reader.pages[idx].indirect_reference.idnum

    def _resolve_object(self, xref: int):
        if xref < 0:
            return self._virtual.get(xref)
        return PdfIndirectRef(xref, 0, self._reader).get_object()

    def xref_get_key(self, xref: int, _xref_key: str):
        cur = self._resolve_object(xref)
        parts = _xref_key.split("/")
        for index_value, part in enumerate(parts):
            if cur is None:
                return ("null", "null")
            if isinstance(cur, PdfIndirectRef):
                cur = cur.get_object()
            if not hasattr(cur, "raw_get"):
                return ("null", "null")
            name = "/" + part
            if name not in cur:
                return ("null", "null")
            if index_value == len(parts) - 1:
                return _pdf_typed(cur.raw_get(name))
            cur = cur[name]
        return _pdf_typed(cur)

    def xref_stream(self, xref: int) -> bytes:
        return self._resolve_object(xref).get_data()

    def xref_object(self, xref: int, compressed: bool = True) -> str:
        return _pdf_obj_str(self._resolve_object(xref))

    def close(self) -> None:
        try:
            self._reader.stream.close()
        except Exception:
            pass
_PDF_WHITESPACE_BYTES = frozenset({0x20, 0x09, 0x0d, 0x0a, 0x0c, 0x00})
_PDF_DELIMITER_BYTES = frozenset(b"()<>[]{}/%")


_PDF_STRING_ESCAPE_BYTES = {0x6E: 0x0A, 0x72: 0x0D, 0x74: 0x09, 0x62: 0x08, 0x66: 0x0C,
            0x28: 0x28, 0x29: 0x29, 0x5C: 0x5C}


def _decode_pdf_name(raw: bytes) -> bytes:
    """Decode #XX escapes in a PDF name token to its canonical bytes."""
    if b"#" not in raw:
        return raw
    out = bytearray()
    index_value = 0
    while index_value < len(raw):
        if raw[index_value] == 0x23 and index_value + 2 < len(raw):
            try:
                out.append(int(raw[index_value + 1:index_value + 3], 16))
                index_value += 3
                continue
            except ValueError:
                pass
        out.append(raw[index_value])
        index_value += 1
    return bytes(out)
