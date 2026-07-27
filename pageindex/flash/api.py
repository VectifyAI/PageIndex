"""Public API for PageIndex Flash. The only supported entry point is :func:`page_index_flash`. Everything else in this package is internal pipeline machinery."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pypdfium2 as pdfium

from .main import extract_toc


def _is_pdfium_password_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "password" in msg or "security" in msg or "encrypted" in msg


def _validate_path(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if not path.is_file():
        raise ValueError(f"PDF path is not a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"PDF file must have a .pdf extension: {path}")
    with path.open("rb") as score_value:
        if score_value.read(5) != b"%PDF-":
            raise ValueError(f"File does not look like a PDF: {path}")
    return str(path)


def _validate_stream(stream: BinaryIO) -> BinaryIO:
    try:
        pos = stream.tell()
        head = stream.read(5)
        stream.seek(pos)
    except Exception as exc:  # noqa: BLE001 - normalize stream capability errors
        raise TypeError("PDF stream must be seekable and readable") from exc
    if head != b"%PDF-":
        raise ValueError("Input stream does not look like a PDF")
    return stream


def _validate_pdf(pdf):
    if isinstance(pdf, (str, Path)):
        handle = _validate_path(Path(pdf))
        restore = None
    elif isinstance(pdf, BytesIO):
        handle = _validate_stream(pdf)
        restore = pdf.tell()
    else:
        raise TypeError("page_index_flash(pdf) expects a PDF path or io.BytesIO stream")

    doc = None
    try:
        doc = pdfium.PdfDocument(handle)
        if len(doc) == 0:
            raise ValueError("PDF contains no pages")
    except pdfium.PdfiumError as exc:
        if _is_pdfium_password_error(exc):
            raise ValueError("PDF is encrypted or password-protected") from exc
        raise ValueError(f"Could not open PDF: {exc}") from exc
    finally:
        if doc is not None:
            doc.close()
        if restore is not None:
            pdf.seek(restore)
    return pdf


def page_index_flash(pdf) -> dict:
    """Build a PageIndex tree structure from a PDF using layout statistics, without an LLM. Args: pdf: path to a PDF file (``str`` or ``pathlib.Path``) or an in-memory binary stream (``io.BytesIO``). Returns: dict with keys ``doc_name``, ``doc_title`` and ``structure`` (a list of nested ``{"title", "start_index", "end_index", "nodes"}`` dicts; page indexes are 1-based). """
    return extract_toc(_validate_pdf(pdf))


__all__ = ["page_index_flash"]
