"""PyPDF2's lazy parse must fail inside the callers' guarded open (#426).

An AES-encrypted PDF raises DependencyError on first object access, not at
PdfReader(), so the guard around the PyPDF2 open used to miss it and the
error surfaced from _page_pass1's page_count instead — an unhandled crash on
a document PDFium reads fine on its own.
"""
import pytest

from PyPDF2.errors import DependencyError


class _LazyBombReader:
    """A reader that parses lazily and fails on first object access, the way
    an AES-encrypted document does without the crypto backend installed."""

    @property
    def pages(self):
        raise DependencyError("PyCryptodome is required for AES algorithm")


def test_pdf_doc_open_fails_at_construction():
    """_PdfDoc resolves the page tree eagerly, so an unreadable document
    raises where both callers already guard the open (the sequential pipeline
    and the parallel worker init) rather than at first page_count."""
    from pageindex.flash.parser_pdfium_charlevel.pdf_objects import _PdfDoc

    with pytest.raises(DependencyError):
        _PdfDoc(_LazyBombReader())


def test_flash_falls_back_to_pdfium_when_pypdf2_channel_is_unreadable(
        sample_pdf, monkeypatch):
    """The PyPDF2 channel only refines what PDFium already extracts, so losing
    it degrades the parse instead of failing the run."""
    from pageindex.flash import page_index_flash
    from pageindex.flash.parser_pdfium_charlevel import pipeline

    baseline = page_index_flash(sample_pdf, summary=False, optimize=False)

    opened = []

    class _BombingPyPDF2:
        @staticmethod
        def PdfReader(*args, **kwargs):
            opened.append(args)
            return _LazyBombReader()

    monkeypatch.setattr(pipeline, "_pypdf2", _BombingPyPDF2)
    degraded = page_index_flash(sample_pdf, summary=False, optimize=False)

    assert opened, "the PyPDF2 channel never opened; the fallback went untested"
    assert degraded == baseline
