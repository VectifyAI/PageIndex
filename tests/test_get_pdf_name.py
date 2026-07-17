from pathlib import Path

from pageindex.index.utils import get_pdf_name


def test_get_pdf_name_str():
    assert get_pdf_name("/some/path/document.pdf") == "document.pdf"


def test_get_pdf_name_pathlib_path():
    # Previously raised UnboundLocalError — Path is neither str nor BytesIO
    assert get_pdf_name(Path("/some/path/document.pdf")) == "document.pdf"


def test_get_pdf_name_pathlib_relative():
    assert get_pdf_name(Path("reports/annual.pdf")) == "annual.pdf"


def test_get_pdf_name_unknown_type_returns_untitled():
    # Any unrecognised type should not crash — returns a safe fallback
    assert get_pdf_name(12345) == "Untitled"
