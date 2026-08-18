"""Pins the pdfium 5.x text-extraction semantics the parser is calibrated to.

pdfium split FPDFFont_GetFontName into GetBaseFontName (/BaseFont, per-face)
and GetFamilyName (old family semantics); the parser uses the per-face names,
which changes word joining and figure-label pickup. These sentinels come from
a 4.30-vs-5.13 corpus A/B and fail if the semantics move again.
"""
from importlib.metadata import version
from pathlib import Path

import pytest

PDF = Path(__file__).parent.parent / "examples" / "documents" / "earthmover.pdf"


@pytest.mark.skipif(int(version("pypdfium2").split(".")[0]) < 5,
                    reason="extraction is pinned to pdfium 5.x font-name semantics")
def test_page_text_pins_pdfium5_semantics():
    from pageindex.flash.main import extract_toc

    page7 = extract_toc(str(PDF))["page_texts"][6]
    assert "p5\nEMD\n1.0" in page7          # figure axis label pdfium 4.x dropped
    assert "break loop\n5: if lbp" in page7  # pseudocode lines no longer glued


def test_optimize_full_fails_fast_without_a_key(tmp_path, monkeypatch):
    """optimize='full' runs LLM expand: with no key configured it must be
    an instant, guided PageIndexAPIError — raised before any PDF work (a
    bogus path proves the ordering) — while the LLM-free spellings and a
    backend-carrying indexing scope stay untouched."""
    from conftest import build_pdf
    from pageindex import PageIndexAPIError
    from pageindex.flash import page_index_flash
    from pageindex.utils import _llm_backend
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CHATGPT_API_KEY", raising=False)

    with pytest.raises(PageIndexAPIError, match="optimize='merge'"):
        page_index_flash(str(tmp_path / "missing.pdf"), summary=False)

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(build_pdf(["1 Introduction", "Body text"]))
    result = page_index_flash(str(pdf), summary=False, optimize="merge")
    assert "structure" in result
    result = page_index_flash(str(pdf), summary=False, optimize=False)
    assert "structure" in result

    token = _llm_backend.set({"api_key": "k"})
    try:
        with pytest.raises(FileNotFoundError):
            page_index_flash(str(tmp_path / "missing.pdf"), summary=False)
    finally:
        _llm_backend.reset(token)
