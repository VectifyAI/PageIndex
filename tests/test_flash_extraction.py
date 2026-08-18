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
