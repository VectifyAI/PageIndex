"""Flash failures explain which extraction stage failed, without model calls."""
import json
import runpy
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import build_pdf


@pytest.fixture
def plain_pdf(tmp_path):
    path = tmp_path / "plain.pdf"
    path.write_bytes(build_pdf(["Alpha body", "", "Gamma body"]))
    return path


def test_page_fallback_keeps_diagnostics_before_early_return(plain_pdf, monkeypatch):
    from pageindex.flash import api

    monkeypatch.setattr(api, "FLAT_TREE_MAX_NODES", 2)
    result = api.page_index_flash(plain_pdf, summary=True, optimize="full")
    assert result["toc_source"] == "pages"
    assert len(result["structure"]) == 3
    assert "page_texts" not in result
    assert "optimize" not in result
    assert result["diagnostics"] == {
        "page_count": 3, "text_page_count": 2, "textless_pages": [2],
        "section_opener_count": 0, "assembled_heading_count": 0,
        "layout_status": "no_headings",
    }
    # No PDF text leaks into the shareable diagnosis.
    assert "Alpha" not in json.dumps(result["diagnostics"])
    reason = api.flash_rejection_reason(result)
    assert "2/3 pages contain extracted text" in reason
    assert "no headings survived outline assembly" in reason
    assert "mode='standard'" in reason


def test_unreadable_pdf_reports_zero_extracted_pages(tmp_path):
    from pageindex.flash import api

    path = tmp_path / "blank.pdf"
    path.write_bytes(build_pdf(["", ""]))
    result = api.page_index_flash(path, summary=False, optimize=False)
    assert result["toc_source"] == "unreadable"
    assert result["diagnostics"]["text_page_count"] == 0
    assert result["diagnostics"]["textless_pages"] == [1, 2]
    reason = api.flash_rejection_reason(result)
    assert "0/2 pages contain extracted text" in reason
    assert "OCR" in reason and "text encoding" in reason
    assert "standard" not in reason


def test_text_counts_precede_line_number_cleanup():
    from pageindex.flash import api

    # These are readable glyphs, but layout cleanup removes the number column.
    pdf = BytesIO(build_pdf([
        [(10, 720 - i * 25, 12, str(i + 1)) for i in range(6)],
        "   ",
    ]))
    result = api.page_index_flash(pdf, summary=False, optimize=False)
    assert result["diagnostics"]["page_count"] == 2
    assert result["diagnostics"]["text_page_count"] == 1
    assert result["diagnostics"]["textless_pages"] == [2]
    # Diagnostics must not make a previously refused document acceptable.
    assert result["toc_source"] == "unreadable"
    assert result["structure"] == []
    reason = api.flash_rejection_reason(result)
    assert "1/2 pages contain extracted text" in reason
    assert "no usable page text remained after layout processing" in reason
    assert "OCR" not in reason and "no text layer" not in reason


def test_section_opener_count_precedes_assembly_mutation(monkeypatch):
    from pageindex.flash import main, heading_detection

    detector = heading_detection.find_section_openers
    observed = {}

    def observe_detector(*args):
        nodes = detector(*args)
        observed["count"] = len(nodes)
        observed["nodes"] = nodes
        return nodes

    monkeypatch.setattr(heading_detection, "find_section_openers", observe_detector)
    pdf = Path(__file__).resolve().parents[1] / "examples/documents/attention-residuals.pdf"
    result = main.extract_toc(pdf, workers=1, use_embedded_toc=False)
    # Run the real assembly: this fixture adds general headings to the same list.
    assert len(observed["nodes"]) > observed["count"] > 0
    assert result["diagnostics"]["section_opener_count"] == observed["count"]


@pytest.mark.parametrize("structured,status", [
    (True, "chapter_validation_failed"),
    (False, "heading_gap_too_large"),
])
def test_validation_failure_keeps_pre_rejection_heading_count(
        plain_pdf, monkeypatch, structured, status):
    from pageindex.flash import main, api

    outline = [SimpleNamespace(child_nodes=[SimpleNamespace(child_nodes=[])])]
    monkeypatch.setattr(main, "assemble_outline", lambda *a: outline)
    monkeypatch.setattr(main, "is_outline_valid", lambda *a: structured)
    monkeypatch.setattr(main, "is_chapter_outline_valid", lambda *a: False)
    monkeypatch.setattr(main, "mark_outline_block_types", lambda *a: None)
    monkeypatch.setattr(main, "compute_max_heading_gap", lambda *a: {"max_gap": 3})
    result = main.extract_toc(plain_pdf, workers=1, use_embedded_toc=False)
    assert result["structure"] == []
    assert result["diagnostics"]["assembled_heading_count"] == 2
    assert result["diagnostics"]["layout_status"] == status
    reason = api.flash_rejection_reason(result)
    assert "assembled headings=2" in reason
    assert ("chapter coverage validation" if structured else
            "gap between headings is too large") in reason


def test_bookmarks_can_rescue_layout_without_erasing_diagnostics(plain_pdf):
    from PyPDF2 import PdfReader, PdfWriter
    from pageindex.flash import api

    writer = PdfWriter()
    for page in PdfReader(plain_pdf).pages:
        writer.add_page(page)
    chapter = writer.add_outline_item("Chapter One", 0)
    writer.add_outline_item("Section A", 0, parent=chapter)
    writer.add_outline_item("Section B", 1, parent=chapter)
    writer.add_outline_item("Chapter Two", 2)
    pdf = BytesIO()
    writer.write(pdf)
    pdf.seek(0)
    result = api.page_index_flash(pdf, summary=False, optimize=False)
    assert result["toc_source"] == "hybrid"
    assert result["structure"][0]["title"] == "Chapter One"
    assert result["diagnostics"]["layout_status"] == "no_headings"
    assert result["diagnostics"]["text_page_count"] == 2
    assert api.flash_rejection_reason(result) is None


def test_local_sdk_exposes_diagnostics_on_refusal(plain_pdf, tmp_path, monkeypatch):
    from pageindex import PageIndexClient, PageIndexAPIError
    from pageindex.flash import api

    monkeypatch.setattr(api, "FLAT_TREE_MAX_NODES", 2)
    client = PageIndexClient(storage_path=str(tmp_path / "store"))
    with pytest.raises(PageIndexAPIError) as caught:
        client.submit_document(str(plain_pdf), mode="flash")
    assert "2/3 pages contain extracted text" in str(caught.value)
    assert "no headings survived outline assembly" in str(caught.value)


def test_cli_exposes_diagnostics_on_refusal(plain_pdf, tmp_path, monkeypatch):
    from pageindex.flash import api

    monkeypatch.setattr(api, "FLAT_TREE_MAX_NODES", 2)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_pageindex.py", "--mode", "flash",
                                     "--pdf_path", str(plain_pdf)])
    script = Path(__file__).resolve().parents[1] / "run_pageindex.py"
    with pytest.raises(ValueError) as caught:
        runpy.run_path(str(script), run_name="__main__")
    assert "2/3 pages contain extracted text" in str(caught.value)
    assert "--mode standard" in str(caught.value)
    assert not (tmp_path / "results").exists()
