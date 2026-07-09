import pymupdf
import pytest
from pathlib import Path
from pageindex.parser.pdf import PdfParser
from pageindex.parser.protocol import ContentNode, ParsedDocument

TEST_PDF = Path("tests/pdfs/deepseek-r1.pdf")

def test_supported_extensions():
    parser = PdfParser()
    assert ".pdf" in parser.supported_extensions()

@pytest.mark.skipif(not TEST_PDF.exists(), reason="Test PDF not available")
def test_parse_returns_parsed_document():
    parser = PdfParser()
    result = parser.parse(str(TEST_PDF))
    assert isinstance(result, ParsedDocument)
    assert len(result.nodes) > 0
    assert result.doc_name != ""

@pytest.mark.skipif(not TEST_PDF.exists(), reason="Test PDF not available")
def test_parse_nodes_are_flat_without_level():
    parser = PdfParser()
    result = parser.parse(str(TEST_PDF))
    for node in result.nodes:
        assert isinstance(node, ContentNode)
        assert node.content is not None
        assert node.tokens >= 0
        assert node.index is not None
        assert node.level is None


def test_cmyk_pixmap_without_alpha_is_saveable_as_png(tmp_path):
    """A CMYK image with no alpha has n==4 -- same as RGBA -- so `pix.n > 4`
    wrongly skips the RGB conversion PNG needs, and pix.save() raises
    'unsupported colorspace for png', silently dropping the image via the
    extractor's bare except. The fix (`pix.n - pix.alpha >= 4`) must convert
    CMYK (4-0=4) while leaving RGBA (4-1=3) untouched."""
    cmyk = pymupdf.Pixmap(pymupdf.csCMYK, pymupdf.Rect(0, 0, 10, 10))
    assert cmyk.n == 4 and cmyk.alpha == 0
    assert cmyk.n - cmyk.alpha >= 4  # the fixed condition: must convert
    converted = pymupdf.Pixmap(pymupdf.csRGB, cmyk)
    converted.save(str(tmp_path / "cmyk.png"))  # must not raise

    rgba = pymupdf.Pixmap(pymupdf.Pixmap(pymupdf.csRGB, pymupdf.Rect(0, 0, 10, 10)), 1)
    assert rgba.n == 4 and rgba.alpha == 1
    assert not (rgba.n - rgba.alpha >= 4)  # unchanged: RGBA needs no conversion
    rgba.save(str(tmp_path / "rgba.png"))  # already saveable as-is


def test_image_paths_are_absolute(tmp_path):
    """Image references must be absolute so they resolve regardless of cwd
    (cwd-relative paths broke after the query ran from another directory)."""
    import os
    import pymupdf
    from pageindex.parser.pdf import PdfParser

    # Build a 1-page PDF with an embedded image (>= _MIN_IMAGE_SIZE).
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 64, 64), False)
    pix.clear_with(128)
    png = tmp_path / "img.png"
    pix.save(str(png))

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(pymupdf.Rect(20, 20, 180, 180), filename=str(png))
    pdf_path = tmp_path / "withimg.pdf"
    doc.save(str(pdf_path))
    doc.close()

    images_dir = tmp_path / "out" / "images"
    result = PdfParser().parse(str(pdf_path), images_dir=str(images_dir))

    img_paths = [im["path"] for n in result.nodes if n.images for im in n.images]
    assert img_paths, "expected at least one extracted image"
    for p in img_paths:
        assert os.path.isabs(p), f"image path not absolute: {p}"
        assert os.path.exists(p), f"image path does not resolve: {p}"
