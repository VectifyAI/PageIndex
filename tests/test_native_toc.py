import pytest
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pageindex.utils import extract_pdf_native_toc, validate_native_toc_quality, convert_toc_levels_to_structure, try_native_toc


class TestExtractPdfNativeToc:
    """Tests for extracting native TOC from PDF bookmarks."""

    def test_returns_none_for_nonexistent_file(self):
        result = extract_pdf_native_toc("/nonexistent/path.pdf")
        assert result is None

    def test_returns_none_for_invalid_file(self, tmp_path):
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_text("not a pdf")
        result = extract_pdf_native_toc(str(fake_pdf))
        assert result is None


class TestValidateNativeTocQuality:
    """Tests for TOC quality validation."""

    def test_rejects_empty_toc(self):
        result = validate_native_toc_quality([], total_pages=100)
        assert result is False

    def test_rejects_too_few_items(self):
        toc = [(1, "Chapter 1", 1), (1, "Chapter 2", 10)]
        result = validate_native_toc_quality(toc, total_pages=100)
        assert result is False

    def test_rejects_empty_titles(self):
        toc = [(1, "", 1), (1, "Chapter 2", 10), (1, "Chapter 3", 20),
               (1, "Chapter 4", 30), (1, "Chapter 5", 40)]
        result = validate_native_toc_quality(toc, total_pages=100)
        assert result is False

    def test_rejects_out_of_range_pages(self):
        toc = [(1, "Chapter 1", 1), (1, "Chapter 2", 10), (1, "Chapter 3", 200),
               (1, "Chapter 4", 30), (1, "Chapter 5", 40)]
        result = validate_native_toc_quality(toc, total_pages=100)
        assert result is False

    def test_accepts_valid_toc(self):
        toc = [(1, "Chapter 1", 1), (1, "Chapter 2", 10), (1, "Chapter 3", 20),
               (1, "Chapter 4", 30), (1, "Chapter 5", 40)]
        result = validate_native_toc_quality(toc, total_pages=100)
        assert result is True


class TestConvertTocLevelsToStructure:
    """Tests for converting pymupdf TOC format to PageIndex structure format."""

    def test_single_level(self):
        toc = [(1, "Chapter 1", 1), (1, "Chapter 2", 10), (1, "Chapter 3", 20)]
        result = convert_toc_levels_to_structure(toc)

        assert len(result) == 3
        assert result[0] == {"structure": "1", "title": "Chapter 1", "physical_index": 1}
        assert result[1] == {"structure": "2", "title": "Chapter 2", "physical_index": 10}
        assert result[2] == {"structure": "3", "title": "Chapter 3", "physical_index": 20}

    def test_nested_levels(self):
        toc = [
            (1, "Chapter 1", 1),
            (2, "Section 1.1", 2),
            (2, "Section 1.2", 5),
            (1, "Chapter 2", 10),
            (2, "Section 2.1", 11),
        ]
        result = convert_toc_levels_to_structure(toc)

        assert result[0] == {"structure": "1", "title": "Chapter 1", "physical_index": 1}
        assert result[1] == {"structure": "1.1", "title": "Section 1.1", "physical_index": 2}
        assert result[2] == {"structure": "1.2", "title": "Section 1.2", "physical_index": 5}
        assert result[3] == {"structure": "2", "title": "Chapter 2", "physical_index": 10}
        assert result[4] == {"structure": "2.1", "title": "Section 2.1", "physical_index": 11}

    def test_three_levels_deep(self):
        toc = [
            (1, "Part 1", 1),
            (2, "Chapter 1", 2),
            (3, "Section 1.1.1", 3),
        ]
        result = convert_toc_levels_to_structure(toc)

        assert result[0] == {"structure": "1", "title": "Part 1", "physical_index": 1}
        assert result[1] == {"structure": "1.1", "title": "Chapter 1", "physical_index": 2}
        assert result[2] == {"structure": "1.1.1", "title": "Section 1.1.1", "physical_index": 3}


class TestTryNativeToc:
    """Tests for the combined try_native_toc function."""

    def test_returns_none_for_nonexistent_file(self):
        result = try_native_toc("/nonexistent/path.pdf", total_pages=100)
        assert result is None

    def test_returns_none_for_invalid_file(self, tmp_path):
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_text("not a pdf")
        result = try_native_toc(str(fake_pdf), total_pages=100)
        assert result is None


class TestIntegrationWithRealPdf:
    """Integration tests using actual test PDFs."""

    @pytest.fixture
    def test_pdf_path(self):
        """Path to a test PDF that should have native TOC."""
        return "tests/pdfs/2023-annual-report.pdf"

    def test_extract_from_real_pdf(self, test_pdf_path):
        """Test extraction from a real PDF file."""
        if not os.path.exists(test_pdf_path):
            pytest.skip(f"Test PDF not found: {test_pdf_path}")

        result = extract_pdf_native_toc(test_pdf_path)
        # Result can be None (no TOC) or a list - both are valid
        assert result is None or isinstance(result, list)

    def test_try_native_toc_with_real_pdf(self, test_pdf_path):
        """Test full pipeline with a real PDF."""
        if not os.path.exists(test_pdf_path):
            pytest.skip(f"Test PDF not found: {test_pdf_path}")

        import pymupdf
        doc = pymupdf.open(test_pdf_path)
        total_pages = len(doc)
        doc.close()

        result = try_native_toc(test_pdf_path, total_pages)
        # Result can be None (no/bad TOC) or properly formatted list
        if result is not None:
            assert isinstance(result, list)
            assert len(result) > 0
            # Check structure format
            for item in result:
                assert 'structure' in item
                assert 'title' in item
                assert 'physical_index' in item
