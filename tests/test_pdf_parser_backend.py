import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pageindex import utils
from pageindex.utils import (
    SUPPORTED_PDF_PARSERS,
    classify_pdf,
    extract_text_from_pdf,
    get_number_of_pages,
    get_page_tokens,
    get_text_of_pages,
)


FIXTURE_PDF = os.path.join(
    os.path.dirname(__file__),
    "..",
    "examples",
    "documents",
    "q1-fy25-earnings.pdf",
)


def _has_pdf_inspector():
    try:
        import pdf_inspector  # noqa: F401
        return True
    except ImportError:
        return False


REQUIRES_PDF_INSPECTOR = unittest.skipUnless(
    _has_pdf_inspector(), "pdf-inspector not installed"
)


class BackendDispatchTest(unittest.TestCase):
    def test_supported_parsers_advertised(self):
        self.assertIn("PyPDF2", SUPPORTED_PDF_PARSERS)
        self.assertIn("PyMuPDF", SUPPORTED_PDF_PARSERS)
        self.assertIn("pdf_inspector", SUPPORTED_PDF_PARSERS)

    def test_get_page_tokens_rejects_unknown_parser(self):
        with self.assertRaises(ValueError):
            get_page_tokens("dummy.pdf", model="gpt-4o", pdf_parser="not_a_parser")

    def test_extract_text_from_pdf_rejects_unknown_parser(self):
        with self.assertRaises(ValueError):
            extract_text_from_pdf("dummy.pdf", pdf_parser="not_a_parser")

    def test_get_text_of_pages_rejects_unknown_parser(self):
        with self.assertRaises(ValueError):
            get_text_of_pages("dummy.pdf", 1, 1, pdf_parser="not_a_parser")

    def test_load_pdf_inspector_raises_friendly_error_when_missing(self):
        # Simulate the package being absent even when it is installed on the
        # dev machine, so this test runs uniformly in CI.
        with patch.dict(sys.modules, {"pdf_inspector": None}):
            with self.assertRaises(ImportError) as ctx:
                utils._load_pdf_inspector()
            self.assertIn("pdf-inspector", str(ctx.exception))

    def test_classify_pdf_returns_none_when_pdf_inspector_missing(self):
        with patch.dict(sys.modules, {"pdf_inspector": None}):
            self.assertIsNone(classify_pdf("dummy.pdf"))

    def test_get_number_of_pages_falls_back_when_pdf_inspector_missing(self):
        # Backend requested but unavailable: fall back to PyPDF2 rather than
        # raise, so opt.pdf_parser stays a soft preference.
        fake_reader = MagicMock()
        fake_reader.pages = [object(), object(), object()]
        with patch.dict(sys.modules, {"pdf_inspector": None}), \
             patch("pageindex.utils.PyPDF2.PdfReader", return_value=fake_reader):
            self.assertEqual(
                get_number_of_pages("dummy.pdf", pdf_parser="pdf_inspector"),
                3,
            )


@REQUIRES_PDF_INSPECTOR
class PdfInspectorBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(FIXTURE_PDF):
            raise unittest.SkipTest(f"fixture PDF missing: {FIXTURE_PDF}")

    def test_classify_pdf_returns_expected_fields(self):
        info = classify_pdf(FIXTURE_PDF)
        self.assertIsNotNone(info)
        self.assertIn("pdf_type", info)
        self.assertIn("page_count", info)
        self.assertIn("pages_needing_ocr", info)
        self.assertGreater(info["page_count"], 0)
        self.assertIn(info["pdf_type"], {"text_based", "mixed", "scanned", "image_based"})

    def test_get_number_of_pages_matches_pypdf2(self):
        expected = get_number_of_pages(FIXTURE_PDF, pdf_parser="PyPDF2")
        actual = get_number_of_pages(FIXTURE_PDF, pdf_parser="pdf_inspector")
        self.assertEqual(actual, expected)

    def test_get_page_tokens_returns_all_pages(self):
        pages = get_page_tokens(FIXTURE_PDF, model="gpt-4o", pdf_parser="pdf_inspector")
        self.assertGreater(len(pages), 0)
        expected = get_number_of_pages(FIXTURE_PDF, pdf_parser="PyPDF2")
        self.assertEqual(len(pages), expected)
        # Every entry must be (text, token_count).
        for text, tokens in pages:
            self.assertIsInstance(text, str)
            self.assertIsInstance(tokens, int)

    def test_get_text_of_pages_tags_selected_range(self):
        text = get_text_of_pages(
            FIXTURE_PDF, 3, 4, tag=True, pdf_parser="pdf_inspector"
        )
        self.assertIn("<start_index_3>", text)
        self.assertIn("<start_index_4>", text)
        self.assertIn("<end_index_3>", text)
        self.assertIn("<end_index_4>", text)

    def test_extract_text_from_pdf_returns_nonempty(self):
        text = extract_text_from_pdf(FIXTURE_PDF, pdf_parser="pdf_inspector")
        self.assertGreater(len(text), 100)


if __name__ == "__main__":
    unittest.main()
