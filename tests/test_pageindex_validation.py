import os
import tempfile
import unittest

from PageIndex import PageIndex, PageIndexConfig


class TestPageIndexValidationParity(unittest.TestCase):
    def _temp_file(self, suffix: str) -> str:
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return path

    def test_requires_either_pdf_or_markdown_path(self):
        with self.assertRaisesRegex(ValueError, "Either --pdf_path or --md_path must be specified"):
            PageIndex(PageIndexConfig())

    def test_rejects_both_pdf_and_markdown_path(self):
        pdf_path = self._temp_file(".pdf")
        md_path = self._temp_file(".md")
        with self.assertRaisesRegex(ValueError, "Only one of --pdf_path or --md_path can be specified"):
            PageIndex(PageIndexConfig(pdf_path=pdf_path, md_path=md_path))

    def test_pdf_requires_pdf_extension(self):
        wrong_extension = self._temp_file(".txt")
        with self.assertRaisesRegex(ValueError, "PDF file must have .pdf extension"):
            PageIndex(PageIndexConfig(pdf_path=wrong_extension))

    def test_pdf_requires_existing_file(self):
        missing_pdf = os.path.join(tempfile.gettempdir(), "missing_pageindex_input.pdf")
        with self.assertRaisesRegex(ValueError, f"PDF file not found: {missing_pdf}"):
            PageIndex(PageIndexConfig(pdf_path=missing_pdf))

    def test_markdown_requires_markdown_extension(self):
        wrong_extension = self._temp_file(".txt")
        with self.assertRaisesRegex(ValueError, "Markdown file must have .md or .markdown extension"):
            PageIndex(PageIndexConfig(md_path=wrong_extension))

    def test_markdown_requires_existing_file(self):
        missing_md = os.path.join(tempfile.gettempdir(), "missing_pageindex_input.md")
        with self.assertRaisesRegex(ValueError, f"Markdown file not found: {missing_md}"):
            PageIndex(PageIndexConfig(md_path=missing_md))

    def test_accepts_valid_pdf_path(self):
        pdf_path = self._temp_file(".pdf")
        page_index = PageIndex(PageIndexConfig(pdf_path=pdf_path))
        self.assertEqual(page_index._doc_kind, "pdf")
        self.assertEqual(page_index._doc_path, pdf_path)

    def test_accepts_valid_markdown_path(self):
        md_path = self._temp_file(".md")
        page_index = PageIndex(PageIndexConfig(md_path=md_path))
        self.assertEqual(page_index._doc_kind, "markdown")
        self.assertEqual(page_index._doc_path, md_path)

if __name__ == "__main__":
    unittest.main()
