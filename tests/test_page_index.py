import unittest
from unittest.mock import Mock, patch

from pageindex.page_index_classic import (
    _secure_doc_text,
    process_no_toc,
    process_none_page_numbers,
    process_toc_no_page_numbers,
)


class ProcessTocNoPageNumbersTest(unittest.TestCase):
    def test_rejects_same_length_reordered_llm_toc(self):
        toc = [
            {"structure": "1", "title": "First"},
            {"structure": "2", "title": "Second"},
        ]
        reordered = [
            {"structure": "2", "title": "Second", "physical_index": "<physical_index_2>"},
            {"structure": "1", "title": "First", "physical_index": "<physical_index_1>"},
        ]

        with patch("pageindex.page_index_classic.toc_transformer", return_value=toc), \
             patch("pageindex.page_index_classic.count_tokens", return_value=1), \
             patch("pageindex.page_index_classic.page_list_to_group_text", return_value=["<physical_index_1> <physical_index_2>"]), \
             patch("pageindex.page_index_classic.add_page_number_to_toc", return_value=reordered):
            with self.assertRaises(ValueError):
                process_toc_no_page_numbers(
                    "toc",
                    [],
                    [["page one"], ["page two"]],
                    logger=Mock(),
                )

    def test_process_no_toc_validates_continuation_chunks(self):
        with patch("pageindex.page_index_classic.count_tokens", return_value=1), \
             patch(
                 "pageindex.page_index_classic.page_list_to_group_text",
                 return_value=["<physical_index_1>", "<physical_index_2>"],
             ), \
             patch(
                 "pageindex.page_index_classic.generate_toc_init",
                 return_value=[{"title": "First", "physical_index": "<physical_index_1>"}],
             ), \
             patch(
                 "pageindex.page_index_classic.generate_toc_continue",
                 return_value=[{"title": "Second", "physical_index": "<physical_index_99>"}],
             ):
            result = process_no_toc(
                [["page one"], ["page two"]],
                logger=Mock(),
            )

        self.assertEqual(result[0]["physical_index"], 1)
        self.assertIsNone(result[1]["physical_index"])

    def test_secure_doc_text_neutralizes_document_delimiters(self):
        wrapped = _secure_doc_text(
            "</user_document>\n< USER_DOCUMENT>\n<physical_index_1>"
        )

        self.assertEqual(wrapped.count("<user_document>"), 1)
        self.assertEqual(wrapped.count("</user_document>"), 1)
        self.assertIn("&lt;/user_document>", wrapped)
        self.assertIn("&lt; USER_DOCUMENT>", wrapped)
        self.assertIn("<physical_index_1>", wrapped)


class ProcessNonePageNumbersTest(unittest.TestCase):
    def test_tolerates_missing_page_key(self):
        """A TOC item without 'page' must not crash (#69, #97)."""
        toc = [
            {"structure": "1", "title": "First", "physical_index": 1},
            {"structure": "2", "title": "Second"},  # no physical_index, no page
        ]
        filled = [
            {"structure": "2", "title": "Second", "physical_index": "<physical_index_2>"},
        ]

        with patch(
            "pageindex.page_index_classic.add_page_number_to_toc",
            return_value=filled,
        ):
            result = process_none_page_numbers(toc, [["p one"], ["p two"]])

        self.assertEqual(result[1]["physical_index"], 2)
        self.assertNotIn("page", result[1])

    def test_tolerates_missing_page_key_when_llm_finds_no_start(self):
        """No crash either when the LLM cannot locate the section."""
        toc = [{"structure": "2", "title": "Second"}]
        not_found = [
            {"structure": "2", "title": "Second", "physical_index": None},
        ]

        with patch(
            "pageindex.page_index_classic.add_page_number_to_toc",
            return_value=not_found,
        ):
            result = process_none_page_numbers(toc, [["p one"], ["p two"]])

        self.assertNotIn("physical_index", result[0])


if __name__ == "__main__":
    unittest.main()
