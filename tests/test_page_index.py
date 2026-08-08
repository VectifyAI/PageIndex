import unittest
from unittest.mock import Mock, patch

from pageindex.page_index import process_toc_no_page_numbers


class ProcessTocNoPageNumbersTest(unittest.TestCase):
    def test_skips_same_length_reordered_llm_toc(self):
        toc = [
            {"structure": "1", "title": "First"},
            {"structure": "2", "title": "Second"},
        ]
        reordered = [
            {"structure": "2", "title": "Second", "physical_index": "<physical_index_2>"},
            {"structure": "1", "title": "First", "physical_index": "<physical_index_1>"},
        ]

        with patch("pageindex.page_index.toc_transformer", return_value=toc), \
             patch("pageindex.page_index.count_tokens", return_value=1), \
             patch("pageindex.page_index.page_list_to_group_text", return_value=["<physical_index_1> <physical_index_2>"]), \
             patch("pageindex.page_index.add_page_number_to_toc", return_value=reordered):
            result = process_toc_no_page_numbers(
                "toc",
                [],
                [["page one"], ["page two"]],
                logger=Mock(),
            )

        self.assertEqual([entry["title"] for entry in result], ["First", "Second"])
        self.assertIsNone(result[0].get("physical_index"))
        self.assertIsNone(result[1].get("physical_index"))

    def test_fills_lenient_formats_and_rejects_out_of_chunk(self):
        toc = [
            {"structure": "1", "title": "First"},
            {"structure": "2", "title": "Second"},
            {"structure": "3", "title": "Third"},
        ]
        llm_result = [
            {"structure": "1", "title": "First", "physical_index": "1"},
            {"structure": "2", "title": "Second", "physical_index": "<physical_index_2>"},
            {"structure": "3", "title": "Third", "physical_index": "<physical_index_99>"},
        ]

        with patch("pageindex.page_index.toc_transformer", return_value=toc), \
             patch("pageindex.page_index.count_tokens", return_value=1), \
             patch("pageindex.page_index.page_list_to_group_text", return_value=["<physical_index_1> <physical_index_2>"]), \
             patch("pageindex.page_index.add_page_number_to_toc", return_value=llm_result):
            result = process_toc_no_page_numbers(
                "toc",
                [],
                [["page one"], ["page two"]],
                logger=Mock(),
            )

        self.assertEqual(result[0]["physical_index"], 1)
        self.assertEqual(result[1]["physical_index"], 2)
        self.assertIsNone(result[2].get("physical_index"))


if __name__ == "__main__":
    unittest.main()
