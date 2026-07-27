import asyncio
import importlib
import inspect
import unittest
from unittest.mock import AsyncMock, patch

from pageindex.page_index import clamp_toc_titles
from pageindex.utils import (
    add_node_text,
    generate_summaries_for_structure,
    partition_text_by_titles,
)

page_index_mod = importlib.import_module("pageindex.page_index")


class ClampTocTitlesTest(unittest.TestCase):
    def test_truncates_long_titles_at_word_boundary(self):
        long_title = (
            "1.1 The iOS version of the software requires that the device's "
            "running memory is not less than 2GB and many more words follow here "
            "to exceed the maximum title character limit substantially"
        )
        toc = [{"title": long_title, "structure": "1.1"}]

        clamp_toc_titles(toc, max_chars=80)

        self.assertLessEqual(len(toc[0]["title"]), 83)  # 80 + "..."
        self.assertTrue(toc[0]["title"].endswith("..."))
        self.assertNotIn("substantially", toc[0]["title"])

    def test_leaves_short_titles_unchanged(self):
        toc = [{"title": "Software Installation", "structure": "1"}]
        clamp_toc_titles(toc)
        self.assertEqual(toc[0]["title"], "Software Installation")


class TocTitlePromptTest(unittest.TestCase):
    def test_toc_prompts_ask_for_concise_titles(self):
        instruction = page_index_mod._TOC_TITLE_INSTRUCTION
        self.assertIn("synthesize a concise title", instruction)
        self.assertIn("Do NOT copy an entire long sentence", instruction)

        init_src = inspect.getsource(page_index_mod.generate_toc_init)
        continue_src = inspect.getsource(page_index_mod.generate_toc_continue)
        self.assertIn("_TOC_TITLE_INSTRUCTION", init_src)
        self.assertIn("_TOC_TITLE_INSTRUCTION", continue_src)
        self.assertNotIn("only fix the space inconsistency", init_src)
        self.assertNotIn("only fix the space inconsistency", continue_src)


class PartitionTextByTitlesTest(unittest.TestCase):
    def test_partitions_same_page_siblings_by_title_anchor(self):
        page = (
            "Software Installation\n"
            "1.1 Memory requirements need 2GB RAM.\n"
            "1.2 Storage requirements need 10GB free space.\n"
            "1.3 Network requirements need Wi-Fi.\n"
        )
        titles = [
            "1.1 Memory requirements",
            "1.2 Storage requirements",
            "1.3 Network requirements",
        ]

        slices = partition_text_by_titles(page, titles)

        self.assertIsNotNone(slices)
        self.assertEqual(len(slices), 3)
        self.assertIn("2GB RAM", slices[0])
        self.assertNotIn("10GB", slices[0])
        self.assertIn("10GB", slices[1])
        self.assertNotIn("Wi-Fi", slices[1])
        self.assertIn("Wi-Fi", slices[2])


class AddNodeTextOverlapTest(unittest.TestCase):
    def test_same_page_children_get_distinct_text(self):
        page_text = (
            "Software Installation preface.\n"
            "1.1 Memory requirements need 2GB RAM.\n"
            "1.2 Storage requirements need 10GB free space.\n"
        )
        pdf_pages = [[page_text]]
        tree = [{
            "title": "Software Installation",
            "start_index": 1,
            "end_index": 1,
            "nodes": [
                {
                    "title": "1.1 Memory requirements",
                    "start_index": 1,
                    "end_index": 1,
                },
                {
                    "title": "1.2 Storage requirements",
                    "start_index": 1,
                    "end_index": 1,
                },
            ],
        }]

        add_node_text(tree, pdf_pages)

        parent, child1, child2 = tree[0], tree[0]["nodes"][0], tree[0]["nodes"][1]
        self.assertIn("preface", parent["text"])
        self.assertNotEqual(child1["text"], child2["text"])
        self.assertIn("2GB RAM", child1["text"])
        self.assertIn("10GB", child2["text"])


class GenerateSummariesDedupeTest(unittest.TestCase):
    def test_reuses_summary_for_identical_text(self):
        structure = [
            {"title": "A", "text": "same text", "node_id": "1"},
            {"title": "B", "text": "same text", "node_id": "2"},
            {"title": "C", "text": "other text", "node_id": "3"},
        ]

        with patch(
            "pageindex.utils.generate_node_summary",
            new_callable=AsyncMock,
            side_effect=["summary-same", "summary-other"],
        ) as mock_summary:
            asyncio.run(generate_summaries_for_structure(structure, model="dummy"))

        self.assertEqual(mock_summary.await_count, 2)
        self.assertEqual(structure[0]["summary"], "summary-same")
        self.assertEqual(structure[1]["summary"], "summary-same")
        self.assertEqual(structure[2]["summary"], "summary-other")

    def test_container_with_duplicate_child_text_skips_llm(self):
        structure = [{
            "title": "Software Installation",
            "text": "dup",
            "nodes": [
                {"title": "1.1 Memory", "text": "dup"},
                {"title": "1.2 Storage", "text": "dup"},
            ],
        }]

        with patch(
            "pageindex.utils.generate_node_summary",
            new_callable=AsyncMock,
            return_value="child-summary",
        ) as mock_summary:
            asyncio.run(generate_summaries_for_structure(structure, model="dummy"))

        # Parent gets container summary; one LLM call for shared child text.
        self.assertEqual(mock_summary.await_count, 1)
        self.assertIn("covering:", structure[0]["summary"])
        self.assertIn("1.1 Memory", structure[0]["summary"])
        self.assertEqual(structure[0]["nodes"][0]["summary"], "child-summary")
        self.assertEqual(structure[0]["nodes"][1]["summary"], "child-summary")


if __name__ == "__main__":
    unittest.main()
