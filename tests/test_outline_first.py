import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pageindex.page_index import page_index_main
from pageindex.utils import get_pdf_outline_tree, structure_to_list


class OutlineFirstTests(unittest.TestCase):
    def test_embedded_outline_builds_a_usable_tree(self):
        pdf_path = Path("examples/documents/PRML.pdf")
        if not pdf_path.exists():
            self.skipTest(f"missing sample PDF: {pdf_path}")

        outline_tree = get_pdf_outline_tree(str(pdf_path))

        self.assertIsInstance(outline_tree, list)
        self.assertTrue(outline_tree, "expected an outline-first tree for PRML.pdf")

        flat_nodes = structure_to_list(outline_tree)
        valid_nodes = [node for node in flat_nodes if node.get("start_index") is not None]

        self.assertGreaterEqual(len(valid_nodes), 5)
        self.assertTrue(all(node["title"] for node in flat_nodes))
        self.assertTrue(
            all(
                node.get("end_index") is not None and node["end_index"] >= node["start_index"]
                for node in valid_nodes
            )
        )

    def test_page_index_main_prefers_outline_tree_over_tree_parser(self):
        pdf_path = "examples/documents/PRML.pdf"
        outline_tree = [{"title": "Outline Root", "start_index": 1, "end_index": 3, "nodes": []}]
        opt = SimpleNamespace(
            model=None,
            if_add_node_id="no",
            if_add_node_text="no",
            if_add_node_summary="no",
            if_add_doc_description="no",
        )

        tree_parser_mock = AsyncMock(side_effect=AssertionError("tree_parser should not run"))

        with patch("pageindex.page_index.get_page_tokens", return_value=[("page", 1)]), \
             patch("pageindex.page_index.get_pdf_outline_tree", return_value=outline_tree), \
             patch("pageindex.page_index.tree_parser", tree_parser_mock), \
             patch("pageindex.page_index.JsonLogger") as logger_cls:
            logger = logger_cls.return_value
            logger.info.return_value = None

            result = page_index_main(pdf_path, opt)

        self.assertEqual(result["structure"], outline_tree)
        tree_parser_mock.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
