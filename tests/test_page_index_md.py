import unittest

from pageindex.page_index_md import extract_nodes_from_markdown


class ExtractNodesFromMarkdownTest(unittest.TestCase):
    def test_skips_bold_heading_with_only_whitespace(self):
        nodes, _ = extract_nodes_from_markdown("**   **\n**Valid heading**")

        self.assertEqual(
            nodes,
            [
                {
                    "node_title": "Valid heading",
                    "line_num": 2,
                    "level": 1,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
