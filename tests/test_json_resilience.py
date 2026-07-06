import unittest

from pageindex.page_index import _as_toc_list, add_page_offset_to_toc_json
from pageindex.utils import extract_json


class JsonExtractionTests(unittest.TestCase):
    def test_extract_json_from_fenced_block(self):
        payload = extract_json('```json\n{"toc_detected": "yes"}\n```')

        self.assertEqual(payload["toc_detected"], "yes")

    def test_extract_json_after_explanatory_text(self):
        payload = extract_json('Here is the JSON:\n{"toc_detected": "no"}')

        self.assertEqual(payload["toc_detected"], "no")

    def test_extract_json_array_with_trailing_text(self):
        payload = extract_json('[{"title": "Item 1", "physical_index": "<physical_index_3>"}]\nDone.')

        self.assertEqual(payload[0]["title"], "Item 1")

    def test_extract_json_python_literals(self):
        payload = extract_json('{"page": None, "valid": True, "done": False}')

        self.assertIsNone(payload["page"])
        self.assertIs(payload["valid"], True)
        self.assertIs(payload["done"], False)

    def test_extract_json_allows_unescaped_newlines_in_strings(self):
        payload = extract_json('{"thinking": "line1\nline2", "answer": "yes"}')

        self.assertEqual(payload["thinking"], "line1\nline2")
        self.assertEqual(payload["answer"], "yes")

    def test_extract_json_preserves_python_literal_words_inside_strings(self):
        payload = extract_json('{"title": "None of the above", "valid": True}')

        self.assertEqual(payload["title"], "None of the above")
        self.assertIs(payload["valid"], True)


class TocFallbackTests(unittest.TestCase):
    def test_as_toc_list_accepts_plain_list(self):
        payload = [{"title": "Item 1", "physical_index": 3}, "bad"]

        self.assertEqual(_as_toc_list(payload), [{"title": "Item 1", "physical_index": 3}])

    def test_as_toc_list_accepts_common_wrappers(self):
        payload = {"toc": [{"title": "Item 1", "physical_index": 3}]}

        self.assertEqual(_as_toc_list(payload), [{"title": "Item 1", "physical_index": 3}])

    def test_as_toc_list_accepts_table_of_contents_wrapper(self):
        payload = {"table_of_contents": [{"title": "Item 1", "physical_index": 3}]}

        self.assertEqual(_as_toc_list(payload), [{"title": "Item 1", "physical_index": 3}])

    def test_page_offset_none_is_noop(self):
        payload = [{"title": "Item 1", "page": 5}]

        self.assertEqual(add_page_offset_to_toc_json(payload, None), payload)


if __name__ == "__main__":
    unittest.main()
