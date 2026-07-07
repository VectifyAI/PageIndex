import importlib
import unittest
from unittest.mock import patch

page_index_module = importlib.import_module("pageindex.page_index")
_as_toc_list = page_index_module._as_toc_list
add_page_offset_to_toc_json = page_index_module.add_page_offset_to_toc_json
check_title_appearance_in_start = page_index_module.check_title_appearance_in_start
toc_transformer = page_index_module.toc_transformer
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

    def test_extract_json_preserves_comma_bracket_text_inside_strings(self):
        payload = extract_json('{"note": "see items,] and more"}')

        self.assertEqual(payload["note"], "see items,] and more")

    def test_extract_json_preserves_comma_brace_text_inside_strings(self):
        payload = extract_json('{"formula": "f(x,} )"}')

        self.assertEqual(payload["formula"], "f(x,} )")

    def test_extract_json_repairs_structural_trailing_commas(self):
        payload = extract_json('{"items": [1, 2,], "meta": {"done": True,}}')

        self.assertEqual(payload["items"], [1, 2])
        self.assertIs(payload["meta"]["done"], True)


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

    def test_toc_transformer_returns_empty_list_for_non_dict_json(self):
        with patch.object(page_index_module, "llm_completion", return_value=('[{"title": "Item 1"}]', "finished")), \
             patch.object(page_index_module, "check_if_toc_transformation_is_complete", return_value="yes"):
            self.assertEqual(toc_transformer("1. Item 1"), [])

    def test_title_start_check_returns_no_for_non_dict_json(self):
        async def fake_completion(model, prompt):
            return '[{"start_begin": "yes"}]'

        with patch.object(page_index_module, "llm_acompletion", side_effect=fake_completion):
            result = __import__("asyncio").run(check_title_appearance_in_start("Item 1", "Item 1 text"))

        self.assertEqual(result, "no")


if __name__ == "__main__":
    unittest.main()
