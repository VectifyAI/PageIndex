import unittest
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

# Patch attributes on the module object: the package exports a *function*
# named page_index that shadows the submodule, and Python 3.10's mock
# resolves the string "pageindex.page_index" to that function.
page_index_module = import_module("pageindex.page_index")

from pageindex.page_index import (
    _parse_physical_index,
    _validate_chunk_physical_indices,
    add_page_number_to_toc,
    fix_incorrect_toc,
    meta_processor,
    process_none_page_numbers,
    process_toc_no_page_numbers,
    process_toc_with_page_numbers,
    validate_and_truncate_physical_indices,
)
from pageindex.utils import convert_physical_index_to_int


def make_page_list(n):
    return [[f"page {i}", 1] for i in range(1, n + 1)]


class PhysicalIndexParseTest(unittest.TestCase):
    def test_accepts_integral_values_in_any_supported_shape(self):
        self.assertEqual(_parse_physical_index("<physical_index_7>"), 7)
        self.assertEqual(_parse_physical_index("physical_index_7"), 7)
        self.assertEqual(_parse_physical_index("<physical_index_7> "), 7)
        self.assertEqual(_parse_physical_index("<physical_index_7>\n"), 7)
        self.assertEqual(_parse_physical_index("<physical_index_7>>"), 7)
        self.assertEqual(_parse_physical_index("7"), 7)
        self.assertEqual(_parse_physical_index(7), 7)
        self.assertEqual(_parse_physical_index(7.0), 7)

    def test_rejects_non_positive_and_malformed_values(self):
        for raw in (None, True, False, 1.9, float("inf"), float("nan"),
                    "-1", "+5", "1_0", " ", "abc", -1, 0,
                    "<physical_index_0>", "<physical_index_-3>", "<physical_index_x>",
                    [7], {"physical_index": 7}):
            self.assertIsNone(_parse_physical_index(raw), msg=repr(raw))


class ConvertPhysicalIndexToIntTest(unittest.TestCase):
    def test_normalizes_list_entries_in_place(self):
        data = [
            {"physical_index": "1"},
            {"physical_index": "<physical_index_2>"},
            {"physical_index": 7.0},
            {"physical_index": True},
            {"physical_index": "<physical_index_x>"},
            {"physical_index": "abc"},
            {"physical_index": None},
            {"title": "no index"},
            "not a dict",
        ]
        convert_physical_index_to_int(data)
        self.assertEqual(data[0]["physical_index"], 1)
        self.assertEqual(data[1]["physical_index"], 2)
        self.assertEqual(data[2]["physical_index"], 7)
        self.assertIsNone(data[3]["physical_index"])
        self.assertIsNone(data[4]["physical_index"])
        self.assertIsNone(data[5]["physical_index"])
        self.assertIsNone(data[6]["physical_index"])
        self.assertNotIn("physical_index", data[7])

    def test_scalar_values(self):
        self.assertEqual(convert_physical_index_to_int("7"), 7)
        self.assertEqual(convert_physical_index_to_int("<physical_index_3>"), 3)
        self.assertEqual(convert_physical_index_to_int(4), 4)
        self.assertIsNone(convert_physical_index_to_int("<physical_index_x>"))
        self.assertIsNone(convert_physical_index_to_int("-1"))
        self.assertIsNone(convert_physical_index_to_int(True))
        self.assertIsNone(convert_physical_index_to_int(-1))
        self.assertIsNone(convert_physical_index_to_int({"weird": 1}))


class ValidateChunkPhysicalIndicesTest(unittest.TestCase):
    def test_requires_exact_marker_pointing_into_chunk(self):
        content = "<physical_index_10> text <physical_index_11>"
        toc = [
            {"title": "A", "physical_index": "<physical_index_10>"},
            {"title": "B", "physical_index": 11},
            {"title": "C", "physical_index": "11"},
            {"title": "D", "physical_index": "<physical_index_12>"},
            {"title": "E", "physical_index": None},
        ]
        _validate_chunk_physical_indices(toc, content)
        self.assertEqual(toc[0]["physical_index"], 10)
        self.assertIsNone(toc[1]["physical_index"])
        self.assertIsNone(toc[2]["physical_index"])
        self.assertIsNone(toc[3]["physical_index"])
        self.assertIsNone(toc[4]["physical_index"])

    def test_drops_non_dict_entries_and_non_list_input(self):
        result = _validate_chunk_physical_indices(
            ["garbage", {"title": "A", "physical_index": "<physical_index_1>"}],
            "<physical_index_1>",
        )
        self.assertEqual(result, [{"title": "A", "physical_index": 1}])
        self.assertEqual(_validate_chunk_physical_indices({"not": "a list"}, ""), [])


class ValidateAndTruncateTest(unittest.TestCase):
    def test_range_boundaries_and_coercion(self):
        toc = [
            {"physical_index": 4},
            {"physical_index": 5},
            {"physical_index": 14},
            {"physical_index": 15},
            {"physical_index": "7"},
            {"physical_index": 7.0},
            {"physical_index": "<physical_index_9>"},
            {"physical_index": True},
            {"physical_index": "abc"},
            {"physical_index": 1.9},
        ]
        validate_and_truncate_physical_indices(toc, 10, start_index=5)
        self.assertIsNone(toc[0]["physical_index"])
        self.assertEqual(toc[1]["physical_index"], 5)
        self.assertEqual(toc[2]["physical_index"], 14)
        self.assertIsNone(toc[3]["physical_index"])
        self.assertEqual(toc[4]["physical_index"], 7)
        self.assertEqual(toc[5]["physical_index"], 7)
        self.assertEqual(toc[6]["physical_index"], 9)
        self.assertIsNone(toc[7]["physical_index"])
        self.assertIsNone(toc[8]["physical_index"])
        self.assertIsNone(toc[9]["physical_index"])


class ProcessTocNoPageNumbersTest(unittest.TestCase):
    def _run(self, toc, llm_results, chunks):
        with patch.object(page_index_module, "toc_transformer", return_value=toc), \
             patch.object(page_index_module, "count_tokens", return_value=1), \
             patch.object(page_index_module, "page_list_to_group_text", return_value=chunks), \
             patch.object(page_index_module, "add_page_number_to_toc", side_effect=llm_results):
            return process_toc_no_page_numbers(
                "toc",
                [],
                [["page one"], ["page two"]],
                logger=Mock(),
            )

    def test_raises_on_length_mismatch(self):
        toc = [{"structure": "1", "title": "First"}, {"structure": "2", "title": "Second"}]
        short_result = [{"structure": "1", "title": "First", "physical_index": "<physical_index_1>"}]
        with self.assertRaises(ValueError):
            self._run(toc, [short_result], ["<physical_index_1>"])

    def test_raises_on_reordered_llm_toc(self):
        toc = [{"structure": "1", "title": "First"}, {"structure": "2", "title": "Second"}]
        reordered = [
            {"structure": "2", "title": "Second", "physical_index": "<physical_index_2>"},
            {"structure": "1", "title": "First", "physical_index": "<physical_index_1>"},
        ]
        with self.assertRaises(ValueError):
            self._run(toc, [reordered], ["<physical_index_1> <physical_index_2>"])

    def test_raises_on_non_dict_entries(self):
        toc = [{"structure": "1", "title": "First"}, {"structure": "2", "title": "Second"}]
        with self.assertRaises(ValueError):
            self._run(toc, [["First", "Second"]], ["<physical_index_1>"])

    def test_fills_lenient_formats_across_chunks_without_overwriting(self):
        toc = [
            {"structure": "1", "title": "First"},
            {"structure": "2", "title": "Second"},
            {"structure": "3", "title": "Third"},
        ]
        chunk1_result = [
            {"structure": "1", "title": "First", "physical_index": "1"},
            {"structure": "2", "title": "Second", "physical_index": None},
            {"structure": "3", "title": "Third", "physical_index": "<physical_index_99>"},
        ]
        chunk2_result = [
            {"structure": "1", "title": "First", "physical_index": "<physical_index_2>"},
            {"structure": "2", "title": "Second", "physical_index": "<physical_index_2>"},
            {"structure": "3", "title": "Third", "physical_index": None},
        ]
        result = self._run(
            toc,
            [chunk1_result, chunk2_result],
            ["<physical_index_1>", "<physical_index_1> <physical_index_2>"],
        )
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["physical_index"], 1)
        self.assertEqual(result[1]["physical_index"], 2)
        self.assertIsNone(result[2].get("physical_index"))


class ProcessNonePageNumbersTest(unittest.TestCase):
    def _toc(self):
        return [
            {"title": "A", "physical_index": 1},
            {"title": "B", "page": "iv"},
            {"title": "C", "physical_index": 4},
        ]

    def test_fills_from_marker_answer_inside_window(self):
        toc = self._toc()
        answer = [{"title": "B", "physical_index": "<physical_index_3>"}]
        with patch.object(page_index_module, "add_page_number_to_toc", return_value=answer):
            process_none_page_numbers(toc, make_page_list(5))
        self.assertEqual(toc[1]["physical_index"], 3)
        self.assertNotIn("page", toc[1])

    def test_ignores_malformed_or_out_of_window_answers(self):
        for bad in ("<physical_index_null>", "<physical_index_x>", "abc", None, "<physical_index_99>"):
            toc = self._toc()
            with patch.object(page_index_module, "add_page_number_to_toc", return_value=[{"title": "B", "physical_index": bad}]):
                process_none_page_numbers(toc, make_page_list(5))
            self.assertNotIn("physical_index", toc[1], msg=repr(bad))
            self.assertEqual(toc[1]["page"], "iv")

    def test_handles_empty_or_malformed_llm_results(self):
        for result in ([], {}, None, ["garbage"]):
            toc = self._toc()
            with patch.object(page_index_module, "add_page_number_to_toc", return_value=result):
                process_none_page_numbers(toc, make_page_list(5))
            self.assertNotIn("physical_index", toc[1], msg=repr(result))

    def test_missing_page_key_does_not_crash(self):
        toc = [
            {"title": "A", "physical_index": 1},
            {"title": "B"},
            {"title": "C", "physical_index": 4},
        ]
        with patch.object(page_index_module, "add_page_number_to_toc", return_value=[{"title": "B", "physical_index": "<physical_index_2>"}]):
            process_none_page_numbers(toc, make_page_list(5))
        self.assertEqual(toc[1]["physical_index"], 2)

    def test_empty_window_skips_llm_call(self):
        toc = [{"title": "Z", "page": "iv"}]
        with patch.object(page_index_module, "add_page_number_to_toc") as llm:
            process_none_page_numbers(toc, make_page_list(5))
        llm.assert_not_called()
        self.assertNotIn("physical_index", toc[0])


class AddPageNumberToTocTest(unittest.TestCase):
    def _call(self, raw_response):
        with patch.object(page_index_module, "llm_completion", return_value=raw_response):
            return add_page_number_to_toc("part", [{"title": "A"}], model=None)

    def test_normalizes_dict_and_non_list_results(self):
        self.assertEqual(
            self._call('{"title": "A", "start": "yes", "physical_index": "<physical_index_2>"}'),
            [{"title": "A", "physical_index": "<physical_index_2>"}],
        )
        self.assertEqual(self._call('null'), [])
        self.assertEqual(self._call('["First", "Second"]'), ["First", "Second"])


class ProcessTocWithPageNumbersTest(unittest.TestCase):
    def test_falls_back_when_no_offset_pairs_survive(self):
        sentinel = [{"title": "A", "physical_index": 2}]
        toc = [{"structure": "1", "title": "A", "page": 3}]
        extractor_result = [{"structure": "1", "title": "A", "physical_index": None}]
        with patch.object(page_index_module, "toc_transformer", return_value=toc), \
             patch.object(page_index_module, "toc_index_extractor", return_value=extractor_result), \
             patch.object(page_index_module, "process_toc_no_page_numbers", return_value=sentinel) as fallback:
            result = process_toc_with_page_numbers(
                "toc", [0], make_page_list(5), toc_check_page_num=2, logger=Mock()
            )
        self.assertIs(result, sentinel)
        fallback.assert_called_once()


class FixIncorrectTocTest(unittest.IsolatedAsyncioTestCase):
    async def test_exceptions_route_to_invalid_results_for_retry(self):
        toc = [
            {"title": "S1", "physical_index": 3},
            {"title": "S2", "physical_index": 5},
        ]
        incorrect = [
            {"list_index": 0, "title": "S1", "physical_index": 3},
            {"list_index": 1, "title": "S2", "physical_index": 5},
        ]

        async def fixer(title, content, model=None):
            if title == "S1":
                raise IndexError("boom")
            return 6

        async def checker(item, page_list, start_index=1, model=None):
            return {"list_index": item["list_index"], "answer": "yes",
                    "title": item["title"], "page_number": item["physical_index"]}

        with patch.object(page_index_module, "single_toc_item_index_fixer", side_effect=fixer), \
             patch.object(page_index_module, "check_title_appearance", side_effect=checker):
            result_toc, invalid = await fix_incorrect_toc(toc, make_page_list(10), incorrect, logger=Mock())

        self.assertEqual(result_toc[1]["physical_index"], 6)
        self.assertEqual(result_toc[0]["physical_index"], 3)
        self.assertEqual([entry["list_index"] for entry in invalid], [0])


class MetaProcessorTest(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_entries_are_removed_not_left_as_none_placeholders(self):
        toc = [
            {"title": "Ch1", "physical_index": 0},
            {"title": "Ch2", "physical_index": 7},
            "garbage",
        ]
        opt = SimpleNamespace(model=None, toc_check_page_num=20)
        with patch.object(page_index_module, "process_no_toc", return_value=toc), \
             patch.object(page_index_module, "verify_toc", new=AsyncMock(return_value=(1.0, []))):
            result = await meta_processor(make_page_list(10), mode="process_no_toc", opt=opt, logger=Mock())
        self.assertEqual(result, [{"title": "Ch2", "physical_index": 7}])


if __name__ == "__main__":
    unittest.main()
