import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from pageindex import PageIndex, PageIndexConfig


class DummyOpt:
    def __init__(self, data):
        self.model = data.get("model")
        self.if_add_node_summary = data.get("if_add_node_summary", "no")
        self.if_add_doc_description = data.get("if_add_doc_description", "no")
        self.if_add_node_text = data.get("if_add_node_text", "no")
        self.if_add_node_id = data.get("if_add_node_id", "yes")


class TestPageIndexUnit(unittest.TestCase):
    def _temp_file(self, suffix: str) -> str:
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return path

    def _stub_modules(self):
        calls = {
            "config_load_input": None,
            "page_index_main": None,
            "md_to_tree": None,
        }

        page_index_module = types.ModuleType("pageindex.page_index")

        def fake_page_index_main(doc, opt):
            calls["page_index_main"] = {"doc": doc, "opt": opt}
            return {"doc_name": "stub-pdf", "structure": []}

        page_index_module.page_index_main = fake_page_index_main

        page_index_md_module = types.ModuleType("pageindex.page_index_md")

        async def fake_md_to_tree(**kwargs):
            calls["md_to_tree"] = kwargs
            return {"doc_name": "stub-md", "line_count": 1, "structure": []}

        page_index_md_module.md_to_tree = fake_md_to_tree

        utils_module = types.ModuleType("pageindex.utils")

        class FakeConfigLoader:
            def load(self, data):
                calls["config_load_input"] = data
                return DummyOpt(data)

        utils_module.ConfigLoader = FakeConfigLoader

        return calls, {
            "pageindex.page_index": page_index_module,
            "pageindex.page_index_md": page_index_md_module,
            "pageindex.utils": utils_module,
        }

    def test_pdf_run_dispatches_and_maps_options(self):
        pdf_path = self._temp_file(".pdf")
        calls, stubbed = self._stub_modules()
        config = PageIndexConfig(
            pdf_path=pdf_path,
            model="test-model",
            toc_check_pages=12,
            max_pages_per_node=7,
            max_tokens_per_node=1234,
            add_node_id=True,
            add_node_summary=False,
            add_doc_description=True,
            add_node_text=False,
        )

        with mock.patch.dict(sys.modules, stubbed):
            page_index = PageIndex(config)
            result = page_index.run()

        self.assertEqual(result["doc_name"], "stub-pdf")
        self.assertEqual(calls["page_index_main"]["doc"], pdf_path)
        self.assertEqual(calls["config_load_input"]["model"], "test-model")
        self.assertEqual(calls["config_load_input"]["toc_check_page_num"], 12)
        self.assertEqual(calls["config_load_input"]["max_page_num_each_node"], 7)
        self.assertEqual(calls["config_load_input"]["max_token_num_each_node"], 1234)
        self.assertEqual(calls["config_load_input"]["if_add_node_id"], "yes")
        self.assertEqual(calls["config_load_input"]["if_add_node_summary"], "no")
        self.assertEqual(calls["config_load_input"]["if_add_doc_description"], "yes")
        self.assertEqual(calls["config_load_input"]["if_add_node_text"], "no")

    def test_markdown_run_dispatches_and_maps_options(self):
        md_path = self._temp_file(".md")
        calls, stubbed = self._stub_modules()
        config = PageIndexConfig(
            md_path=md_path,
            model="test-model",
            if_thinning=True,
            thinning_threshold=777,
            summary_token_threshold=333,
            add_node_id=False,
            add_node_summary=True,
            add_doc_description=False,
            add_node_text=True,
        )

        with mock.patch.dict(sys.modules, stubbed):
            page_index = PageIndex(config)
            result = page_index.run()

        self.assertEqual(result["doc_name"], "stub-md")
        self.assertEqual(calls["config_load_input"]["model"], "test-model")
        self.assertEqual(calls["config_load_input"]["if_add_node_id"], "no")
        self.assertEqual(calls["config_load_input"]["if_add_node_summary"], "yes")
        self.assertEqual(calls["config_load_input"]["if_add_doc_description"], "no")
        self.assertEqual(calls["config_load_input"]["if_add_node_text"], "yes")
        self.assertEqual(calls["md_to_tree"]["md_path"], md_path)
        self.assertTrue(calls["md_to_tree"]["if_thinning"])
        self.assertEqual(calls["md_to_tree"]["min_token_threshold"], 777)
        self.assertEqual(calls["md_to_tree"]["summary_token_threshold"], 333)

    def test_run_and_save_writes_expected_output_file(self):
        md_path = self._temp_file(".md")
        output_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: os.path.isdir(output_dir) and os.rmdir(output_dir))
        calls, stubbed = self._stub_modules()
        config = PageIndexConfig(md_path=md_path, output_dir=output_dir)

        with mock.patch.dict(sys.modules, stubbed):
            page_index = PageIndex(config)
            output_file = page_index.run_and_save()

        self.addCleanup(lambda: os.path.exists(output_file) and os.remove(output_file))
        self.assertTrue(os.path.exists(output_file))
        self.assertTrue(output_file.endswith("_structure.json"))
        self.assertIn(os.path.basename(output_dir), output_file)


if __name__ == "__main__":
    unittest.main()
