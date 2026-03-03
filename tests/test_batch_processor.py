"""
Tests for pageindex/batch_processor.py

All tests use unittest.mock and tempfile — no API key required.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch


# Fake document structure returned by a mocked page_index_main
def _fake_result(doc_name, description=None):
    result = {
        "doc_name": doc_name,
        "structure": [
            {
                "title": "Introduction",
                "start_index": 1,
                "end_index": 2,
                "node_id": "0000",
                "summary": "An overview of the document covering revenue growth.",
                "nodes": [
                    {
                        "title": "Background",
                        "start_index": 1,
                        "end_index": 1,
                        "node_id": "0001",
                        "summary": "Historical context and market analysis.",
                    }
                ],
            },
            {
                "title": "Financial Results",
                "start_index": 3,
                "end_index": 5,
                "node_id": "0002",
                "summary": "Quarterly earnings showed strong performance.",
            },
        ],
    }
    if description:
        result["doc_description"] = description
    return result


# ---------------------------------------------------------------------------
# process_batch — input validation
# ---------------------------------------------------------------------------

class TestProcessBatchValidation(unittest.TestCase):

    def test_empty_list_raises(self):
        from pageindex.batch_processor import process_batch
        with self.assertRaises(ValueError, msg="Should raise on empty list"):
            process_batch([])

    def test_non_pdf_raises(self):
        from pageindex.batch_processor import process_batch
        with self.assertRaises(ValueError, msg="Should raise on non-PDF file"):
            process_batch(["document.md"])

    def test_mixed_pdf_and_non_pdf_raises(self):
        from pageindex.batch_processor import process_batch
        with self.assertRaises(ValueError):
            process_batch(["report.pdf", "notes.docx"])


# ---------------------------------------------------------------------------
# process_batch — success path
# ---------------------------------------------------------------------------

class TestProcessBatchSuccess(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # Create two fake PDF files (content irrelevant — page_index_main is mocked)
        self.pdf1 = os.path.join(self.tmp.name, "report.pdf")
        self.pdf2 = os.path.join(self.tmp.name, "earnings.pdf")
        for p in (self.pdf1, self.pdf2):
            open(p, "w").close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_two_docs_processed_successfully(self):
        from pageindex.batch_processor import process_batch

        side_effects = [
            _fake_result("report.pdf", description="Annual report."),
            _fake_result("earnings.pdf"),
        ]

        with patch("pageindex.batch_processor.page_index_main",
                   side_effect=side_effects):
            summary = process_batch(
                [self.pdf1, self.pdf2],
                output_dir=self.tmp.name,
            )

        self.assertEqual(summary["processed"], ["report.pdf", "earnings.pdf"])
        self.assertEqual(summary["failed"], [])

    def test_structure_files_written(self):
        from pageindex.batch_processor import process_batch

        with patch("pageindex.batch_processor.page_index_main",
                   side_effect=[_fake_result("report.pdf"), _fake_result("earnings.pdf")]):
            process_batch([self.pdf1, self.pdf2], output_dir=self.tmp.name)

        self.assertTrue(os.path.isfile(os.path.join(self.tmp.name, "report_structure.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.tmp.name, "earnings_structure.json")))

    def test_kb_index_written(self):
        from pageindex.batch_processor import process_batch

        with patch("pageindex.batch_processor.page_index_main",
                   side_effect=[_fake_result("report.pdf"), _fake_result("earnings.pdf")]):
            summary = process_batch([self.pdf1, self.pdf2], output_dir=self.tmp.name)

        kb_path = summary["kb_index_path"]
        self.assertTrue(os.path.isfile(kb_path))

        with open(kb_path) as f:
            kb = json.load(f)

        self.assertEqual(kb["total_documents"], 2)
        self.assertEqual(len(kb["documents"]), 2)
        doc_names = [d["doc_name"] for d in kb["documents"]]
        self.assertIn("report.pdf", doc_names)
        self.assertIn("earnings.pdf", doc_names)
        self.assertTrue(all(d["status"] == "success" for d in kb["documents"]))

    def test_doc_description_stored_in_index(self):
        from pageindex.batch_processor import process_batch

        with patch("pageindex.batch_processor.page_index_main",
                   return_value=_fake_result("report.pdf", description="A great report.")):
            process_batch([self.pdf1], output_dir=self.tmp.name)

        with open(os.path.join(self.tmp.name, "kb_index.json")) as f:
            kb = json.load(f)

        self.assertEqual(kb["documents"][0]["doc_description"], "A great report.")

    def test_structure_json_content_is_correct(self):
        from pageindex.batch_processor import process_batch

        fake = _fake_result("report.pdf")
        with patch("pageindex.batch_processor.page_index_main", return_value=fake):
            process_batch([self.pdf1], output_dir=self.tmp.name)

        with open(os.path.join(self.tmp.name, "report_structure.json")) as f:
            saved = json.load(f)

        self.assertEqual(saved["doc_name"], "report.pdf")
        self.assertEqual(len(saved["structure"]), 2)


# ---------------------------------------------------------------------------
# process_batch — error handling
# ---------------------------------------------------------------------------

class TestProcessBatchErrors(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.good_pdf = os.path.join(self.tmp.name, "good.pdf")
        open(self.good_pdf, "w").close()
        self.missing_pdf = os.path.join(self.tmp.name, "missing.pdf")  # not created

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_goes_to_failed(self):
        from pageindex.batch_processor import process_batch

        with patch("pageindex.batch_processor.page_index_main",
                   return_value=_fake_result("good.pdf")):
            summary = process_batch(
                [self.good_pdf, self.missing_pdf],
                output_dir=self.tmp.name,
            )

        self.assertIn("good.pdf", summary["processed"])
        self.assertEqual(len(summary["failed"]), 1)
        self.assertIn("missing.pdf", summary["failed"][0]["doc"])
        self.assertEqual(summary["failed"][0]["error"], "File not found.")

    def test_processing_exception_goes_to_failed(self):
        from pageindex.batch_processor import process_batch

        with patch("pageindex.batch_processor.page_index_main",
                   side_effect=RuntimeError("LLM error")):
            summary = process_batch([self.good_pdf], output_dir=self.tmp.name)

        self.assertEqual(summary["processed"], [])
        self.assertEqual(len(summary["failed"]), 1)
        self.assertIn("LLM error", summary["failed"][0]["error"])

    def test_kb_index_still_written_on_partial_failure(self):
        from pageindex.batch_processor import process_batch

        good_pdf2 = os.path.join(self.tmp.name, "good2.pdf")
        open(good_pdf2, "w").close()

        with patch("pageindex.batch_processor.page_index_main",
                   side_effect=[_fake_result("good.pdf"), RuntimeError("crash")]):
            summary = process_batch(
                [self.good_pdf, good_pdf2],
                output_dir=self.tmp.name,
            )

        self.assertTrue(os.path.isfile(summary["kb_index_path"]))
        with open(summary["kb_index_path"]) as f:
            kb = json.load(f)
        # Only the successful doc appears in the index
        self.assertEqual(kb["total_documents"], 1)
        self.assertEqual(len(summary["processed"]), 1)
        self.assertEqual(len(summary["failed"]), 1)

    def test_output_dir_is_created_if_absent(self):
        from pageindex.batch_processor import process_batch

        new_dir = os.path.join(self.tmp.name, "new_subdir")
        self.assertFalse(os.path.exists(new_dir))

        with patch("pageindex.batch_processor.page_index_main",
                   return_value=_fake_result("good.pdf")):
            process_batch([self.good_pdf], output_dir=new_dir)

        self.assertTrue(os.path.isdir(new_dir))


# ---------------------------------------------------------------------------
# KnowledgeBaseSearch — setup helpers
# ---------------------------------------------------------------------------

def _write_kb(tmp_dir, docs):
    """Write fake structure files + kb_index.json into tmp_dir.

    docs: list of (doc_name, structure_dict)
    """
    kb_docs = []
    for doc_name, structure in docs:
        stem = os.path.splitext(doc_name)[0]
        out_file = f"{stem}_structure.json"
        with open(os.path.join(tmp_dir, out_file), "w") as f:
            json.dump({"doc_name": doc_name, "structure": structure}, f)
        kb_docs.append({
            "doc_name": doc_name,
            "doc_description": "",
            "output_file": out_file,
            "status": "success",
        })

    kb_index = {
        "created_at": "2026-03-01T00:00:00+00:00",
        "total_documents": len(kb_docs),
        "documents": kb_docs,
    }
    kb_path = os.path.join(tmp_dir, "kb_index.json")
    with open(kb_path, "w") as f:
        json.dump(kb_index, f)
    return kb_path


_STRUCTURE_A = [
    {
        "title": "Revenue Growth",
        "start_index": 1,
        "end_index": 3,
        "node_id": "0000",
        "summary": "Quarterly revenue increased by 20% driven by new markets.",
        "nodes": [
            {
                "title": "Market Expansion",
                "start_index": 2,
                "end_index": 3,
                "node_id": "0001",
                "summary": "Expansion into Asia and Europe.",
            }
        ],
    }
]

_STRUCTURE_B = [
    {
        "title": "Risk Factors",
        "start_index": 1,
        "end_index": 2,
        "node_id": "0000",
        "summary": "Operational and market risks are discussed.",
    }
]


# ---------------------------------------------------------------------------
# KnowledgeBaseSearch — initialisation
# ---------------------------------------------------------------------------

class TestKnowledgeBaseSearchInit(unittest.TestCase):

    def test_raises_if_index_missing(self):
        from pageindex.batch_processor import KnowledgeBaseSearch
        with self.assertRaises(FileNotFoundError):
            KnowledgeBaseSearch("/nonexistent/kb_index.json")

    def test_loads_successfully(self):
        from pageindex.batch_processor import KnowledgeBaseSearch
        with tempfile.TemporaryDirectory() as tmp:
            kb_path = _write_kb(tmp, [("doc_a.pdf", _STRUCTURE_A)])
            kb = KnowledgeBaseSearch(kb_path)
            self.assertIsNotNone(kb)


# ---------------------------------------------------------------------------
# KnowledgeBaseSearch.list_documents
# ---------------------------------------------------------------------------

class TestListDocuments(unittest.TestCase):

    def test_returns_all_successful_docs(self):
        from pageindex.batch_processor import KnowledgeBaseSearch
        with tempfile.TemporaryDirectory() as tmp:
            kb_path = _write_kb(tmp, [
                ("doc_a.pdf", _STRUCTURE_A),
                ("doc_b.pdf", _STRUCTURE_B),
            ])
            kb = KnowledgeBaseSearch(kb_path)
            names = kb.list_documents()
        self.assertIn("doc_a.pdf", names)
        self.assertIn("doc_b.pdf", names)
        self.assertEqual(len(names), 2)

    def test_excludes_failed_docs(self):
        from pageindex.batch_processor import KnowledgeBaseSearch
        with tempfile.TemporaryDirectory() as tmp:
            # Write one good doc manually, then inject a failed entry into index
            kb_path = _write_kb(tmp, [("doc_a.pdf", _STRUCTURE_A)])
            with open(kb_path) as f:
                kb_data = json.load(f)
            kb_data["documents"].append({
                "doc_name": "bad.pdf",
                "doc_description": "",
                "output_file": "bad_structure.json",
                "status": "failed",
            })
            with open(kb_path, "w") as f:
                json.dump(kb_data, f)

            kb = KnowledgeBaseSearch(kb_path)
            names = kb.list_documents()

        self.assertNotIn("bad.pdf", names)
        self.assertIn("doc_a.pdf", names)


# ---------------------------------------------------------------------------
# KnowledgeBaseSearch.get_document
# ---------------------------------------------------------------------------

class TestGetDocument(unittest.TestCase):

    def test_returns_correct_structure(self):
        from pageindex.batch_processor import KnowledgeBaseSearch
        with tempfile.TemporaryDirectory() as tmp:
            kb_path = _write_kb(tmp, [("doc_a.pdf", _STRUCTURE_A)])
            kb = KnowledgeBaseSearch(kb_path)
            doc = kb.get_document("doc_a.pdf")

        self.assertEqual(doc["doc_name"], "doc_a.pdf")
        self.assertEqual(len(doc["structure"]), 1)
        self.assertEqual(doc["structure"][0]["title"], "Revenue Growth")

    def test_raises_key_error_for_unknown_doc(self):
        from pageindex.batch_processor import KnowledgeBaseSearch
        with tempfile.TemporaryDirectory() as tmp:
            kb_path = _write_kb(tmp, [("doc_a.pdf", _STRUCTURE_A)])
            kb = KnowledgeBaseSearch(kb_path)
            with self.assertRaises(KeyError):
                kb.get_document("nonexistent.pdf")


# ---------------------------------------------------------------------------
# KnowledgeBaseSearch.search
# ---------------------------------------------------------------------------

class TestSearch(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.kb_path = _write_kb(self.tmp.name, [
            ("doc_a.pdf", _STRUCTURE_A),
            ("doc_b.pdf", _STRUCTURE_B),
        ])

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_query_raises(self):
        from pageindex.batch_processor import KnowledgeBaseSearch
        kb = KnowledgeBaseSearch(self.kb_path)
        with self.assertRaises(ValueError):
            kb.search("")
        with self.assertRaises(ValueError):
            kb.search("   ")

    def test_title_match_found(self):
        from pageindex.batch_processor import KnowledgeBaseSearch
        kb = KnowledgeBaseSearch(self.kb_path)
        results = kb.search("Revenue Growth")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["title"], "Revenue Growth")
        self.assertIn("title", results[0]["match_on"])

    def test_summary_match_found(self):
        from pageindex.batch_processor import KnowledgeBaseSearch
        kb = KnowledgeBaseSearch(self.kb_path)
        # "markets" appears only in a summary, not in a title
        results = kb.search("new markets")
        self.assertTrue(len(results) > 0)
        self.assertIn("summary", results[0]["match_on"])

    def test_case_insensitive(self):
        from pageindex.batch_processor import KnowledgeBaseSearch
        kb = KnowledgeBaseSearch(self.kb_path)
        upper = kb.search("REVENUE GROWTH")
        lower = kb.search("revenue growth")
        self.assertEqual(len(upper), len(lower))

    def test_no_results_returns_empty_list(self):
        from pageindex.batch_processor import KnowledgeBaseSearch
        kb = KnowledgeBaseSearch(self.kb_path)
        results = kb.search("xyzzy_no_match_ever")
        self.assertEqual(results, [])

    def test_title_match_scores_higher_than_summary_only(self):
        from pageindex.batch_processor import KnowledgeBaseSearch
        kb = KnowledgeBaseSearch(self.kb_path)
        # "expansion" appears in both title ("Market Expansion") and summary
        results = kb.search("expansion")
        title_hits = [r for r in results if "title" in r["match_on"]]
        summary_only_hits = [r for r in results if r["match_on"] == ["summary"]]
        if title_hits and summary_only_hits:
            self.assertGreater(title_hits[0]["score"], summary_only_hits[0]["score"])

    def test_top_k_limits_results(self):
        from pageindex.batch_processor import KnowledgeBaseSearch
        kb = KnowledgeBaseSearch(self.kb_path)
        # "e" appears everywhere — many matches expected
        results = kb.search("e", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_result_contains_required_fields(self):
        from pageindex.batch_processor import KnowledgeBaseSearch
        kb = KnowledgeBaseSearch(self.kb_path)
        results = kb.search("revenue")
        self.assertTrue(len(results) > 0)
        for r in results:
            self.assertIn("doc_name", r)
            self.assertIn("title", r)
            self.assertIn("start_index", r)
            self.assertIn("end_index", r)
            self.assertIn("score", r)
            self.assertIn("match_on", r)

    def test_searches_across_multiple_documents(self):
        from pageindex.batch_processor import KnowledgeBaseSearch
        kb = KnowledgeBaseSearch(self.kb_path)
        # "risk" only in doc_b, "revenue" only in doc_a
        risk_results = kb.search("risk")
        revenue_results = kb.search("revenue")
        risk_docs = {r["doc_name"] for r in risk_results}
        revenue_docs = {r["doc_name"] for r in revenue_results}
        self.assertIn("doc_b.pdf", risk_docs)
        self.assertIn("doc_a.pdf", revenue_docs)
        self.assertNotIn("doc_b.pdf", revenue_docs)

    def test_nested_nodes_are_searchable(self):
        from pageindex.batch_processor import KnowledgeBaseSearch
        kb = KnowledgeBaseSearch(self.kb_path)
        # "Market Expansion" is a child node inside doc_a
        results = kb.search("Market Expansion")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["title"], "Market Expansion")


if __name__ == "__main__":
    unittest.main(verbosity=2)
