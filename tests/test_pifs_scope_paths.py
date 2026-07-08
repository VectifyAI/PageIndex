import json
import tempfile
import unittest
from types import SimpleNamespace

from tests.pifs_markdown_fixture import register_markdown


class BrowseBackend:
    def __init__(self, document_ids, *, file_refs_by_document_id=None):
        self.document_ids = list(document_ids)
        self.file_refs_by_document_id = dict(file_refs_by_document_id or {})

    def available_channels(self):
        return ("summary",)

    def search_channel(self, channel, query, *, limit=10, filters=None):
        file_ref_filter = set((filters or {}).get("file_ref") or [])
        document_ids = self.document_ids
        if file_ref_filter:
            document_ids = [
                document_id
                for document_id in document_ids
                if self.file_refs_by_document_id.get(document_id) in file_ref_filter
            ]
        return [
            SimpleNamespace(
                document_id=document_id,
                snippet=f"{channel} candidate {rank}: {query}",
                score=1.0 - rank * 0.01,
                sources=[{"channel": channel, "rank": rank}],
            )
            for rank, document_id in enumerate(document_ids[:limit], 1)
        ]


def _payload(output):
    return json.loads(output)


class PIFSScopePathTest(unittest.TestCase):
    def test_tree_browse_and_stat_accept_metadata_scope_paths(self):
        from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path

            root = Path(tmp)
            filesystem = PageIndexFileSystem(workspace=root / "workspace")
            refs = {
                "doc_aapl_2024": register_markdown(
                    filesystem,
                    root,
                    "doc_aapl_2024",
                    "/documents/sec-filings",
                    title="aapl-2024.md",
                    metadata={"year": 2024, "ticker": "AAPL", "doc_type": "10-K"},
                ),
                "doc_aapl_2023": register_markdown(
                    filesystem,
                    root,
                    "doc_aapl_2023",
                    "/documents/sec-filings",
                    title="aapl-2023.md",
                    metadata={"year": 2023, "ticker": "AAPL", "doc_type": "10-K"},
                ),
                "doc_msft_2024": register_markdown(
                    filesystem,
                    root,
                    "doc_msft_2024",
                    "/documents/sec-filings",
                    title="msft-2024.md",
                    metadata={"year": 2024, "ticker": "MSFT", "doc_type": "10-K"},
                ),
            }
            filesystem.semantic_retrieval_backend = BrowseBackend(
                ["doc_msft_2024", "doc_aapl_2024", "doc_aapl_2023"],
                file_refs_by_document_id=refs,
            )
            executor = PIFSCommandExecutor(filesystem)

            tree_root = _payload(executor.execute("tree /documents -L 1"))
            self.assertTrue(tree_root["success"])

            root_folders = tree_root["data"]["tree"]["folders"]
            self.assertTrue(any(item["path"] == "/documents/sec-filings" for item in root_folders))
            self.assertEqual(
                [item["name"] for item in root_folders if item["type"] == "metadata_axis"],
                ["@doc_type", "@ticker", "@year"],
            )

            tree_year = _payload(executor.execute("tree /documents/@year"))
            self.assertTrue(tree_year["success"])
            self.assertEqual(
                [item["path"] for item in tree_year["data"]["tree"]["folders"]],
                ["/documents/@year/2024", "/documents/@year/2023"],
            )

            browse = _payload(
                executor.execute('browse /documents/@year/2024/@ticker/AAPL "risk factors"')
            )
            self.assertTrue(browse["success"])
            self.assertEqual(
                [item["path"] for item in browse["data"]["documents"]],
                ["/documents/@year/2024/@ticker/AAPL/aapl-2024.md"],
            )
            self.assertEqual(browse["data"]["scope"]["path"], "/documents/@year/2024/@ticker/AAPL")
            self.assertEqual(browse["data"]["scope"]["folder_path"], "/documents")
            self.assertEqual(
                browse["data"]["scope"]["metadata_filter"],
                {"year": "2024", "ticker": "AAPL"},
            )

            locator = browse["data"]["documents"][0]["path"]
            stat = _payload(executor.execute(f"stat {locator}"))
            self.assertTrue(stat["success"])
            self.assertEqual(stat["data"]["document"]["path"], locator)
            self.assertEqual(stat["data"]["document"]["title"], "aapl-2024.md")

    def test_metadata_scope_file_locators_round_trip_to_file_commands(self):
        from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path

            root = Path(tmp)
            filesystem = PageIndexFileSystem(workspace=root / "workspace")
            refs = {
                "doc_3m_2018": register_markdown(
                    filesystem,
                    root,
                    "doc_3m_2018",
                    "/docs/filings",
                    title="3M_2018_10K.md",
                    text="cash flow evidence\nnet sales evidence",
                    metadata={"company": "3M"},
                ),
                "doc_other": register_markdown(
                    filesystem,
                    root,
                    "doc_other",
                    "/docs/filings",
                    title="Other_2018_10K.md",
                    text="other cash flow",
                    metadata={"company": "Other"},
                ),
            }
            filesystem.semantic_retrieval_backend = BrowseBackend(
                ["doc_3m_2018", "doc_other"],
                file_refs_by_document_id=refs,
            )
            executor = PIFSCommandExecutor(filesystem)

            tree = _payload(executor.execute("tree /docs/@company/3M"))
            self.assertTrue(tree["success"])
            self.assertEqual(
                [item["path"] for item in tree["data"]["tree"]["files"]],
                ["/docs/@company/3M/3M_2018_10K.md"],
            )

            browse = _payload(executor.execute('browse /docs/@company/3M "cash flow"'))
            self.assertTrue(browse["success"])
            locator = browse["data"]["documents"][0]["path"]
            self.assertEqual(locator, "/docs/@company/3M/3M_2018_10K.md")

            stat = _payload(executor.execute(f"stat {locator}"))
            self.assertTrue(stat["success"])
            self.assertEqual(stat["data"]["document"]["path"], locator)
            self.assertEqual(stat["data"]["document"]["title"], "3M_2018_10K.md")

            structure = _payload(executor.execute(f"cat {locator} --structure"))
            self.assertTrue(structure["success"])
            self.assertEqual(structure["data"]["document"]["path"], locator)
            self.assertTrue(structure["data"]["structure"])

            grep = _payload(executor.execute(f"grep cash {locator}"))
            self.assertTrue(grep["success"])
            self.assertEqual(grep["data"]["document"]["path"], locator)
            self.assertEqual(grep["data"]["matches"][0]["line"], 1)

            scope_only = _payload(executor.execute("cat /docs/@company/3M --structure"))
            self.assertFalse(scope_only["success"])
            self.assertIn("scope, not a file locator", scope_only["error"]["message"])
            self.assertIn("tree /docs/@company/3M", scope_only["error"]["message"])
            self.assertIn('browse /docs/@company/3M "<query>"', scope_only["error"]["message"])

    def test_tree_depth_does_not_bypass_file_pagination(self):
        from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path

            root = Path(tmp)
            filesystem = PageIndexFileSystem(workspace=root / "workspace")
            for index in range(55):
                register_markdown(
                    filesystem,
                    root,
                    f"doc_{index:02d}",
                    "/docs",
                    title=f"report-{index:02d}.md",
                    metadata={"company": "3M"},
                )
            executor = PIFSCommandExecutor(filesystem)

            first_page = _payload(executor.execute("tree /docs/@company/3M -L 100"))
            self.assertTrue(first_page["success"])
            self.assertEqual(len(first_page["data"]["tree"]["files"]), 50)
            self.assertEqual(
                first_page["data"]["pagination"],
                {"page": 1, "page_size": 50, "has_more": True, "next_page": 2},
            )

            second_page = _payload(executor.execute("tree /docs/@company/3M -L 100 --page 2"))
            self.assertTrue(second_page["success"])
            self.assertEqual(len(second_page["data"]["tree"]["files"]), 5)
            self.assertEqual(
                second_page["data"]["pagination"],
                {"page": 2, "page_size": 50, "has_more": False, "next_page": None},
            )

    def test_duplicate_file_leaves_in_scope_return_disambiguated_locators(self):
        from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path

            root = Path(tmp)
            filesystem = PageIndexFileSystem(workspace=root / "workspace")
            refs = {
                "doc_a": register_markdown(
                    filesystem,
                    root,
                    "doc_a",
                    "/docs/a",
                    title="report.md",
                    metadata={"company": "3M"},
                ),
                "doc_b": register_markdown(
                    filesystem,
                    root,
                    "doc_b",
                    "/docs/b",
                    title="report.md",
                    metadata={"company": "3M"},
                ),
            }
            filesystem.semantic_retrieval_backend = BrowseBackend(
                ["doc_a", "doc_b"],
                file_refs_by_document_id=refs,
            )
            executor = PIFSCommandExecutor(filesystem)

            tree = _payload(executor.execute("tree /docs/@company/3M"))
            self.assertTrue(tree["success"])
            paths = [item["path"] for item in tree["data"]["tree"]["files"]]
            self.assertEqual(len(paths), 2)
            self.assertEqual(len(set(paths)), 2)

            for path in paths:
                stat = _payload(executor.execute(f"stat {path}"))
                self.assertTrue(stat["success"])
                self.assertEqual(stat["data"]["document"]["path"], path)

            browse = _payload(executor.execute('browse /docs/@company/3M "report"'))
            self.assertTrue(browse["success"])
            browse_paths = [item["path"] for item in browse["data"]["documents"]]
            self.assertEqual(len(browse_paths), 2)
            self.assertEqual(len(set(browse_paths)), 2)

    def test_scope_paths_reject_duplicate_and_unknown_metadata_fields(self):
        from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path

            root = Path(tmp)
            filesystem = PageIndexFileSystem(workspace=root / "workspace")
            register_markdown(
                filesystem,
                root,
                "doc_aapl_2024",
                "/documents",
                metadata={"year": 2024, "ticker": "AAPL"},
            )
            executor = PIFSCommandExecutor(filesystem)

            duplicate = _payload(executor.execute("tree /documents/@ticker/AAPL/@ticker/MSFT"))
            self.assertFalse(duplicate["success"])
            self.assertIn("can appear only once", duplicate["error"]["message"])

            unknown = _payload(executor.execute("tree /documents/@sector"))
            self.assertFalse(unknown["success"])
            self.assertIn("Unknown metadata axis", unknown["error"]["message"])

    def test_tree_metadata_value_pagination_uses_page_flag(self):
        from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path

            root = Path(tmp)
            filesystem = PageIndexFileSystem(workspace=root / "workspace")
            for index in range(55):
                ticker = f"T{index:02d}"
                register_markdown(
                    filesystem,
                    root,
                    f"doc_{ticker}",
                    "/documents",
                    title=f"{ticker}.md",
                    metadata={"ticker": ticker},
                )

            executor = PIFSCommandExecutor(filesystem)

            first_page = _payload(executor.execute("tree /documents/@ticker"))
            self.assertTrue(first_page["success"])
            self.assertEqual(len(first_page["data"]["tree"]["folders"]), 50)
            self.assertEqual(
                first_page["data"]["pagination"],
                {"page": 1, "page_size": 50, "has_more": True, "next_page": 2},
            )
            self.assertEqual(
                [item["value"] for item in first_page["data"]["tree"]["folders"][:3]],
                ["T00", "T01", "T02"],
            )

            second_page = _payload(executor.execute("tree /documents/@ticker --page 2"))
            self.assertTrue(second_page["success"])
            self.assertEqual(
                [item["value"] for item in second_page["data"]["tree"]["folders"]],
                ["T50", "T51", "T52", "T53", "T54"],
            )
            self.assertEqual(
                second_page["data"]["pagination"],
                {"page": 2, "page_size": 50, "has_more": False, "next_page": None},
            )

    def test_tree_child_paths_keep_active_metadata_scope(self):
        from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path

            root = Path(tmp)
            filesystem = PageIndexFileSystem(workspace=root / "workspace")
            register_markdown(
                filesystem,
                root,
                "doc_sec_2024",
                "/documents/sec",
                title="sec-2024.md",
                metadata={"year": 2024, "ticker": "AAPL"},
            )
            register_markdown(
                filesystem,
                root,
                "doc_sec_2023",
                "/documents/sec",
                title="sec-2023.md",
                metadata={"year": 2023, "ticker": "AAPL"},
            )
            register_markdown(
                filesystem,
                root,
                "doc_other_2024",
                "/documents/other",
                title="other-2024.md",
                metadata={"year": 2024, "ticker": "MSFT"},
            )

            executor = PIFSCommandExecutor(filesystem)

            tree = _payload(executor.execute("tree /documents/@year/2024 -L 1"))
            self.assertTrue(tree["success"])
            sec_node = next(
                item
                for item in tree["data"]["tree"]["folders"]
                if item["name"] == "sec"
            )
            self.assertEqual(sec_node["path"], "/documents/sec/@year/2024")
            self.assertEqual(sec_node["file_count"], 1)

            scope_stat = _payload(executor.execute("stat /documents/sec/@year/2024"))
            self.assertFalse(scope_stat["success"])
            self.assertIn("scope, not a file locator", scope_stat["error"]["message"])

            scoped_tree = _payload(executor.execute("tree /documents/sec/@year/2024"))
            self.assertTrue(scoped_tree["success"])
            locator = scoped_tree["data"]["tree"]["files"][0]["path"]
            self.assertEqual(locator, "/documents/sec/@year/2024/sec-2024.md")

            stat = _payload(executor.execute(f"stat {locator}"))
            self.assertTrue(stat["success"])
            self.assertEqual(stat["data"]["document"]["path"], locator)

    def test_core_browse_semantic_files_accepts_metadata_scoped_path(self):
        from pageindex.filesystem import PageIndexFileSystem

        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path

            root = Path(tmp)
            filesystem = PageIndexFileSystem(workspace=root / "workspace")
            refs = {
                "doc_aapl": register_markdown(
                    filesystem,
                    root,
                    "doc_aapl",
                    "/documents/sec",
                    title="aapl.md",
                    metadata={"ticker": "AAPL"},
                ),
                "doc_msft": register_markdown(
                    filesystem,
                    root,
                    "doc_msft_direct",
                    "/documents",
                    title="msft-direct.md",
                    metadata={"ticker": "MSFT"},
                ),
                "doc_msft_descendant": register_markdown(
                    filesystem,
                    root,
                    "doc_msft_descendant",
                    "/documents/sec",
                    title="msft-descendant.md",
                    metadata={"ticker": "MSFT"},
                ),
            }
            filesystem.semantic_retrieval_backend = BrowseBackend(
                ["doc_msft_direct", "doc_msft_descendant", "doc_aapl"],
                file_refs_by_document_id=refs,
            )

            payload = filesystem.browse_semantic_files(
                "/documents/@ticker/AAPL",
                "query",
                recursive=False,
            )

            self.assertEqual(
                [item["document_id"] for item in payload["data"]],
                ["doc_aapl"],
            )
            self.assertEqual(payload["scope"], "/documents/@ticker/AAPL")

    def test_core_browse_semantic_files_prefers_in_scope_physical_membership(self):
        from pageindex.filesystem import PageIndexFileSystem

        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path

            root = Path(tmp)
            filesystem = PageIndexFileSystem(workspace=root / "workspace")
            file_ref = register_markdown(
                filesystem,
                root,
                "doc_shared",
                "/adocs",
                title="shared.md",
                metadata={"ticker": "AAPL"},
            )
            filesystem.store.attach_file_to_folder(file_ref, "/zdocs")
            filesystem.semantic_retrieval_backend = BrowseBackend(
                ["doc_shared"],
                file_refs_by_document_id={"doc_shared": file_ref},
            )

            payload = filesystem.browse_semantic_files(
                "/zdocs/@ticker/AAPL",
                "query",
                recursive=False,
            )

            self.assertEqual([item["document_id"] for item in payload["data"]], ["doc_shared"])
            self.assertEqual(payload["data"][0]["folder_path"], "/zdocs")
            self.assertTrue(payload["data"][0]["path"].startswith("/zdocs/"))

    def test_one_document_is_reachable_through_multiple_metadata_virtual_paths(self):
        from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path

            root = Path(tmp)
            filesystem = PageIndexFileSystem(workspace=root / "workspace")
            file_ref = register_markdown(
                filesystem,
                root,
                "doc_contract",
                "/documents/contracts",
                title="apple-renewal.md",
                metadata={"vendor": "Apple", "year": 2024, "doc_type": "contract"},
            )
            filesystem.semantic_retrieval_backend = BrowseBackend(
                ["doc_contract"],
                file_refs_by_document_id={"doc_contract": file_ref},
            )
            executor = PIFSCommandExecutor(filesystem)

            tree_root = _payload(executor.execute("tree /documents -L 1"))
            self.assertTrue(tree_root["success"])
            root_nodes = tree_root["data"]["tree"]["folders"]
            self.assertIn(
                "/documents/@vendor",
                [item["path"] for item in root_nodes if item["type"] == "metadata_axis"],
            )

            tree_vendor = _payload(executor.execute("tree /documents/@vendor"))
            self.assertTrue(tree_vendor["success"])
            self.assertEqual(
                [
                    item["type"]
                    for item in tree_vendor["data"]["tree"]["folders"]
                    if item["path"] == "/documents/@vendor/Apple"
                ],
                ["metadata_value"],
            )

            vendor = _payload(
                executor.execute('browse /documents/@vendor/Apple "renewal contract"')
            )
            year = _payload(executor.execute('browse /documents/@year/2024 "renewal contract"'))

            self.assertTrue(vendor["success"])
            self.assertTrue(year["success"])
            self.assertEqual(
                [item["path"] for item in vendor["data"]["documents"]],
                ["/documents/@vendor/Apple/apple-renewal.md"],
            )
            self.assertEqual(
                [item["path"] for item in year["data"]["documents"]],
                ["/documents/@year/2024/apple-renewal.md"],
            )
            self.assertEqual(
                vendor["data"]["scope"]["metadata_filter"],
                {"vendor": "Apple"},
            )
            self.assertEqual(
                year["data"]["scope"]["metadata_filter"],
                {"year": "2024"},
            )
