import json
import tempfile
from pathlib import Path

import pytest


def write_pageindex_client_doc(workspace: Path, doc_id: str, doc: dict) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / f"{doc_id}.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    meta = {
        doc_id: {
            "type": doc.get("type", ""),
            "doc_name": doc.get("doc_name", ""),
            "doc_description": doc.get("doc_description", ""),
            "path": doc.get("path", ""),
        }
    }
    if doc.get("type") == "pdf":
        meta[doc_id]["page_count"] = doc.get("page_count")
    elif doc.get("type") == "md":
        meta[doc_id]["line_count"] = doc.get("line_count")
    (workspace / "_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class RecordingMetadataGenerator:
    values = {
        "summary": "Generated retrieval summary.",
        "doc_type": "technical_note",
        "domain": "documentation",
        "topic": "pageindex extraction",
    }

    def __init__(self):
        self.calls = []

    def generate(self, request, *, fields):
        self.calls.append((request, list(fields)))
        return {field: self.values[field] for field in fields if field in self.values}


def test_pageindex_structure_options_report_failed_register_build(monkeypatch):
    from pageindex import PageIndexClient
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "report.md"
        source.write_text("# Report\n\nCached structure is not built yet.", encoding="utf-8")
        filesystem = PageIndexFileSystem(workspace=Path(tmp) / "workspace")

        def fail_index(*args, **kwargs):
            raise RuntimeError("index failed")

        monkeypatch.setattr(PageIndexClient, "index", fail_index)
        filesystem.register_file(
            storage_uri=source.as_uri(),
            source_path="docs/report.md",
            external_id="dsid_structural_missing",
            title="Structural report",
            content=source.read_text(encoding="utf-8"),
        )
        executor = PIFSCommandExecutor(filesystem, json_output=True)

        structure = json.loads(executor.execute("cat dsid_structural_missing --structure"))
        node = json.loads(executor.execute("cat dsid_structural_missing --node 0001"))
        pages = json.loads(executor.execute("cat dsid_structural_missing --page 1-2"))
        stat = json.loads(executor.execute("stat dsid_structural_missing"))

        assert structure["data"]["mode"] == "structure"
        assert structure["data"]["available"] is False
        assert structure["data"]["status"] == "failed"
        assert "PageIndexClient workspace" in structure["data"]["message"]
        assert stat["data"]["pageindex_tree_status"] == "failed"

        assert node["data"]["mode"] == "node"
        assert node["data"]["available"] is False
        assert node["data"]["node_id"] == "0001"

        assert pages["data"]["mode"] == "page"
        assert pages["data"]["available"] is False
        assert pages["data"]["pages"] == "1-2"

        assert "cp" not in executor.allowed_commands()
        assert "mkdir" not in executor.allowed_commands()


def test_register_pdf_markdown_uses_pageindex_extracted_text_for_metadata_and_fts(monkeypatch):
    from pageindex import PageIndexClient
    from pageindex.filesystem import PageIndexFileSystem

    def fake_index(self, file_path, mode="auto"):
        suffix = Path(file_path).suffix.lower()
        doc_id = f"doc_{suffix.lstrip('.')}"
        if suffix == ".pdf":
            doc = {
                "id": doc_id,
                "type": "pdf",
                "path": str(Path(file_path).resolve()),
                "doc_name": "report.pdf",
                "doc_description": "",
                "page_count": 2,
                "structure": [{"title": "Report", "node_id": "0001", "nodes": []}],
                "pages": [
                    {"page": 1, "content": "PageIndex PDF extracted alpha text."},
                    {"page": 2, "content": "Second PageIndex PDF extracted beta text."},
                ],
            }
        else:
            doc = {
                "id": doc_id,
                "type": "md",
                "path": str(Path(file_path).resolve()),
                "doc_name": "notes",
                "doc_description": "",
                "line_count": 3,
                "structure": [
                    {
                        "title": "Notes",
                        "node_id": "0001",
                        "line_num": 1,
                        "text": "# Notes\n\nPageIndex Markdown extracted gamma text.",
                        "nodes": [],
                    }
                ],
            }
        write_pageindex_client_doc(self.workspace, doc_id, doc)
        self.documents[doc_id] = doc
        return doc_id

    monkeypatch.setattr(PageIndexClient, "index", fake_index)
    with tempfile.TemporaryDirectory() as tmp:
        source_pdf = Path(tmp) / "report.pdf"
        source_md = Path(tmp) / "notes.md"
        source_pdf.write_bytes(b"%PDF-1.4\n% test fixture\n")
        source_md.write_text("# Notes\n\nCaller markdown content", encoding="utf-8")
        generator = RecordingMetadataGenerator()
        filesystem = PageIndexFileSystem(
            workspace=Path(tmp) / "workspace",
            metadata_generator=generator,
        )

        filesystem.register_file(
            storage_uri=source_pdf.as_uri(),
            source_path="docs/report.pdf",
            external_id="dsid_pdf_extracted",
            title="PDF extracted",
            content="CALLER PDF CONTENT MUST NOT REACH GENERATOR",
        )
        filesystem.register_file(
            storage_uri=source_md.as_uri(),
            source_path="docs/notes.md",
            external_id="dsid_md_extracted",
            title="Markdown extracted",
            content="CALLER MD CONTENT MUST NOT REACH GENERATOR",
        )

        pdf_request = generator.calls[0][0]
        md_request = generator.calls[1][0]
        pdf_stat = filesystem.store.file_info("dsid_pdf_extracted")
        md_stat = filesystem.store.file_info("dsid_md_extracted")

        assert "PageIndex PDF extracted alpha text" in pdf_request.text
        assert "Second PageIndex PDF extracted beta text" in pdf_request.text
        assert "CALLER PDF CONTENT" not in pdf_request.text
        assert "PageIndex Markdown extracted gamma text" in md_request.text
        assert "CALLER MD CONTENT" not in md_request.text
        assert "PageIndex PDF extracted alpha text" in Path(
            pdf_stat["text_artifact_path"]
        ).read_text(encoding="utf-8")
        assert "PageIndex Markdown extracted gamma text" in Path(
            md_stat["text_artifact_path"]
        ).read_text(encoding="utf-8")
        assert [r.external_id for r in filesystem.search("alpha beta", limit=5)] == [
            "dsid_pdf_extracted"
        ]
        assert [r.external_id for r in filesystem.search("gamma", limit=5)] == [
            "dsid_md_extracted"
        ]
        assert filesystem.search("CALLER", limit=5) == []


def test_register_text_metadata_generation_keeps_caller_content_without_pageindex(monkeypatch):
    from pageindex import PageIndexClient
    from pageindex.filesystem import PageIndexFileSystem

    def fail_index(*args, **kwargs):
        raise AssertionError("PageIndexClient.index should not be called for text files")

    monkeypatch.setattr(PageIndexClient, "index", fail_index)
    with tempfile.TemporaryDirectory() as tmp:
        generator = RecordingMetadataGenerator()
        filesystem = PageIndexFileSystem(
            workspace=Path(tmp) / "workspace",
            metadata_generator=generator,
        )

        filesystem.register_file(
            storage_uri="file:///tmp/readme.txt",
            source_path="docs/readme.txt",
            external_id="dsid_text_generation",
            title="Text generation",
            content="Plain text caller content stays authoritative.",
            content_type="text/plain",
        )

        stat = filesystem.store.file_info("dsid_text_generation")

        assert generator.calls[0][0].text == "Plain text caller content stays authoritative."
        assert stat["pageindex_doc_id"] is None
        assert stat["pageindex_tree_status"] == "not_built"
        assert Path(stat["text_artifact_path"]).read_text(
            encoding="utf-8"
        ) == "Plain text caller content stays authoritative."


def test_register_pdf_markdown_cache_miss_invokes_pageindex_client_index(monkeypatch):
    from pageindex import PageIndexClient
    from pageindex.filesystem import PageIndexFileSystem

    calls: list[str] = []

    def fake_index(self, file_path, mode="auto"):
        calls.append(str(file_path))
        doc_id = f"doc_{Path(file_path).suffix.lstrip('.')}"
        doc_type = "pdf" if Path(file_path).suffix == ".pdf" else "md"
        doc = {
            "id": doc_id,
            "type": doc_type,
            "path": str(Path(file_path).resolve()),
            "doc_name": Path(file_path).name,
            "doc_description": "",
            "structure": [{"title": Path(file_path).stem, "node_id": "0001", "nodes": []}],
        }
        if doc_type == "pdf":
            doc["page_count"] = 1
            doc["pages"] = [{"page": 1, "content": "Page one text"}]
        else:
            doc["line_count"] = 1
        write_pageindex_client_doc(self.workspace, doc_id, doc)
        self.documents[doc_id] = doc
        return doc_id

    monkeypatch.setattr(PageIndexClient, "index", fake_index)
    with tempfile.TemporaryDirectory() as tmp:
        source_pdf = Path(tmp) / "report.pdf"
        source_md = Path(tmp) / "notes.md"
        source_pdf.write_bytes(b"%PDF-1.4\n% test fixture\n")
        source_md.write_text("# Notes", encoding="utf-8")
        filesystem = PageIndexFileSystem(workspace=Path(tmp) / "workspace")

        filesystem.register_file(
            storage_uri=str(source_pdf),
            source_path="docs/report.pdf",
            external_id="dsid_pdf_build",
            title="PDF build",
            content="pdf text",
        )
        filesystem.register_file(
            storage_uri=source_md.as_uri(),
            source_path="docs/notes.md",
            external_id="dsid_md_build",
            title="Markdown build",
            content=source_md.read_text(encoding="utf-8"),
        )

        pdf_stat = filesystem.store.file_info("dsid_pdf_build")
        md_stat = filesystem.store.file_info("dsid_md_build")

        assert calls == [str(source_pdf.resolve()), str(source_md.resolve())]
        assert pdf_stat["pageindex_doc_id"] == "doc_pdf"
        assert pdf_stat["pageindex_tree_status"] == "built"
        assert md_stat["pageindex_doc_id"] == "doc_md"
        assert md_stat["pageindex_tree_status"] == "built"


def test_cat_structure_page_reuses_pageindex_client_cache_without_indexing(monkeypatch):
    from pageindex import PageIndexClient
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.commands import PIFSCommandError

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "report.pdf"
        source.write_bytes(b"%PDF-1.4\n% test fixture\n")
        workspace = Path(tmp) / "workspace"
        filesystem = PageIndexFileSystem(workspace=workspace)
        write_pageindex_client_doc(
            filesystem.pageindex_client_workspace,
            "doc_cached_pdf",
            {
                "id": "doc_cached_pdf",
                "type": "pdf",
                "path": str(source.resolve()),
                "doc_name": "report.pdf",
                "doc_description": "",
                "page_count": 2,
                "structure": [
                    {
                        "title": "Introduction",
                        "node_id": "0001",
                        "text": "Intro section text",
                        "nodes": [
                            {
                                "title": "Findings",
                                "node_id": "0002",
                                "physical_index": 2,
                                "nodes": [],
                            }
                        ],
                    }
                ],
                "pages": [
                    {"page": 1, "content": "Page one text"},
                    {"page": 2, "content": "Page two text"},
                ],
            },
        )

        def fail_index(*args, **kwargs):
            raise AssertionError("PageIndexClient.index should not be called on cache hit")

        monkeypatch.setattr(PageIndexClient, "index", fail_index)
        filesystem.register_file(
            storage_uri=source.as_uri(),
            source_path="docs/report.pdf",
            external_id="dsid_structural_cached",
            title="Cached structural report",
            content="text artifact remains available for grep, not cat --all",
        )
        executor = PIFSCommandExecutor(filesystem, json_output=True)

        structure = json.loads(executor.execute("cat dsid_structural_cached --structure"))
        pages = json.loads(executor.execute("cat dsid_structural_cached --page 1-2"))
        stat = json.loads(executor.execute("stat dsid_structural_cached"))

        assert structure["data"]["available"] is True
        assert structure["data"]["pageindex_doc_id"] == "doc_cached_pdf"
        assert structure["data"]["structure"][0]["title"] == "Introduction"
        assert structure["data"]["structure"][1]["title"] == "Findings"
        assert structure["data"]["structure_pagination"]["limit"] == 25
        assert "text" not in structure["data"]["structure"][0]
        assert "text" not in structure["data"]["structure"][1]

        assert pages["data"]["available"] is True
        assert pages["data"]["text"] == "Page one text\n\nPage two text"
        with pytest.raises(PIFSCommandError, match="target-first"):
            executor.execute("cat --page 1-2 dsid_structural_cached")
        with pytest.raises(PIFSCommandError, match="one file target"):
            executor.execute("cat dsid_structural_cached --page 1 2")

        assert stat["data"]["pageindex_doc_id"] == "doc_cached_pdf"
        assert stat["data"]["pageindex_tree_status"] == "built"


def test_cat_node_reads_pageindex_client_structure_without_custom_pifs_artifact():
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "notes.md"
        source.write_text("# Notes\n\nBody", encoding="utf-8")
        filesystem = PageIndexFileSystem(workspace=Path(tmp) / "workspace")
        write_pageindex_client_doc(
            filesystem.pageindex_client_workspace,
            "doc_cached_md",
            {
                "id": "doc_cached_md",
                "type": "md",
                "path": str(source.resolve()),
                "doc_name": "notes",
                "doc_description": "",
                "line_count": 3,
                "structure": [
                    {
                        "title": "Notes",
                        "node_id": "0001",
                        "line_num": 1,
                        "text": "# Notes\n\nBody",
                        "nodes": [],
                    }
                ],
            },
        )
        filesystem.register_file(
            storage_uri=source.as_uri(),
            source_path="docs/notes.md",
            external_id="dsid_md_cached",
            title="Cached markdown notes",
            content=source.read_text(encoding="utf-8"),
        )
        executor = PIFSCommandExecutor(filesystem, json_output=True)

        node = json.loads(executor.execute("cat dsid_md_cached --node 0001"))

        assert node["data"]["available"] is True
        assert node["data"]["pageindex_doc_id"] == "doc_cached_md"
        assert node["data"]["node"]["title"] == "Notes"
        assert node["data"]["text"] == "# Notes\n\nBody"
        assert "text" not in node["data"]["node"]


def test_cat_structure_page_node_and_text_outputs_are_hard_limited():
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.commands import PIFSCommandError

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "report.pdf"
        source.write_bytes(b"%PDF-1.4\n% test fixture\n")
        filesystem = PageIndexFileSystem(workspace=Path(tmp) / "workspace")
        structure_nodes = [
            {
                "title": f"Section {index}",
                "node_id": f"{index:04d}",
                "start_index": index,
                "end_index": index,
                "text": f"node {index} text",
                "nodes": [],
            }
            for index in range(1, 31)
        ]
        write_pageindex_client_doc(
            filesystem.pageindex_client_workspace,
            "doc_limited_pdf",
            {
                "id": "doc_limited_pdf",
                "type": "pdf",
                "path": str(source.resolve()),
                "doc_name": "report.pdf",
                "doc_description": "",
                "page_count": 10,
                "structure": structure_nodes,
                "pages": [
                    {"page": index, "content": f"Page {index} text"}
                    for index in range(1, 11)
                ],
            },
        )
        filesystem.register_file(
            storage_uri=source.as_uri(),
            source_path="docs/report.pdf",
            external_id="dsid_limited_pdf",
            title="Limited structural report",
            content="text artifact remains available for grep",
        )
        text_content = "\n".join(f"line {index}" for index in range(1, 106))
        filesystem.register_file(
            storage_uri="file:///tmp/long.txt",
            source_path="docs/long.txt",
            external_id="dsid_long_text",
            title="Long text",
            content=text_content,
        )
        executor = PIFSCommandExecutor(filesystem, json_output=True)

        first_structure = json.loads(executor.execute("cat dsid_limited_pdf --structure"))
        assert len(first_structure["data"]["structure"]) == 25
        assert first_structure["data"]["structure_pagination"]["has_more"] is True
        assert first_structure["data"]["structure_pagination"]["next_offset"] == 25

        second_structure = json.loads(
            executor.execute("cat dsid_limited_pdf --structure --offset 25")
        )
        assert len(second_structure["data"]["structure"]) == 5
        assert second_structure["data"]["structure"][0]["node_id"] == "0026"

        pages = json.loads(executor.execute("cat dsid_limited_pdf --page 1-3"))
        assert pages["data"]["text"] == "Page 1 text\n\nPage 2 text\n\nPage 3 text"
        assert pages["data"]["page_pagination"]["limit"] == 3
        with pytest.raises(PIFSCommandError, match="at most 3"):
            executor.execute("cat dsid_limited_pdf --page 1-4")

        nodes = json.loads(
            executor.execute("cat dsid_limited_pdf --node 0001,0002,0003,0004,0005")
        )
        assert nodes["data"]["node_ids"] == ["0001", "0002", "0003", "0004", "0005"]
        with pytest.raises(PIFSCommandError, match="at most 5"):
            executor.execute("cat dsid_limited_pdf --node 0001,0002,0003,0004,0005,0006")

        text = json.loads(executor.execute("cat dsid_long_text --all"))
        assert "line 100" in text["data"]["text"]
        assert "line 101" not in text["data"]["text"]
        assert text["data"]["pagination"]["has_more"] is True
        assert text["data"]["pagination"]["next_range"] == "101-105"
        with pytest.raises(PIFSCommandError, match="at most 100"):
            executor.execute("cat dsid_long_text --range 1-101")


def test_tree_folder_behavior_is_preserved():
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

    with tempfile.TemporaryDirectory() as tmp:
        filesystem = PageIndexFileSystem(workspace=Path(tmp) / "workspace")
        filesystem.register_file(
            storage_uri="file:///tmp/report.txt",
            source_path="docs/report.txt",
            folder_path="/docs/reports",
            external_id="dsid_folder_tree",
            title="Folder report",
            content="folder tree behavior remains intact",
        )
        executor = PIFSCommandExecutor(filesystem, json_output=True)

        folder_tree = json.loads(executor.execute("tree /docs --depth 2"))

        assert folder_tree["data"]["path"] == "/docs"
        assert folder_tree["data"]["folders"][0]["path"] == "/docs/reports"


def test_tree_does_not_read_file_internal_pageindex_structure():
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.commands import PIFSCommandError

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "report.pdf"
        source.write_bytes(b"%PDF-1.4\n% test fixture\n")
        filesystem = PageIndexFileSystem(workspace=Path(tmp) / "workspace")
        write_pageindex_client_doc(
            filesystem.pageindex_client_workspace,
            "doc_tree_is_folder_only",
            {
                "id": "doc_tree_is_folder_only",
                "type": "pdf",
                "path": str(source.resolve()),
                "doc_name": "report.pdf",
                "doc_description": "",
                "page_count": 1,
                "structure": [
                    {"title": "Introduction", "node_id": "0001", "nodes": []}
                ],
                "pages": [{"page": 1, "content": "Page one text"}],
            },
        )
        filesystem.register_file(
            storage_uri=source.as_uri(),
            source_path="docs/report.pdf",
            external_id="dsid_tree_is_folder_only",
            title="Cached structural report",
            content="text artifact remains available",
        )
        executor = PIFSCommandExecutor(filesystem, json_output=True)

        with pytest.raises(PIFSCommandError):
            executor.execute("tree dsid_tree_is_folder_only")

        structure = json.loads(executor.execute("cat dsid_tree_is_folder_only --structure"))
        assert structure["data"]["structure"][0]["title"] == "Introduction"


def test_cat_all_is_limited_to_text_files():
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.commands import PIFSCommandError

    with tempfile.TemporaryDirectory() as tmp:
        filesystem = PageIndexFileSystem(workspace=Path(tmp) / "workspace")
        filesystem.register_file(
            storage_uri="file:///tmp/readme.txt",
            source_path="docs/readme.txt",
            external_id="dsid_text_file",
            title="Text readme",
            content="plain text body",
        )
        filesystem.register_file(
            storage_uri="file:///tmp/report.pdf",
            source_path="docs/report.pdf",
            external_id="dsid_pdf_file",
            title="PDF report",
            content="extracted text should not be served through cat --all",
        )
        filesystem.register_file(
            storage_uri="file:///tmp/notes.md",
            source_path="docs/notes.md",
            external_id="dsid_md_file",
            title="Markdown notes",
            content="markdown text should use PageIndex structure reads",
        )
        filesystem.register_file(
            storage_uri="file:///tmp/data.json",
            source_path="docs/data.json",
            external_id="dsid_json_file",
            title="JSON record",
            content='{"body":"json"}',
            content_type="application/json",
        )
        executor = PIFSCommandExecutor(filesystem, json_output=True)

        text = json.loads(executor.execute("cat dsid_text_file --all"))
        assert text["data"]["text"] == "plain text body"

        with pytest.raises(PIFSCommandError, match="only supported for txt/text files"):
            executor.execute("cat dsid_pdf_file --all")
        with pytest.raises(ValueError, match="not supported for PDF/Markdown"):
            filesystem.open("dsid_pdf_file")
        with pytest.raises(PIFSCommandError, match="only supported for txt/text files"):
            executor.execute("cat dsid_md_file --all")
        with pytest.raises(ValueError, match="not supported for PDF/Markdown"):
            filesystem.open("dsid_md_file")
        with pytest.raises(PIFSCommandError, match="only supported for txt/text files"):
            executor.execute("cat dsid_json_file --all")
        assert filesystem.open("dsid_json_file").text == '{"body":"json"}'
        for command in (
            "head dsid_pdf_file",
            "tail dsid_pdf_file",
            "sed -n 1,1p dsid_pdf_file",
            "head dsid_md_file",
            "tail dsid_md_file",
            "sed -n 1,1p dsid_md_file",
        ):
            with pytest.raises(PIFSCommandError, match="only supported for txt/text files"):
                executor.execute(command)


def test_pageindex_structure_commands_are_limited_to_pdf_and_markdown():
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.commands import PIFSCommandError

    with tempfile.TemporaryDirectory() as tmp:
        filesystem = PageIndexFileSystem(workspace=Path(tmp) / "workspace")
        filesystem.register_file(
            storage_uri="file:///tmp/readme.txt",
            source_path="docs/readme.txt",
            external_id="dsid_text_only",
            title="Text readme",
            content="plain text body",
        )
        executor = PIFSCommandExecutor(filesystem, json_output=True)

        for command in (
            "cat dsid_text_only --structure",
            "cat dsid_text_only --page 1",
            "cat dsid_text_only --node 0001",
        ):
            with pytest.raises(PIFSCommandError, match="only supported for PDF/Markdown"):
                executor.execute(command)


def test_existing_pageindex_status_allows_legacy_record_without_format_suffix():
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.commands import PIFSCommandError

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "uploaded"
        source.write_text("# Uploaded\n\nBody", encoding="utf-8")
        filesystem = PageIndexFileSystem(workspace=Path(tmp) / "workspace")
        file_ref = filesystem.register_file(
            storage_uri=source.as_uri(),
            source_path="uploads/uploaded",
            external_id="dsid_legacy_pageindex",
            title="Legacy PageIndex record",
            content="text/plain is only a weak default here",
        )
        write_pageindex_client_doc(
            filesystem.pageindex_client_workspace,
            "doc_legacy_pageindex",
            {
                "id": "doc_legacy_pageindex",
                "type": "md",
                "path": str(source.resolve()),
                "doc_name": "uploaded",
                "doc_description": "",
                "line_count": 3,
                "structure": [
                    {"title": "Uploaded", "node_id": "0001", "text": "Body", "nodes": []}
                ],
            },
        )
        filesystem.store.update_pageindex_pointer(
            file_ref,
            pageindex_doc_id="doc_legacy_pageindex",
            pageindex_tree_status="built",
        )
        executor = PIFSCommandExecutor(filesystem, json_output=True)

        structure = json.loads(executor.execute("cat dsid_legacy_pageindex --structure"))
        assert structure["data"]["structure"][0]["title"] == "Uploaded"
        with pytest.raises(PIFSCommandError, match="only supported for txt/text files"):
            executor.execute("cat dsid_legacy_pageindex --all")


def test_read_commands_do_not_link_pageindex_cache_when_pointer_is_missing(monkeypatch):
    from pageindex import PageIndexClient
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "late.md"
        source.write_text("# Late\n\nBody", encoding="utf-8")
        filesystem = PageIndexFileSystem(workspace=Path(tmp) / "workspace")

        def fail_index(*args, **kwargs):
            raise RuntimeError("index failed")

        monkeypatch.setattr(PageIndexClient, "index", fail_index)
        filesystem.register_file(
            storage_uri=source.as_uri(),
            source_path="docs/late.md",
            external_id="dsid_late_cache",
            title="Late cache",
            content=source.read_text(encoding="utf-8"),
        )
        write_pageindex_client_doc(
            filesystem.pageindex_client_workspace,
            "doc_late_cache",
            {
                "id": "doc_late_cache",
                "type": "md",
                "path": str(source.resolve()),
                "doc_name": "late",
                "doc_description": "",
                "line_count": 3,
                "structure": [
                    {"title": "Late", "node_id": "0001", "text": "Body", "nodes": []}
                ],
            },
        )
        executor = PIFSCommandExecutor(filesystem, json_output=True)

        structure = json.loads(executor.execute("cat dsid_late_cache --structure"))
        stat = json.loads(executor.execute("stat dsid_late_cache"))

        assert structure["data"]["available"] is False
        assert stat["data"]["pageindex_doc_id"] is None
        assert stat["data"]["pageindex_tree_status"] == "failed"
