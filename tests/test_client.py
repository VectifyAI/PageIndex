import json

import pytest

from pageindex import client


def test_normalize_retrieve_model():
    assert client._normalize_retrieve_model("") == ""
    assert client._normalize_retrieve_model("gpt-4") == "gpt-4"
    assert client._normalize_retrieve_model("openai/gpt-4") == "openai/gpt-4"
    assert client._normalize_retrieve_model("litellm/anthropic/claude") == "litellm/anthropic/claude"
    assert client._normalize_retrieve_model("anthropic/claude") == "litellm/anthropic/claude"


def test_client_init_sets_api_key_alias_and_workspace(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CHATGPT_API_KEY", "alias")

    c = client.PageIndexClient(workspace=str(tmp_path), model="m", retrieve_model="provider/model")

    assert c.model == "m"
    assert c.retrieve_model == "litellm/provider/model"
    assert c.workspace == tmp_path
    assert c.documents == {}

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client.PageIndexClient(api_key="key")
    assert client.os.environ["OPENAI_API_KEY"] == "key"


def test_make_meta_entry_for_pdf_md_and_other():
    assert client.PageIndexClient._make_meta_entry({"type": "pdf", "page_count": 5})["page_count"] == 5
    assert client.PageIndexClient._make_meta_entry({"type": "md", "line_count": 6})["line_count"] == 6
    assert "page_count" not in client.PageIndexClient._make_meta_entry({"type": "txt"})


def test_read_json_errors_and_meta_rebuild(monkeypatch, tmp_path, capsys):
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    (tmp_path / "doc.json").write_text(json.dumps({"type": "md", "line_count": 3}), encoding="utf-8")
    (tmp_path / client.META_INDEX).write_text("[]", encoding="utf-8")

    c = client.PageIndexClient(workspace=str(tmp_path))

    assert c._read_json(tmp_path / "bad.json") is None
    assert "corrupt bad.json" in capsys.readouterr().out
    assert c._read_meta() is None
    assert c._rebuild_meta()["doc"]["line_count"] == 3


def test_workspace_save_load_and_lazy_load(tmp_path):
    c = client.PageIndexClient(workspace=str(tmp_path))
    c.documents["d"] = {
        "type": "pdf",
        "doc_name": "Doc",
        "path": "relative.pdf",
        "page_count": 1,
        "structure": [{"title": "A", "text": "drop"}],
        "pages": [{"page": 1, "content": "body"}],
    }

    c._save_doc("d")

    assert "structure" not in c.documents["d"]
    saved = json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))
    assert saved["structure"] == [{"title": "A"}]

    loaded = client.PageIndexClient(workspace=str(tmp_path))
    assert loaded.documents["d"]["path"].endswith("relative.pdf")
    loaded._ensure_doc_loaded("d")
    assert loaded.documents["d"]["structure"] == [{"title": "A"}]
    assert loaded.documents["d"]["pages"] == [{"page": 1, "content": "body"}]

    loaded._ensure_doc_loaded("missing")

    loaded.documents["empty"] = {"type": "md"}
    assert loaded._ensure_doc_loaded("empty") is None
    (tmp_path / "nopages.json").write_text(json.dumps({"structure": []}), encoding="utf-8")
    loaded.documents["nopages"] = {"type": "pdf"}
    loaded._ensure_doc_loaded("nopages")
    assert "pages" not in loaded.documents["nopages"]

    no_workspace = client.PageIndexClient()
    no_workspace.documents["d"] = {"type": "md", "structure": []}
    assert json.loads(no_workspace.get_document_structure("d")) == []
    assert json.loads(no_workspace.get_page_content("d", "1")) == []


def test_get_methods_delegate_and_ensure_loaded(monkeypatch, tmp_path):
    c = client.PageIndexClient(workspace=str(tmp_path))
    c.documents["d"] = {"type": "md", "line_count": 1, "structure": [{"line_num": 1, "text": "x"}]}
    calls = []
    monkeypatch.setattr(c, "_ensure_doc_loaded", lambda doc_id: calls.append(doc_id))

    assert json.loads(c.get_document("d"))["doc_id"] == "d"
    assert json.loads(c.get_document_structure("d")) == [{"line_num": 1}]
    assert json.loads(c.get_page_content("d", "1")) == [{"page": 1, "content": "x"}]
    assert calls == ["d", "d"]


def test_index_missing_and_unsupported_file(tmp_path):
    c = client.PageIndexClient()

    with pytest.raises(FileNotFoundError):
        c.index(str(tmp_path / "missing.md"))

    txt = tmp_path / "file.txt"
    txt.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file format"):
        c.index(str(txt))


def test_index_markdown_sync_and_running_loop(monkeypatch, tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# Title\nbody\n", encoding="utf-8")

    async def fake_md_to_tree(**_kwargs):
        return {"doc_name": "doc", "doc_description": "desc", "line_count": 2, "structure": [{"title": "T"}]}

    monkeypatch.setattr(client, "md_to_tree", fake_md_to_tree)

    c = client.PageIndexClient()
    doc_id = c.index(str(path), mode="md")
    assert c.documents[doc_id]["type"] == "md"

    async def run_inside_loop():
        c2 = client.PageIndexClient()
        return c2.index(str(path), mode="md"), c2

    doc_id2, c2 = client.asyncio.run(run_inside_loop())
    assert c2.documents[doc_id2]["line_count"] == 2

    workspace = tmp_path / "workspace"
    c3 = client.PageIndexClient(workspace=str(workspace))
    doc_id3 = c3.index(str(path), mode="md")
    assert (workspace / f"{doc_id3}.json").exists()


def test_index_pdf_with_mocked_indexer_and_reader(monkeypatch, tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"pdf")
    monkeypatch.setattr(client, "page_index", lambda **_kwargs: {"doc_name": "PDF", "doc_description": "desc", "structure": []})

    class Page:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class Reader:
        def __init__(self, _file):
            self.pages = [Page("one"), Page(None)]

    monkeypatch.setattr(client.PyPDF2, "PdfReader", Reader)

    c = client.PageIndexClient()
    doc_id = c.index(str(path), mode="pdf")
    assert c.documents[doc_id]["pages"] == [{"page": 1, "content": "one"}, {"page": 2, "content": ""}]
