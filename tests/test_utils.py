import asyncio
import importlib
import json
from io import BytesIO
from types import SimpleNamespace

import pytest

from pageindex import utils


def sample_tree():
    return [
        {
            "title": "Root",
            "node_id": "1",
            "text": "root text",
            "nodes": [{"title": "Leaf", "node_id": "2", "text": "leaf text", "nodes": []}],
        }
    ]


def test_json_helpers_parse_fences_none_trailing_commas_and_bad_input():
    assert utils.get_json_content("```json\n{\"a\": 1}\n```") == '{"a": 1}'
    assert utils.extract_json('```json\n{"a": None,}\n```') == {"a": None}
    assert utils.extract_json("not json") == {}
    assert utils.extract_json(None) == {}


def test_tree_helpers_cover_lists_dicts_and_leaf_detection():
    tree = sample_tree()

    assert [node["title"] for node in utils.get_nodes(tree)] == ["Root", "Leaf"]
    assert utils.get_nodes({"title": "Solo"}) == [{"title": "Solo"}]
    assert [node["node_id"] for node in utils.structure_to_list(tree)] == ["1", "2"]
    assert utils.structure_to_list({"node_id": "solo"}) == [{"node_id": "solo"}]
    assert utils.get_leaf_nodes(tree) == [{"title": "Leaf", "node_id": "2", "text": "leaf text"}]
    assert utils.get_leaf_nodes({"title": "Leaf", "nodes": []}) == [{"title": "Leaf"}]
    assert utils.is_leaf_node(tree, "2") is True
    assert utils.is_leaf_node(tree, "1") is False
    assert utils.is_leaf_node(tree, "missing") is False
    assert utils.get_last_node(tree)["title"] == "Root"


def test_write_node_id_and_mapping():
    tree = [{"title": "A", "nodes": [{"title": "B", "nodes": []}]}]

    utils.write_node_id(tree)

    assert tree[0]["node_id"] == "0000"
    assert tree[0]["nodes"][0]["node_id"] == "0001"
    assert utils.create_node_mapping(tree)["0001"]["title"] == "B"


def test_file_name_helpers(monkeypatch):
    assert utils.sanitize_filename("a/b") == "a-b"
    assert utils.get_pdf_name("/tmp/name.pdf") == "name.pdf"

    class Meta:
        title = "bad/name"

    class Reader:
        metadata = Meta()

        def __init__(self, _stream):
            pass

    monkeypatch.setattr(utils.PyPDF2, "PdfReader", Reader)
    assert utils.get_pdf_name(BytesIO(b"pdf")) == "bad-name"

    class NoTitleReader:
        metadata = None

        def __init__(self, _stream):
            pass

    monkeypatch.setattr(utils.PyPDF2, "PdfReader", NoTitleReader)
    assert utils.get_pdf_name(BytesIO(b"pdf")) == "Untitled"


def test_pdf_text_helpers_with_mock_reader(monkeypatch):
    class Page:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class Meta:
        title = "Title"

    class Reader:
        metadata = Meta()
        pages = [Page("one"), Page("two")]

        def __init__(self, _path):
            pass

    monkeypatch.setattr(utils.PyPDF2, "PdfReader", Reader)

    assert utils.extract_text_from_pdf("doc.pdf") == "onetwo"
    assert utils.get_pdf_title("doc.pdf") == "Title"
    assert "<start_index_1>" in utils.get_text_of_pages("doc.pdf", 1, 2)
    assert utils.get_text_of_pages("doc.pdf", 1, 2, tag=False) == "onetwo"
    assert utils.get_number_of_pages("doc.pdf") == 2

    class NoTitleReader(Reader):
        metadata = None

    monkeypatch.setattr(utils.PyPDF2, "PdfReader", NoTitleReader)
    assert utils.get_pdf_title("doc.pdf") == "Untitled"


def test_list_to_tree_and_preface():
    flat = [
        {"structure": "1", "title": "A", "start_index": 1, "end_index": 2},
        {"structure": "1.1", "title": "B", "start_index": 2, "end_index": 2},
        {"structure": "2.1", "title": "Orphan", "start_index": 3, "end_index": 4},
    ]

    tree = utils.list_to_tree(flat)
    assert tree[0]["nodes"][0]["title"] == "B"
    assert tree[1]["title"] == "Orphan"
    assert "nodes" not in tree[1]
    assert utils.add_preface_if_needed([]) == []
    data = [{"physical_index": 3, "title": "A"}]
    assert utils.add_preface_if_needed(data)[0]["title"] == "Preface"


def test_text_page_and_post_processing_helpers(monkeypatch):
    pages = [("one", 1), ("two", 1), ("three", 1)]
    assert utils.get_text_of_pdf_pages(pages, 1, 2) == "onetwo"
    assert "<physical_index_2>" in utils.get_text_of_pdf_pages_with_labels(pages, 2, 3)
    assert utils.get_first_start_page_from_text("x<start_index_4>") == 4
    assert utils.get_first_start_page_from_text("none") == -1
    assert utils.get_last_start_page_from_text("<start_index_1><start_index_9>") == 9
    assert utils.get_last_start_page_from_text("none") == -1

    structure = [
        {"structure": "1", "title": "A", "physical_index": 1, "appear_start": "yes"},
        {"structure": "2", "title": "B", "physical_index": 3, "appear_start": "no"},
    ]
    assert utils.post_processing(structure, 5)[0]["end_index"] == 3

    monkeypatch.setattr(utils, "list_to_tree", lambda _: [])
    fallback = utils.post_processing([{"title": "A", "physical_index": 1, "appear_start": "no"}], 1)
    assert fallback == [{"title": "A", "start_index": 1, "end_index": 1}]


def test_clean_remove_convert_and_format_helpers():
    tree = [{"text": "x", "page_number": 1, "start_index": 1, "end_index": 2, "nodes": [{"text": "y"}]}]
    assert utils.remove_fields(tree, ["text"]) == [{"page_number": 1, "start_index": 1, "end_index": 2, "nodes": [{}]}]
    assert utils.clean_structure_post(tree)[0] == {"text": "x", "nodes": [{"text": "y"}]}
    assert utils.clean_structure_post({"page_number": 1}) == {}
    assert utils.remove_structure_text(tree) == [{"nodes": [{}]}]
    assert utils.remove_structure_text({"text": "x"}) == {}

    data = [{"physical_index": "<physical_index_12>"}, {"physical_index": "physical_index_3"}]
    assert utils.convert_physical_index_to_int(data) == [{"physical_index": 12}, {"physical_index": 3}]
    assert utils.convert_physical_index_to_int([{"no": "key"}, "raw"]) == [{"no": "key"}, "raw"]
    assert utils.convert_physical_index_to_int("<physical_index_5>") == 5
    assert utils.convert_physical_index_to_int("physical_index_6") == 6
    assert utils.convert_physical_index_to_int("bad") is None
    assert utils.convert_page_to_int([{"page": "4"}, {"page": "iv"}]) == [{"page": 4}, {"page": "iv"}]

    node = {"title": "A", "nodes": [{"title": "B", "nodes": []}], "extra": 1}
    assert utils.format_structure(node, ["title", "nodes"]) == {"title": "A", "nodes": [{"title": "B"}]}
    assert utils.format_structure([{"title": "A", "extra": 1}], ["title"]) == [{"title": "A"}]
    assert utils.reorder_dict({"b": 2, "a": 1}, ["a"]) == {"a": 1}
    assert utils.reorder_dict({"a": 1}, None) == {"a": 1}
    assert utils.format_structure({"a": 1}, None) == {"a": 1}


def test_add_node_text_variants():
    node = {"start_index": 1, "end_index": 2, "nodes": [{"start_index": 2, "end_index": 2}]}
    pages = [("one", 1), ("two", 1)]

    utils.add_node_text(node, pages)
    assert node["text"] == "onetwo"
    assert node["nodes"][0]["text"] == "two"

    nodes = [{"start_index": 1, "end_index": 1}]
    utils.add_node_text_with_labels(nodes, pages)
    assert "<physical_index_1>" in nodes[0]["text"]

    node_with_child = {"start_index": 1, "end_index": 1, "nodes": [{"start_index": 2, "end_index": 2}]}
    utils.add_node_text_with_labels(node_with_child, pages)
    assert "<physical_index_2>" in node_with_child["nodes"][0]["text"]


def test_llm_wrappers_success_and_failure(monkeypatch):
    class Choice:
        finish_reason = "length"
        message = SimpleNamespace(content="ok")

    class Response:
        choices = [Choice()]

    monkeypatch.setattr(utils.litellm, "completion", lambda **_: Response())
    assert utils.llm_completion("litellm/model", "prompt") == "ok"
    assert utils.llm_completion("model", "prompt", return_finish_reason=True) == ("ok", "max_output_reached")

    class FinishedChoice(Choice):
        finish_reason = "stop"

    class FinishedResponse:
        choices = [FinishedChoice()]

    monkeypatch.setattr(utils.litellm, "completion", lambda **_: FinishedResponse())
    assert utils.llm_completion("model", "prompt", chat_history=[{"role": "assistant", "content": "old"}], return_finish_reason=True) == ("ok", "finished")

    calls = {"count": 0}

    def fail(**_):
        calls["count"] += 1
        raise RuntimeError("nope")

    monkeypatch.setattr(utils.litellm, "completion", fail)
    monkeypatch.setattr(utils.time, "sleep", lambda _seconds: None)
    assert utils.llm_completion("model", "prompt") == ""
    assert calls["count"] == 10
    assert utils.llm_completion("model", "prompt", return_finish_reason=True) == ("", "error")


def test_async_llm_wrappers_and_summary_generation(monkeypatch):
    class Choice:
        message = SimpleNamespace(content="async ok")

    class Response:
        choices = [Choice()]

    async def fake_acompletion(**_):
        return Response()

    monkeypatch.setattr(utils.litellm, "acompletion", fake_acompletion)
    assert asyncio.run(utils.llm_acompletion("litellm/model", "prompt")) == "async ok"

    async def fake_llm(model, prompt):
        return "summary"

    monkeypatch.setattr(utils, "llm_acompletion", fake_llm)
    node = {"text": "body"}
    assert asyncio.run(utils.generate_node_summary(node)) == "summary"
    tree = [{"text": "a", "nodes": [{"text": "b", "nodes": []}]}]
    assert asyncio.run(utils.generate_summaries_for_structure(tree))[0]["summary"] == "summary"


def test_async_llm_failure_path(monkeypatch):
    async def fail(**_):
        raise RuntimeError("bad")

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(utils.litellm, "acompletion", fail)
    monkeypatch.setattr(utils.asyncio, "sleep", no_sleep)
    assert asyncio.run(utils.llm_acompletion("model", "prompt")) == ""


def test_description_config_logging_and_print_helpers(monkeypatch, tmp_path, capsys):
    structure = [{"title": "A", "text": "hide", "summary": "s", "nodes": [{"title": "B", "prefix_summary": "p"}]}]
    assert utils.create_clean_structure_for_description(structure) == [
        {"title": "A", "summary": "s", "nodes": [{"title": "B", "prefix_summary": "p"}]}
    ]
    assert utils.create_clean_structure_for_description("raw") == "raw"
    monkeypatch.setattr(utils, "llm_completion", lambda model, prompt: "doc desc")
    assert utils.generate_doc_description(structure) == "doc desc"

    cfg = tmp_path / "config.yaml"
    cfg.write_text("a: 1\nb: two\n", encoding="utf-8")
    loader = utils.ConfigLoader(cfg)
    assert loader.load().a == 1
    assert loader.load({"a": 2}).a == 2
    assert loader.load(SimpleNamespace(b="three")).b == "three"
    with pytest.raises(TypeError):
        loader.load([])
    with pytest.raises(ValueError, match="Unknown config keys"):
        loader.load({"c": 3})

    monkeypatch.chdir(tmp_path)
    logger = utils.JsonLogger("/tmp/doc.pdf")
    logger.info({"event": "ok"})
    logger.error("bad")
    logger.debug("debug")
    logger.exception("oops")
    log_data = json.loads((tmp_path / logger._filepath()).read_text(encoding="utf-8"))
    assert log_data[0] == {"event": "ok"}
    assert log_data[-1] == {"message": "oops"}

    utils.print_toc([{"title": "A", "nodes": [{"title": "B"}]}])
    utils.print_json({"x": "a" * 50}, max_len=3)
    utils.print_json(["x", 3], max_len=3)
    utils.print_tree([{"node_id": "1", "title": "A", "summary": "hello", "nodes": []}])
    utils.print_tree([{"node_id": "1", "title": "A", "nodes": [{"node_id": "2", "title": "B"}]}])
    utils.print_wrapped("hello world", width=5)
    out = capsys.readouterr().out
    assert "A" in out and "hello..." in out and "[1]" in out


def test_count_token_and_check_token_limit(monkeypatch, capsys):
    monkeypatch.setattr(utils.litellm, "token_counter", lambda model=None, text="": len(text.split()))
    assert utils.count_tokens("") == 0
    assert utils.count_tokens("one two") == 2
    utils.check_token_limit([{"text": "one two three", "node_id": "n", "start_index": 1, "end_index": 2, "title": "T"}], limit=2)
    assert "Node ID: n" in capsys.readouterr().out


def test_env_alias_reload(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CHATGPT_API_KEY", "alias-key")
    importlib.reload(utils)
    assert utils.os.environ["OPENAI_API_KEY"] == "alias-key"


def test_get_page_tokens_parsers(monkeypatch, tmp_path):
    class PdfPage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class PdfReader:
        pages = [PdfPage("one"), PdfPage("two")]

        def __init__(self, _path):
            pass

    monkeypatch.setattr(utils.PyPDF2, "PdfReader", PdfReader)
    monkeypatch.setattr(utils.litellm, "token_counter", lambda model=None, text="": len(text.split()))
    assert utils.get_page_tokens("doc.pdf") == [("one", 1), ("two", 1)]

    class MuPage:
        def __init__(self, text):
            self._text = text

        def get_text(self):
            return self._text

    monkeypatch.setattr(utils.pymupdf, "open", lambda *args, **kwargs: [MuPage("alpha beta")])
    assert utils.get_page_tokens(BytesIO(b"pdf"), pdf_parser="PyMuPDF") == [("alpha beta", 2)]
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"pdf")
    assert utils.get_page_tokens(str(path), pdf_parser="PyMuPDF") == [("alpha beta", 2)]

    with pytest.raises(ValueError, match="Unsupported PDF parser"):
        utils.get_page_tokens("doc.pdf", pdf_parser="bad")


def test_leaf_and_node_helpers_on_empty_lists():
    assert utils.get_nodes([]) == []
    assert utils.structure_to_list([]) == []
    assert utils.get_leaf_nodes([]) == []
    assert utils.create_node_mapping([]) == {}
