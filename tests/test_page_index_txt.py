import asyncio
import os
import tempfile

import pytest

from pageindex.page_index_txt import txt_to_tree


def _run(coro):
    return asyncio.run(coro)


def _write_tmp(content, suffix=".txt", encoding="utf-8"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "w", encoding=encoding) as f:
        f.write(content)
    return path


def test_txt_to_tree_parses_plain_text_into_single_node():
    path = _write_tmp("Hello world.\nThis is a plain text document.\n")
    try:
        result = _run(txt_to_tree(txt_path=path, if_add_node_summary="no", if_add_doc_description="no"))
    finally:
        os.unlink(path)

    assert result["doc_name"] == os.path.splitext(os.path.basename(path))[0]
    assert isinstance(result["structure"], list)
    assert len(result["structure"]) == 1
    root = result["structure"][0]
    assert root["text"].startswith("Hello world.")
    assert "This is a plain text document." in root["text"]


def test_txt_to_tree_preserves_utf8_content():
    path = _write_tmp("héllo wörld — 你好\n", encoding="utf-8")
    try:
        result = _run(txt_to_tree(txt_path=path, if_add_node_summary="no", if_add_doc_description="no"))
    finally:
        os.unlink(path)

    assert "héllo wörld" in result["structure"][0]["text"]
    assert "你好" in result["structure"][0]["text"]


def test_txt_to_tree_includes_line_count():
    path = _write_tmp("line1\nline2\nline3\n")
    try:
        result = _run(txt_to_tree(txt_path=path, if_add_node_summary="no", if_add_doc_description="no"))
    finally:
        os.unlink(path)

    assert result["line_count"] == 4


def test_txt_to_tree_exposed_from_package():
    from pageindex import txt_to_tree as exported
    assert exported is txt_to_tree


def test_client_dispatches_txt_extension():
    import inspect
    from pageindex import client
    src = inspect.getsource(client)
    assert "txt_to_tree" in src
    assert ".txt" in src
