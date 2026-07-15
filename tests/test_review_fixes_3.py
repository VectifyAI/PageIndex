"""Regression tests for the PR #272 review findings #9-#15."""
import asyncio
import sqlite3
import threading
import warnings

import pytest

from pageindex.errors import CollectionNotFoundError, IndexingError


# ── #9: below-range physical_index must not wrap to the last page ────────────

def test_check_title_appearance_rejects_below_range_index(monkeypatch):
    from pageindex.index import page_index as pi

    async def _fail(*a, **k):
        raise AssertionError("LLM must not be called for a below-range index")

    monkeypatch.setattr(pi, "llm_acompletion", _fail)
    item = {"title": "Intro", "physical_index": 0, "list_index": 3}
    # A single-page page_list: without the guard, page_list[0-1] silently
    # reads this (last) page instead of erroring.
    result = asyncio.run(
        pi.check_title_appearance(item, [("last page text", 10)], start_index=1)
    )
    assert result["answer"] == "no"
    assert result["page_number"] is None


# ── #6: page_index_main must coerce legacy 'yes'/'no' string flags ───────────

def test_page_index_main_coerces_legacy_string_flags():
    from types import SimpleNamespace
    from pageindex.index.page_index import page_index_main

    opt = SimpleNamespace(model=None, if_add_node_id='yes', if_add_node_text='no',
                          if_add_node_summary='no', if_add_doc_description='no')
    # Coercion runs before input validation, which rejects the non-PDF path.
    with pytest.raises(ValueError, match="Unsupported input type"):
        page_index_main('not-a-pdf.txt', opt)
    assert opt.if_add_node_id is True
    assert opt.if_add_node_text is False
    assert opt.if_add_node_summary is False
    assert opt.if_add_doc_description is False


# ── #10: per-index max_concurrency above the ceiling must warn ────────────────

def test_max_concurrency_scope_warns_above_ceiling():
    from pageindex.config import (
        max_concurrency_scope,
        set_max_concurrency,
        _process_wide_max_concurrency,
    )

    original = _process_wide_max_concurrency()
    set_max_concurrency(5)
    try:
        with pytest.warns(UserWarning, match="exceeds the process-wide ceiling"):
            with max_concurrency_scope(20):
                pass
        # A narrowing value is the supported use and must stay silent.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with max_concurrency_scope(3):
                pass
    finally:
        set_max_concurrency(original)


# ── #11: non-streaming legacy chat_completions needs a long read timeout ─────

def test_legacy_chat_completions_nonstream_timeout(monkeypatch):
    from pageindex import cloud_api as ca

    captured = {}

    class FakeResp:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": []}

    def fake_request(method, url, headers=None, **kwargs):
        captured.clear()
        captured.update(kwargs)
        return FakeResp()

    monkeypatch.setattr(ca.requests, "request", fake_request)
    api = ca.LegacyCloudAPI(api_key="pi-test")

    api.chat_completions(messages=[{"role": "user", "content": "q"}], stream=False)
    assert captured["timeout"] == 300

    api.chat_completions(messages=[{"role": "user", "content": "q"}], stream=True)
    assert captured["timeout"] == 120  # between-chunks timeout, unchanged


# ── #12: query_stream must not run the doc-id listing on the loop thread ─────

def test_query_stream_lists_docs_off_event_loop(monkeypatch):
    import pageindex.backend.cloud as cloud_mod
    from pageindex.backend.cloud import CloudBackend

    backend = CloudBackend(api_key="pi-test")
    seen = {}

    def fake_get_all(collection):
        seen["thread"] = threading.current_thread()
        return ["d1"]

    monkeypatch.setattr(backend, "_get_all_doc_ids", fake_get_all)

    class FakeResponse:
        status_code = 200
        text = ""

        def iter_lines(self, decode_unicode=True):
            yield "data: [DONE]"

        def close(self):
            pass

    monkeypatch.setattr(cloud_mod.requests, "post", lambda *a, **k: FakeResponse())

    async def _run():
        async for _ in backend.query_stream("col", "q"):  # doc_ids=None → list all
            pass
        return threading.current_thread()

    loop_thread = asyncio.run(_run())
    assert seen["thread"] is not loop_thread


# ── #13: an unexplained IntegrityError must surface as a PageIndexError ──────

class _RaceStorage:
    """Collection exists at the fail-fast check, then save trips the FK."""

    def __init__(self, collections_after_start):
        self._after = collections_after_start
        self._calls = 0

    def list_collections(self):
        self._calls += 1
        return ["col"] if self._calls == 1 else self._after

    def find_document_by_hash(self, collection, file_hash):
        return None

    def save_document(self, *a, **k):
        raise sqlite3.IntegrityError("FOREIGN KEY constraint failed")


def _make_backend(tmp_path, monkeypatch, storage):
    import pageindex.backend.local as local_mod
    from pageindex.backend.local import LocalBackend

    backend = LocalBackend(storage=storage, files_dir=str(tmp_path / "files"))

    class FakeParsed:
        doc_name = "doc"
        nodes = []

    class FakeParser:
        def parse(self, path, model=None, images_dir=None):
            return FakeParsed()

    monkeypatch.setattr(backend, "_resolve_parser", lambda p: FakeParser())
    monkeypatch.setattr(
        local_mod, "build_index", lambda parsed, model=None, opt=None: {"structure": []}
    )
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    return backend, str(pdf)


def test_concurrent_collection_delete_raises_collection_not_found(tmp_path, monkeypatch):
    backend, pdf = _make_backend(tmp_path, monkeypatch, _RaceStorage([]))
    with pytest.raises(CollectionNotFoundError):
        backend.add_document("col", pdf)


def test_unexplained_integrity_error_wrapped_as_indexing_error(tmp_path, monkeypatch):
    backend, pdf = _make_backend(tmp_path, monkeypatch, _RaceStorage(["col"]))
    with pytest.raises(IndexingError):
        backend.add_document("col", pdf)


# ── #14: legacy `from pageindex.utils import config` must keep working ───────

def test_legacy_config_alias_importable():
    from types import SimpleNamespace
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # module-level deprecation shim warning
        from pageindex.utils import config
    assert config is SimpleNamespace


# ── #15: star import must bind the legacy pre-SDK names ──────────────────────

def test_star_import_binds_legacy_names():
    ns = {}
    exec("from pageindex import *", ns)
    for name in (
        "page_index",
        "page_index_main",
        "tree_parser",
        "ConfigLoader",
        "llm_completion",
        "llm_acompletion",
    ):
        assert name in ns, f"{name} missing from star import"
