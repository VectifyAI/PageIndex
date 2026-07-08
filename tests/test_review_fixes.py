"""Regression tests for the directly-fixable PR #272 review findings."""
import asyncio

import pytest


# ── #1: page_index() must not capture the imported IndexConfig into opt ───────
def test_page_index_wrapper_does_not_capture_indexconfig(monkeypatch):
    import pageindex.index.page_index as pi

    captured = {}

    def fake_main(doc, opt):
        captured["opt"] = opt
        return "ok"

    monkeypatch.setattr(pi, "page_index_main", fake_main)
    # Previously raised ValidationError (IndexConfig extra='forbid') because
    # locals() captured the just-imported IndexConfig class.
    result = pi.page_index("dummy.pdf", model="gpt-4o")
    assert result == "ok"
    assert captured["opt"].model == "gpt-4o"


# ── #2: process_none_page_numbers tolerates items with no 'page' key ──────────
def test_process_none_page_numbers_tolerates_missing_page(monkeypatch):
    import pageindex.index.page_index as pi

    monkeypatch.setattr(
        pi, "add_page_number_to_toc",
        lambda pages, item, model: [{"physical_index": "<physical_index_2>"}],
    )
    toc = [
        {"title": "A", "physical_index": 1},
        {"title": "B"},  # no physical_index AND no 'page' -> used to KeyError
    ]
    page_list = [("p1", 1), ("p2", 1), ("p3", 1)]
    result = pi.process_none_page_numbers(toc, page_list)  # must not raise
    assert result is toc
    assert toc[1]["physical_index"] == 2


# ── P4: a real RuntimeError from the coroutine is not masked ──────────────────
def test_run_async_propagates_worker_runtimeerror():
    from pageindex.index.pipeline import _run_async

    async def boom():
        raise RuntimeError("real indexing error")

    async def outer():
        # Inside a running loop -> _run_async uses the worker-thread path; the
        # real error must surface, not a bogus "asyncio.run() cannot be called".
        with pytest.raises(RuntimeError, match="real indexing error"):
            _run_async(boom())

    asyncio.run(outer())


# ── #9: FileTypeError also subclasses ValueError ─────────────────────────────
def test_filetypeerror_is_valueerror():
    from pageindex.errors import FileTypeError, PageIndexError

    assert issubclass(FileTypeError, ValueError)
    assert issubclass(FileTypeError, PageIndexError)


# ── #4: md_to_tree coerces legacy 'yes'/'no' flags ───────────────────────────
def test_md_coerce_bool():
    from pageindex.index.page_index_md import _coerce_bool

    assert _coerce_bool("no") is False   # the whole point: 'no' is NOT truthy
    assert _coerce_bool("yes") is True
    assert _coerce_bool("YES") is True
    assert _coerce_bool(True) is True
    assert _coerce_bool(False) is False


# ── P6: __all__ includes the legacy top-level exports ────────────────────────
def test_all_includes_legacy_exports():
    import pageindex

    for name in ("page_index", "md_to_tree", "get_document",
                 "get_document_structure", "get_page_content"):
        assert name in pageindex.__all__, f"{name} missing from __all__"
        assert hasattr(pageindex, name), f"{name} not importable"


# ── P1: keyless local providers pass validation ──────────────────────────────
def test_validate_llm_provider_skips_keyless_providers():
    from pageindex.client import LocalClient

    # These raised PageIndexError("API key not configured...") before the fix.
    LocalClient._validate_llm_provider("ollama/llama3")
    LocalClient._validate_llm_provider("lm_studio/some-model")


# ── #3/P3: missing doc must raise, not return an empty structure ─────────────
def test_local_get_document_structure_missing_raises(tmp_path):
    from pageindex.backend.local import LocalBackend
    from pageindex.storage.sqlite import SQLiteStorage
    from pageindex.errors import DocumentNotFoundError

    backend = LocalBackend(
        storage=SQLiteStorage(str(tmp_path / "t.db")),
        files_dir=str(tmp_path / "f"), model="gpt-4o",
    )
    backend.get_or_create_collection("c")
    with pytest.raises(DocumentNotFoundError):
        backend.get_document_structure("c", "ghost")


# ── #5: delete_collection drops the cached folder_id ─────────────────────────
def test_cloud_delete_collection_clears_folder_cache(monkeypatch):
    from pageindex.backend.cloud import CloudBackend

    backend = CloudBackend(api_key="pi-test")
    backend._folder_id_cache["papers"] = "folder-123"
    monkeypatch.setattr(backend, "_request", lambda *a, **k: {})
    backend.delete_collection("papers")
    assert "papers" not in backend._folder_id_cache


# ── #6: querying an empty collection raises instead of sending doc_id:[] ──────
def test_cloud_query_empty_collection_raises(monkeypatch):
    from pageindex.backend.cloud import CloudBackend

    backend = CloudBackend(api_key="pi-test")
    monkeypatch.setattr(backend, "_get_all_doc_ids", lambda col: [])
    with pytest.raises(ValueError, match="no documents"):
        backend.query("empty", "q")  # doc_ids=None -> resolves to []


# ── #10: CLI bool flags still parse legacy yes/no (a bare 'no' must be False) ──
def test_cli_bool_coerces_legacy_yes_no():
    import run_pageindex

    assert run_pageindex._cli_bool("no") is False   # legacy off-switch
    assert run_pageindex._cli_bool("yes") is True
    assert run_pageindex._cli_bool("false") is False
    assert run_pageindex._cli_bool(True) is True     # bare flag -> const=True
