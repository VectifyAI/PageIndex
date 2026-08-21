"""Pins the pdfium 5.x text-extraction semantics the parser is calibrated to.

pdfium split FPDFFont_GetFontName into GetBaseFontName (/BaseFont, per-face)
and GetFamilyName (old family semantics); the parser uses the per-face names,
which changes word joining and figure-label pickup. These sentinels come from
a 4.30-vs-5.13 corpus A/B and fail if the semantics move again.
"""
from importlib.metadata import version
from pathlib import Path

import pytest

PDF = Path(__file__).parent.parent / "examples" / "documents" / "earthmover.pdf"


def test_read_bookmarks_same_on_pdfium_4_and_5():
    """Users may install pypdfium2 4.x or 5.x (floor >=4.30) — the bookmark
    reader has one branch per major and both must yield the same entries.
    The CI pdfium-4 leg runs this against the 4.x branch; everywhere else
    it pins the 5.x branch to the same values."""
    from pageindex.flash.embedded_toc import read_bookmarks

    pdf = Path(__file__).parent.parent / "examples" / "documents" / "attention-residuals.pdf"
    bookmarks = read_bookmarks(str(pdf))
    assert len(bookmarks) == 22
    assert bookmarks[0] == {"title": "Introduction", "level": 1, "page": 2}
    assert bookmarks[2] == {"title": "Training Deep Networks via Residuals",
                            "level": 2, "page": 3}


@pytest.mark.skipif(int(version("pypdfium2").split(".")[0]) < 5,
                    reason="extraction is pinned to pdfium 5.x font-name semantics")
def test_page_text_pins_pdfium5_semantics():
    from pageindex.flash.main import extract_toc

    page7 = extract_toc(str(PDF))["page_texts"][6]
    assert "p5\nEMD\n1.0" in page7          # figure axis label pdfium 4.x dropped
    assert "break loop\n5: if lbp" in page7  # pseudocode lines no longer glued


def test_page_mode_walk_uses_merged_surrogate_census():
    """The page-mode unicode walk must consume the char census char_extract
    built (astral chars merged to one entry at the high-surrogate slot).
    Re-reading the textpage split them back into two lone-surrogate slots,
    desynced the walk against their one-char cmap targets, and silently
    dropped every patch on any page containing an astral char."""
    from pageindex.flash.parser_pdfium_charlevel.unicode_apply import (
        _apply_font_unicode)

    astral = {"i": 0, "ch": "\U0001d44e", "is_gen": False}   # slots 0-1 merged
    unmapped = {"i": 2, "ch": "\x00", "is_gen": False}       # PDFium found no unicode
    raw_chars = [astral, unmapped]
    show_codes = [(7, (5, 6), 100.0)]
    map_cache = {7: (1, {5: "\U0001d44e", 6: "β"})}

    # objects vs show ops count differs -> page mode.
    _apply_font_unicode(raw_chars, [], show_codes, None, map_cache)

    assert astral["ch"] == "\U0001d44e"
    assert unmapped["ch"] == "β"


def test_optimize_full_keyless_reports_file_errors_first(tmp_path, monkeypatch):
    """No credential pre-check: a bad path is a FileNotFoundError even
    keyless (validation runs first), and the LLM-free spellings still run
    end to end."""
    from conftest import build_pdf
    from pageindex.flash import page_index_flash
    import litellm  # noqa: F401 — first import may load a .env; delenv after it
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CHATGPT_API_KEY", raising=False)

    with pytest.raises(FileNotFoundError):
        page_index_flash(str(tmp_path / "missing.pdf"), summary=False)

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(build_pdf(["1 Introduction", "Body text"]))
    result = page_index_flash(str(pdf), summary=False, optimize="merge")
    assert "structure" in result
    result = page_index_flash(str(pdf), summary=False, optimize=False)
    assert "structure" in result


def test_empty_outline_gate_carries_page_texts(tmp_path):
    """The gate's bookmark-built trees feed the same summary/expand passes
    as detected ones, so its result must carry the per-page text too."""
    from conftest import build_pdf
    from pageindex.flash.main import extract_toc

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(build_pdf(["Alpha body", "Beta body"]))
    result = extract_toc(str(pdf))
    assert result["structure"] == []  # the short-document gate fired
    assert len(result["page_texts"]) == 2
    assert "Alpha" in result["page_texts"][0]


def test_propose_children_clamps_to_loaded_pages(monkeypatch):
    """A tree from another parser may overrun the loaded pages; the span is
    clamped instead of IndexErroring into a silently frozen node."""
    import asyncio
    from types import SimpleNamespace
    import pageindex.tree_optimize as tree_optimize

    seen = {}

    async def fake_ask(model, prompt):
        seen["prompt"] = prompt
        return {"subsections": []}

    monkeypatch.setattr(tree_optimize, "ask_model", fake_ask)
    node = {"title": "T", "start_index": 1, "end_index": 3, "node_id": "n1"}
    out = asyncio.run(tree_optimize.propose_children(
        node, ["page one", "page two"], SimpleNamespace(model="m")))
    assert out == []
    assert "<page_2>" in seen["prompt"] and "<page_3>" not in seen["prompt"]

    seen.clear()
    node = {"title": "T", "start_index": 3, "end_index": 4, "node_id": "n2"}
    out = asyncio.run(tree_optimize.propose_children(
        node, ["page one", "page two"], SimpleNamespace(model="m")))
    assert out == [] and "prompt" not in seen  # fully beyond: no model call


def test_bootstrap_reimport_is_not_swallowed(monkeypatch):
    # An unguarded caller script re-imported by a spawn worker must die loudly,
    # not fall back to a silent full sequential rerun in every worker.
    import multiprocessing
    import sys

    from pageindex.flash import parser_pdfium_parallel as mod

    class BoomExecutor:
        def __init__(self, *a, **k):
            pass

        def map(self, *a, **k):
            raise RuntimeError("start a new process before bootstrapping")

        def shutdown(self, *a, **k):
            pass

    monkeypatch.setattr(mod, "ProcessPoolExecutor", BoomExecutor)
    cur = multiprocessing.current_process()

    monkeypatch.setattr(cur, "_inheriting", True, raising=False)
    with pytest.raises(RuntimeError):
        mod.parse_charlevel_meta_parallel(str(PDF), workers=2, min_pages=1)
    assert hasattr(sys.modules["__main__"], "__file__")  # window restored on error

    monkeypatch.delattr(cur, "_inheriting")
    out, meta = mod.parse_charlevel_meta_parallel(str(PDF), workers=2, min_pages=1)
    assert len(out) == len(meta) > 0  # normal failures still fall back sequentially


def test_pool_construction_failure_falls_back_sequential(monkeypatch):
    """Restricted environments (no working POSIX semaphores) refuse the pool
    at construction, before any work is mapped; indexing must take the
    sequential path, not die — except in a bootstrapping spawn child, where
    a sequential rerun would duplicate the whole run per worker."""
    import multiprocessing

    from pageindex.flash import parser_pdfium_parallel as mod

    class RefusedExecutor:
        def __init__(self, *a, **k):
            raise OSError("Function not implemented")

    monkeypatch.setattr(mod, "ProcessPoolExecutor", RefusedExecutor)
    out, meta = mod.parse_charlevel_meta_parallel(str(PDF), workers=2,
                                                  min_pages=1)
    assert len(out) == len(meta) > 0

    cur = multiprocessing.current_process()
    monkeypatch.setattr(cur, "_inheriting", True, raising=False)
    with pytest.raises(OSError):
        mod.parse_charlevel_meta_parallel(str(PDF), workers=2, min_pages=1)


def test_submit_document_refuses_during_bootstrap(tmp_path, monkeypatch):
    import multiprocessing

    from pageindex import PageIndexAPIError, PageIndexLocalClient

    c = PageIndexLocalClient(storage_path=str(tmp_path))
    monkeypatch.setattr(
        multiprocessing.current_process(), "_inheriting", True, raising=False
    )
    with pytest.raises(PageIndexAPIError, match="__main__"):
        c.submit_document("whatever.pdf")


def test_unguarded_script_parses_parallel_without_reexecution(tmp_path):
    # spawn workers must not re-run an unguarded caller script: one completion,
    # no dead-worker noise (dying workers would trip the sequential fallback).
    import os
    import subprocess
    import sys

    marker = tmp_path / "runs.txt"
    script = tmp_path / "unguarded.py"
    script.write_text(
        "from pageindex.flash.parser_pdfium_parallel import parse_charlevel_meta_parallel\n"
        f"out, meta = parse_charlevel_meta_parallel({str(PDF)!r}, workers=2, min_pages=1)\n"
        "assert len(out) == len(meta) > 0\n"
        f"open({str(marker)!r}, 'a').write('ran\\n')\n"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)}
    res = subprocess.run(
        [sys.executable, str(script)], capture_output=True, env=env, timeout=120
    )
    assert res.returncode == 0, res.stderr.decode()
    assert marker.read_text() == "ran\n"
    assert b"Traceback" not in res.stderr


def test_optimize_wins_over_deprecated_optimize_expand(tmp_path, monkeypatch):
    """Explicit optimize= beats optimize_expand; legacy True still honors it."""
    from conftest import build_pdf
    from pageindex.flash import page_index_flash
    import litellm  # noqa: F401 — first import may load a .env; delenv after it
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CHATGPT_API_KEY", raising=False)

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(build_pdf(["1 Introduction", "Body text"]))
    # explicit "merge" wins even when the deprecated flag says expand
    with pytest.warns(DeprecationWarning):
        result = page_index_flash(str(pdf), summary=False,
                                  optimize="merge", optimize_expand=True)
    assert "structure" in result
    with pytest.warns(DeprecationWarning):
        result = page_index_flash(str(pdf), summary=False,
                                  optimize=True, optimize_expand=False)
    assert "structure" in result
    # optimize=None means unset ("full"), not off
    from pageindex.flash import api as flash_api
    seen = {}

    def fake_optimize(structure, pages, do_expand, model):
        seen["do_expand"] = do_expand
        return {"merges": 0}

    monkeypatch.setattr(flash_api, "_optimize", fake_optimize)
    monkeypatch.setattr(flash_api, "extract_toc",
                        lambda pdf, use_embedded_toc=True: {
                            "structure": [{"title": "T", "start_index": 1,
                                           "end_index": 1, "nodes": []}],
                            "page_texts": ["body"]})
    page_index_flash(str(pdf), summary=False, optimize=None)
    assert seen["do_expand"] is True


def test_lone_surrogate_from_broken_tounicode_is_replaced(monkeypatch):
    """An unpaired UTF-16 surrogate leaves as U+FFFD, not a str that crashes utf-8 save."""
    import json
    from io import BytesIO

    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c
    from conftest import build_pdf
    from pageindex.flash.parser_pdfium_charlevel.char_extract import (
        _extract_raw_chars)

    orig = pdfium_c.FPDFText_GetUnicode
    monkeypatch.setattr(pdfium_c, "FPDFText_GetUnicode",
                        lambda tp, i: 0xD83D if i == 0 else orig(tp, i))
    pdf = pdfium.PdfDocument(BytesIO(build_pdf(["Hello broken cmap"])))
    page = pdf[0]
    # hold the textpage: GC finalizes an unreferenced one mid-extraction,
    # closing the handle so every per-char call reads back 0
    text_page = page.get_textpage()
    raw_chars, _objects = _extract_raw_chars(page, text_page.raw)
    text = "".join(char["ch"] for char in raw_chars)
    assert "\ud83d" not in text
    assert text.startswith("�ello")
    json.dumps(text)  # the save-time crash this guards against


def test_lone_surrogate_targets_never_patched_into_chars():
    """A surrogate-band code with no cmap entry (chr fallback) must not patch a lone surrogate back in."""
    from pageindex.flash.parser_pdfium_charlevel.unicode_apply import (
        _apply_font_unicode)

    char = {"i": 0, "ch": "X", "is_gen": False}
    show_codes = [(7, (0xD8, 0x3D), 100.0)]
    map_cache = {7: (2, {})}  # Identity map, no ToUnicode: target = chr(0xD83D)

    _apply_font_unicode([char], [], show_codes, None, map_cache)

    assert char["ch"] == "�"


def test_lone_surrogate_from_single_byte_map_is_replaced():
    """The single-byte branch scrubs mapped lone surrogates like the
    two-byte branch does."""
    from pageindex.flash.parser_pdfium_charlevel.unicode_apply import (
        _apply_font_unicode)

    char = {"i": 0, "ch": "X", "is_gen": False}
    show_codes = [(7, (0x41,), 100.0)]
    map_cache = {7: (1, {0x41: "\ud83d"})}

    _apply_font_unicode([char], [], show_codes, None, map_cache)

    assert char["ch"] == "�"


def test_anonymous_main_overlapping_windows_restore(monkeypatch):
    """The last window out must restore the true originals, not a mid-window snapshot."""
    import sys
    import threading

    from pageindex.flash.parser_pdfium_parallel import _anonymous_main

    main = sys.modules["__main__"]
    spec = object()
    monkeypatch.setattr(main, "__file__", "sentinel-file", raising=False)
    monkeypatch.setattr(main, "__spec__", spec, raising=False)
    a_in, b_in, a_out = (threading.Event() for _ in range(3))
    errors = []

    def first():
        try:
            with _anonymous_main():
                a_in.set()
                assert b_in.wait(5)
            a_out.set()
        except BaseException as exc:  # in-thread failures only warn in pytest
            errors.append(exc)

    def second():
        try:
            assert a_in.wait(5)
            with _anonymous_main():
                b_in.set()
                assert a_out.wait(5)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert not errors
    assert main.__spec__ is spec
    assert main.__file__ == "sentinel-file"


def test_optimize_full_skips_expand_without_page_texts(tmp_path, monkeypatch):
    """A bookmark-only extraction (no page_texts) skips expand; merge still runs."""
    from conftest import build_pdf
    from pageindex.flash import api as flash_api

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    calls = {}

    def fake_optimize(structure, pages, do_expand, model):
        calls["pages"] = pages
        calls["do_expand"] = do_expand
        return {"merges": 0}

    monkeypatch.setattr(flash_api, "_optimize", fake_optimize)
    monkeypatch.setattr(flash_api, "extract_toc",
                        lambda pdf, use_embedded_toc=True: {
                            "structure": [{"title": "T", "start_index": 1,
                                           "end_index": 1, "nodes": []}]})
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(build_pdf(["x"]))
    result = flash_api.page_index_flash(str(pdf), summary=False)
    assert calls == {"pages": [], "do_expand": False}
    assert result["optimize"] == {"merges": 0}


def test_optimize_full_skips_expand_on_textless_pages(tmp_path, monkeypatch):
    """Scanned PDFs yield page_texts of empty strings; expand still skips —
    proposals against empty text are all rejected, so the calls are waste."""
    from conftest import build_pdf
    from pageindex.flash import api as flash_api

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    calls = {}

    def fake_optimize(structure, pages, do_expand, model):
        calls["do_expand"] = do_expand
        return {"merges": 0}

    monkeypatch.setattr(flash_api, "_optimize", fake_optimize)
    monkeypatch.setattr(flash_api, "extract_toc",
                        lambda pdf, use_embedded_toc=True: {
                            "structure": [{"title": "T", "start_index": 1,
                                           "end_index": 2, "nodes": []}],
                            "page_texts": ["", ""]})
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(build_pdf(["x"]))
    result = flash_api.page_index_flash(str(pdf), summary=False)
    assert calls == {"do_expand": False}
    assert result["optimize"] == {"merges": 0}


def test_optimize_expand_warning_names_the_behavior_change(tmp_path,
                                                           monkeypatch):
    """The deprecation must say the optimize pass now runs, not just that
    the parameter was renamed."""
    from conftest import build_pdf
    from pageindex.flash import api as flash_api

    monkeypatch.setattr(flash_api, "extract_toc",
                        lambda pdf, use_embedded_toc=True: {"structure": []})
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(build_pdf(["x"]))
    with pytest.warns(DeprecationWarning, match="now runs"):
        flash_api.page_index_flash(str(pdf), summary=False,
                                   optimize_expand=False)


# ── run_pageindex.py flash branch (twelfth review) ──

SCRIPT = Path(__file__).resolve().parent.parent / "run_pageindex.py"


def _run_flash_cli(monkeypatch, tmp_path, argv, structure):
    """Drive run_pageindex.py in-process with a stubbed flash indexer."""
    import runpy
    import sys

    import pageindex.flash

    pdf = tmp_path / "t.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")
    captured = {}

    def fake_flash(path, **kw):
        captured.update(kw)
        return {"structure": structure}
    monkeypatch.setattr(pageindex.flash, "page_index_flash", fake_flash)
    monkeypatch.setattr(sys, "argv",
                        ["run_pageindex.py", "--pdf_path", str(pdf), *argv])
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(SCRIPT), run_name="__main__")
    return captured


def test_flash_cli_summary_model_follows_config_chain(monkeypatch, tmp_path):
    """--model must not outrank a file-supplied summary_model: the flash
    branch resolves through ConfigLoader's chain like the standard and
    markdown branches, and like the --summary-model help promises."""
    import pageindex.utils as U

    cfg = tmp_path / "config.yaml"
    cfg.write_text((Path(U.__file__).parent / "config.yaml").read_text()
                   + "\nsummary_model: yaml-summary\n")
    orig = U.ConfigLoader.__init__
    monkeypatch.setattr(U.ConfigLoader, "__init__",
                        lambda self, default_path=None: orig(self, str(cfg)))
    captured = _run_flash_cli(
        monkeypatch, tmp_path, ["--model", "cli-model"],
        [{"title": "T", "start_index": 1, "end_index": 1}])
    assert captured["summary_model"] == "yaml-summary"
    assert captured["optimize_model"] == "yaml-summary"


def test_flash_cli_rejects_empty_structure(monkeypatch, tmp_path):
    """A PDF flash cannot structure must error like the SDK does, not write
    "structure": [] and exit 0 with a success message."""
    with pytest.raises(ValueError, match="try --mode standard"):
        _run_flash_cli(monkeypatch, tmp_path, [], [])
    assert not (tmp_path / "results").exists()
