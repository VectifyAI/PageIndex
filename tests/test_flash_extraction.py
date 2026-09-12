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


def test_no_heading_result_carries_page_texts(tmp_path):
    """A document that yields no headings still carries its per-page text,
    so the page-node fallback and the summary/expand passes can read it."""
    from conftest import build_pdf
    from pageindex.flash.main import extract_toc

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(build_pdf(["Alpha body", "Beta body"]))
    result = extract_toc(str(pdf))
    assert result["structure"] == []  # two body-only pages: nothing to detect
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


def _run_flash_cli(monkeypatch, tmp_path, argv, structure, toc_source=None):
    """Drive run_pageindex.py in-process with a stubbed flash indexer."""
    import runpy
    import sys

    import pageindex.flash

    pdf = tmp_path / "t.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")
    captured = {}

    def fake_flash(path, **kw):
        captured.update(kw)
        result = {"structure": structure}
        if toc_source:
            result["toc_source"] = toc_source
        return result
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


def test_flash_cli_rejects_oversized_flat_tree(monkeypatch, tmp_path):
    """The CLI applies the same refusal policy as the local client, worded
    for its own flag."""
    from pageindex.flash.api import FLAT_TREE_MAX_NODES

    nodes = [{"title": f"Page {n}", "start_index": n, "end_index": n, "nodes": []}
             for n in range(1, FLAT_TREE_MAX_NODES + 2)]
    with pytest.raises(ValueError, match="no layout structure.*--mode standard"):
        _run_flash_cli(monkeypatch, tmp_path, [], nodes, toc_source="pages")
    assert not (tmp_path / "results").exists()


def _layout_pdf(pages, landscape=False):
    """Uncompressed PDF, per page a 20pt heading line then 11pt body lines:
    ``pages`` is a list of ``(heading, [body line, ...])``."""
    width, height = (792, 612) if landscape else (595, 842)
    objs = {1: "<</Type/Catalog/Pages 2 0 R>>", 3: "<</Font<</F1 5 0 R>>>>",
            5: "<</Type/Font/Subtype/Type1/BaseFont/Helvetica"
               "/Encoding/WinAnsiEncoding>>"}
    kids, nxt = [], 6
    for heading, lines in pages:
        ops, y = [(20, height - 80, heading)], height - 120
        for line in lines:
            ops.append((11, y, line))
            y -= 16
        stream = "".join(f"BT /F1 {size} Tf 72 {top} Td ({text}) Tj ET\n"
                         for size, top, text in ops)
        objs[nxt] = f"<</Length {len(stream)}>>\nstream\n{stream}endstream"
        objs[nxt + 1] = (f"<</Type/Page/MediaBox[0 0 {width} {height}]"
                         f"/Resources 3 0 R/Parent 2 0 R/Contents {nxt} 0 R>>")
        kids.append(nxt + 1)
        nxt += 2
    objs[2] = (f"<</Type/Pages/Count {len(kids)}"
               f"/Kids[{' '.join(f'{k} 0 R' for k in kids)}]>>")
    out, offsets = bytearray(b"%PDF-1.7\n"), {}
    for num in sorted(objs):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n{objs[num]}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {nxt}\n0000000000 65535 f \n".encode()
    for num in range(1, nxt):
        out += (f"{offsets[num]:010d} 00000 n \n" if num in offsets
                else "0000000000 65535 f \n").encode()
    out += (f"trailer\n<</Size {nxt}/Root 1 0 R>>\nstartxref\n{xref}\n"
            "%%EOF\n").encode()
    return bytes(out)


DECK_TITLES = ["Revenue Overview", "Operating Expenses", "Customer Growth",
               "Product Roadmap", "Regional Performance", "Engineering Metrics",
               "Risk Factors", "Outlook and Guidance"]


def _deck_pdf():
    return _layout_pdf(
        [(title, [f"Detail {n} for slide {slide} with a few more words of body text"
                  for n in range(1, 6)]) for slide, title in enumerate(DECK_TITLES, 1)],
        landscape=True)


def test_landscape_deck_is_structured(tmp_path):
    """A text-light landscape deck is an ordinary document: every slide title
    becomes a node."""
    from pageindex.flash.main import extract_toc

    pdf = tmp_path / "deck.pdf"
    pdf.write_bytes(_deck_pdf())
    result = extract_toc(str(pdf))
    # slide 1 is spent on the document title, as on any title page
    assert [node["title"] for node in result["structure"]] == DECK_TITLES[1:]
    assert result["toc_source"] == "detected"


def test_short_document_headings_are_detected(tmp_path):
    """Three pages of heading-plus-a-line are enough for detection."""
    from pageindex.flash.main import extract_toc

    pdf = tmp_path / "memo.pdf"
    pdf.write_bytes(_layout_pdf([("Mission", ["The launch is named Skylark."]),
                                 ("Budget", ["The budget is 420 euros."]),
                                 ("Team", ["The lead is Ada."])]))
    result = extract_toc(str(pdf))
    assert [node["title"] for node in result["structure"]] == ["Budget", "Team"]
    assert result["doc_title"] == "Mission"


def test_toc_source_is_always_present(tmp_path):
    """The pure-detected path (no bookmark pass) labels its result too, so
    callers can rely on the key."""
    from pageindex.flash import page_index_flash

    pdf = tmp_path / "memo.pdf"
    pdf.write_bytes(_layout_pdf([("Mission", ["The launch is named Skylark."]),
                                 ("Budget", ["The budget is 420 euros."]),
                                 ("Team", ["The lead is Ada."])]))
    result = page_index_flash(str(pdf), summary=False, optimize=False,
                              use_embedded_toc=False)
    assert result["toc_source"] == "detected"
    assert [node["title"] for node in result["structure"]] == ["Budget", "Team"]


def test_no_hierarchy_falls_back_to_page_nodes(tmp_path):
    """Two pages leave one heading after the title claims the other; with no
    hierarchy to infer, the pages themselves are the tree, labelled as such."""
    from pageindex.flash import page_index_flash

    pdf = tmp_path / "two.pdf"
    pdf.write_bytes(_layout_pdf([("Mission", ["The launch is named Skylark."]),
                                 ("Budget", ["The budget is 420 euros."])]))
    result = page_index_flash(str(pdf), summary=False, optimize=False)
    assert result["toc_source"] == "pages"
    assert result["structure"] == [
        {"title": "Page 1", "node_id": "0000", "start_index": 1, "end_index": 1},
        {"title": "Page 2", "node_id": "0001", "start_index": 2, "end_index": 2},
    ]
    assert "page_texts" not in result


def test_flat_fallback_over_limit_skips_model_passes(tmp_path, monkeypatch):
    """A flat tree larger than the managed pipelines accept is returned
    unsummarized: neither optimize nor summary may spend model calls on it."""
    from conftest import build_pdf
    from pageindex.flash import api as flash_api

    monkeypatch.setattr(flash_api, "FLAT_TREE_MAX_NODES", 2)
    monkeypatch.setattr(flash_api, "_optimize", lambda *a, **k: pytest.fail(
        "optimize ran on a refused flat tree"))
    monkeypatch.setattr(flash_api, "_summarize", lambda *a, **k: pytest.fail(
        "summary ran on a refused flat tree"))
    pdf = tmp_path / "letter.pdf"
    pdf.write_bytes(build_pdf(["Alpha body", "Beta body", "Gamma body"]))
    result = flash_api.page_index_flash(str(pdf), summary=True, summary_model="m")
    assert result["toc_source"] == "pages"
    assert [node["title"] for node in result["structure"]] == [
        "Page 1", "Page 2", "Page 3"]
    assert "page_texts" not in result


def test_textless_pdf_is_unreadable(tmp_path):
    """A PDF with no text on any page has nothing to index: an empty
    structure labelled unreadable, no page nodes."""
    from conftest import build_pdf
    from pageindex.flash import page_index_flash

    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(build_pdf(["", ""]))
    result = page_index_flash(str(pdf), summary=False, optimize=False)
    assert result["structure"] == []
    assert result["toc_source"] == "unreadable"


def test_flash_rejection_reason():
    from pageindex.flash.api import FLAT_TREE_MAX_NODES, flash_rejection_reason

    node = {"title": "T", "start_index": 1, "end_index": 1, "nodes": []}
    assert flash_rejection_reason(
        {"structure": [node], "toc_source": "detected"}) is None
    assert flash_rejection_reason(
        {"structure": [node] * 2, "toc_source": "pages"}) is None
    unreadable = flash_rejection_reason({"structure": [], "toc_source": "unreadable"})
    assert "no text layer" in unreadable and "standard" not in unreadable
    oversized = {"structure": [node] * (FLAT_TREE_MAX_NODES + 1),
                 "toc_source": "pages"}
    flat = flash_rejection_reason(oversized)
    assert "no layout structure" in flat and "mode='standard'" in flat
    assert "--mode standard" in flash_rejection_reason(
        oversized, standard_hint="--mode standard")
    assert "could not extract" in flash_rejection_reason({"structure": []})


FLASH_DATA = Path(__file__).parent / "data" / "flash"   # see its make_fixtures.py


def test_page_fallback_covers_every_page(tmp_path):
    """Pages without text still get a node: the flat tree covers the whole
    document, so no page is unreachable."""
    from conftest import build_pdf
    from pageindex.flash import page_index_flash

    pdf = tmp_path / "sparse.pdf"
    pdf.write_bytes(build_pdf(["", "Alpha body", "", "Delta body"]))
    result = page_index_flash(str(pdf), summary=False, optimize=False)
    assert result["toc_source"] == "pages"
    assert [(n["title"], n["start_index"], n["end_index"])
            for n in result["structure"]] == [
        ("Page 1", 1, 1), ("Page 2", 2, 2), ("Page 3", 3, 3), ("Page 4", 4, 4)]
    assert all("nodes" not in n for n in result["structure"])


@pytest.mark.parametrize("name", ["hi_report.pdf", "ar_report.pdf"])
def test_non_latin_document_is_indexed(name):
    """Script never decides whether a document is indexable."""
    from pageindex.flash import page_index_flash
    from pageindex.flash.api import flash_rejection_reason

    result = page_index_flash(str(FLASH_DATA / name), summary=False, optimize=False)
    assert result["structure"]
    assert flash_rejection_reason(result) is None


def test_japanese_headings_are_kept():
    from pageindex.flash.main import extract_toc

    result = extract_toc(str(FLASH_DATA / "ja_report.pdf"), use_embedded_toc=False)
    assert [n["title"] for n in result["structure"]] == [
        "財務ハイライト", "リスク要因", "今後の見通し"]


def test_cross_script_headings_are_kept():
    from pageindex.flash.main import extract_toc

    result = extract_toc(str(FLASH_DATA / "zh_body_en_headings.pdf"),
                         use_embedded_toc=False)
    assert [n["title"] for n in result["structure"]] == [
        "Financial Review", "Risk Factors", "Business Outlook"]


@pytest.mark.parametrize("name", ["hi_report.pdf", "ar_report.pdf"])
def test_non_latin_headings_are_detected(name):
    from pageindex.flash.main import extract_toc

    result = extract_toc(str(FLASH_DATA / name), use_embedded_toc=False)
    assert result["toc_source"] == "detected"
    assert len(result["structure"]) == 3


def test_landscape_deck_title_is_the_slide_heading(tmp_path):
    from pageindex.flash.main import extract_toc

    pdf = tmp_path / "deck.pdf"
    pdf.write_bytes(_deck_pdf())
    assert extract_toc(str(pdf))["doc_title"] == DECK_TITLES[0]
