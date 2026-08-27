"""Public API for PageIndex Flash. The only supported entry point is :func:`page_index_flash`. Everything else in this package is internal pipeline machinery."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pypdfium2 as pdfium

from .main import extract_toc


def _is_pdfium_password_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "password" in msg or "security" in msg or "encrypted" in msg


def _validate_path(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if not path.is_file():
        raise ValueError(f"PDF path is not a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"PDF file must have a .pdf extension: {path}")
    with path.open("rb") as score_value:
        if score_value.read(5) != b"%PDF-":
            raise ValueError(f"File does not look like a PDF: {path}")
    return str(path)


def _validate_stream(stream: BinaryIO) -> BinaryIO:
    try:
        pos = stream.tell()
        head = stream.read(5)
        stream.seek(pos)
    except Exception as exc:  # noqa: BLE001 - normalize stream capability errors
        raise TypeError("PDF stream must be seekable and readable") from exc
    if head != b"%PDF-":
        raise ValueError("Input stream does not look like a PDF")
    return stream


def _validate_pdf(pdf):
    if isinstance(pdf, (str, Path)):
        handle = _validate_path(Path(pdf))
        restore = None
    elif isinstance(pdf, BytesIO):
        handle = _validate_stream(pdf)
        restore = pdf.tell()
    else:
        raise TypeError("page_index_flash(pdf) expects a PDF path or io.BytesIO stream")

    doc = None
    try:
        doc = pdfium.PdfDocument(handle)
        if len(doc) == 0:
            raise ValueError("PDF contains no pages")
    except pdfium.PdfiumError as exc:
        if _is_pdfium_password_error(exc):
            raise ValueError("PDF is encrypted or password-protected") from exc
        raise ValueError(f"Could not open PDF: {exc}") from exc
    finally:
        if doc is not None:
            doc.close()
        if restore is not None:
            pdf.seek(restore)
    return pdf


async def _summarize(structure, page_list, model, concurrency=None, max_words=None):
    from ..utils import summarize_tree
    await summarize_tree(structure, page_list, model=model, concurrency=concurrency,
                         max_words=max_words)


async def _optimize_async(structure, page_texts, do_expand, model, on_final=None,
                          concurrency=None):
    """Merge/expand refinement after extraction, overlapped with the summaries
    when `on_final` is passed; without it the caller runs them after.

    Beyond the merge the default path runs anyway, this adds LLM expand and
    reports before/after search-cost metrics. Expand reads the same page text
    the summaries use.
    """
    from ..tree_optimize import optimize
    lines = [[line_text.strip() for line_text in (page_text or "").splitlines()
              if line_text.strip()]
             for page_text in page_texts]
    outcome = await optimize(structure, page_texts, lines, model=model,
                             do_expand=do_expand, page_count=len(page_texts),
                             on_final=on_final, concurrency=concurrency)
    return {"merges": outcome["merges"], "expands": outcome["expands"],
            "same_page_merges": outcome["same_page_merges"],
            "same_page_dropped": outcome["same_page_dropped"],
            "kept_collapsed": outcome["kept_collapsed"],
            "before": outcome["before"], "after": outcome["after"]}


def _optimize(structure, page_texts, do_expand, model, concurrency=None):
    import asyncio
    return asyncio.run(_optimize_async(structure, page_texts, do_expand, model,
                                       concurrency=concurrency))


async def _optimize_and_summarize(structure, page_texts, optimize_model, summary_model,
                                  concurrency, max_words=None):
    """Expand and summarize on one loop: a node is summarized as soon as
    expand can no longer change it, a parent once its children are done."""
    from ..utils import SummaryScheduler
    scheduler = SummaryScheduler(structure, [(text, 0) for text in page_texts],
                                 model=summary_model, concurrency=concurrency,
                                 max_words=max_words)
    report = await _optimize_async(structure, page_texts, True, optimize_model,
                                   on_final=scheduler.mark_final,
                                   concurrency=concurrency)
    await scheduler.finish()
    return report


def page_index_flash(pdf, summary=True, summary_model=None,
                     optimize: str | bool | None = None, optimize_expand=None,
                     optimize_model=None, summary_concurrency=None,
                     use_embedded_toc=True, summary_max_words=None) -> dict:
    """Build a PageIndex tree structure from a PDF using layout statistics. The tree extraction itself uses no LLM; by default an LLM writes node summaries and expands the tree (``summary=False, optimize=False`` runs fully LLM-free). Args: pdf: path to a PDF file (``str`` or ``pathlib.Path``) or an in-memory binary stream (``io.BytesIO``). summary: if True, generate LLM summaries for each node (requires ``summary_model``). summary_model: the LLM model identifier to use for summary generation. optimize: ``"full"`` for merge + LLM expand (a model unreachable after the retry ladder — a missing credential included — fails the run loudly from expand itself; a per-prompt rejection leaves just that node collapsed), ``"merge"`` for deterministic merge only, ``False`` to disable. ``True`` is accepted as ``"full"`` for backward compatibility; defaults to ``"full"``. Expand needs readable page text, so a bookmark-only or scanned PDF runs the merge half only (``expands`` reports 0). optimize_expand: deprecated — use ``optimize``. Honored only when ``optimize`` is not passed (or is the legacy ``True``): ``False`` maps to ``"merge"``, ``True`` to ``"full"``. optimize_model: the LLM model for expand (defaults to the summary model). summary_concurrency: cap on simultaneous indexing model calls per lane: the summaries, and expand up to its own ceiling of 32; None uses the library defaults (64 and 32). summary_max_words: word cap each node summary is asked to stay within; None uses the library default (150). use_embedded_toc: if True, consume the PDF's embedded bookmarks when trustworthy: deep bookmarks become the frame and the detected sections they lack are grafted back in after noise filtering, coarse ones become the chapter frame with detected nodes re-hung under them (deeper sparse entries are filled in when the page text confirms them, and garbled extracted titles are repaired from the bookmark strings), garbage ones are ignored; adds a ``toc_source`` key to the result. On by default; pass False for the pure detected structure. Returns: dict with keys ``doc_name``, ``doc_title``, ``structure`` (a list of nested ``{"title", "start_index", "end_index", "nodes"}`` dicts; page indexes are 1-based) and ``has_abstract_or_references_section`` (True when a top-level entry is an abstract or references heading). With ``optimize`` an ``optimize`` key reports merge/expand counts and before/after search-cost metrics. """
    if optimize_expand is not None:
        import warnings
        warnings.warn(
            "optimize_expand is deprecated: pass optimize='full', 'merge', "
            "or False. When optimize is not passed it maps onto it (False "
            "-> 'merge', True -> 'full'), so the optimize pass now runs "
            "where the old optimize=False default ran nothing.",
            DeprecationWarning, stacklevel=2)
    if optimize is None or optimize is True:
        # legacy spellings only — an explicit 'full'/'merge' wins
        optimize = "merge" if optimize_expand is False else "full"
    if not optimize:
        optimize = False
    elif optimize not in ("full", "merge"):
        raise ValueError(
            f"optimize must be 'full', 'merge', or False, got {optimize!r}")
    result = extract_toc(_validate_pdf(pdf), use_embedded_toc=use_embedded_toc)
    structure = result.get("structure", [])
    if summary and structure and summary_model is None:
        from ..utils import ConfigLoader
        cfg = ConfigLoader().load()
        summary_model = getattr(cfg, 'summary_model', None) or cfg.model
    # bookmark-only extractions carry no page_texts and scanned ones
    # only empty strings; expand needs text
    pages = result.pop("page_texts", None) or []
    do_expand = optimize == "full" and any(pages)
    if optimize and structure and summary and do_expand:
        import asyncio
        result["optimize"] = asyncio.run(_optimize_and_summarize(
            structure, pages, optimize_model=optimize_model or summary_model,
            summary_model=summary_model, concurrency=summary_concurrency,
            max_words=summary_max_words))
        return result
    if optimize and structure:
        result["optimize"] = _optimize(structure, pages, do_expand,
                                       optimize_model or summary_model,
                                       concurrency=summary_concurrency)
    if summary and structure:
        import asyncio
        page_list = [(text, 0) for text in pages]
        asyncio.run(_summarize(structure, page_list, summary_model,
                               concurrency=summary_concurrency,
                               max_words=summary_max_words))
    elif structure:
        from ..utils import strip_internal_keys
        strip_internal_keys(structure)   # summarize_tree does this on its way out
    return result


__all__ = ["page_index_flash"]
