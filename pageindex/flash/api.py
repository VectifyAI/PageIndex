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


def _thin(structure):
    from ..utils import page_level_thinning, write_node_id
    page_level_thinning(structure)
    write_node_id(structure)


async def _summarize(structure, page_list, model):
    from ..utils import summarize_tree
    await summarize_tree(structure, page_list, model=model)


def _optimize(structure, page_texts, do_expand, model):
    """Merge/expand refinement between extraction and summaries.

    Supersedes ``_thin``: merge collapses everything thinning would, but keeps
    the dropped titles as ``key_items``. Summaries run after, so they describe
    the final tree. Expand reads the same page text the summaries use.
    """
    import asyncio
    from ..tree_optimize import optimize
    lines = [[line_text.strip() for line_text in (page_text or "").splitlines()
              if line_text.strip()]
             for page_text in page_texts]
    outcome = asyncio.run(optimize(structure, page_texts, lines, model=model,
                                   do_expand=do_expand,
                                   page_count=len(page_texts)))
    return {"merges": outcome["merges"], "expands": outcome["expands"],
            "before": outcome["before"], "after": outcome["after"]}


def page_index_flash(pdf, summary=True, summary_model=None,
                     optimize=False, optimize_expand=True,
                     optimize_model=None) -> dict:
    """Build a PageIndex tree structure from a PDF using layout statistics, without an LLM. Args: pdf: path to a PDF file (``str`` or ``pathlib.Path``) or an in-memory binary stream (``io.BytesIO``). summary: if True, generate LLM summaries for each node (requires ``summary_model``). summary_model: the LLM model identifier to use for summary generation. optimize: if True, refine the tree for search cost (merge + expand) before summaries. optimize_expand: if False, optimization only merges - deterministic, no LLM calls. optimize_model: the LLM model for expand (defaults to the summary model). Returns: dict with keys ``doc_name``, ``doc_title``, ``structure`` (a list of nested ``{"title", "start_index", "end_index", "nodes"}`` dicts; page indexes are 1-based) and ``has_abstract_or_references_section`` (True when a top-level entry is an abstract or references heading). With ``optimize`` an ``optimize`` key reports merge/expand counts and before/after search-cost metrics. """
    result = extract_toc(_validate_pdf(pdf))
    structure = result.get("structure", [])
    if optimize and structure:
        result["optimize"] = _optimize(structure, result.get("page_texts") or [],
                                       optimize_expand,
                                       optimize_model or summary_model)
    elif structure:
        _thin(structure)
    if summary and structure:
        import asyncio
        from ..utils import ConfigLoader
        if summary_model is None:
            cfg = ConfigLoader().load()
            summary_model = getattr(cfg, 'summary_model', None) or cfg.model
        page_texts = result.pop("page_texts", [])
        page_list = [(text, 0) for text in page_texts]
        asyncio.run(_summarize(structure, page_list, summary_model))
    else:
        result.pop("page_texts", None)
    return result


__all__ = ["page_index_flash"]
