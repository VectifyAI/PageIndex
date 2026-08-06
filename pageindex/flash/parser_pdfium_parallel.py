"""Per-page parallel driver for the charlevel parser.

Wraps the UNMODIFIED per-page pipeline (``_page_pass1`` / ``_page_pass2`` /
``_page_spans``) in a process pool. PDFium's FFI is not thread-safe and its
handles are process-local, so parallelism uses processes, each opening its
own copy of the document.

Parity contract: per-page processing depends on no cross-page state
except the document-wide identity-matrix Type-3 extent union. An empty union
makes ``_apply_type3_sizes`` a no-op, so per-page == whole-document exactly.
Workers run pass 1 + pass 2 per page assuming the union stays empty and
poison the run the moment any page accumulates an extent; the driver then
discards the parallel attempt and reruns the document on the sequential
path, which is the source of truth. Any other worker failure falls back the
same way, so this entry can only ever return sequential-identical output.

Worker startup pays the full package import chain plus its own document
open; ``min_pages`` routes documents too small to amortize that to the
sequential path directly.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Union

import pypdfium2 as pdfium
import PyPDF2 as _pypdf2  # declared dependency (also imported by pageindex.utils/client)

from .model import Span
from .parser_pdfium_charlevel import (
    parse_charlevel_meta,
    _PdfDoc,
    _page_pass1,
    _page_pass2,
    _page_spans,
)

_MIN_PARALLEL_PAGES = 64


class _Type3Detected(Exception):
    """A page accumulated an identity-matrix Type-3 extent: the document
    needs the cross-page font sizing only the sequential path performs."""


# Per-worker state, set once by _init_worker in each spawned process.
_worker_pdf = None
_worker_pdf_doc = None
_worker_font_maps: dict = {}


def _init_worker(kind: str, payload) -> None:
    global _worker_pdf, _worker_pdf_doc, _worker_font_maps
    # Open the document exactly as parse_charlevel_meta does, including
    # the guarded PyPDF2 open and its separate bytes copy.
    if kind == "path":
        _worker_pdf = pdfium.PdfDocument(payload)
    else:
        _worker_pdf = pdfium.PdfDocument(BytesIO(payload))
    _worker_pdf_doc = None
    if _pypdf2 is not None:
        try:
            if kind == "path":
                _worker_pdf_doc = _PdfDoc(_pypdf2.PdfReader(payload))
            else:
                _worker_pdf_doc = _PdfDoc(_pypdf2.PdfReader(BytesIO(payload)))
        except Exception:
            _worker_pdf_doc = None
    _worker_font_maps = {}


def _run_page(page_idx: int):
    type3_ext: dict = {}
    page, raw_chars, page_vb, page_rot = _page_pass1(
        _worker_pdf, _worker_pdf_doc, page_idx, type3_ext, _worker_font_maps)
    try:
        if type3_ext:
            raise _Type3Detected(page_idx)
        merged = _page_pass2(raw_chars, page_vb, {})
        spans = _page_spans(merged)
    finally:
        page.close()
    return spans, (page_vb, page_rot)


def parse_charlevel_meta_parallel(
    doc_handle: Union[str, Path, BytesIO],
    workers: int | None = None,
    min_pages: int = _MIN_PARALLEL_PAGES,
) -> tuple[list[list[Span]], list]:
    """Parallel-when-possible variant of ``parse_charlevel_meta``.

    Returns the same ``(pages, page_meta)`` with identical content for
    every input. ``workers`` caps the pool size (default: CPU count - 1).
    """
    if isinstance(doc_handle, (str, Path)):
        src = ("path", str(doc_handle))
    elif isinstance(doc_handle, BytesIO):
        src = ("bytes", doc_handle.getvalue())
    else:
        # An already-open PdfDocument cannot be reopened per worker.
        return parse_charlevel_meta(doc_handle)

    probe = pdfium.PdfDocument(BytesIO(src[1]) if src[0] == "bytes" else src[1])
    n_pages = len(probe)
    probe.close()

    max_w = max(1, (os.cpu_count() or 2) - 1)
    w = max(1, min(workers if workers is not None else max_w, max_w, n_pages))
    if w <= 1 or n_pages < min_pages:
        return parse_charlevel_meta(doc_handle)

    executor = ProcessPoolExecutor(
        max_workers=w,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=src,
    )
    try:
        results = list(executor.map(_run_page, range(n_pages)))
    except Exception:
        # _Type3Detected or any worker/pool failure. Cancel what is queued
        # and rerun sequentially; in-flight pages finish in their workers
        # and are discarded (separate processes, no shared PDFium state).
        executor.shutdown(wait=False, cancel_futures=True)
        return parse_charlevel_meta(doc_handle)
    executor.shutdown()

    out = [spans for spans, _meta_entry in results]
    meta = [meta_entry for _spans, meta_entry in results]
    return out, meta


__all__ = ["parse_charlevel_meta_parallel"]
