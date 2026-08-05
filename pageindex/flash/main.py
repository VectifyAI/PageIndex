"""
End-to-end orchestrator for the TOC extraction pipeline. Pipeline order: 1. parse character-level spans and page viewport metadata 2. cluster spans into lines 3. compute page statistics 4. detect columns and recluster lines with column awareness 5. remove line-number artifacts and recompute statistics 6. compute document-level statistics 7. cluster lines into blocks and assign reading order 8. classify headers, footers, watermarks, TOC-like pages, captions, references, and body paragraphs 9. detect the document title 10. collect heading candidates and assemble the final outline The ordering is load-bearing: title selection, labeled-section detection,
heading candidate collection, and outline assembly each consume annotations
from the previous stages. ``to_pageindex_tree`` serializes the final outline
into the JSON shape that ``run_pageindex.py`` writes.
"""

from __future__ import annotations

import json
import re
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

# (re is used by the title-reject regex below)

from .blocks import cluster_lines_into_blocks, BlockClusterContext
from .classification import is_body_paragraph, detect_header_footer, HeaderFooterContext, mark_watermarks, mark_toc_and_boilerplate
from .labels import detect_captions, build_caption_regions, CaptionContext
from .model import Rect, numbering_kind, block_text, deaccented_text, Block
from .outline_assembly import (
    build_heading_from_block, is_landscape_or_empty, is_outline_valid, is_chapter_outline_valid, mark_outline_block_types, assemble_outline, compute_max_heading_gap, has_table_or_prominent, OutlineNode, outline_to_dict_tree,
)
from .parser_pdfium_parallel import parse_charlevel_meta_parallel
from .phases import assign_reading_order, PageView, process_page
PageView = PageView  # re-export for type hints
from .stats import compute_doc_stats
from .title import detect_title


# --------------------------------------------------------------------------- #
# References-section dictionary (load once) #
# --------------------------------------------------------------------------- #


_DICT_PATH = Path(__file__).parent / "data" / "dictionaries.json"


def _normalize_text_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).strip().split()).lower()


_REFS_DICT_RAW = json.loads(_DICT_PATH.read_text(encoding="utf-8"))
REFERENCES_KEYWORDS = frozenset(_normalize_text_key(text_value) for text_value in _REFS_DICT_RAW.get("references", []) if text_value)


# --------------------------------------------------------------------------- #
# Document container #
# --------------------------------------------------------------------------- #


class DocumentState:
    """Document-level extraction state: pages, document statistics, and recurring-text frequency map. """

    __slots__ = ("primary_slot", "secondary_slot", "tertiary_slot")

    def __init__(self, pages: list[PageView]):
        self.primary_slot = pages
        self.secondary_slot = None      # set after document statistics are computed
        self.tertiary_slot: dict = {}


# --------------------------------------------------------------------------- #
# References-section detection #
# --------------------------------------------------------------------------- #


def find_references(doc: DocumentState) -> Optional[tuple[int, Block]]:
    """Return ``(page_num, block)`` for the first references heading in reading order."""
    for page in doc.primary_slot:
        for block in (page.secondary_slot or []):
            if block.type != 0:
                continue
            normalized = deaccented_text(block)
            if not normalized or len(normalized) > 80:
                continue
            if normalized in REFERENCES_KEYWORDS:
                return page.page_index, block
            # Allow short numbered prefix: "12. References"
            parts = normalized.split()
            if 1 <= len(parts) <= 4 and parts[-1] in REFERENCES_KEYWORDS:
                return page.page_index, block
    return None


def mark_references(doc: DocumentState, ref: Optional[tuple[int, Block]]) -> None:
    """Tag the references heading itself + everything after as type=3."""
    if ref is None:
        return
    ref_page, ref_block = ref
    seen = False
    for page in doc.primary_slot:
        if page.page_index < ref_page:
            continue
        for block in (page.secondary_slot or []):
            if not seen and block is ref_block:
                seen = True
                block.type = 3
                continue
            if seen:
                block.type = 3


# --------------------------------------------------------------------------- #
# Repeated-text accumulator #
# --------------------------------------------------------------------------- #


def page_by_block_lookup(pages, block) -> Optional[PageView]:
    """Find which page owns ``block``. Used for wrapping labeled blocks."""
    for page in pages:
        if block in (page.secondary_slot or []):
            return page
    return None


# --------------------------------------------------------------------------- #
# End-to-end entry point #
# --------------------------------------------------------------------------- #


def extract_toc(
    doc_handle: Union[str, Path, BytesIO],
    workers: Optional[int] = None,
    use_embedded_toc: bool = True,
) -> dict:
    """Run the full pipeline. Returns a dict shaped like:: { "doc_name": "...", "doc_title": "...", "structure": [ {"title": "...", "start_index": 1, "end_index": 3, "nodes": [...]}, ... ], "has_abstract_or_references_section": False } ``has_abstract_or_references_section`` is True when any TOP-LEVEL outline entry is an abstract-keyword heading or carries the prominent-heading flag (a references-keyword heading, plain or numbered). The near-empty bail and the valid-outline branch both report False. ``workers`` sets the process count for the per-page parallel parser: None = auto (CPU count - 1), 1 forces the sequential path; output is identical either way. ``use_embedded_toc`` consumes the PDF's embedded bookmarks when trustworthy: deep bookmarks become the frame with the detected sections they lack grafted back in, coarse ones become the chapter frame with detected nodes re-hung under them, garbage ones are ignored; adds ``toc_source`` to the result. On by default; pass False for the pure detected structure. """
    # ----- 1) Parse PDF -> flat spans per page --------------------------
    # per-page (view box, /Rotate) comes from the same engine (PDFium) that
    # produced the block coordinates, so the geometry frame is consistent.
    parsed, page_meta = parse_charlevel_meta_parallel(doc_handle, workers=workers)

    # ----- 2) Per-page layout classification ----------------------------------
    # Heading coordinate projection uses the page viewport.
    pages: list[PageView] = []
    for index_value, spans in enumerate(parsed):
        viewport_box_value, rot = page_meta[index_value]
        viewport_x0, viewport_y0, viewport_x1, viewport_y1 = viewport_box_value
        # page bbox uses DISPLAYED (post-/Rotate) dims.
        viewport_width, viewport_height = abs(viewport_x1 - viewport_x0), abs(viewport_y1 - viewport_y0)
        page_width, page_height = (viewport_height, viewport_width) if rot % 180 == 90 else (viewport_width, viewport_height)
        page_bbox = Rect(0, page_width, page_height, 0)
        page = process_page(spans, page_num=index_value + 1, page_bbox=page_bbox)
        if viewport_box_value is not None:
            page.viewport_box, page.rot = viewport_box_value, rot
        pages.append(page)

    # ----- 3) Document-level stats --------------------------------------
    doc = DocumentState(pages)
    doc.secondary_slot = compute_doc_stats(pages)

    # ----- 4) Block clustering per page, then reading order -------------
    for page in pages:
        ctx = BlockClusterContext(doc.secondary_slot, page.bounds, page.primary_slot, page.lines, page.tertiary_slot)
        page.blocks = cluster_lines_into_blocks(ctx)
        assign_reading_order(page, page.blocks)

    # ----- Early empty-outline gate ------------------------------------
    # Short, near-empty, unsupported-script, or mostly-landscape documents
    # emit an empty outline rather than a fabricated structure.
    if (doc.secondary_slot.state_slot <= 300 or doc.secondary_slot.previous_slot <= 200
            or doc.secondary_slot.tertiary_slot in (0, 2, 10) or is_landscape_or_empty(doc)):
        if isinstance(doc_handle, (str, Path)):
            doc_name = Path(str(doc_handle)).name
        else:
            doc_name = "document.pdf"
        result = {
            "doc_name": doc_name,
            "doc_title": None,
            "structure": [],
            "has_abstract_or_references_section": False,
        }
        # Bookmarks need no extracted text, so they can still structure a
        # document this gate wrote off as unreadable.
        if use_embedded_toc:
            from .embedded_toc import apply_embedded_toc
            result["structure"], result["toc_source"] = apply_embedded_toc(
                [], doc_handle, len(pages),
                page_texts=["\n".join(block_text(block)
                                      for block in (page.secondary_slot or []))
                            for page in pages],
            )
        return result

    # ----- 5) Classification: header / footer / watermark / TOC pages ---
    detect_header_footer(HeaderFooterContext(doc, 1))                              # HEADER
    detect_header_footer(HeaderFooterContext(doc, 2))                              # FOOTER
    mark_watermarks(doc)
    mark_toc_and_boilerplate(doc)

    # ----- 6) Body-paragraph flagging (post-classification) -------------
    # Populates body-paragraph flags, page substantive-body flags,
    # and per-page body-style hashes.
    from .model import dominant_style_of as span_style_hash
    for page in pages:
        for block in (page.output_slot or []):
            if block.type == 0:
                block.is_body_paragraph = is_body_paragraph(doc.secondary_slot, page, block)
                if block.is_body_paragraph:
                    page.state_slot = True
                    # The empty style hash is significant for later page-level
                    # membership checks, so it must be retained.
                    page.style_slot.add(span_style_hash(block))

    # ----- 7) Title selection ------------------------------------------
    # Title selection and title-echo marking must run before labeled-section
    # detection and heading collection so title blocks are excluded from both.
    from .classification import bounded_edit_distance, _normalize_text_key
    doc_title: Optional[str] = None
    title_winner = detect_title(doc)
    if title_winner is not None:
        # Emit the full joined title string, preserving inter-block spaces.
        doc_title = title_winner.to_string()
        title_winner.page.auxiliary_slot = True
        for block in title_winner.output_slot:
            block.type = 3
        title_norm = _normalize_text_key(title_winner.to_string()).lower()
        # The body-paragraph break exits only the inner block loop; later
        # pages are still scanned for title-echo headers.
        for candidate_page in doc.primary_slot:
            for candidate_block in (candidate_page.output_slot or []):
                if candidate_block.type != 0:
                    continue
                normalized = deaccented_text(candidate_block).lower()
                if (len(normalized) > 20 and len(title_norm) > 20 and (
                        normalized.startswith(title_norm)
                        or title_norm.startswith(normalized)
                        or title_norm.endswith(normalized))):
                    candidate_page.auxiliary_slot = True
                    candidate_block.type = 3
                    continue
                threshold = 0.2 * min(len(normalized), len(title_norm))
                if bounded_edit_distance(normalized, title_norm, threshold) < threshold:
                    candidate_page.auxiliary_slot = True
                    candidate_block.type = 3
                elif candidate_block.is_body_paragraph:
                    break

    # ----- 8) Keyword-labeled section detection -------------------------
    # Labeled section regions are built here but extended after heading collection.
    caption_context = CaptionContext(doc)
    detect_captions(caption_context)

    # ----- 9) General heading collection --------------------------------
    # Heading collection runs before labeled regions claim their body blocks.
    # The start page skips the title page when a title was found.
    page_lookup: dict[int, int] = {}
    for page in pages:
        for block in (page.secondary_slot or []):
            page_lookup[id(block)] = page.page_index

    from .heading_detection import find_section_openers as _find_section_openers
    title_page_idx = title_winner.page.page_index if title_winner is not None else 0
    section_openers = _find_section_openers(doc, title_page_idx)

    # ----- 10) Extend labeled sections and claim body blocks ------------
    # Each labeled heading keeps its label type; body blocks are marked with
    # the used-as-heading flag so heading collection skips claimed caption/section bodies.
    # The head block type is preserved; claimed body blocks are not retyped.
    caption_regions = build_caption_regions(caption_context)
    for caption_region in caption_regions:
        head_block = caption_region.primary_slot
        head_block.state_slot = head_block.marker_slot
        for body_block in caption_region.output_slot:
            body_block.measure_slot = True

    # NOTE: References-section detection -- intentionally absent ---------
    # Bulk-marking everything after a references heading would hide later
    # appendix headings in some documents, so references detection remains off.
    # ref = find_references(doc)
    # mark_references(doc, ref)

    # NOTE: Ghost-text histogram -- intentionally absent -----------------
    # Recurring text is counted during header/footer/watermark marking. A
    # second doc-wide pass would double-count headers and pollute title scoring.

    # ----- 11) Outline assembly and validation gate ---------------------
    outline_nodes = assemble_outline(doc, section_openers)
    # Validate the assembled outline. Structured outlines must cover enough
    # chapters; unstructured outlines are filtered by script and density gap.
    # The abstract/references signal rides along with this gate: it is False on
    # the valid-outline branch, and on the other branch it is read off the
    # possibly-emptied list once the density filter has run.
    if is_outline_valid(doc, outline_nodes):
        if not is_chapter_outline_valid(doc, outline_nodes):
            outline_nodes = []
        has_abstract_or_references = False
    else:
        mark_outline_block_types(outline_nodes)
        page_count = len(doc.primary_slot)
        if doc.secondary_slot.tertiary_slot == 7 or (
            page_count >= 3
            and compute_max_heading_gap(outline_nodes, 1)["max_gap"] > (0.65 if doc.secondary_slot.tertiary_slot == 4 else 0.85) * page_count
        ):
            outline_nodes = []
        has_abstract_or_references = has_table_or_prominent(outline_nodes)
    if outline_nodes:
        structure = outline_to_dict_tree(outline_nodes, total_pages=len(pages))
    else:
        structure = []

    # ----- 12) Output ---------------------------------------------------
    if isinstance(doc_handle, (str, Path)):
        doc_name = Path(str(doc_handle)).name
    else:
        doc_name = "document.pdf"

    page_texts = []
    for page in pages:
        parts = []
        for block in (page.secondary_slot or []):
            parts.append(block_text(block))
        page_texts.append("\n".join(parts))

    result = {
        "doc_name": doc_name,
        "doc_title": doc_title,
        "structure": structure,
        "has_abstract_or_references_section": has_abstract_or_references,
        "page_texts": page_texts,
    }
    if use_embedded_toc:
        from .embedded_toc import apply_embedded_toc
        result["structure"], result["toc_source"] = apply_embedded_toc(
            structure, doc_handle, len(pages), page_texts=page_texts
        )
    return result


__all__ = ["extract_toc", "DocumentState", "find_references", "mark_references"]
