"""Embedded PDF bookmark (outline dictionary) consumption.

Opt-in feature gated behind ``extract_toc(use_embedded_toc=True)``; the
default path never imports this module.

Bookmarks are read with PDFium, validated, and classified into tiers:

  * FULL      deep and dense enough to be the frame: the bookmark tree
              replaces the detected hierarchy, then detected sections it
              lacks are grafted back in by page range, after dropping
              recurring-label and overlong-title noise.
  * SKELETON  coarse but reliable (typically chapter-only): top-level
              entries pin the chapter frame and detected nodes are
              re-hung under them by page range. With page text available,
              deeper sparse entries are verified against it and filled
              in, and a detected title that disagrees with a same-page
              entry is repaired from the bookmark string (bookmark
              titles never pass through text extraction, so they are
              immune to its garbling).
  * IGNORE    absent or untrustworthy (non-monotonic targets, generic
              "Page 12"-style titles, near-empty, or an enumeration:
              most titles collapse to one template plus a counter, like
              per-page labels or file-name codes): detection stands.
              Templates whose word is a numbering unit ("Chapter N",
              "Part N") are real if coarse frames, not enumerations.

Title comparisons everywhere use a normalized key: lowercased, LaTeX
spans dropped, punctuation stripped, and roman-numeral tokens converted
to digits, so "II. The General Assembly" and "2. The General Assembly"
compare equal. Displayed titles are never rewritten by normalization.

Section boundary semantics: a bookmark target carries no reliable
on-page position, so a section runs onto the page where the next entry
starts (the boundary page is shared; slack, never truncation). Parent
``end_index`` is then promoted to the subtree maximum, the same union
semantics the detected-outline output layer uses.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import pypdfium2 as pdfium

IGNORE, SKELETON, FULL = 1, 2, 3

# Same-page titles at or above this similarity refer to the same section.
_REPAIR_SIMILARITY = 0.7

# Backfill noise filters: a normalized title recurring this often across the
# detected tree is a running label, not a section; a title longer than this
# is a paragraph lead picked up as a heading.
_BACKFILL_DUP_MIN = 3
_BACKFILL_MAX_TITLE = 100

_GENERIC_TITLE = re.compile(
    r"^(?:(?:page|slide|folie|document\s+page)\s*)?\d+$", re.IGNORECASE
)
_LATEX_SPAN = re.compile(r"\$.*?\$")
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_ROMAN = re.compile(r"m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})")
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
_DIGIT_RUN = re.compile(r"[0-9]+")
_ROMAN_EXCLUDED = frozenset({"di", "div", "li", "liv", "mi", "mix", "xi"})

# Words that number a CONTENT unit: a majority template built on one of
# these ("Chapter N") is a genuine coarse frame, not a filing enumeration.
_CONTENT_UNITS = frozenset({
    "chapter", "chap", "ch", "part", "pt", "section", "sec", "book",
    "appendix", "unit", "lesson",
    "章", "部", "节", "篇", "卷", "第章", "第部", "第节", "第篇", "第卷",
})


def _roman_to_arabic(token: str) -> str:
    total = 0
    prev = 0
    for char in reversed(token):
        value = _ROMAN_VALUES[char]
        total = total - value if value < prev else total + value
        prev = max(prev, value)
    return str(total)


def _is_roman(token: str) -> bool:
    return bool(_ROMAN.fullmatch(token)) and token not in _ROMAN_EXCLUDED


def _normalize_title(title: str) -> str:
    tokens = _TOKEN.findall(_LATEX_SPAN.sub("", title.lower()))
    return "".join(
        _roman_to_arabic(token) if _is_roman(token) else token
        for token in tokens
    )


def _title_template(title: str) -> str:
    """Collapse every counter (digit run or roman token) to ``#``."""
    tokens = _TOKEN.findall(_LATEX_SPAN.sub("", title.lower()))
    return "".join(
        "#" if _is_roman(token) else _DIGIT_RUN.sub("#", token)
        for token in tokens
    )


def read_bookmarks(doc_handle: Union[str, Path, BytesIO]) -> list[dict]:
    """Extract raw bookmark entries as {title, level, page}, all 1-based.

    Returns [] when the document has no outline, the handle is not one
    PDFium can open, or the outline is unreadable; the caller treats all
    three as IGNORE.
    """
    stream = doc_handle if isinstance(doc_handle, BytesIO) else None
    restore = stream.tell() if stream is not None else None
    doc = None
    try:
        handle = str(doc_handle) if isinstance(doc_handle, Path) else doc_handle
        doc = pdfium.PdfDocument(handle)
        entries = []
        for item in doc.get_toc():
            if item.page_index is None:
                continue
            title = (item.title or "").strip()
            if not title:
                continue
            entries.append(
                {"title": title, "level": item.level + 1, "page": item.page_index + 1}
            )
        return entries
    except (pdfium.PdfiumError, OSError, ValueError, TypeError):
        return []
    finally:
        if doc is not None:
            doc.close()
        if stream is not None and restore is not None:
            stream.seek(restore)


def validate_bookmarks(entries: list[dict], n_pages: int) -> list[dict]:
    """Drop unusable entries, then re-stack levels over the survivors.

    Bookmarks must walk forward through the document: an entry whose
    target page precedes an earlier entry's is dropped. Level re-stacking
    (position in the pruned ancestor stack) closes the nesting gaps the
    drops leave behind.
    """
    kept = []
    last_page = 0
    for entry in entries:
        if not 1 <= entry["page"] <= n_pages:
            continue
        if entry["page"] < last_page:
            continue
        kept.append(entry)
        last_page = entry["page"]

    out = []
    stack: list[int] = []
    for entry in kept:
        while stack and stack[-1] >= entry["level"]:
            stack.pop()
        stack.append(entry["level"])
        out.append({"title": entry["title"], "level": len(stack), "page": entry["page"]})
    return out


def classify_bookmarks(entries: list[dict], n_pages: int) -> int:
    """Pick the consumption tier for a validated bookmark set.

    Beyond the generic-title and duplicate checks, a set where most
    titles collapse to a single template with a counter is a filing
    enumeration (per-page labels, file-name codes) and is ignored --
    unless the template is one content-numbering word ("Chapter N"),
    which is a genuine coarse frame.
    """
    if len(entries) < 3:
        return IGNORE
    generic = sum(1 for entry in entries if _GENERIC_TITLE.match(entry["title"]))
    if 2 * generic >= len(entries):
        return IGNORE
    templates = Counter(_title_template(entry["title"]) for entry in entries)
    top_template, top_count = templates.most_common(1)[0]
    if ("#" in top_template and 2 * top_count >= len(entries)
            and (top_template.count("#") >= 2
                 or top_template.replace("#", "") not in _CONTENT_UNITS)):
        return IGNORE
    distinct = {_normalize_title(entry["title"]) for entry in entries}
    if 4 * len(distinct) <= len(entries):
        return IGNORE
    max_level = max(entry["level"] for entry in entries)
    if max_level >= 2 and len(entries) >= max(6, n_pages // 20):
        return FULL
    return SKELETON


def _finalize(roots: list[dict], n_pages: int) -> list[dict]:
    """End fill (shared boundary), union promotion, node ids, leaf cleanup."""
    flat: list[dict] = []

    def _collect(nodes: list[dict]) -> None:
        for node in nodes:
            flat.append(node)
            _collect(node.get("nodes") or [])

    _collect(roots)

    for index, node in enumerate(flat):
        boundary = flat[index + 1]["start_index"] if index + 1 < len(flat) else n_pages
        node["end_index"] = max(node["start_index"], boundary)

    def _promote(nodes: list[dict]) -> int:
        end = 0
        for node in nodes:
            children = node.get("nodes")
            if children:
                node["end_index"] = max(node["end_index"], _promote(children))
            end = max(end, node["end_index"])
        return end

    _promote(roots)

    for index, node in enumerate(flat):
        node["node_id"] = str(index).zfill(4)
        if "nodes" in node and not node["nodes"]:
            del node["nodes"]
    return roots


def _build_bookmark_nodes(entries: list[dict]) -> list[dict]:
    roots: list[dict] = []
    stack: list[dict] = []
    for entry in entries:
        node = {
            "title": entry["title"],
            "node_id": "",
            "start_index": entry["page"],
            "end_index": entry["page"],
            "nodes": [],
        }
        while len(stack) >= entry["level"]:
            stack.pop()
        (stack[-1]["nodes"] if stack else roots).append(node)
        stack.append(node)
    return roots


def bookmarks_to_structure(entries: list[dict], n_pages: int) -> list[dict]:
    """Nest validated entries into the output tree shape."""
    return _finalize(_build_bookmark_nodes(entries), n_pages)


def _same_heading(node: dict, chapter: dict) -> bool:
    if node["start_index"] != chapter["start_index"]:
        return False
    node_norm = _normalize_title(node["title"])
    chapter_norm = _normalize_title(chapter["title"])
    if not node_norm or not chapter_norm:
        return False
    return (node_norm == chapter_norm or node_norm.endswith(chapter_norm)
            or chapter_norm.endswith(node_norm)
            or _similarity(node_norm, chapter_norm) >= _REPAIR_SIMILARITY)


def _subtree_max_start(node: dict) -> int:
    deepest = node["start_index"]
    for child in node.get("nodes") or []:
        deepest = max(deepest, _subtree_max_start(child))
    return deepest


def _similarity(a_norm: str, b_norm: str) -> float:
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()


def _title_in_text(entry: dict, page_norms: list[str]) -> bool:
    """The entry's normalized title appears on its target page or a neighbour."""
    norm = _normalize_title(entry["title"])
    if not norm:
        return False
    page = entry["page"]
    window_start = max(0, page - 2)
    window_end = min(len(page_norms), page + 1)
    return any(norm in page_norms[page_pos]
               for page_pos in range(window_start, window_end))


def _repair_titles(structure: list[dict], entries: list[dict]) -> None:
    """Overwrite garbled extracted titles with same-page bookmark strings.

    Only fires when the two titles are fuzzy-similar but neither is a
    suffix of the other: a suffix pair is the numbering-prefix case, where
    the extracted title is the richer one and must be kept.
    """
    entries_by_page: dict[int, list[dict]] = {}
    for entry in entries:
        entries_by_page.setdefault(entry["page"], []).append(entry)

    def visit(node: dict) -> None:
        node_norm = _normalize_title(node["title"])
        best, best_ratio = None, _REPAIR_SIMILARITY
        for entry in entries_by_page.get(node["start_index"], ()):
            entry_norm = _normalize_title(entry["title"])
            if not node_norm or not entry_norm:
                continue
            if node_norm.endswith(entry_norm) or entry_norm.endswith(node_norm):
                continue
            ratio = _similarity(node_norm, entry_norm)
            if ratio >= best_ratio:
                best, best_ratio = entry, ratio
        if best is not None:
            node["title"] = best["title"]
        for child in node.get("nodes") or []:
            visit(child)

    for node in structure:
        visit(node)


def _find_entry_node(root: dict, entry: dict) -> Optional[dict]:
    """Locate a node in ``root``'s subtree that already represents ``entry``."""
    entry_norm = _normalize_title(entry["title"])

    def visit(node: dict) -> Optional[dict]:
        if abs(node["start_index"] - entry["page"]) <= 1:
            node_norm = _normalize_title(node["title"])
            if entry_norm and node_norm and (
                    node_norm.endswith(entry_norm)
                    or entry_norm.endswith(node_norm)
                    or _similarity(node_norm, entry_norm) >= _REPAIR_SIMILARITY):
                return node
        for child in node.get("nodes") or []:
            found = visit(child)
            if found is not None:
                return found
        return None

    return visit(root)


def merge_bookmark_skeleton(
    structure: list[dict], entries: list[dict], n_pages: int,
    page_texts: Optional[list[str]] = None,
) -> list[dict]:
    """Re-hang the detected tree under the bookmark chapter frame (SKELETON tier).

    Top-level entries partition the document by start page. Each detected
    node whose subtree fits inside one chapter range moves under that
    chapter whole; a node whose subtree spans chapter boundaries (the
    detected hierarchy disagrees with the bookmark frame) is dissolved --
    its heading stays behind as a leaf and its children are placed
    individually. A node that duplicates its chapter heading (same start
    page, normalized-equal or numbering-prefixed title) dissolves into it
    without leaving a leaf. Nodes wholly before the first chapter stay at
    the root. All end_index values are recomputed for the merged tree, so
    detected ends derived from the heading-at-page-top signal may gain one
    page of slack.

    With ``page_texts``, two extra passes run: detected titles are
    repaired from same-page bookmark strings, and deeper bookmark entries
    with no counterpart in the merged tree are inserted under their
    chapter (or their nearest inserted ancestor) -- but only when the
    title is non-generic and actually appears in the page text.
    """
    chapters = [entry for entry in entries if entry["level"] == 1]
    if not chapters:
        return structure
    if page_texts:
        _repair_titles(structure, entries)

    chapter_nodes = [
        {
            "title": entry["title"],
            "node_id": "",
            "start_index": entry["page"],
            "end_index": entry["page"],
            "nodes": [],
        }
        for entry in chapters
    ]
    range_end = [
        chapters[index + 1]["page"] if index + 1 < len(chapters) else n_pages + 1
        for index in range(len(chapters))
    ]

    def chapter_index(page: int):
        idx = None
        for index, chapter in enumerate(chapters):
            if page >= chapter["page"]:
                idx = index
            else:
                break
        return idx

    front: list[dict] = []

    def place(node: dict) -> None:
        children = node.get("nodes") or []
        idx = chapter_index(node["start_index"])
        if idx is None:
            if _subtree_max_start(node) < chapters[0]["page"]:
                front.append(node)
            else:
                front.append(dict(node, nodes=[]))
                for child in children:
                    place(child)
            return
        target = chapter_nodes[idx]
        if _subtree_max_start(node) < range_end[idx]:
            if _same_heading(node, target):
                target["nodes"].extend(children)
            else:
                target["nodes"].append(node)
            return
        if not _same_heading(node, target):
            target["nodes"].append(dict(node, nodes=[]))
        for child in children:
            place(child)

    for root in structure:
        place(root)

    if page_texts:
        page_norms = [_normalize_title(text or "") for text in page_texts]
        node_for_entry: dict[int, dict] = {}
        chapter_pos = 0
        stack: list[tuple[int, int]] = []
        for index, entry in enumerate(entries):
            while stack and stack[-1][0] >= entry["level"]:
                stack.pop()
            parent_index = stack[-1][1] if stack else None
            stack.append((entry["level"], index))
            if entry["level"] == 1:
                node_for_entry[index] = chapter_nodes[chapter_pos]
                chapter_pos += 1
                continue
            idx = chapter_index(entry["page"])
            if idx is None:
                continue
            existing = _find_entry_node(chapter_nodes[idx], entry)
            if existing is not None:
                node_for_entry[index] = existing
                continue
            if _GENERIC_TITLE.match(entry["title"]):
                continue
            if not _title_in_text(entry, page_norms):
                continue
            node = {
                "title": entry["title"],
                "node_id": "",
                "start_index": entry["page"],
                "end_index": entry["page"],
                "nodes": [],
            }
            parent = chapter_nodes[idx]
            if parent_index is not None and parent_index in node_for_entry:
                parent = node_for_entry[parent_index]
            siblings = parent.setdefault("nodes", [])
            pos = len(siblings)
            for sibling_pos, sibling in enumerate(siblings):
                if sibling["start_index"] > entry["page"]:
                    pos = sibling_pos
                    break
            siblings.insert(pos, node)
            node_for_entry[index] = node

    return _finalize(front + chapter_nodes, n_pages)


def merge_bookmark_tree(
    structure: list[dict], entries: list[dict], n_pages: int
) -> list[dict]:
    """Graft filtered detected sections into the bookmark tree (FULL tier).

    The bookmark tree is the frame. Each flat bookmark node owns the page
    range up to the next entry's start, and every detected node is
    anchored to the deepest bookmark section active at its start page. A
    detected subtree that fits inside its anchor's range moves under it
    whole; one that spans ranges is dissolved, leaving a leaf, with its
    children placed individually. A node that duplicates its anchor
    heading dissolves into it. Two noise filters prune the whole detected
    tree before placement: titles recurring across it (running labels,
    not sections) and overlong titles (paragraph leads picked up as
    headings) are spliced out, their children promoted into their place.
    Nodes wholly before the first bookmark stay at the root.
    """
    roots = _build_bookmark_nodes(entries)
    flat: list[dict] = []

    def collect(nodes: list[dict]) -> None:
        for node in nodes:
            flat.append(node)
            collect(node["nodes"])

    collect(roots)
    if not flat:
        return structure

    range_end = [
        flat[pos + 1]["start_index"] if pos + 1 < len(flat) else n_pages + 1
        for pos in range(len(flat))
    ]

    def anchor_index(page: int):
        idx = None
        for pos, node in enumerate(flat):
            if page >= node["start_index"]:
                idx = pos
            else:
                break
        return idx

    title_counts: Counter = Counter()

    def count_titles(nodes: list[dict]) -> None:
        for node in nodes:
            title_counts[_normalize_title(node["title"])] += 1
            count_titles(node.get("nodes") or [])

    count_titles(structure)

    def filtered_out(node: dict) -> bool:
        norm = _normalize_title(node["title"])
        if norm and title_counts[norm] >= _BACKFILL_DUP_MIN:
            return True
        return len(node["title"]) > _BACKFILL_MAX_TITLE

    def prune(nodes: list[dict]) -> list[dict]:
        kept: list[dict] = []
        for node in nodes:
            children = prune(node.get("nodes") or [])
            if filtered_out(node):
                kept.extend(children)
            else:
                node["nodes"] = children
                kept.append(node)
        return kept

    def insert_by_page(siblings: list[dict], node: dict) -> None:
        pos = len(siblings)
        for sibling_pos, sibling in enumerate(siblings):
            if sibling["start_index"] > node["start_index"]:
                pos = sibling_pos
                break
        siblings.insert(pos, node)

    front: list[dict] = []

    def place(node: dict) -> None:
        children = node.get("nodes") or []
        idx = anchor_index(node["start_index"])
        if idx is None:
            if _subtree_max_start(node) < flat[0]["start_index"]:
                front.append(node)
            else:
                front.append(dict(node, nodes=[]))
                for child in children:
                    place(child)
            return
        target = flat[idx]
        if _subtree_max_start(node) < range_end[idx]:
            if _same_heading(node, target):
                for child in children:
                    insert_by_page(target["nodes"], child)
            else:
                insert_by_page(target["nodes"], node)
            return
        if not _same_heading(node, target):
            insert_by_page(target["nodes"], dict(node, nodes=[]))
        for child in children:
            place(child)

    for root in prune(structure):
        place(root)
    return _finalize(front + roots, n_pages)


def apply_embedded_toc(
    structure: list[dict], doc_handle: Union[str, Path, BytesIO], n_pages: int,
    page_texts: Optional[list[str]] = None,
) -> tuple[list[dict], str]:
    """Classify the document's bookmarks and apply the matching tier.

    Returns ``(structure, source)`` with source one of ``"bookmarks"``
    (FULL: bookmark frame with filtered detected grafts), ``"hybrid"``
    (SKELETON re-hang), or ``"detected"`` (IGNORE; input returned
    untouched). ``page_texts`` (extracted text per page, 0-indexed)
    enables the SKELETON-tier title-repair and verified fill-in passes.
    """
    entries = validate_bookmarks(read_bookmarks(doc_handle), n_pages)
    tier = classify_bookmarks(entries, n_pages)
    if tier == FULL:
        return merge_bookmark_tree(structure, entries, n_pages), "bookmarks"
    if tier == SKELETON:
        return merge_bookmark_skeleton(structure, entries, n_pages,
                                       page_texts=page_texts), "hybrid"
    return structure, "detected"


__all__ = [
    "IGNORE", "SKELETON", "FULL",
    "read_bookmarks", "validate_bookmarks", "classify_bookmarks",
    "bookmarks_to_structure", "merge_bookmark_skeleton", "merge_bookmark_tree",
    "apply_embedded_toc",
]
