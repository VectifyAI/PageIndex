"""Embedded PDF bookmark (outline dictionary) consumption.

Opt-in feature gated behind ``extract_toc(use_embedded_toc=True)``; the
default path never imports this module.

Bookmarks are read with PDFium, validated, and classified into tiers:

  * FULL      deep and dense enough to be the frame: the bookmark tree
              replaces the detected hierarchy, then detected sections it
              lacks are grafted back in reading order, under the open
              section their numbering, the printed table of contents or
              their style names, after dropping recurring-label and
              overlong-title noise.
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

Section boundary semantics: a bookmark target's on-page position only
orders headings on its page, so a section runs onto the page where the
next entry starts (the boundary page is shared; slack, never truncation). Parent
``end_index`` is then promoted to the subtree maximum, the same union
semantics the detected-outline output layer uses.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from bisect import bisect_right
from collections import Counter
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional, Union

import pypdfium2 as pdfium

IGNORE, SKELETON, FULL = 1, 2, 3

# Same-page titles at or above this similarity refer to the same section.
_REPAIR_SIMILARITY = 0.7
_DISSOLVE_SIMILARITY = 0.9

# A climb past a same-page frame heading must leave it at least this much
# of the page as content, or it is claiming the section is empty.
_MIN_SECTION_GAP = 0.15

# A bookmark's /XYZ top sits at most this far below its printed heading's top
_DEST_TOL = 0.02
# Reading order jumping this far back up a page has moved to the next column
_COLUMN_JUMP = 0.05

# Backfill noise filters: a normalized title recurring this often across the
# detected tree is a running label, not a section (unless printed like a
# bookmark's heading and never on adjacent pages: a chapter's "Notes"); a
# title longer than this is a paragraph lead picked up as a heading.
_BACKFILL_DUP_MIN = 3
_BACKFILL_MAX_TITLE = 100

_GENERIC_TITLE = re.compile(
    r"^(?:(?:page|slide|folie|document\s+page)\s*)?\d+$", re.IGNORECASE
)
_LATEX_SPAN = re.compile(r"\$.*?\$")
_PAREN_TAIL = re.compile(r"\s*\([^()]*\)\s*$")
_CJK_ENUMERATOR = re.compile(r"^\s*[（(]?[一二三四五六七八九十百]+[、)）]")
_SECTION_KEY = re.compile(
    r"^\s*(?:([A-Za-z]+)\.?\s+(\d{1,3}[A-Z]?(?:\.\d+)*)(?![\w-]|\.\d)"
    r"|(\d{1,3}(?:\.\d+)*)\.\s|(\d{1,3}(?:\.\d+)+)(?![\w-]|\.\d)"
    r"|(第[一二三四五六七八九十百\d]+[章节条]|[一二三四五六七八九十百]+、))")
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


def _title_tokens(title: str) -> list[str]:
    return [
        _roman_to_arabic(token) if _is_roman(token) else token
        for token in _TOKEN.findall(_LATEX_SPAN.sub("", title.lower()))
    ]


@lru_cache(maxsize=1 << 16)
def _normalize_title(title: str) -> str:
    return "".join(_title_tokens(title))


def _title_template(title: str) -> str:
    """Collapse every counter (digit run or roman token) to ``#``."""
    tokens = _TOKEN.findall(_LATEX_SPAN.sub("", title.lower()))
    return "".join(
        "#" if _is_roman(token) else _DIGIT_RUN.sub("#", token)
        for token in tokens
    )


def _dest_xy(doc, page_idx: int, view: Callable[[], tuple],
             boxes: dict[int, Optional[tuple]]) -> tuple[Optional[float], Optional[float]]:
    """(x, y) of an /XYZ target as page fractions from the top-left, None where absent."""
    try:
        mode, pos = view()
        if mode != 1 or len(pos) < 2 or pos[1] in (None, 0):
            return None, None
        if page_idx not in boxes:  # loading a page parses it: once per page
            page = doc[page_idx]
            # get_bbox inherits the page tree's boxes; get_cropbox does not (pdfium bug 1786)
            boxes[page_idx] = None if page.get_rotation() % 360 else page.get_bbox()
            page.close()
        box = boxes[page_idx]
        if box is None:
            return None, None
        left, bottom, right, top = box
        y = (top - pos[1]) / ((top - bottom) or 1)
        x = (pos[0] - left) / ((right - left) or 1) if pos[0] not in (None, 0) else None
    except Exception:
        return None, None
    if not -0.05 <= y <= 1.05:
        return None, None
    return (x if x is not None and 0 <= x <= 1 else None), y


def read_bookmarks(doc_handle: Union[str, Path, BytesIO],
                   page_boxes: Optional[list[tuple]] = None) -> list[dict]:
    """Extract raw bookmark entries as {title, level, page, x, y}, level and
    page 1-based, x and y the target's position (None unless /XYZ).

    ``page_boxes`` is flash's per-page (view box, /Rotate), the frame its
    headings' ``_y`` uses; without it each target page is loaded once.

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
        boxes: dict[int, Optional[tuple]] = {
            i: None if rot % 360 else tuple(box)
            for i, (box, rot) in enumerate(page_boxes or ()) if box is not None}
        for item in doc.get_toc():
            if hasattr(item, "get_dest"):
                dest = item.get_dest()
                if not dest:
                    continue
                page_idx, title, view = dest.get_index(), item.get_title(), dest.get_view
            else:
                page_idx, title = item.page_index, item.title
                view = lambda item=item: (item.view_mode, item.view_pos)
            if page_idx is None:
                continue
            title = (title or "").strip()
            if not title:
                continue
            x, y = _dest_xy(doc, page_idx, view, boxes)
            entries.append({"title": title, "level": item.level + 1,
                            "page": page_idx + 1, "x": x, "y": y})
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
        out.append({"title": entry["title"], "level": len(stack), "page": entry["page"],
                    "x": entry.get("x"), "y": entry.get("y")})
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


# "3D", "2.5D", "10-K" and "2019 Survey" are words, not section numbers
_NUMERAL_RE = re.compile(
    r"^([A-Z]\.(?:\d+(?:\.\d+)*)?|\d{1,3}(?:\.\d+)*)(?![\w-]|\.\d)")


def _numeral_prefix(title: str) -> Optional[str]:
    match = _NUMERAL_RE.match(title.strip())
    return match.group(1) if match else None


def _numeral_child(parent_title: str, child_title: str) -> bool:
    """'1'->'1.1', '1.1'->'1.1.1', 'D.'->'D.1' are parent->child pairs."""
    parent = _numeral_prefix(parent_title)
    child = _numeral_prefix(child_title)
    if not parent or not child:
        return False
    if parent.endswith(".") and len(parent) == 2:
        return child.startswith(parent) and len(child) > 2
    return child.startswith(parent + ".") and len(child) > len(parent)


def _numbered_under(parent_title: str, title: str) -> bool:
    """``title`` is numbered directly under ``parent_title`` ("9.3.2.1", "9.3.2")."""
    parent, mine = _numeral_prefix(parent_title), _numeral_prefix(title)
    return (parent is not None and mine is not None
            and mine.rstrip(".").rpartition(".")[0] == parent.rstrip("."))


def _numeral_after(title: str, other: str) -> bool:
    """``title`` is numbered after ``other`` in the same scheme ("4" after "3.1")."""
    def key(numeral: str) -> tuple:
        head, *rest = numeral.rstrip(".").split(".")
        return (head if head.isalpha() else int(head), *map(int, rest))

    mine, theirs = _numeral_prefix(title), _numeral_prefix(other)
    if not mine or not theirs:
        return False
    mine_key, theirs_key = key(mine), key(theirs)
    return type(mine_key[0]) is type(theirs_key[0]) and mine_key > theirs_key


def _numbered_apart(title: str, other: str) -> bool:
    """Both titles carry section numbers and they differ ("13.6.1" vs "13.6")."""
    mine, theirs = _numeral_prefix(title), _numeral_prefix(other)
    return mine is not None and theirs is not None and mine != theirs


def _ends_apart(title: str, other: str) -> bool:
    """Both titles end in a number and the numbers differ ("Part I" vs "Part II")."""
    mine, theirs = _title_tokens(title), _title_tokens(other)
    return (bool(mine and theirs) and mine[-1].isdigit()
            and theirs[-1].isdigit() and mine[-1] != theirs[-1])


def _numeral_token(token: str) -> bool:
    return token.isdigit() or len(token) == 1 or not re.search("[a-z]", token)


def _numbered_variant(title: str, base_norm: str) -> bool:
    """``title`` is ``base_norm`` behind numbering only ("5", "Item 7", "第三章")."""
    tokens = _title_tokens(title)
    rest = len("".join(tokens)) - len(base_norm)
    head: list[str] = []
    for token in tokens:
        if rest <= 0:
            break
        head.append(token)
        rest -= len(token)
    if rest != 0:
        return False
    if head and not _numeral_token(head[0]):
        head = head[1:]
    return bool(head) and all(_numeral_token(token) for token in head)


def _numbered_print(title: str, frame_title: str) -> bool:
    """``title`` is ``frame_title`` behind numbering ("9 Conclusion")."""
    base = _normalize_title(frame_title)
    return (bool(base) and _normalize_title(title).endswith(base)
            and _numbered_variant(title, base))


def _continued(title: str, frame_title: str) -> bool:
    """``title`` carries ``frame_title`` onto a later page ("... (Continued)")."""
    return (bool(_PAREN_TAIL.search(title))
            and _same_text(_PAREN_TAIL.sub("", title), frame_title))


def _same_text(title: str, other: str) -> bool:
    """Equal once normalized, letter spacing ("R E F E R E N C E S") aside."""
    return (_normalize_title(title) == _normalize_title(other)
            or _normalize_title("".join(title.split()))
            == _normalize_title("".join(other.split())))


def _section_key(title: str) -> Optional[tuple[str, str]]:
    match = _SECTION_KEY.match(title)
    if not match:
        return None
    label, *numbers = match.groups()
    return (label or "").lower(), next(number for number in numbers if number)


def _shared_head(tokens: list[str], other: list[str]) -> int:
    count = 0
    while count < min(len(tokens), len(other)) and tokens[count] == other[count]:
        count += 1
    return count


def _reference_resolved(tokens: list[str], frame_tokens: list[str]) -> bool:
    """The print resolves a reference the bookmark kept ("Table 1" for "tab:results")."""
    if _numeral_token(frame_tokens[0]):
        return False
    body = list(tokens)
    while len(body) > 1 and _numeral_token(body[0]) and body[0] != frame_tokens[0]:
        body.pop(0)
    head = _shared_head(body, frame_tokens)
    foot = _shared_head(body[head:][::-1], frame_tokens[head:][::-1])
    printed = body[head:len(body) - foot]
    booked = frame_tokens[head:len(frame_tokens) - foot]
    return (head + foot >= 2 and 0 < len(printed) <= 3 and printed[-1].isdigit()
            and bool(booked) and not any(token.isdigit() for token in booked))


def _printed_as(title: str, frame_title: str) -> bool:
    """``title`` is ``frame_title`` printed headed, annotated, cut or renamed."""
    tokens, frame_tokens = _title_tokens(title), _title_tokens(frame_title)
    tail = len(frame_tokens)
    if not tokens or not tail:
        return False
    numeral = _numeral_prefix(frame_title)
    if frame_tokens in (tokens[-tail:],
                        _title_tokens(_PAREN_TAIL.sub("", title))[-tail:]):
        return numeral is not None or _numeral_prefix(title) is None
    if (len(tokens) == tail > 1 and tokens[1:] == frame_tokens[1:]
            and _numeral_token(tokens[0]) != _numeral_token(frame_tokens[0])):
        return True
    if _reference_resolved(tokens, frame_tokens):
        return True
    if numeral is not None and _numeral_prefix(title) is None:
        core = frame_tokens[len(_title_tokens(numeral)):]
        if len(tokens) < len(core) and core[:len(tokens)] == tokens:
            return True
    if _numeral_prefix(title) != numeral:
        return False
    if len(tokens) < tail and frame_tokens[:len(tokens)] == tokens:
        return True
    if tokens[:tail] == frame_tokens:
        bare = _LATEX_SPAN.sub("", title)
        words = list(_TOKEN.finditer(bare))
        if len(words) > tail and re.match(r"[a-z]{2,}\b(?![.)])",
                                          bare[words[tail].start():]):
            return True
        remark = tokens[tail:]
        return (len(remark) <= 3 and title.rstrip().endswith(".")
                and not any(token.isdigit() for token in remark)
                and any(len(token) > 1 for token in remark))  # initials ("B.A.") remark nothing
    rest = iter(tokens)
    if (len(tokens) > tail > 1 and tokens[0] == frame_tokens[0]
            and tokens[-1] == frame_tokens[-1]
            and all(token in rest for token in frame_tokens)):
        return True
    key = _section_key(title)
    return key is not None and key == _section_key(frame_title)


def _extends(title: str, other: str) -> bool:
    """One title is the other with words added ("Mathematics, B.A." and "Mathematics")."""
    short, longer = sorted((_title_tokens(title), _title_tokens(other)), key=len)
    return len(short) < len(longer) and longer[:len(short)] == short


def _enumerated_alike(title: str, other: str) -> bool:
    """Both titles open with a section number, or both with "一、"-style ones."""
    return ((_numeral_prefix(title) is not None
             and _numeral_prefix(other) is not None)
            or bool(_CJK_ENUMERATOR.match(title) and _CJK_ENUMERATOR.match(other)))


def _same_heading(node: dict, chapter: dict,
                  similarity: float = _REPAIR_SIMILARITY) -> bool:
    node_norm = _normalize_title(node["title"])
    chapter_norm = _normalize_title(chapter["title"])
    node_squeezed = _normalize_title("".join(node["title"].split()))
    if not node_norm or not chapter_norm:
        return False
    # A bookmark can target the page before its printed title
    if node["start_index"] == chapter["start_index"] + 1:
        return _same_text(node["title"], chapter["title"])
    if node["start_index"] != chapter["start_index"]:
        return False
    return (_same_text(node["title"], chapter["title"])
            or _numbered_print(node["title"], chapter["title"])
            or any(chapter_norm.endswith(norm)
                   and _numbered_variant(chapter["title"], norm)
                   for norm in (node_norm, node_squeezed))
            or (not _numbered_apart(node["title"], chapter["title"])
                and (_similarity(node_norm, chapter_norm) >= similarity
                     or (_similarity(node_squeezed, _normalize_title(
                             "".join(chapter["title"].split()))) >= similarity
                         and not _ends_apart(node["title"], chapter["title"])))))


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
    page, normalized-equal or numbering-prefixed title; or the next page,
    normalized-equal title) dissolves into it without leaving a leaf.
    Nodes wholly before the first chapter stay at the root. All end_index
    values are recomputed for the merged tree, so detected ends derived
    from the heading-at-page-top signal may gain one page of slack.

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


_NUMBER_WORD = (r"(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[- ]?)?"
                r"(?:one|two|three|four|five|six|seven|eight|nine)"
                r"|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen"
                r"|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety")
# A heading's label line printed on its own: "Chapter 4", "TWENTY", "第三章"
_LABEL_ONLY = re.compile(
    r"^\s*(?:(?:chapter|chap\.?|part|section|appendix|lecture|unit|book|module)"
    r"\s+(?:\d+|[ivxlc]+|[a-z])\.?"
    r"|第[一二三四五六七八九十百零\d]+[章部篇节]|" + _NUMBER_WORD + r")\s*$", re.IGNORECASE)
_BACK_WORDS = re.compile(
    r"\b(?:references|bibliography|acknowledge?ments?|appendix|appendices|annex(?:es)?"
    r"|index|glossary|supplementary)\b|参考文献|致\s*谢|附\s*录", re.IGNORECASE)
_APPENDIX_LETTER = re.compile(r"^[A-H](?:\.\d+)*[.:]?\s+[A-Z]")
# Flash heading-candidate types the merge reads
_KEYWORD_TYPE, _REFERENCES_TYPE = 6, 7


def _numeral_depth(title: str) -> int:
    numeral = _numeral_prefix(title)
    return 0 if numeral is None else len(numeral.rstrip(".").split("."))


def _toc_same(title: str, other: str) -> bool:
    return _same_text(title, other) or _numbered_print(title, other) or _numbered_print(other, title)


# Section numbers as addresses: "4.3" is the third child of "4"
_CJK_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}
_CJK_UNIT = {"章": ("chapter",), "部分": ("label", "part"), "篇": ("label", "part"),
             "部": ("label", "part")}
_ADDR_CJK_LABEL = re.compile(
    r"第\s*([0-9]{1,3}|[零〇一二三四五六七八九十百]{1,4})\s*(章|节|節|条|條|部分|篇|部|讲|講|课|課)")
_ADDR_LABEL = re.compile(
    r"(?i:(chapter|part|section|item|note|article|appendix|lecture|unit|schedule|exhibit"
    r"|annex|book|volume))\s+"
    r"([0-9]{1,3}(?:\.[0-9]{1,3})*|[IVXLC]{1,6}|[A-Z])([A-Z])?(?=$|[\s.:,;\-–—)])")
_ADDR_WORD_LABEL = re.compile(r"([A-Z][a-z]{2,15})\s+([0-9]{1,3})([A-Z])?\s*[:.\-–—](?=\s|$)")
_CAPTION_WORDS = frozenset({"table", "figure", "fig", "equation", "algorithm", "listing",
                            "page", "box", "example", "exercise", "problem"})
_ADDR_DIGIT = re.compile(r"(\(?)\s*(?:§\s*)?([0-9]{1,3}(?:\s*\.\s*[0-9]{1,3})*)(?:\s*([.、):]))?")
_ADDR_ROMAN = re.compile(r"(\(?)\s*([IVXLCDM]{1,7}|[ivxlcdm]{1,7})(?:\s*([.、):]))?")
_ADDR_LETTER = re.compile(r"(\(?)\s*([A-Za-z])((?:\s*\.\s*[0-9]{1,3})*)(?:\s*([.、):]))?")
_ADDR_CJK = re.compile(r"(\(?)\s*([零〇一二三四五六七八九十百]{1,4})(?:\s*([.、):]))?")
# Label words whose numbers are section numbers ("Section 4" is address "4")
_NUMBERED_LABELS = frozenset({("chapter",), ("label", "section"), ("label", "article")})


def _roman_number(text: str) -> Optional[int]:
    text = text.lower()
    return int(_roman_to_arabic(text)) if text and _ROMAN.fullmatch(text) else None


def _cjk_number(text: str) -> Optional[int]:
    if text.isdigit():
        return int(text)
    total = current = 0
    for char in text:
        if char in "十百":
            total += (current or 1) * (10 if char == "十" else 100)
            current = 0
        elif char in _CJK_DIGIT:
            current = _CJK_DIGIT[char]
        else:
            return None
    return total + current


def _readings(text: Optional[str]) -> list[tuple[tuple, tuple]]:
    """The addresses a heading line can mean, as [(scheme, numbers)]; a lone "I." reads as Roman and as a letter."""
    if not text:
        return []
    t = unicodedata.normalize("NFKC", text).lstrip(" \t​﻿")
    m = _ADDR_CJK_LABEL.match(t)
    if m and (value := _cjk_number(m.group(1))) is not None:
        return [(_CJK_UNIT.get(m.group(2), ("label", m.group(2))), (value,))]
    m = _ADDR_LABEL.match(t)
    if m:
        word, number, suffix = m.group(1).lower(), m.group(2), m.group(3)
        parts: list[float]
        if number[0].isdigit():
            parts = [int(x) for x in number.split(".")]
        elif len(number) > 1 or word in ("part", "book", "volume", "lecture", "unit"):
            if (roman := _roman_number(number)) is None:
                return []
            parts = [roman]
        else:
            parts = [ord(number.upper()) - 64]
        if suffix:
            parts[-1] += (ord(suffix.upper()) - 64) / 100  # Item 1 < 1A < 1B < 2
        return [(("chapter",) if word == "chapter" else ("label", word), tuple(parts))]
    m = _ADDR_WORD_LABEL.match(t)
    if m and m.group(1).lower() not in _CAPTION_WORDS:
        value = int(m.group(2)) + ((ord(m.group(3)) - 64) / 100 if m.group(3) else 0)
        return [(("label", m.group(1).lower()), (value,))]

    def tail_ok(m, need_close: bool) -> bool:
        rest = t[m.end():]
        if not rest.strip() or (rest[0].isascii() and (rest[0].isalnum() or rest[0] in "-/%,+*=<>$&")):
            return False
        bracket, close = m.group(1), m.groups()[-1] or ""
        if bracket and close != ")":
            return False
        return not need_close or bool(close and rest[0].isspace())

    m = _ADDR_DIGIT.match(t)
    if m and tail_ok(m, False):
        values = tuple(int(x) for x in re.split(r"\s*\.\s*", m.group(2)))
        close = (m.group(3) or "") if len(values) == 1 else ""
        return [((bool(m.group(1)), "digit", close), values)]
    m = _ADDR_ROMAN.match(t)
    if m and (roman := _roman_number(m.group(2))) is not None:
        numeral, close = m.group(2), m.group(3) or ""
        kind = "ROMAN" if numeral.isupper() else "roman"
        if len(numeral) > 1 and (tail_ok(m, True) or (
                numeral.isupper() and tail_ok(m, False) and roman <= 39
                and t[m.end():][:1].isspace())):
            return [((bool(m.group(1)), kind, close), (roman,))]
        if len(numeral) == 1 and numeral.upper() in "IVX" and tail_ok(m, True):
            letter = "LETTER" if numeral.isupper() else "letter"
            return [((bool(m.group(1)), kind, close), (roman,)),
                    ((bool(m.group(1)), letter, close), (ord(numeral.upper()) - 64,))]
    m = _ADDR_LETTER.match(t)
    if m and tail_ok(m, not m.group(3)):
        values = (ord(m.group(2).upper()) - 64, *(int(x) for x in re.findall(r"[0-9]+", m.group(3))))
        kind = "LETTER" if m.group(2).isupper() else "letter"
        close = (m.group(4) or "") if len(values) == 1 else ""
        return [((bool(m.group(1)), kind, close), values)]
    m = _ADDR_CJK.match(t)
    if m and (value := _cjk_number(m.group(2))) is not None and m.group(3) and tail_ok(m, False):
        return [((bool(m.group(1)), "cjk", m.group(3)), (value,))]
    return []


def _family(scheme: tuple) -> tuple:
    return scheme if scheme[0] in ("chapter", "label") else scheme[:2]


def _style_after(lines: Optional[list], head: str) -> Optional[str]:
    """Style of the line after the lines printing ``head`` (detection's ``_lines``), None unless head ends a line."""
    seq, want, got = lines or [], _normalize_title(head), ""
    for k, (text, _) in enumerate(seq[:-1]):
        got += _normalize_title(text)
        if got == want:
            return seq[k + 1][1]
        if not want.startswith(got):
            return None
    return None


def _larger_alike(style: Optional[str], other: Optional[str]) -> bool:
    """``style`` differs from ``other`` in size alone and is larger by over half a point.

    Fingerprints read "face weight size", "|C" when caps-heavy; small caps are set small by design.
    """
    if not style or not other or " " not in style or " " not in other:
        return False
    face, size = style.rsplit(" ", 1)
    other_face, other_size = other.rsplit(" ", 1)
    size, _, caps = size.partition("|")
    other_size, _, other_caps = other_size.partition("|")
    if face != other_face or caps != other_caps:
        return False
    try:
        return float(size) > float(other_size) + 0.5
    except ValueError:
        return False


def _list_families(readings: list) -> set:
    """The list numberings among an address's readings ("2." digit, "（三）" CJK), label words aside."""
    return {_family(scheme) for scheme, _ in readings if scheme[0] not in ("chapter", "label")}


def _relation(sa: tuple, a: tuple, sh: tuple, h: tuple):
    """Heading address h against bookmark address a: "ancestor", ("fork", k) (h is a child of a[:k]), "stop", or None."""
    if _family(sa) != _family(sh):
        return None
    if len(a) < len(h) and h[:len(a)] == a:
        return "ancestor"
    k = next((i for i in range(min(len(a), len(h))) if a[i] != h[i]), None)
    if k is None or a[k] > h[k] or len(h) != k + 1:
        return "stop"
    if k == len(a) - 1 and len(h) == 1 and sa[-1] != sh[-1]:
        return "stop"  # siblings are numbered alike: "1." after "1)" is another list
    return ("fork", k)


def _unique_key(scheme: tuple, values: tuple) -> tuple:
    family = _family(scheme)
    return ((False, "digit") if family in _NUMBERED_LABELS else family, values)


def merge_bookmark_tree(
    structure: list[dict], entries: list[dict], n_pages: int,
    toc: Optional[list[dict]] = None,
    line_of: Optional[Callable[[int, str], Optional[str]]] = None,
) -> list[dict]:
    """Graft filtered detected sections into the bookmark tree (FULL tier).

    The bookmark tree is the frame; the detected headings it lacks are
    placed by one walk in reading order. ``toc`` is the printed table of
    contents (``read_printed_toc``); ``line_of(page, title)`` returns the
    printed line of a bookmark's heading on its target page, for
    bookmarks whose title carries no section number.

    Copies. A detected node whose normalized title matches
    a frame heading starting on its page -- equal (letter spacing aside),
    apart by a numbering prefix (also when a numbered bookmark prints
    letter-spaced), or apart by character-level garble under the same
    section number (letter-spaced too, unless the two end in different
    numbers: "Part I", "Part II") -- or equals one starting on the page
    before, is that bookmark's printed heading and dissolves without a
    trace. A bookmark no node has matched yet also takes the first
    node on its page that prints its title after a header (for a numbered
    bookmark, any text before its number: "Chapter 2", a merged parent
    heading), before a parenthetical, a short remark or a lowercase
    continuation ("... for Fiscal 2015"), cut off (its number may go too),
    with words inserted inside it ("Line of business and Corporate
    equity"), with a number for its label word ("A. Proofs" for "Appendix
    Proofs"), with a short reference resolved where the bookmark kept a
    label ("Table 1" for "tab:wordnet-results"), or in new words under the
    same section key ("Note 26", "2.", "2.1", "第十二条", "二、"), or the
    first node on the next page that prints it behind numbering ("9
    Conclusion"). One that has its heading takes, from the next page, only
    a copy under the same section number (a running header), not "I.
    Findings" after "3 Findings", or its title carried on with a
    parenthetical ("J15ZJ (15 Ton) (Continued)"). A node printed at a
    bookmark's target that runs on past the bookmark's title into a
    capitalized heading has that heading glued on ("Item 1C. Cybersecurity
    Risk management and strategy"): it keeps only the rest, which never
    dissolves back into that bookmark, and takes the style of its own
    line; its detected children printed in that style are its peers.
    Dissolving holds fuzzy matches to a higher bar than title repair: at
    the repair bar, wordy-overlapping titles of distinct sections
    false-match, and a false dissolve deletes a real heading, while an
    escaped garbled duplicate only leaves a stray leaf. A print that adds
    words to a bookmark's title away from its target is another heading,
    and initials are no remark ("Mathematics, B.A." below "Mathematics").
    A bookmark learns style and position from its copies; detection
    exports the ``_style``, ``_lines`` (each line's text and style),
    ``_y``, ``_type`` and ``_prominent`` keys and ``extract_toc`` strips
    them after the merge.

    Slots. Each bookmark takes a place in the detected reading order: at
    its first copy, else after the headings printed above its /XYZ target
    on its page (column first on a two-column page: a heading's column is
    where reading order jumps back up the page, a bookmark's is its
    target's x), else at its page's start -- never between a same-page
    detected heading and its children. A same-page heading numbered
    before the bookmark goes ahead of it; a label line ("Chapter 4") or a
    heading numbered inside the bookmark goes with it.

    Walk. One stack holds the open sections. A bookmark closes those at
    or below its level and opens its own. A detected heading goes under
    one open section, at most one level below the heading before it and
    never so high that it closes the parent of the next bookmark. Its
    parent is, by the first evidence that holds:

      * the top level, for a heading the detector typed References or
        Acknowledgments (or flagged prominent) when only back matter
        follows;
      * the open bookmark whose address is the heading's minus its last
        part ("5.3" for "5.3.1"), over a box title detected above it;
      * its detected parent, while that is open and spans no later
        bookmark (or numbers the heading directly);
      * the parent its section number names. A number is an address
        ("4.3" is the third child of "4") read within its own family:
        digits, Roman, letters, CJK, or a label word ("Chapter 8"); a
        lone "I." reads as Roman and as a letter. Against the nearest open bookmark
        numbered alike, the heading is its descendant (parent: that
        bookmark), forks off after it at some depth (parent: the
        ancestor there, when bookmark levels minus address lengths agree
        for that family within the top-level section), or cannot follow
        it (no evidence); only a fork may cross an unnumbered open
        bookmark. With none open, the next bookmark numbered as the
        heading's younger sibling names the parent ("1 Introduction"
        before bookmark "2 Related Work"). Either way the next numbered
        bookmark in the parent's section must come after the heading. A
        bookmark's address is read from its copy, else its title, else
        the numbered line printing its title at its target; one unlike
        all its numbered siblings' is dropped (a table row taken for
        the heading), and an address a bookmark holds is no evidence for
        a heading unless the bookmarks themselves repeat it;
      * the parent the printed table of contents lists it under, when
        that contents nests the bookmarks it lists the way they nest
        themselves -- all of them, or those under that parent;
      * else a climb from the deepest open bookmark, past a candidate
        only on positive evidence the heading is its peer: a numbering
        prefix outside the candidate's and after it, or the same style
        fingerprint (face, weight and size as one hash; a larger raw size
        in a different face is not seniority) that leaves the candidate
        real content below its heading on a shared page and no
        same-style detected child enumerated like the heading, or a style
        that differs from the candidate's in size alone and is larger
        ("EXHIBIT B" after a smaller "8. Miscellaneous"); never past a
        candidate numbered in another list style ("2." below "（三）").
        A climb must reach a numbering parent or the top level (just below
        the root when one root bookmark titles the whole document); one
        stopped halfway leaves the heading at the deepest open bookmark.

    Two noise filters prune the whole detected tree before placement:
    titles recurring across it (running labels, not sections) and
    overlong titles (paragraph leads picked up as headings) are spliced
    out, their children promoted into their place. A recurring title
    printed in the style of a bookmark's own heading and never on
    adjacent pages is a section each chapter repeats (Notes, Exercises,
    Loan modifications) and stays. Headings before the first bookmark
    stay at the root.
    """
    roots = _build_bookmark_nodes(entries)
    flat: list[dict] = []
    parent_pos: list[Optional[int]] = []
    last_pos: list[int] = []

    def collect(nodes: list[dict], parent: Optional[int]) -> None:
        for node in nodes:
            pos = len(flat)
            flat.append(node)
            parent_pos.append(parent)
            last_pos.append(pos)
            collect(node["nodes"], pos)
            last_pos[pos] = len(flat) - 1

    collect(roots, None)
    if not flat:
        return structure
    toc = toc or []
    levels = [entry["level"] for entry in entries]
    dest_y = [entry.get("y") for entry in entries]
    dest_x = [entry.get("x") for entry in entries]
    frame_ids = {id(frame) for frame in flat}
    # The top level a heading can reach: below a lone root titling the document
    top_level = 2 if len(roots) == 1 else 1

    def anchor_index(page: int):
        idx = None
        for pos, node in enumerate(flat):
            if page >= node["start_index"]:
                idx = pos
            else:
                break
        return idx

    def off_target(node: dict, b: int) -> bool:
        """The heading is printed on bookmark b's page, away from b's target."""
        return (node["start_index"] == flat[b]["start_index"] and dest_y[b] is not None
                and node.get("_y") is not None and abs(node["_y"] - dest_y[b]) > _DEST_TOL)

    def dissolve_target(node: dict, idx: int,
                        matched: dict[int, Optional[str]]) -> Optional[int]:
        numeral = _numeral_prefix(node["title"])
        same = []
        pos = idx
        while pos >= 0 and flat[pos]["start_index"] >= node["start_index"] - 1:
            # a print adding words to a bookmark's title away from its target is another heading
            if _same_heading(node, flat[pos], _DISSOLVE_SIMILARITY) and not (
                    _extends(node["title"], flat[pos]["title"]) and off_target(node, pos)):
                same.append(pos)
            pos -= 1
        if same:
            # "7.1.2" and "7.2.2" print two same-titled bookmarks: each takes the first one free
            pos = next((p for p in reversed(same) if numeral is not None
                        and matched.get(p, numeral) == numeral), same[0])
            matched.setdefault(pos, numeral)
            return pos
        pos = idx
        while pos >= 0 and flat[pos]["start_index"] >= node["start_index"] - 1:
            frame_title = flat[pos]["title"]
            same_page = flat[pos]["start_index"] == node["start_index"]
            free = pos not in matched or (
                not same_page and numeral is not None and matched[pos] == numeral)
            if same_page:
                takes = free and _printed_as(node["title"], frame_title)
            else:
                takes = ((free and _numbered_print(node["title"], frame_title))
                         or _continued(node["title"], frame_title))
            if takes:
                matched.setdefault(pos, numeral)
                return pos
            pos -= 1
        return None

    def _style_climbs(node: dict, cand: dict) -> bool:
        return (node.get("_style") is not None
                and node["_style"] == cand.get("_style"))

    def _claims_empty(node: dict, cand: dict) -> bool:
        if cand["start_index"] != node["start_index"]:
            return False
        node_y = node.get("_y")
        cand_y = cand.get("_y")
        if node_y is None or cand_y is None:
            return True
        return node_y - cand_y < _MIN_SECTION_GAP

    title_pages: dict[str, list[int]] = {}
    # Styles a bookmark's own printed heading uses
    section_styles: set[str] = set()
    style_matches: dict[int, Optional[str]] = {}

    def survey(nodes: list[dict]) -> None:
        for node in nodes:
            norm = _normalize_title(node["title"])
            title_pages.setdefault(norm, []).append(node["start_index"])
            idx = anchor_index(node["start_index"])
            if (node.get("_style") is not None and idx is not None
                    and dissolve_target(node, idx, style_matches) is not None):
                section_styles.add(node["_style"])
            survey(node.get("nodes") or [])

    survey(structure)

    def filtered_out(node: dict) -> bool:
        norm = _normalize_title(node["title"])
        pages = sorted(title_pages.get(norm, ()))
        if (norm and len(pages) >= _BACKFILL_DUP_MIN
                and not (node.get("_style") in section_styles
                         and all(b - a > 1 for a, b in zip(pages, pages[1:])))):
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

    reading: list[dict] = []
    det_parent: dict[int, Optional[dict]] = {}
    last_desc: dict[int, int] = {}

    def read(nodes: list[dict], parent: Optional[dict]) -> None:
        for node in nodes:
            reading.append(node)
            det_parent[id(node)] = parent
            read(node.get("nodes") or [], node)
            last_desc[id(node)] = len(reading) - 1

    read(prune(structure), None)
    read_index = {id(node): pos for pos, node in enumerate(reading)}

    # A print at a bookmark's target that runs on past its title carries the next heading
    on_page: dict[int, list[int]] = {}
    for b, frame in enumerate(flat):
        on_page.setdefault(frame["start_index"], []).append(b)
    bookmarked = {(frame["start_index"], _normalize_title(frame["title"])) for frame in flat}
    stripped: dict[int, int] = {}
    for node in reading:
        if node.get("_y") is None or (node["start_index"], _normalize_title(node["title"])) in bookmarked:
            continue
        title = " ".join(node["title"].split())
        for b in on_page.get(node["start_index"], ()):
            head = " ".join(flat[b]["title"].split())
            if (dest_y[b] is not None and abs(node["_y"] - dest_y[b]) <= _DEST_TOL
                    and title.startswith(head + " ")):
                if (_same_heading(node, flat[b], _DISSOLVE_SIMILARITY)
                        or _printed_as(node["title"], flat[b]["title"])):
                    break  # the bookmark's print with a remark ("... Not applicable.")
                rest = title[len(head):].strip()
                first = rest[:1]
                if ((first.isupper() or first.isdigit() or not (first.isascii() or first.islower()))
                        and re.search(r"[^\W\d_]", rest)):
                    node["title"] = rest
                    stripped[id(node)] = b
                    style = _style_after(node.get("_lines"), head)
                    if style is not None:
                        node["_style"] = style
                        for child in node.get("nodes") or []:
                            if child.get("_style") == style:
                                det_parent[id(child)] = None  # printed alike: a peer
                break

    # A heading's column: reading order jumping back up its page starts the next one
    column: dict[int, int] = {}
    page_column: dict[int, int] = {}
    multi_column: set[int] = set()
    last_seen: dict[int, float] = {}
    for node in reading:
        page, y = node["start_index"], node.get("_y")
        col = page_column.get(page, 0)
        if y is not None and page in last_seen and y < last_seen[page] - _COLUMN_JUMP:
            col += 1
            multi_column.add(page)
        page_column[page] = column[id(node)] = col
        if y is not None:
            last_seen[page] = y

    # Without a target x, a bookmark's column comes from its page's bookmarks jumping back up
    bookmark_column: list[int] = []
    for b, frame in enumerate(flat):
        same_page = b > 0 and flat[b - 1]["start_index"] == frame["start_index"]
        col = bookmark_column[b - 1] if same_page else 0
        y, above = dest_y[b], dest_y[b - 1] if same_page else None
        if y is not None and above is not None and y < above - _COLUMN_JUMP:
            col += 1
        bookmark_column.append(col)

    def before_target(node: dict, b: int) -> bool:
        """The heading is printed above bookmark b's target on b's page."""
        y, x = dest_y[b], dest_x[b]
        if y is None or node.get("_y") is None:
            return False
        if flat[b]["start_index"] not in multi_column:
            return node["_y"] < y - _DEST_TOL
        col = (1 if x >= 0.45 else 0) if x is not None else bookmark_column[b]
        return (column[id(node)], node["_y"]) < (col, y - _DEST_TOL)

    copy_of: dict[int, int] = {}
    found: dict[int, Optional[str]] = {}
    first_copy: dict[int, int] = {}
    for pos, node in enumerate(reading):
        idx = anchor_index(node["start_index"])
        target = None if idx is None else dissolve_target(node, idx, found)
        if target is not None and stripped.get(id(node)) == target:
            target = None  # what follows a bookmark's title is not its heading
        if target is not None:
            copy_of[id(node)] = target
            first_copy.setdefault(target, pos)
            frame = flat[target]
            for key in ("_style", "_y"):
                if frame.get(key) is None and node.get(key) is not None:
                    frame[key] = node[key]

    def numbered_inside(title: str, b: int) -> bool:
        """Numbered under bookmark b or beside a numbered bookmark below it ("6.2" beside "6.1")."""
        mine = _numeral_prefix(title)
        if mine is None or "." not in mine.rstrip("."):
            return False
        if _numeral_child(flat[b]["title"], title):
            return True
        parent = mine.rstrip(".").rpartition(".")[0]
        return any((theirs := _numeral_prefix(flat[p]["title"])) is not None
                   and theirs.rstrip(".").rpartition(".")[0] == parent
                   for p in range(b + 1, last_pos[b] + 1))

    # Bookmark b comes right before reading[slots[b]]
    slots: list[int] = []
    prev = 0
    for b, frame in enumerate(flat):
        page = frame["start_index"]
        if b in first_copy and first_copy[b] >= prev:
            slot = first_copy[b]
        else:
            slot = prev
            while slot < len(reading) and (
                    reading[slot]["start_index"] < page
                    or (reading[slot]["start_index"] == page
                        and before_target(reading[slot], b))):
                slot += 1
            # an imprecise target never splits a same-page section from its children
            for head in range(prev, slot):
                if (reading[head]["start_index"] == page
                        and id(reading[head]) not in copy_of
                        and last_desc[id(reading[head])] >= slot):
                    slot = head
                    break
        # a same-page heading numbered before the bookmark goes ahead ("9.1.2" read after "9.2")
        ahead = slot + 1 if b in first_copy and slot == first_copy[b] else slot
        while (ahead < len(reading) and reading[ahead]["start_index"] == page
               and id(reading[ahead]) not in copy_of
               and _numeral_after(frame["title"], reading[ahead]["title"])
               and not _numeral_child(frame["title"], reading[ahead]["title"])
               and (_numeral_depth(reading[ahead]["title"])
                    >= _numeral_depth(frame["title"]))):
            ahead += 1
            slot = ahead
        # a label line or a heading numbered inside the bookmark goes with it
        while (slot > prev and reading[slot - 1]["start_index"] == page
               and id(reading[slot - 1]) not in copy_of
               and not (slot - 1 == prev and b > 0
                        and flat[b - 1]["start_index"] == page)
               and (_LABEL_ONLY.match(reading[slot - 1]["title"])
                    or numbered_inside(reading[slot - 1]["title"], b))):
            slot -= 1
        slots.append(slot)
        prev = slot

    # Bookmarks ("b", position) and surviving headings ("d", node) in order
    seq: list[tuple] = []
    b = 0
    for pos in range(len(reading) + 1):
        while b < len(flat) and slots[b] == pos:
            seq.append(("b", b))
            b += 1
        if pos < len(reading) and id(reading[pos]) not in copy_of:
            seq.append(("d", reading[pos]))
    seq_pos = {id(ref): k for k, (kind, ref) in enumerate(seq) if kind == "d"}
    bookmark_seq_pos = {ref: k for k, (kind, ref) in enumerate(seq) if kind == "b"}
    # A heading sits no higher than the next bookmark's level, or that bookmark's parent closes
    next_level = [1] * len(seq)
    upcoming = 1
    for k in range(len(seq) - 1, -1, -1):
        next_level[k] = upcoming
        if seq[k][0] == "b":
            upcoming = levels[seq[k][1]]

    def later_bookmarks(node: dict):
        return (ref for kind, ref in seq[seq_pos[id(node)] + 1:] if kind == "b")

    def back_matter_parent(node: dict, floor: int) -> Optional[int]:
        """The top level for a detected References/Acknowledgments heading when only back matter follows."""
        if not (node.get("_type") == _REFERENCES_TYPE or node.get("_prominent")
                or (node.get("_type") == _KEYWORD_TYPE
                    and "acknowledg" in node["title"].lower())):
            return None
        top = max(top_level, floor)
        for ref in later_bookmarks(node):
            if levels[ref] <= top:
                printed = reading[first_copy[ref]]["title"] if ref in first_copy else ""
                if not any(_BACK_WORDS.search(t) or _APPENDIX_LETTER.match(t)
                           for t in (flat[ref]["title"], printed)):
                    return None
                break
        return top - 2

    def spans(node: dict) -> bool:
        """A bookmark lands inside the detected subtree, so its nesting is not trusted."""
        start, end = read_index[id(node)], last_desc[id(node)]
        j = bisect_right(slots, start)
        return j < len(slots) and (
            slots[j] <= end or _subtree_max_start(node) >= flat[j]["start_index"])

    def bookmark_readings(b: int) -> list:
        """Bookmark b's addresses: from its copy, else its title, else its line at its target."""
        if b in first_copy:
            got = _readings(reading[first_copy[b]]["title"])
            if got:
                return got
        got = _readings(flat[b]["title"])
        if got or line_of is None:
            return got
        line = line_of(flat[b]["start_index"], flat[b]["title"])
        return _readings(line) if line else []

    addresses = [bookmark_readings(b) for b in range(len(flat))]
    # An address unlike all its numbered siblings' is misread (a table row taken for the heading)
    siblings: dict[Optional[int], list[int]] = {}
    for b in range(len(flat)):
        siblings.setdefault(parent_pos[b], []).append(b)
    unlike = [
        b for b, readings in enumerate(addresses)
        if readings and (others := [scheme for s in siblings[parent_pos[b]] if s != b
                                    for scheme, _ in addresses[s][:1]])
        and not any(_family(scheme) == _family(other)
                    for scheme, _ in readings for other in others)]
    for b in unlike:
        addresses[b] = []
    # An address is unique unless the bookmarks repeat it (numbering restarting per part)
    key_count = Counter(_unique_key(scheme, values)
                        for readings in addresses for scheme, values in readings)
    taken = {key for key, count in key_count.items() if count == 1}
    restarts = {key[0] for key, count in key_count.items() if count > 1}

    def free_address(scheme: tuple, values: tuple) -> bool:
        key = _unique_key(scheme, values)
        return key not in taken or key[0] in restarts

    def top_of(b: int) -> int:
        while (up := parent_pos[b]) is not None and levels[up] >= top_level:
            b = up
        return b

    # Bookmark level minus address length per numbering family, also per top-level section
    offsets: dict[tuple, Counter] = {}
    for b, readings in enumerate(addresses):
        for scheme, values in readings[:1]:
            for key in (_family(scheme), (_family(scheme), top_of(b))):
                offsets.setdefault(key, Counter())[levels[b] - len(values)] += 1

    def depth_agrees(scheme: tuple, a: tuple, b: int) -> bool:
        got = offsets.get((_family(scheme), top_of(b)))
        if not got or sum(got.values()) < 3:
            got = offsets.get(_family(scheme))
            if not got or sum(got.values()) < 3:
                return True
        offset, count = got.most_common(1)[0]
        return levels[b] - len(a) == offset and count >= 0.8 * sum(got.values())

    def ordered(first: tuple, then: tuple) -> bool:
        """Address ``first`` comes before ``then`` in reading order."""
        common = min(len(first), len(then))
        return (first[:common] < then[:common]
                or (first[:common] == then[:common] and len(first) < len(then)))

    def follows_in_scope(node: dict, sh: tuple, h: tuple, p: int) -> bool:
        """The next numbered bookmark inside the parent's section comes after h."""
        for ref in later_bookmarks(node):
            if levels[ref] <= p + 1:
                return True
            for scheme, n in addresses[ref]:
                if _family(scheme) == _family(sh):
                    return ordered(h, n)
        return True

    def after_previous(node: dict, sh: tuple, h: tuple, p: int) -> bool:
        """The last numbered bookmark inside the parent's section comes before h."""
        start = bookmark_seq_pos[stack[p][3]] + 1 if p >= 0 else 0
        for kind, ref in reversed(seq[start:seq_pos[id(node)]]):
            if kind != "b":
                continue
            for scheme, a in addresses[ref]:
                if _family(scheme) == _family(sh):
                    return ordered(a, h)
        return True

    def younger_sibling_parent(node: dict, sh: tuple, h: tuple,
                               floor: int) -> Optional[int]:
        """The parent of the next bookmark numbered as h's younger sibling ("2 Related Work" for "1 Introduction")."""
        for ref in later_bookmarks(node):
            if not any(_family(scheme) == _family(sh) for scheme, _ in addresses[ref]):
                continue
            if not any(_family(scheme) == _family(sh)
                       and (len(h) > 1 or scheme[-1] == sh[-1])
                       and len(n) == len(h) and n[:-1] == h[:-1] and n[-1] > h[-1]
                       for scheme, n in addresses[ref]):
                return None
            parent = parent_pos[ref]
            p = -1 if parent is None else next(
                (k for k in range(len(stack))
                 if stack[k][2] == "b" and stack[k][3] == parent), None)
            if (p is None or not after_previous(node, sh, h, p)
                    or (p + 2 if p >= 0 else 1) < floor):
                return None
            return p
        return None

    def address_parent(node: dict, sh: tuple, h: tuple,
                       floor: int) -> Optional[int]:
        """The parent address h names against the nearest open bookmark numbered alike."""
        crossed = False
        for kb in range(len(stack) - 1, -1, -1):
            if stack[kb][2] != "b":
                continue
            ref = stack[kb][3]
            if not addresses[ref]:
                crossed = True  # a number leaves an unnumbered section only by a fork
                continue
            rels = [(rel, sa, a) for sa, a in addresses[ref]
                    if (rel := _relation(sa, a, sh, h)) is not None]
            if not rels:
                continue
            if len({rel for rel, _, _ in rels}) > 1:
                return None
            rel, sa, a = rels[0]
            if rel == "stop":
                return None
            if rel == "ancestor":
                if crossed:
                    return None
                p = kb
            else:
                p = kb - (len(a) - rel[1])
                if p < -1 or not depth_agrees(sa, a, ref):
                    return None
                for j in range(max(p, 0), kb):
                    if stack[j][2] != "b":
                        continue
                    xs = [x for sx, x in addresses[stack[j][3]] if _family(sx) == _family(sa)]
                    if xs and not any(x == a[:len(x)] and kb - j == len(a) - len(x)
                                      for x in xs):
                        return None
            if not follows_in_scope(node, sh, h, p) or (p + 2 if p >= 0 else 1) < floor:
                return None
            return p
        return younger_sibling_parent(node, sh, h, floor)

    def numbered_parent(node: dict, floor: int) -> Optional[int]:
        """The one parent the heading's free addresses name, else None."""
        got = {p for sh, h in _readings(node["title"]) if free_address(sh, h)
               for p in [address_parent(node, sh, h, floor)] if p is not None}
        return got.pop() if len(got) == 1 else None

    def direct_parent(node: dict, floor: int) -> Optional[int]:
        """The open bookmark addressed as the heading minus its last part ("5.3" for "5.3.1")."""
        for sh, h in _readings(node["title"]):
            if len(h) < 2 or not free_address(sh, h):
                continue
            for kb in range(len(stack) - 1, -1, -1):
                if stack[kb][2] != "b":
                    continue
                if not addresses[stack[kb][3]]:
                    break
                rels = [(rel, a) for sa, a in addresses[stack[kb][3]]
                        if (rel := _relation(sa, a, sh, h)) is not None]
                if not rels:
                    continue
                if (all(rel == "ancestor" and len(a) == len(h) - 1 for rel, a in rels)
                        and follows_in_scope(node, sh, h, kb) and kb + 2 >= floor):
                    return kb
                break
        return None

    listed = [i for i, item in enumerate(toc) if item["page"] is not None]

    def near(i: int, page: int) -> bool:
        return abs(toc[i]["page"] - page) <= 1

    def prints(i: int, b: int) -> bool:
        """TOC entry i is bookmark b, by its title or its printed heading's."""
        return (_toc_same(toc[i]["title"], flat[b]["title"])
                or (b in first_copy
                    and _toc_same(toc[i]["title"], reading[first_copy[b]]["title"])))

    toc_bookmark: dict[int, int] = {}
    for i in listed:
        b = next((b for b in range(len(flat)) if near(i, flat[b]["start_index"])
                  and b not in toc_bookmark.values() and prints(i, b)), None)
        if b is not None:
            toc_bookmark[i] = b

    def toc_agrees(i: int, b: int) -> bool:
        """The TOC lists bookmark b under the bookmark it nests under."""
        p = toc[i]["parent"]
        got = None if p is None else toc_bookmark[p]
        return got == parent_pos[b] or (got is None and parent_pos[b] == 0 and len(roots) == 1)

    checked = agreed = 0
    # TOC parent entry (None: the top) -> every bookmark listed under it agrees
    group_agrees: dict[Optional[int], bool] = {}
    for i, b in toc_bookmark.items():
        p = toc[i]["parent"]
        if p is not None and p not in toc_bookmark:
            continue
        ok = toc_agrees(i, b)
        checked += 1
        agreed += ok
        group_agrees[p] = group_agrees.get(p, True) and ok
    toc_trusted = checked >= 3 and agreed == checked

    def group_trusted(p: Optional[int]) -> bool:
        """The TOC nests the bookmarks under entry p, and p itself, as the bookmarks do."""
        if not group_agrees.get(p):
            return False
        if p is None:
            return True
        pp = toc[p]["parent"]
        return (pp is None or pp in toc_bookmark) and toc_agrees(p, toc_bookmark[p])

    def toc_parent(node: dict, floor: int) -> Optional[int]:
        """The open bookmark the printed TOC lists the heading under."""
        i = next((i for i in listed if near(i, node["start_index"])
                  and _toc_same(toc[i]["title"], node["title"])), None)
        if i is None or i in toc_bookmark:
            return None  # unlisted, or listed as a bookmark: this is its print
        p = toc[i]["parent"]
        if not (toc_trusted or group_trusted(p)):
            return None
        if p is None:
            k = 0 if len(roots) == 1 else -1
        else:
            k = next((j for j in range(len(stack) - 1, -1, -1) if stack[j][2] == "b" and (
                _toc_same(toc[p]["title"], stack[j][0]["title"])
                or (stack[j][3] in first_copy and _toc_same(
                    toc[p]["title"], reading[first_copy[stack[j][3]]]["title"])))), None)
            if k is None:
                return None
        if (stack[k][1] + 1 if k >= 0 else 1) < floor:
            return None
        return k

    for frame in flat:
        frame["nodes"] = []
    out_roots: list[dict] = []
    stack: list[list] = []  # [node, depth, "b" | "d", bookmark position | detected node]

    def attach(node: dict, k: int) -> None:
        (out_roots if k < 0 else stack[k][0]["nodes"]).append(node)

    def evidence(node: dict, b: int) -> str:
        cand = flat[b]
        title, cand_title = node["title"], cand["title"]
        if _numeral_prefix(cand_title) is not None and _numeral_prefix(title) is not None:
            if _numeral_child(cand_title, title):
                return "parent"
            return "peer" if _numeral_after(title, cand_title) else "stop"
        mine, theirs = _list_families(_readings(title)), _list_families(addresses[b])
        if mine and theirs and not mine & theirs:
            return "stop"  # siblings are numbered alike: "2." is no peer of "（三）"
        if _larger_alike(node.get("_style"), cand.get("_style")):
            return "peer"
        if (not _style_climbs(node, cand) or _claims_empty(node, cand)
                or any(id(child) not in frame_ids and _style_climbs(node, child)
                       and _enumerated_alike(title, child["title"])
                       for child in cand["nodes"])):
            return "stop"
        return "peer"

    def choose(node: dict, floor: int) -> int:
        """Stack index of the heading's parent (-1: the root)."""
        k = back_matter_parent(node, floor)
        if k is not None:
            return min(k, len(stack) - 1)
        k = direct_parent(node, floor)
        if k is not None:
            return k
        k = len(stack) - 1
        while k >= 0 and stack[k][2] == "d":
            ref = stack[k][3]
            if det_parent[id(node)] is ref and (
                    not spans(ref) or _numbered_under(ref["title"], node["title"])):
                return k
            k -= 1
        if k < 0:
            return -1
        for named in (numbered_parent, toc_parent):
            got = named(node, floor)
            if got is not None:
                return got
        anchor = k
        while True:
            if parent_pos[stack[k][3]] is None and len(roots) == 1:
                return k
            verdict = evidence(node, stack[k][3])
            if verdict == "parent":
                return k
            below = k - 1
            if verdict == "stop" or (stack[below][1] + 1 if below >= 0 else 1) < floor:
                return anchor
            if below < 0:
                return -1
            k = below

    for k, (kind, ref) in enumerate(seq):
        if kind == "b":
            while stack and stack[-1][1] >= levels[ref]:
                stack.pop()
            while stack and not (stack[-1][2] == "b" and stack[-1][3] == parent_pos[ref]):
                stack.pop()
            attach(flat[ref], len(stack) - 1)
            stack.append([flat[ref], len(stack) + 1, "b", ref])
        else:
            parent = choose(ref, next_level[k])
            node = dict(ref, nodes=[])
            attach(node, parent)
            del stack[parent + 1:]
            stack.append([node, len(stack) + 1, "d", ref])
    return _finalize(out_roots, n_pages)


# A printed TOC line: title, dot leaders or a space, page number
_TOC_ENTRY = re.compile(
    r"^(?P<title>.*?\S)(?:\s*[.·•…_]{2,}\s*|\s+)(?P<num>\d{1,4}|[ivxlcdm]{1,7})$", re.IGNORECASE)
_TOC_PAGE_NUMBER = re.compile(r"^(?:\d{1,4}|[ivxlcdm]{1,7})$", re.IGNORECASE)
# A label cell printed apart from its title cell: "Item 1A.", "Part II", "2.", "第三节"
_TOC_LABEL = re.compile(
    r"^\s*(?:(?:item|part|chapter|section|article|schedule|note|appendix)\s*[\dIVXLC]+[A-Z]?"
    r"|\d{1,3}(?:\.\d{1,3})*|[A-Z]|[IVXLC]{1,6}|第[一二三四五六七八九十\d]+[章节部篇])"
    r"\s*[.:、．]?\s*$", re.IGNORECASE)


def _toc_items(pages: list[list[tuple]]) -> list[dict]:
    """Printed TOC entries as {page, title, x, num}, label cells and wrapped titles joined."""
    items = []
    for page, rows in enumerate(pages, 1):
        rows = [list(row) for row in rows]  # [text, x0, x1, y, style]
        joined: set[int] = set()
        for i, row in enumerate(rows):
            if not _TOC_LABEL.match(row[0]):
                continue
            cells = [j for j, cell in enumerate(rows) if j != i and j not in joined
                     and abs(cell[3] - row[3]) < 0.004 and cell[1] > row[2] - 0.01
                     and not _TOC_PAGE_NUMBER.match(cell[0].strip())
                     and not _TOC_LABEL.match(cell[0])]
            if cells:
                j = min(cells, key=lambda j: rows[j][1])
                row[0], row[2] = row[0].strip() + " " + rows[j][0].strip(), rows[j][2]
                joined.add(j)
        rows = [row for j, row in enumerate(rows) if j not in joined]
        numbers = [row for row in rows if _TOC_PAGE_NUMBER.match(row[0].strip())]
        pending = None  # (title so far, x0, y, style) of a wrapped entry
        for text, x0, x1, y, style in rows:
            text = text.strip()
            if _TOC_PAGE_NUMBER.match(text):
                continue
            m = _TOC_ENTRY.match(text)
            num = None
            if m:
                title, num = m["title"], m["num"]
            else:
                beside = [row for row in numbers if abs(row[3] - y) < 0.006 and row[1] > x1 - 0.01]
                if beside:
                    title, num = text, beside[0][0].strip()
            wraps = (pending is not None and pending[3] == style
                     and 0 < y - pending[2] < 0.03 and x0 >= pending[1] - 0.005)
            if num is None:
                pending = ((pending[0] + " " + text, pending[1], y, style) if wraps
                           else (text, x0, y, style))
                continue
            if wraps:
                title, x0 = pending[0] + " " + title, pending[1]
            pending = None
            title = re.sub(r"[\s.·•…_]+$", "", title)
            if len(_normalize_title(title)) >= 3:
                items.append({"page": page, "title": title, "x": x0,
                              "num": int(num) if num.isdigit() else None})
    return items


def _toc_levels(items: list[dict], gap: float = 0.008) -> list[int]:
    """Levels by indent, indents closer than ``gap`` sharing one."""
    ranks, prev, level = {}, None, 0
    for x in sorted({item["x"] for item in items}):
        if prev is None or x - prev > gap:
            level += 1
        ranks[x], prev = level, x
    return [ranks[item["x"]] for item in items]


def read_printed_toc(pages: list[list[tuple]], entries: list[dict]) -> list[dict]:
    """The printed TOC as [{title, parent entry index, PDF page}] from each page's
    (text, x0, x1, y, style) TOC lines; [] under five entries or none a bookmark."""
    items = _toc_items(pages)
    if len(items) < 5:
        return []
    def ends(title: str) -> set[str]:  # titles _toc_same can match end alike
        return {_normalize_title(title)[-1:], _normalize_title("".join(title.split()))[-1:]}

    by_end: dict[str, list[int]] = {}
    for j, entry in enumerate(entries):
        for end in ends(entry["title"]):
            by_end.setdefault(end, []).append(j)
    offsets = Counter(entries[j]["page"] - item["num"] for item in items if item["num"] is not None
                      for j in sorted({j for end in ends(item["title"]) for j in by_end.get(end, ())})
                      if _toc_same(item["title"], entries[j]["title"]))
    if not offsets:
        return []
    offset = offsets.most_common(1)[0][0]
    levels = _toc_levels(items)
    # An indent takes the level of the bookmarks listed at it; indents no bookmark sits at keep their order
    known: dict[int, Counter] = {}
    for item, rank in zip(items, levels):
        if item["num"] is None:
            continue
        page = item["num"] + offset
        j = next((j for end in ends(item["title"]) for j in by_end.get(end, ())
                  if abs(entries[j]["page"] - page) <= 1 and _toc_same(item["title"], entries[j]["title"])), None)
        if j is not None:
            known.setdefault(rank, Counter())[entries[j]["level"]] += 1
    if known:
        level_of = {rank: votes.most_common(1)[0][0] for rank, votes in known.items()}
        first = min(level_of)

        def calibrated(rank: int) -> int:
            below = max((r for r in level_of if r <= rank), default=None)
            return level_of[below] + rank - below if below is not None else level_of[first] - (first - rank)

        levels = [calibrated(rank) for rank in levels]
    toc, open_entries = [], []
    for i, (item, level) in enumerate(zip(items, levels)):
        while open_entries and levels[open_entries[-1]] >= level:
            open_entries.pop()
        toc.append({"title": item["title"], "parent": open_entries[-1] if open_entries else None,
                    "page": None if item["num"] is None else item["num"] + offset})
        open_entries.append(i)
    return toc


# A heading line led by its section number, and a label printed on the line above its title
_LINE_NUMBER = re.compile(
    r"^\s*((?:(?:chapter|part|section|appendix|item|note|article|lecture|unit)\s+)?"
    r"(?:[0-9]{1,3}(?:\s*\.\s*[0-9]{1,3})*[A-Z]?|[IVXLC]{1,6}|[A-Za-z](?:\s*\.\s*[0-9]{1,3})*)"
    r"|第\s*[0-9一二三四五六七八九十百]+\s*[章节条部分篇]|[一二三四五六七八九十]+\s*、"
    r"|[（(]\s*[0-9一二三四五六七八九十]+\s*[)）])\s*[.:：、)）]?\s*(.*)$", re.IGNORECASE)
_LINE_LABEL = re.compile(
    r"^\s*(?:chapter|part|lecture|unit|appendix)\s+[0-9IVXLC]+\s*$"
    r"|^\s*第\s*[0-9一二三四五六七八九十百]+\s*[章部分篇]\s*$", re.IGNORECASE)


@lru_cache(maxsize=1 << 16)
def _line_key(text: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).lower())


def _heading_line_reader(page_texts: list[str]) -> Callable[[int, str], Optional[str]]:
    """line_of(page, title): the numbered line printing title on page or the next, None if numbers disagree."""
    cache: dict[int, list[str]] = {}

    def lines(page: int) -> list[str]:
        if page not in cache:
            text = page_texts[page - 1] if 1 <= page <= len(page_texts) else ""
            cache[page] = [line.strip() for line in (text or "").splitlines()]
        return cache[page]

    def prints(rest: str, title: str) -> bool:
        a, b = _line_key(rest), _line_key(title)
        return bool(a) and bool(b) and (a == b or (len(a) >= 8 and b.startswith(a)))

    def line_of(page: int, title: str) -> Optional[str]:
        found = []
        for at in (page, page + 1):
            text = lines(at)
            for j, line in enumerate(text):
                m = _LINE_NUMBER.match(line)
                if m and m.group(2) and m.group(1).strip() and prints(m.group(2), title):
                    found.append((m.group(1).strip(), line))
                elif prints(line, title) and j > 0 and _LINE_LABEL.match(text[j - 1]):
                    found.append((text[j - 1].strip(), text[j - 1] + " " + line))
        return found[0][1] if found and len({number for number, _ in found}) == 1 else None

    return line_of


def apply_embedded_toc(
    structure: list[dict], doc_handle: Union[str, Path, BytesIO], n_pages: int,
    page_texts: Optional[list[str]] = None,
    toc_lines: Optional[list[list[tuple]]] = None,
    page_boxes: Optional[list[tuple]] = None,
) -> tuple[list[dict], str]:
    """Classify the document's bookmarks and apply the matching tier.

    Returns ``(structure, source)`` with source one of ``"bookmarks"``
    (FULL: bookmark frame with filtered detected grafts), ``"hybrid"``
    (SKELETON re-hang), or ``"detected"`` (IGNORE; input returned
    untouched). ``page_texts`` (extracted text per page, 0-indexed)
    enables the SKELETON-tier title-repair and verified fill-in passes
    and FULL-tier section numbers read off the page; ``toc_lines`` (see
    ``read_printed_toc``) adds the printed TOC; ``page_boxes`` (see
    ``read_bookmarks``) places bookmark targets without loading pages.
    """
    entries = validate_bookmarks(read_bookmarks(doc_handle, page_boxes), n_pages)
    tier = classify_bookmarks(entries, n_pages)
    if tier == FULL:
        toc = read_printed_toc(toc_lines, entries) if toc_lines else []
        line_of = _heading_line_reader(page_texts) if page_texts else None
        return merge_bookmark_tree(structure, entries, n_pages,
                                   toc=toc, line_of=line_of), "bookmarks"
    if tier == SKELETON:
        return merge_bookmark_skeleton(structure, entries, n_pages,
                                       page_texts=page_texts), "hybrid"
    return structure, "detected"


__all__ = [
    "IGNORE", "SKELETON", "FULL",
    "read_bookmarks", "validate_bookmarks", "classify_bookmarks",
    "bookmarks_to_structure", "merge_bookmark_skeleton", "merge_bookmark_tree",
    "read_printed_toc", "apply_embedded_toc",
]
