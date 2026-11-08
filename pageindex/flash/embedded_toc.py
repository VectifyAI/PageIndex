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

from .heading_detection.keyword_tables import _DICTS as _FLASH_DICTS, ENGLISH_WORD_TO_NUMBER, EQUATION_KEYWORDS
from .outline_assembly.candidates import _viewport_y_fraction

IGNORE, SKELETON, FULL = 1, 2, 3

# Same-page titles at or above this similarity refer to the same section.
_REPAIR_SIMILARITY = 0.7
_DISSOLVE_SIMILARITY = 0.9

# A climb past a same-page frame heading must leave it at least this much
# of the page as content, or it is claiming the section is empty.
_MIN_SECTION_GAP = 0.15

# A bookmark's /XYZ top sits at most this far below its printed heading's top
_DEST_TOL = 0.02

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
             boxes: dict[int, tuple]) -> tuple[Optional[float], Optional[float]]:
    """(x, y) of an /XYZ target as page fractions from the top-left, None where absent; y as flash
    places its headings (rotation included), x on unrotated pages."""
    try:
        mode, pos = view()
        if mode != 1 or len(pos) < 2 or pos[1] in (None, 0):
            return None, None
        if page_idx not in boxes:  # loading a page parses it: once per page
            page = doc[page_idx]
            # get_bbox inherits the page tree's boxes; get_cropbox does not (pdfium bug 1786)
            boxes[page_idx] = (page.get_bbox(), page.get_rotation())
            page.close()
        box, rotation = boxes[page_idx]
        upright = rotation % 360 == 0
        if not upright and pos[0] in (None, 0):
            return None, None
        y = _viewport_y_fraction(box, rotation, pos[0] or 0.0, pos[1])
        left, _, right, _ = box
        x = (pos[0] - left) / ((right - left) or 1) if upright and pos[0] not in (None, 0) else None
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
        boxes = {i: (tuple(box), rot) for i, (box, rot) in enumerate(page_boxes or ()) if box is not None}
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


def _numeral_prefix(title: str) -> Optional[str]:
    """A title's dotted section number, "3.2.1", "A." or "A.1", as its reading gives it ("3D", "2019 Survey" are words)."""
    for scheme, values in _readings(title):
        if len(scheme) < 3 or scheme[0]:
            continue
        if scheme[1] == "digit":
            return ".".join(map(str, values))
        if scheme[1] == "LETTER" and (len(values) > 1 or scheme[2] == "."):
            return ".".join([chr(64 + values[0])] + [str(value) for value in values[1:]]) + ("." * (len(values) == 1))
    return None


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


def _section_key(title: str) -> Optional[tuple]:
    """The section a title's number names ("Item 7", "4.2", "4.", "第三章", "一、"); a bare "4" is too weak."""
    for scheme, values in _readings(title)[:1]:
        if (scheme[0] in ("chapter", "label") or scheme[1] == "cjk" or len(values) > 1
                or (scheme[1] == "digit" and scheme[2] == ".")):
            return _family(scheme), values
    return None


def _numbered_outside(parent_title: str, title: str) -> bool:
    """Both name sections in one numbering and ``title``'s lies outside ``parent_title``'s ("NOTE 19", "NOTE 18")."""
    parent, mine = _section_key(parent_title), _section_key(title)
    return (parent is not None and mine is not None and parent[0] == mine[0]
            and mine[1][:len(parent[1])] != parent[1])


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
                and len(remark[-1]) > 1)  # a remark's period ends a word, not initials ("B.S.")
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


def _says_same(title: str, other: str) -> bool:
    """Half the words of the title with fewer share their first five letters with the other's ("FX Risk", "foreign exchange risk")."""
    mine, theirs = ({token[:5] for token in _title_tokens(text) if len(token) >= 4} for text in (title, other))
    return bool(mine and theirs) and 2 * len(mine & theirs) >= min(len(mine), len(theirs))


def _enumerated_alike(title: str, other: str) -> bool:
    """Both titles open with a section number, or both with "一、"-style ones."""
    def cjk(text: str) -> bool:
        return any(scheme[1:2] == ("cjk",) for scheme, _ in _readings(text))

    return ((_numeral_prefix(title) is not None
             and _numeral_prefix(other) is not None)
            or (cjk(title) and cjk(other)))


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


def _flash_words(*keys: str) -> frozenset:
    """Lowercased entries of flash's multilingual keyword dictionaries."""
    return frozenset(" ".join(word.lower().split()) for key in keys for word in _FLASH_DICTS.get(key, []))


def _wide(word: str) -> bool:
    return any(unicodedata.east_asian_width(char) in "WF" for char in word)


def _any_word(words) -> str:
    """A regex alternation of ``words``, longest first; CJK may be letter-spaced."""
    return "|".join((r"\s*".join(map(re.escape, word.replace(" ", ""))) if _wide(word)
                     else r"\s+".join(map(re.escape, word.split())))
                    for word in sorted(words, key=len, reverse=True))


# Label words: flash's dictionaries, plus the ones they lack
_CHAPTER_WORDS = _flash_words("chapter_words")
_APPENDIX_WORDS = _flash_words("appendix_keywords", "appendices_dict")
_VOLUME_WORDS = _flash_words("volume_words")
_LABEL_WORDS = _CHAPTER_WORDS | _APPENDIX_WORDS | _VOLUME_WORDS | {
    "chap", "chap.", "part", "section", "item", "note", "article", "lecture", "unit", "schedule",
    "exhibit", "book", "module"}
_BACK_SET = _flash_words("references") | _APPENDIX_WORDS | {
    "acknowledgment", "acknowledgments", "acknowledgement", "acknowledgements",
    "index", "glossary", "supplementary", "致谢"}
# flash's number words (one to twenty), with the tens past twenty it lacks
_TENS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_NUMBER_WORD = ("(?:(?:" + "|".join(_TENS) + ")[- ]?)?(?:"
                + "|".join(word for word, value in ENGLISH_WORD_TO_NUMBER.items() if value < 10) + ")|"
                + "|".join([word for word, value in ENGLISH_WORD_TO_NUMBER.items() if value >= 10] + list(_TENS)))
_NUMBER_WORD_LINE = re.compile(r"^\s*(?:" + _NUMBER_WORD + r")\s*$", re.IGNORECASE)
_BACK_WORDS = re.compile(
    r"(?<!\w)(?:" + _any_word({word for word in _BACK_SET if not _wide(word)}) + r")(?!\w)|"
    + _any_word({word for word in _BACK_SET if _wide(word)}), re.IGNORECASE)
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
    r"(?i:(" + _any_word(_LABEL_WORDS) + r"))\s+"
    r"([0-9]{1,3}(?:\.[0-9]{1,3})*|[IVXLC]{1,6}|[A-Z])([A-Z])?(?=$|[\s.:,;\-–—)])")
_ADDR_WORD_LABEL = re.compile(r"([A-Z][a-z]{2,15})\s+([0-9]{1,3})([A-Z])?\s*[:.\-–—](?=\s|$)")
_CAPTION_WORDS = _flash_words("ai_section_keywords", "table_keywords", "chart_keywords") | frozenset(
    EQUATION_KEYWORDS) | {"algorithm", "listing", "page", "box", "example", "exercise", "problem"}
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
    return _lead(text)[0]


def _label_only(text: Optional[str], bare: bool = False) -> bool:
    """A label printed on its own: "Chapter 4", "第三章", "TWENTY"; ``bare``, also an unbracketed number
    ("2.", "IV"), which above a heading may be a folio."""
    for scheme, _ in _lead(text, alone=True)[0]:
        if scheme[0] == "chapter" or (scheme[0] == "label" and (scheme[1] in _LABEL_WORDS or _wide(scheme[1]))):
            return True
        if bare and scheme[0] is False:
            return True
    return bool(_NUMBER_WORD_LINE.match(text or ""))


def _lead(text: Optional[str], alone: bool = False) -> tuple[list[tuple[tuple, tuple]], str]:
    """The addresses a line's leading number or label can mean, and the text after it; ``alone``,
    the number or label is all the line holds."""
    if not text:
        return [], ""
    raw = text.lstrip(" \t​﻿")
    t = unicodedata.normalize("NFKC", raw)

    def read(readings: list, end: int) -> tuple[list, str]:
        rest = t[end:].lstrip(" .:：、．)）")
        return (readings, rest) if not alone or not rest.strip() else ([], t)

    # "②", "⑵", "❷": an enclosed number is a list style of its own, not the digit NFKC makes of it
    if raw[:1] and unicodedata.category(raw[0]) == "No":
        rest = unicodedata.normalize("NFKC", raw[1:]).lstrip(" .:：、．)）")
        if alone or rest:
            return read([((False, "enclosed", ""), (int(unicodedata.numeric(raw[0])),))], len(t) - len(rest))

    m = _ADDR_CJK_LABEL.match(t)
    if m and (value := _cjk_number(m.group(1))) is not None:
        return read([(_CJK_UNIT.get(m.group(2), ("label", m.group(2))), (value,))], m.end())
    m = _ADDR_LABEL.match(t)
    if m:
        word, number, suffix = " ".join(m.group(1).lower().split()), m.group(2), m.group(3)
        parts: list[float]
        if number[0].isdigit():
            parts = [int(x) for x in number.split(".")]
        elif len(number) > 1 or word in _VOLUME_WORDS or word in ("part", "book", "lecture", "unit"):
            if (roman := _roman_number(number)) is None:
                return [], t
            parts = [roman]
        else:
            parts = [ord(number.upper()) - 64]
        if suffix:
            parts[-1] += (ord(suffix.upper()) - 64) / 100  # Item 1 < 1A < 1B < 2
        return read([(("chapter",) if word in _CHAPTER_WORDS else ("label", word), tuple(parts))], m.end())
    m = _ADDR_WORD_LABEL.match(t)
    if m and m.group(1).lower() not in _CAPTION_WORDS:
        value = int(m.group(2)) + ((ord(m.group(3)) - 64) / 100 if m.group(3) else 0)
        return read([(("label", m.group(1).lower()), (value,))], m.end())

    def tail_ok(m, need_close: bool) -> bool:
        rest = t[m.end():]
        bracket, close = m.group(1), m.groups()[-1] or ""
        if bracket and close != ")":
            return False
        if alone:
            return not rest.strip(" .:：、．")
        if not rest.strip():
            return False
        if need_close:  # a lone letter or numeral: "A. Proofs", not "A Proofs" or "U.S."
            return bool(close and rest[0].isspace())
        # "3D", "2019 Survey", "10-K" are words; after "5.1、" or "1." the title may start anyhow
        return bool(close) or not (rest[0].isascii() and (rest[0].isalnum() or rest[0] in "-/%,+*=<>$&"))

    m = _ADDR_DIGIT.match(t)
    if m and tail_ok(m, False):
        values = tuple(int(x) for x in re.split(r"\s*\.\s*", m.group(2)))
        close = (m.group(3) or "") if len(values) == 1 else ""
        return read([((bool(m.group(1)), "digit", close), values)], m.end())
    m = _ADDR_ROMAN.match(t)
    if m and (roman := _roman_number(m.group(2))) is not None:
        numeral, close = m.group(2), m.group(3) or ""
        kind = "ROMAN" if numeral.isupper() else "roman"
        if len(numeral) > 1 and (tail_ok(m, True) or (
                numeral.isupper() and tail_ok(m, False) and roman <= 39
                and t[m.end():][:1].isspace())):
            return read([((bool(m.group(1)), kind, close), (roman,))], m.end())
        if len(numeral) == 1 and numeral.upper() in "IVX" and tail_ok(m, True):
            letter = "LETTER" if numeral.isupper() else "letter"
            return read([((bool(m.group(1)), kind, close), (roman,)),
                         ((bool(m.group(1)), letter, close), (ord(numeral.upper()) - 64,))], m.end())
    m = _ADDR_LETTER.match(t)
    if m and tail_ok(m, not m.group(3)):
        values = (ord(m.group(2).upper()) - 64, *(int(x) for x in re.findall(r"[0-9]+", m.group(3))))
        kind = "LETTER" if m.group(2).isupper() else "letter"
        close = (m.group(4) or "") if len(values) == 1 else ""
        return read([((bool(m.group(1)), kind, close), values)], m.end())
    m = _ADDR_CJK.match(t)
    if m and (value := _cjk_number(m.group(2))) is not None and m.group(3) and tail_ok(m, False):
        return read([((bool(m.group(1)), "cjk", m.group(3)), (value,))], m.end())
    return [], t


def _family(scheme: tuple) -> tuple:
    return scheme if scheme[0] in ("chapter", "label") else scheme[:2]


def _marks(node: dict) -> tuple[bool, bool]:
    """Whether a heading's print opens like a list item ("• iPhones") and closes like a run-in lead ("Model.", "Subsidiary banks:")."""
    lines = node.get("_lines") or [[node["title"]]]
    return (lines[0][0].strip().startswith(tuple("•·▪●◦■□➢►-–—*")),
            lines[-1][0].strip().endswith((".", ":", "：", "。")))


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

    Fingerprints are flash's ``block_style_signature``, "face weight size caps-heavy"; small caps
    are set small by design.
    """
    try:
        (face, size, caps), (other_face, other_size, other_caps) = (s.rsplit(" ", 2) for s in (style, other))
        return face == other_face and caps == other_caps and float(size) > float(other_size) + 0.5
    except (AttributeError, ValueError):
        return False


def _style_size(style: Optional[str]) -> Optional[tuple[str, str, str]]:
    """A style fingerprint's face, size and caps flag; None without a size."""
    parts = style.rsplit(" ", 2) if style else []
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 and parts[1].replace(".", "", 1).isdigit() else None


_CODE = re.compile(r"([A-Z][0-9][0-9A-Z])(?:\.[0-9A-Z]{1,4})?\s")
_CODE_RANGE = re.compile(r"\(([A-Z][0-9][0-9A-Z])-([A-Z][0-9][0-9A-Z])\)\s*$")


def _code_within(title: str, frame_title: str) -> bool:
    """``title`` opens with a classification code ("I60", "S03.0") inside the range ``frame_title`` ends with ("(I60-I69)")."""
    code, span = _CODE.match(title.strip() + " "), _CODE_RANGE.search(frame_title)
    return code is not None and span is not None and span.group(1) <= code.group(1) <= span.group(2)


def _list_families(readings: list) -> set:
    """The list numberings and depths among an address's readings ("2." digit, "（三）" CJK, "2.1.2" three deep), label words aside."""
    return {(_family(scheme), len(values)) for scheme, values in readings if scheme[0] not in ("chapter", "label")}


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
    columns: Optional[list[list[tuple]]] = None,
    page_lines: Optional[Callable[[int], list[str]]] = None,
) -> list[dict]:
    """Graft filtered detected sections into the bookmark tree (FULL tier).

    The bookmark tree is the frame; the detected headings it lacks are
    placed by one walk in reading order. ``toc`` is the printed table of
    contents (``read_printed_toc``); ``line_of(page, title)`` returns the
    printed line of a bookmark's heading on its target page, for
    bookmarks whose title carries no section number; ``columns`` holds
    flash's column regions per page as (x0, x1, top, bottom) page
    fractions in its reading order, the order a heading's ``_col``
    indexes; ``page_lines(page)`` returns a page's text lines.

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
    escaped garbled duplicate only leaves a stray leaf; a print takes a
    bookmark it prints exactly over one it only resembles. A print that adds
    words to a bookmark's title away from its target is another heading,
    and initials are no remark ("Mathematics, B.A." below "Mathematics").
    A bookmark still without a print takes the first free node at its
    target sharing half the four-letter-plus words of the title with fewer ("Non-U.S. dollar foreign exchange
    risk" for "Non-U.S. Dollar FX Risk"); one with a heading glued on only
    lends its style. A bookmark learns style and position from its copies; detection
    exports the ``_style``, ``_lines`` (each line's text and style),
    ``_y``, ``_content_follows`` (real content after the heading on its
    page before the next heading, in flash's reading order),
    ``_prose_follows`` (whether that content opens with a body
    paragraph), ``_col``
    (its column region), ``_type`` and ``_prominent`` keys and
    ``extract_toc`` strips them after the merge; sizes a tenth of a point
    apart in one face are one size ("9.9", "10.0"). Label words (chapter,
    appendix, volume, references, table) come from flash's multilingual
    keyword dictionaries, plus the few they lack.

    Slots. Each bookmark takes a place in the detected reading order: at
    its first copy, else after the headings flash reads before its /XYZ
    target on its page (by column region, then height: a target sits in
    the first region not left of its x nor wholly above it; one without an
    x is placed by height alone), else at its page's start -- never
    between a same-page
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
        bookmark (or numbers the heading directly), unless their numbers
        set them apart ("NOTE 19" detected under "NOTE 18");
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
        the numbered line printing its title at its target; a reading
        unlike all its numbered siblings' is dropped (a table row taken
        for the heading, "V." read as a letter beside "IV." and "VI."),
        and an address a bookmark holds is no evidence for
        a heading unless the bookmarks themselves repeat it;
      * the parent the printed table of contents lists it under, when
        that contents nests the bookmarks it lists the way they nest
        themselves -- all of them, or those under that parent;
      * else a climb from the deepest open bookmark, past a candidate
        only on positive evidence the heading is its peer: a numbering
        prefix outside the candidate's and after it, or the same style
        fingerprint (face, weight and size as one hash; a larger raw size
        in a different face is not seniority) that leaves the candidate
        real content below its heading on a shared page (what flash reads
        between the two, columns included, else a vertical gap) and no
        same-style detected child enumerated like the heading, or a style
        that differs from the candidate's in size alone and is larger
        ("EXHIBIT B" after a smaller "8. Miscellaneous"); never past a
        candidate numbered in another list style ("2." below "（三）"), or
        unnumbered among siblings numbered so ("3.2." below an "APPENDIX"
        that follows "VII. REFERENCES").
        A climb must reach a numbering parent or the top level (just below
        the root when one root bookmark titles the whole document); one
        stopped halfway sets the heading beside the bookmark it climbed
        past, unless it reads as a part of the deepest open section:
        numbered where that bookmark is not (or not where it is), a
        run-in lead or list item beside one that is
        neither ("Subsidiary banks:"), heading a table or list rather than
        text, coded inside the range the deepest bookmark's title names
        ("I60" under "(I60-I69)"), under a title the document repeats,
        beside another heading printed alike in that section (a layer the
        bookmarks leave out, unless under the same parent they leave out a
        heading printed so before its first child), or followed in the page text, before the next
        bookmark at that level, by a line each section of that level
        prints ("Requirements for the Minor" after "Eligibility"). It stays
        beside that bookmark when the bookmarks print the level below and
        never like the heading, numbered like the level it climbed to
        ("Overall Summary" in bold beside a bookmark whose children print
        in bold italic). Where the bookmarks mostly keep the heading's
        print, the level model built from their own prints sets the level
        they skipped it at. An empty section is read at the bookmark's print,
        not its target page.
        One the next bookmark's open parent stops keeps its height, beside
        that bookmark, unless the two are numbered in another list style
        or depth ("3." is no peer of "1.1.2").

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

    def walk(nodes: list[dict]):
        for node in nodes:
            yield node
            yield from walk(node.get("nodes") or [])

    # Sizes a tenth of a point apart in one face are one size set two ways ("9.9", "10.0"): each run takes its most used
    used: dict[tuple, Counter] = {}
    for node in walk(structure):
        parts = _style_size(node.get("_style"))
        if parts:
            used.setdefault((parts[0], parts[2]), Counter())[parts[1]] += 1
    resized: dict[str, str] = {}
    for (face, caps), counts in used.items():
        run: list[str] = []
        for size in sorted(counts, key=float) + [None]:
            if run and (size is None or float(size) - float(run[-1]) > 0.15):
                top = max(run, key=counts.__getitem__)
                resized.update({f"{face} {s} {caps}": f"{face} {top} {caps}" for s in run})
                run = []
            if size is not None:
                run.append(size)
    for node in walk(structure):
        if node.get("_style") in resized:
            node["_style"] = resized[node["_style"]]
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
            # "7.1.2" and "7.2.2" print two same-titled bookmarks: each takes the first one free;
            # else the one printed exactly ("Sociology, B.A." over a similar "Sociology, B.S.")
            exact = next((p for p in same if _same_text(node["title"], flat[p]["title"])), same[0])
            pos = next((p for p in reversed(same) if numeral is not None
                        and matched.get(p, numeral) == numeral), exact)
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

    def _claims_empty(node: dict, b: int) -> bool:
        """Bookmark b would hold nothing were the heading its peer."""
        cand = flat[b]
        printed = reading[first_copy[b]] if b in first_copy else None
        if (printed is not None and printed["start_index"] == node["start_index"]
                and printed.get("_content_follows") is not None):
            # flash read the page: the heading precedes b's print, or follows it with nothing between
            pos = read_index.get(id(node), -1)
            return pos < first_copy[b] or (pos == first_copy[b] + 1 and not printed["_content_follows"])
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
                        style = resized.get(style, style)
                        node["_style"] = style
                        for child in node.get("nodes") or []:
                            if child.get("_style") == style:
                                det_parent[id(child)] = None  # printed alike: a peer
                break

    def before_target(node: dict, b: int) -> bool:
        """flash reads the heading before bookmark b's target on b's page: by column region, then height,
        in the first region not left of the target's x nor wholly above it; by height alone without an x."""
        page, x, y = flat[b]["start_index"], dest_x[b], dest_y[b]
        if y is None or node.get("_y") is None:
            return False
        regions = columns[page - 1] if x is not None and columns and 0 < page <= len(columns) else []
        region = next((i for i, (_, right, _, bottom) in enumerate(regions)
                       if x <= right and bottom >= y - _DEST_TOL), None)
        if region is None or node.get("_col") is None:
            return node["_y"] < y - _DEST_TOL
        return (node["_col"], node["_y"]) < (region, y - _DEST_TOL)

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
    # A bookmark worded unlike its print: the heading at its target saying mostly its words is the print,
    # one with a heading glued on only lends its look
    at_target: dict[int, list[int]] = {}
    for pos, node in enumerate(reading):
        if id(node) not in copy_of and id(node) not in stripped and node.get("_y") is not None:
            at_target.setdefault(node["start_index"], []).append(pos)
    for b, frame in enumerate(flat):
        if b in first_copy or dest_y[b] is None or frame.get("_style") is not None:
            continue
        pos = next((p for p in at_target.get(frame["start_index"], ())
                    if id(reading[p]) not in copy_of and abs(reading[p]["_y"] - dest_y[b]) <= _DEST_TOL), None)
        if pos is None:
            continue
        node = reading[pos]
        glued = _extends(node["title"], frame["title"])
        if not glued and not _says_same(node["title"], frame["title"]):
            continue
        if not glued:
            copy_of[id(node)] = b
            first_copy[b] = pos
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
               and (_label_only(reading[slot - 1]["title"])
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
    # A reading unlike all its numbered siblings' is misread (a table row taken for the heading, "V." as a letter)
    siblings: dict[Optional[int], list[int]] = {}
    for b in range(len(flat)):
        siblings.setdefault(parent_pos[b], []).append(b)
    addresses[:] = [
        [(scheme, values) for scheme, values in readings if _family(scheme) in others]
        if (others := {_family(scheme) for s in siblings[parent_pos[b]] if s != b
                       for scheme, _ in addresses[s][:1]}) else readings
        for b, readings in enumerate(addresses)]
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
                    # this numbering's depth is no evidence here; its next number still is
                    return younger_sibling_parent(node, sh, h, floor)
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

    # The list numberings each bookmark level uses, for an unnumbered member ("APPENDIX" after "VII.")
    level_families = {parent: set().union(*(_list_families(addresses[s]) for s in kids))
                      for parent, kids in siblings.items()}
    # How each bookmark level is printed and written, from the bookmarks' own copies, and the levels each style prints
    level_styles: dict[int, set] = {}
    style_levels: dict[str, Counter] = {}
    level_traits: dict[int, list] = {}
    for b, frame in enumerate(flat):
        printed = reading[first_copy[b]] if b in first_copy else frame
        level_traits.setdefault(levels[b], []).append((bool(addresses[b]), _marks(printed)))
        if frame.get("_style") is not None:
            level_styles.setdefault(levels[b], set()).add(frame["_style"])
            style_levels.setdefault(frame["_style"], Counter())[levels[b]] += 1
    level_list = sorted(level_traits)
    unbookmarked = Counter(ref.get("_style") for kind, ref in seq if kind == "d")

    def schemes(readings: list) -> set:
        """How a title is numbered: each reading's family (label words included) and depth."""
        return {(_family(scheme), len(values)) for scheme, values in readings}

    level_schemes: dict[int, set] = {}
    scheme_levels: dict[tuple, set] = {}
    for b in range(len(flat)):
        for key in schemes(addresses[b]):
            level_schemes.setdefault(levels[b], set()).add(key)
            scheme_levels.setdefault(key, set()).add(levels[b])
    bookmarked_titles = {_normalize_title(title) for b, frame in enumerate(flat) for title in (
        frame["title"], (reading[first_copy[b]] if b in first_copy else frame)["title"])}
    unbookmarked_titles = Counter(_normalize_title(ref["title"]) for ref in reading if id(ref) not in copy_of)
    copies_of: dict[str, list[int]] = {}
    for pos, ref in enumerate(reading):
        copies_of.setdefault(_normalize_title(ref["title"]), []).append(pos)

    def printed_unlike_below(node: dict, deepest: int, peer: int) -> bool:
        """The bookmarks print the level below ``deepest``, never like the heading, which is numbered like ``peer``'s level."""
        below = level_styles.get(levels[deepest] + 1)
        kin = _list_families(addresses[peer]) or level_families[parent_pos[peer]]
        return (bool(below) and node.get("_style") not in below
                and (not kin or bool(_list_families(_readings(node["title"])) & kin)))

    def no_skip(node: dict) -> bool:
        """A never-bookmarked title recurring in the document: left out on purpose, not skipped."""
        norm = _normalize_title(node["title"])
        return unbookmarked_titles[norm] > 1 and norm not in bookmarked_titles

    def level_posterior(node: dict) -> dict:
        """Odds of each level the bookmarks skipped the heading at, from where they keep its style, or "minor"."""
        style = node.get("_style")
        seen = style_levels.get(style) if style is not None else None
        if not seen or no_skip(node):
            return {"minor": 1.0}
        kept = sum(seen.values())
        skip = (kept + 1) / (kept + unbookmarked[style] + 2)
        readings = _readings(node["title"])
        numbered, mine, marks = bool(readings), schemes(readings), _marks(node)
        score = {}
        for level in level_list:
            written = level_traits[level]
            out = (all(theirs != numbered for theirs, _ in written)  # a level is out when written unlike the heading
                   or any(m and not any(t[i] for _, t in written) for i, m in enumerate(marks))
                   or (mine and level_schemes.get(level) and not mine & level_schemes[level])
                   or (mine and all(key in scheme_levels and level not in scheme_levels[key] for key in mine)))
            score[level] = 0.0 if out else skip * (seen[level] + 1) / (kept + len(level_list))
        score["minor"] = 1 - skip
        total = sum(score.values())
        return {h: p / total for h, p in score.items()}

    def looks(node: dict, b: int) -> Optional[bool]:
        """Whether bookmark b prints like the heading; None when either has no print or only caps differ."""
        mine, theirs = node.get("_style"), flat[b].get("_style")
        if mine is None or theirs is None:
            return None
        if mine == theirs:
            return True
        return None if mine.rsplit(" ", 1)[0] == theirs.rsplit(" ", 1)[0] else False

    def owners_at(bookmarks: list) -> Optional[tuple[int, list[int]]]:
        """The shallowest bookmark level whose sections hold the given bookmarks' places one apiece, and those sections."""
        chains = []
        for b in bookmarks:
            if b is None or b < 0:
                return None
            chain = {}
            while b is not None:
                chain[levels[b]] = b
                b = parent_pos[b]
            chains.append(chain)
        for level in sorted(set(levels)):
            owners = [chain.get(level) for chain in chains]
            if None in owners:
                return None
            if len(set(owners)) == len(owners):
                return level, owners
        return None

    text_owners: Optional[dict[str, list[Optional[int]]]] = None

    def owners_of_text(page_lines: Callable[[int], list[str]]) -> dict[str, list[Optional[int]]]:
        """Each heading title's lines in the page text, owned by the bookmark whose title line comes before them; read once."""
        nonlocal text_owners
        if text_owners is None:
            text_owners = {}
            owner: Optional[int] = None
            for p in range(1, n_pages + 1):
                starting = list(on_page.get(p, ()))
                for line in page_lines(p):
                    norm = _normalize_title(line)
                    while starting and norm and norm == _normalize_title(flat[starting[0]]["title"]):
                        owner = starting.pop(0)
                    if norm in title_pages:
                        text_owners.setdefault(norm, []).append(None if starting else owner)  # None: a start unseen
                if starting:
                    owner = starting[-1]
        return text_owners
    text_levels: dict[str, Optional[int]] = {}

    def text_part_level(norm: str, page_lines: Callable[[int], list[str]]) -> Optional[int]:
        """The bookmark level whose sections each print this heading line once, three times or more."""
        if norm not in text_levels:
            owners = [b for b in owners_of_text(page_lines).get(norm, ()) if b is not None]
            got = owners_at(owners) if len(owners) >= 3 else None
            text_levels[norm] = got[0] if got else None
        return text_levels[norm]

    def text_continues(node: dict, peer: int) -> bool:
        """The page text after the heading, before the next bookmark at peer's level, prints a part each section of that level prints."""
        if page_lines is None:
            return False
        level = levels[peer]
        stop = next((ref for kind, ref in seq[seq_pos[id(node)] + 1:] if kind == "b" and levels[ref] <= level), None)
        last_page = flat[stop]["start_index"] if stop is not None else n_pages
        stop_norm = _normalize_title(flat[stop]["title"]) if stop is not None else None
        mine, started = _normalize_title(node["title"]), False
        for p in range(node["start_index"], last_page + 1):
            for line in page_lines(p):
                norm = _normalize_title(line)
                if not started:
                    started = bool(norm) and (norm == mine or (len(norm) >= 8 and mine.startswith(norm)))
                elif norm and norm == stop_norm:
                    return False
                elif text_part_level(norm, page_lines) == level:
                    return True
        return False

    # Prints the bookmarks leave out under a parent: a heading between it and its first child, printed exactly like that child
    skipped_under: set[tuple[str, int]] = set()
    last, between = None, []
    for kind, ref in seq:
        if kind == "d":
            between.append(ref)
            continue
        style = flat[ref].get("_style")
        if (style is not None and last is not None and parent_pos[ref] == last and ref in first_copy
                and _same_text(reading[first_copy[ref]]["title"], flat[ref]["title"])):
            skipped_under.update((style, last) for d in between if d.get("_style") == style)
        last, between = ref, []

    def son(node: dict, a: int, peer: int) -> bool:
        """Evidence that a heading the climb would set beside bookmark ``peer`` is a part of bookmark ``a``'s section."""
        if bool(_readings(node["title"])) != bool(addresses[peer]):
            return True  # numbered one way, the sibling the other
        printed = reading[first_copy[peer]] if peer in first_copy else flat[peer]
        if any(mark and not theirs for mark, theirs in zip(_marks(node), _marks(printed))):
            return True  # a run-in lead or a list item beside a bookmark that is neither
        if node.get("_prose_follows") is False:
            return True  # it heads a table or a list, not text
        if _code_within(node["title"], flat[a]["title"]):
            return True  # coded inside the range a's title names ("I60" under "(I60-I69)")
        if len(copies_of.get(_normalize_title(node["title"]), ())) > 1:
            return True  # a title the document repeats
        style = node.get("_style")
        # another heading printed alike in a's section: a layer the bookmarks leave out, unless they leave that print out beside peer
        for kind, ref in ([] if (style, parent_pos[peer]) in skipped_under else seq[bookmark_seq_pos[a] + 1:]):
            if kind == "b" and levels[ref] <= levels[a]:
                break
            if style is not None and ref is not node and (flat[ref] if kind == "b" else ref).get("_style") == style:
                return True
        return text_continues(node, peer)

    def modeled_stop(node: dict, floor: int, anchor: int, k: int) -> int:
        """A stopped climb's parent: the level the heading's look ranks first, up to the nearest bookmark printed alike."""
        top = next((j for j in range(anchor, -1, -1)
                    if _same_text(node["title"], flat[stack[j][3]]["title"])
                    or _numbered_print(node["title"], flat[stack[j][3]]["title"])), -1)
        # beside the bookmark it climbed past, unless the bookmarks print the level below unlike it or it reads as a part
        halfway = k != anchor and k >= top and (printed_unlike_below(node, stack[anchor][3], stack[k + 1][3])
                                                or not son(node, stack[anchor][3], stack[k + 1][3]))
        best = k if halfway else anchor
        post = level_posterior(node)
        if 2 * post["minor"] >= 1:
            return best
        best_score = post.get(levels[stack[anchor][3]] + 1, 0.0)  # a child of the deepest sits one level below it
        if halfway:
            best_score = max(best_score, post.get(levels[stack[k + 1][3]], 0.0))
        mine = _list_families(_readings(node["title"]))
        for j in range(anchor, top, -1):
            b = stack[j][3]
            if parent_pos[b] is None and len(roots) == 1:
                break
            kin = _list_families(addresses[b]) or level_families[parent_pos[b]]
            if j <= k and ((mine and kin and not mine & kin) or _claims_empty(node, b)):
                break
            p = j - 1
            if (stack[p][1] + 1 if p >= 0 else 1) < floor:
                break
            score = post.get(levels[b], 0.0)
            alike = looks(node, b)
            if alike:
                return p if score and score >= best_score else best  # a tie goes to the look nearby
            if alike is None and score > best_score:
                best, best_score = p, score
        return best

    def periodic_parent(node: dict, floor: int) -> Optional[int]:
        """A title in most sections of a bookmark level, sitting alike in each, is a part of that level's section."""
        copies = copies_of.get(_normalize_title(node["title"]), ())
        if len(copies) < 3:
            return None
        got = owners_at([bisect_right(slots, pos) - 1 for pos in copies])  # the bookmark read last before each copy
        if got is None:
            return None
        level, owners = got
        sections = [b for b in range(min(owners), max(owners) + 1) if levels[b] == level]
        if 2 * len(owners) <= len(sections):
            return None  # most sections of that level go without it: not a part each repeats
        if len({bisect_right(slots, pos) - 1 == o for pos, o in zip(copies, owners)}) > 1:
            return None  # some copies open their section, others follow a part of it: they sit unalike
        k = next((j for j in range(len(stack) - 1, -1, -1)
                  if stack[j][2] == "b" and levels[stack[j][3]] == level), None)
        if k is None or (stack[k][1] + 1) < floor:
            return None
        return k

    def evidence(node: dict, b: int) -> str:
        cand = flat[b]
        title, cand_title = node["title"], cand["title"]
        if _numeral_prefix(cand_title) is not None and _numeral_prefix(title) is not None:
            if _numeral_child(cand_title, title):
                return "parent"
            return "peer" if _numeral_after(title, cand_title) else "stop"
        mine = _list_families(_readings(title))
        kin = _list_families(addresses[b]) or level_families[parent_pos[b]]
        if mine and kin and not mine & kin:
            return "stop"  # siblings are numbered alike: "2." is no peer of "（三）", nor "3.2." of "VII."
        if _larger_alike(node.get("_style"), cand.get("_style")):
            return "peer"
        if (not _style_climbs(node, cand) or _claims_empty(node, b)
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
            if det_parent[id(node)] is ref and not _numbered_outside(ref["title"], node["title"]) and (
                    not spans(ref) or _numbered_under(ref["title"], node["title"])):
                return k
            k -= 1
        if k < 0:
            return -1
        for named in (numbered_parent, toc_parent, periodic_parent):
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
            if verdict == "stop":
                return modeled_stop(node, floor, anchor, k)
            if (stack[below][1] + 1 if below >= 0 else 1) < floor:
                # the next bookmark keeps the section open: the heading stays beside it, if numbered alike
                mine = _list_families(_readings(node["title"]))
                kin = _list_families(addresses[next(later_bookmarks(node))])
                return anchor if mine and kin and not mine & kin else k
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
def _toc_items(pages: list[list[tuple]]) -> list[dict]:
    """Printed TOC entries as {page, title, x, num}, label cells and wrapped titles joined."""
    items = []
    for page, rows in enumerate(pages, 1):
        rows = [list(row) for row in rows]  # [text, x0, x1, y, style]
        joined: set[int] = set()
        for i, row in enumerate(rows):
            if not _label_only(row[0], bare=True):  # a label cell printed apart: "Item 1A.", "2.", "第三节"
                continue
            cells = [j for j, cell in enumerate(rows) if j != i and j not in joined
                     and abs(cell[3] - row[3]) < 0.004 and cell[1] > row[2] - 0.01
                     and not _TOC_PAGE_NUMBER.match(cell[0].strip())
                     and not _label_only(cell[0], bare=True)]
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


@lru_cache(maxsize=1 << 16)
def _line_key(text: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).lower())


def _page_line_reader(page_texts: list[str]) -> Callable[[int], list[str]]:
    """lines(page): the page's text lines, stripped."""
    cache: dict[int, list[str]] = {}

    def lines(page: int) -> list[str]:
        if page not in cache:
            text = page_texts[page - 1] if 1 <= page <= len(page_texts) else ""
            cache[page] = [line.strip() for line in (text or "").splitlines()]
        return cache[page]

    return lines


def _heading_line_reader(page_texts: list[str]) -> Callable[[int, str], Optional[str]]:
    """line_of(page, title): the numbered line printing title on page or the next, None if numbers disagree."""
    lines = _page_line_reader(page_texts)

    def prints(rest: str, title: str) -> bool:
        a, b = _line_key(rest), _line_key(title)
        return bool(a) and bool(b) and (a == b or (len(a) >= 8 and b.startswith(a)))

    def line_of(page: int, title: str) -> Optional[str]:
        found = []
        for at in (page, page + 1):
            text = lines(at)
            for j, line in enumerate(text):
                readings, rest = _lead(line)
                if readings and rest and prints(rest, title):
                    found.append((tuple(readings), line))
                elif prints(line, title) and j > 0 and _label_only(text[j - 1]):
                    found.append((tuple(_readings(text[j - 1] + " " + line)), text[j - 1] + " " + line))
        return found[0][1] if found and len({number for number, _ in found}) == 1 else None

    return line_of


def apply_embedded_toc(
    structure: list[dict], doc_handle: Union[str, Path, BytesIO], n_pages: int,
    page_texts: Optional[list[str]] = None,
    toc_lines: Optional[list[list[tuple]]] = None,
    page_boxes: Optional[list[tuple]] = None,
    page_columns: Optional[list[list[tuple]]] = None,
) -> tuple[list[dict], str]:
    """Classify the document's bookmarks and apply the matching tier.

    Returns ``(structure, source)`` with source one of ``"bookmarks"``
    (FULL: bookmark frame with filtered detected grafts), ``"hybrid"``
    (SKELETON re-hang), or ``"detected"`` (IGNORE; input returned
    untouched). ``page_texts`` (extracted text per page, 0-indexed)
    enables the SKELETON-tier title-repair and verified fill-in passes
    and FULL-tier section numbers read off the page; ``toc_lines`` (see
    ``read_printed_toc``) adds the printed TOC; ``page_boxes`` (see
    ``read_bookmarks``) places bookmark targets without loading pages;
    ``page_columns`` (flash's column regions per page, in reading order)
    reads them in its reading order.
    """
    entries = validate_bookmarks(read_bookmarks(doc_handle, page_boxes), n_pages)
    tier = classify_bookmarks(entries, n_pages)
    if tier == FULL:
        toc = read_printed_toc(toc_lines, entries) if toc_lines else []
        line_of = _heading_line_reader(page_texts) if page_texts else None
        return merge_bookmark_tree(structure, entries, n_pages, toc=toc, line_of=line_of, columns=page_columns,
                                   page_lines=_page_line_reader(page_texts) if page_texts else None), "bookmarks"
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
