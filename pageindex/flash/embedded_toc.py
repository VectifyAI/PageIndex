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
_DISSOLVE_SIMILARITY = 0.9

# A climb past a same-page frame heading must leave it at least this much
# of the page as content, or it is claiming the section is empty.
_MIN_SECTION_GAP = 0.15

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


def _normalize_title(title: str) -> str:
    return "".join(_title_tokens(title))


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
            if hasattr(item, "get_dest"):
                dest = item.get_dest()
                page_idx = dest.get_index() if dest else None
                title = item.get_title()
            else:
                page_idx, title = item.page_index, item.title
            if page_idx is None:
                continue
            title = (title or "").strip()
            if not title:
                continue
            entries.append(
                {"title": title, "level": item.level + 1, "page": page_idx + 1}
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


# "3D", "2.5D", "10-K" and "2019 Survey" are words, not section numbers
_NUMERAL_RE = re.compile(
    r"^([A-Z]\.\d*(?:\.\d+)*|[A-Z]\.$|\d{1,3}(?:\.\d+)*)(?![\w-]|\.\d)")


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
                and not any(token.isdigit() for token in remark))
    rest = iter(tokens)
    if (len(tokens) > tail > 1 and tokens[0] == frame_tokens[0]
            and tokens[-1] == frame_tokens[-1]
            and all(token in rest for token in frame_tokens)):
        return True
    key = _section_key(title)
    return key is not None and key == _section_key(frame_title)


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


def merge_bookmark_tree(
    structure: list[dict], entries: list[dict], n_pages: int
) -> list[dict]:
    """Graft filtered detected sections into the bookmark tree (FULL tier).

    The bookmark tree is the frame. Each flat bookmark node owns the page
    range up to the next entry's start, and every detected node is
    anchored to the deepest bookmark section active at its start page,
    unless it is read before the printed heading of a bookmark starting
    on that page or the one before: then it anchors to the bookmark
    before (if that one starts there too) and goes ahead of the later
    bookmark among its siblings, as an Abstract does before Introduction.
    A detected subtree that fits inside its anchor's range moves under it
    whole; one that spans ranges is dissolved, leaving a leaf, with its
    children placed individually -- a child numbered directly under the
    leaf ("9.3.2.1" under "9.3.2") stays with it while no bookmark starts
    in between. A node whose normalized title matches
    a frame heading starting on its page -- equal (letter spacing aside),
    apart by a numbering prefix (also when a numbered bookmark prints
    letter-spaced), or apart by character-level garble under the same
    section number (letter-spaced too, unless the two end in different
    numbers: "Part I", "Part II") -- or equals one starting on the page
    before, dissolves without a trace, its children re-placed
    individually. A bookmark no node has matched yet also takes the first
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
    parenthetical ("J15ZJ (15 Ton) (Continued)").
    Dissolving holds fuzzy matches to a higher bar than title repair: at
    the repair bar, wordy-overlapping titles of distinct sections
    false-match, and a false dissolve deletes a real heading, while an
    escaped garbled duplicate only leaves a stray leaf.

    A surviving numbered node whose section number belongs to a section
    ending on its page joins that section. Any other node attaches along
    its anchor's ancestor chain -- the open sections at its page, the only
    parents page containment allows. It climbs past a candidate only on
    positive evidence that it is the candidate's peer, not its child: a
    numbering prefix outside the candidate's numbering and after it, or
    the same style fingerprint -- face, weight and size as one hash; a
    larger raw size in a different face is not seniority (margin-box
    titles run larger-regular than bold subsection headings). Frame nodes
    learn style and position from the duplicates that dissolve into them;
    detection exports the ``_style``/``_y`` keys and ``extract_toc``
    strips them after the merge. A same-page climb must also leave the
    candidate real content below its heading -- climbing past it on a
    near-zero gap would claim the section is empty, the signature of
    margin-box noise rather than of a tail section. Numbering-driven
    climbs are exempt: the numbering itself proves the level relation. A
    climb must reach a numbering parent or the top level (just below the
    root when one root bookmark titles the whole document); one stopped
    halfway -- by missing evidence, an earlier number, a same-style
    detected heading under the candidate enumerated like the node, or a
    candidate with content after the node's page -- leaves the node at
    the anchor.

    Two noise filters prune the whole detected tree before placement:
    titles recurring across it (running labels, not sections) and
    overlong titles (paragraph leads picked up as headings) are spliced
    out, their children promoted into their place. A recurring title
    printed in the style of a bookmark's own heading and never on
    adjacent pages is a section each chapter repeats (Notes, Exercises,
    Loan modifications) and stays. Nodes wholly before the first bookmark
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
    frame_index = {id(frame): pos for pos, frame in enumerate(flat)}

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

    # bookmark position -> section number of the heading it took
    matched_frames: dict[int, Optional[str]] = {}

    def dissolve_target(node: dict, idx: int,
                        matched: dict[int, Optional[str]]) -> Optional[int]:
        numeral = _numeral_prefix(node["title"])
        pos = idx
        while pos >= 0 and flat[pos]["start_index"] >= node["start_index"] - 1:
            if _same_heading(node, flat[pos], _DISSOLVE_SIMILARITY):
                matched.setdefault(pos, numeral)
                return pos
            pos -= 1
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

    def boundary_parent(node: dict, idx: int) -> Optional[int]:
        found = None
        for pos in range(idx):
            after = last_pos[pos] + 1
            if (after <= idx
                    and flat[after]["start_index"] >= node["start_index"]
                    and _numeral_child(flat[pos]["title"], node["title"])):
                found = pos
        return found

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

    def choose_parent(node: dict, idx: int) -> Optional[int]:
        pos: Optional[int] = idx
        while pos is not None:
            cand = flat[pos]
            if _subtree_max_start(cand) > node["start_index"]:
                return idx
            if parent_pos[pos] is None and len(roots) == 1:
                return pos
            if (_numeral_prefix(cand["title"]) is not None
                    and _numeral_prefix(node["title"]) is not None):
                if _numeral_child(cand["title"], node["title"]):
                    return pos
                if not _numeral_after(node["title"], cand["title"]):
                    return idx
                pos = parent_pos[pos]
                continue
            if (not _style_climbs(node, cand) or _claims_empty(node, cand)
                    or any(id(child) not in frame_index
                           and _style_climbs(node, child)
                           and _enumerated_alike(node["title"], child["title"])
                           for child in cand["nodes"])):
                return idx
            pos = parent_pos[pos]
        return None

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

    read_pos: dict[int, int] = {}
    heading_pos: dict[int, int] = {}

    def insert_by_page(siblings: list[dict], node: dict, read_at: int) -> None:
        pos = len(siblings)
        for sibling_pos, sibling in enumerate(siblings):
            if (sibling["start_index"] > node["start_index"]
                    or (sibling["start_index"] == node["start_index"]
                        and heading_pos.get(frame_index.get(id(sibling), -1), -1)
                        > read_at)):
                pos = sibling_pos
                break
        siblings.insert(pos, node)

    def read_before(node: dict, idx: Optional[int]) -> Optional[int]:
        """Step past bookmarks starting here whose heading is read after the node."""
        page = node["start_index"]
        while (idx is not None and heading_pos.get(idx, -1) > read_pos[id(node)]
               and flat[idx]["start_index"] >= page - 1
               and (idx == 0 or flat[idx - 1]["start_index"] >= page - 1)):
            idx = idx - 1 if idx else None
        return idx

    front: list[dict] = []
    # Bookmarks a heading climbed out of; later headings stay out
    closed: set[int] = set()

    def open_anchor(idx: Optional[int]) -> Optional[int]:
        while idx is not None and idx in closed:
            idx = parent_pos[idx]
        return idx

    def place(node: dict, holder: Optional[tuple[dict, int]] = None) -> None:
        children = node.get("nodes") or []
        read_at = read_pos[id(node)]
        idx = read_before(node, anchor_index(node["start_index"]))
        if idx is None:
            if _subtree_max_start(node) < flat[0]["start_index"]:
                front.append(node)
            else:
                front.append(dict(node, nodes=[]))
                for child in children:
                    place(child)
            return
        matched = dissolve_target(node, idx, matched_frames)
        if matched is not None:
            frame = flat[matched]
            for key in ("_style", "_y"):
                if frame.get(key) is None and node.get(key) is not None:
                    frame[key] = node[key]
            for child in children:
                place(child)
            return
        if (holder is not None and holder[1] == idx
                and _numbered_under(holder[0]["title"], node["title"])):
            siblings = holder[0]["nodes"]
        else:
            chosen = boundary_parent(node, idx)
            if chosen is None or chosen in closed:
                anchor = open_anchor(idx)
                chosen = None if anchor is None else choose_parent(node, anchor)
                pos = anchor
                while pos is not None and pos != chosen:
                    closed.add(pos)
                    pos = parent_pos[pos]
            siblings = roots if chosen is None else flat[chosen]["nodes"]
        if _subtree_max_start(node) < range_end[idx]:
            insert_by_page(siblings, node, read_at)
            return
        leaf = dict(node, nodes=[])
        insert_by_page(siblings, leaf, read_at)
        for child in children:
            place(child, (leaf, idx))

    pruned = prune(structure)
    reading: list[dict] = []

    def read(nodes: list[dict]) -> None:
        for node in nodes:
            reading.append(node)
            read(node.get("nodes") or [])

    read(pruned)
    found: dict[int, Optional[str]] = {}
    for pos, node in enumerate(reading):
        read_pos[id(node)] = pos
        idx = anchor_index(node["start_index"])
        target = None if idx is None else dissolve_target(node, idx, found)
        if target is not None:
            heading_pos.setdefault(target, pos)
    for root in pruned:
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
