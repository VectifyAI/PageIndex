"""Heading candidate collection, filtering, and keyword screening."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

from ..labels import extract_structural_number
from ..model import numbering_text, numbering_kind, block_text, is_caps_heavy, Block
from ..stats import column_index_of
from ..tokens import set_case_fold, TrieConfig, build_trie, tokenize_block, trie_full_match


# English section keywords loaded into a case-folded trie matching tokenized
# block text exactly.
SECTION_KEYWORD_TRIE = build_trie(
    [
        "acknowledgements", "acknowledgments",
        "background",
        "conclusion", "conclusions",
        "discussion",
        "introduction",
        "materials and methods",
        "method", "methods",
        "results",
    ],
    set_case_fold(TrieConfig(), True),
)


def heading_order_key(block: Block, page_lookup: dict[int, int]) -> tuple:
    """Sort by page, then column index and reading position."""
    return (
        page_lookup.get(id(block), 1),
        column_index_of(block),
        -block.top_edge(),
        -block.bottom_edge(),
        block.left_edge(),
        block.right_edge(),
    )


# --------------------------------------------------------------------------- #
# Heading candidate gates #
# --------------------------------------------------------------------------- #


# Block types excluded from heading candidacy:
# 1 header, 2 footer, 3 references body, 9 TOC page content,
# 12 watermark/caption, 13 claimed labeled-section body, 99 title.
_NON_HEADING_TYPES = frozenset({1, 2, 3, 9, 12, 13, 99})

_DOT_LEADER_RE = re.compile(r"\.{4,}\s*\d+\s*$")
_CAPTION_LABEL_RE = re.compile(
    r"^\s*(?:figure|fig\.?|table|tab\.?|algorithm|alg\.?|equation|eq\.?|listing)\s+\d",
    re.IGNORECASE,
)
# Equation labels like "(1)", "(2.3)", "(a)", "(i)", "(*)" -- parenthesised
# short labels that the numbering detector mistakes for "1." section starts.
_EQUATION_LABEL_RE = re.compile(
    r"^\s*[\[\(]\s*(?:[0-9]+(?:\.\d+)?[a-z]?|[a-z]|[ivx]+)\s*[\)\]]\s*$",
    re.IGNORECASE,
)
# Parenthesised body fragments like "(current cost)", "(estimated cost)".
_PAREN_FRAGMENT_RE = re.compile(r"^\s*[\[\(][^\]\)]{1,40}[\]\)]\s*$")


def _looks_like_pseudo_code(text: str) -> bool:
    """Reject pseudo-code and math fragments that can resemble numbered headings."""
    if any(pat.search(text) for pat in _PSEUDO_CODE_PATTERNS):
        return True
    # No alphabetic word of >= 3 letters? Reject.
    if not re.search(r"[A-Za-zÀ-ÿ一-鿿가-힯]{3,}", text):
        return True
    # Bullet-list item: "1. long flowing prose..."
    if _BULLET_LIST_RE.match(text) and len(text) > 80:
        return True
    # Parenthesised fragment: "(current cost)", "(maximum flow)"
    if _PAREN_FRAGMENT_RE.match(text):
        return True
    # Author-block heuristics
    if any(pat.search(text) for pat in _AUTHOR_PATTERNS):
        return True
    return False


def is_heading_candidate(block: Block, body_size: float, body_bold: bool) -> bool:
    """Return whether a block has the visual and textual shape of a heading."""
    if block.type in _NON_HEADING_TYPES:
        return False
    if block.line_count() > 6:
        return False
    text = block_text(block).strip()
    if len(text) < 2 or len(text) > 200:
        return False
    if _DOT_LEADER_RE.search(text):
        return False
    if _CAPTION_LABEL_RE.match(text):
        return False
    if _EQUATION_LABEL_RE.match(text):
        return False
    if _looks_like_pseudo_code(text):
        return False
    marker_type = getattr(block, "marker_slot", 0)
    if marker_type == 4:
        return True
    block_size = block.avg_font_size() or body_size
    size_gain = block_size / max(body_size, 1e-3)
    if size_gain >= 1.08:
        return True
    if block.bold_frac() > 0.5 and not body_bold and size_gain >= 0.95:
        return True
    if block.line_count() >= 1 and numbering_kind(block.line()) != 0 and len(text) <= 120 and (
        block.bold_frac() > 0.3 or size_gain >= 1.0
    ):
        return True
    if is_caps_heavy(block) and len(text) <= 80 and size_gain >= 1.0:
        return True
    return False


# --------------------------------------------------------------------------- #
# Style buckets + level assignment #
# --------------------------------------------------------------------------- #


def _style_key(block: Block) -> tuple[str, float, bool]:
    """Hashable signature for grouping headings into hierarchy levels."""
    first_span = block.line().primary_slot[0] if block.line().primary_slot else None
    font = first_span.font_name if first_span else ""
    return (font, round(block.avg_font_size(), 1), block.bold_frac() > 0.5)


def _numbering_depth(block: Block) -> Optional[int]:
    """Return the section-numbering depth, e.g. ``1.2.3-> 3. None if the block doesn't start with a digit-style number (only digit chains use ``.``-separated depth; Roman / letter labels return 1). """
    if numbering_kind(block.line()) != 1:
        return None
    text = numbering_text(block.line())
    if not text:
        return None
    if re.match(r"^\d+(?:\.\d+)*$", text):
        return text.count(".") + 1
    return 1


def collect_headings(doc) -> list[Block]:
    """Walk all pages, gather heading-candidate blocks in reading order."""
    body_size = doc.secondary_slot.primary_slot
    body_bold = doc.secondary_slot.tertiary_slot == 0
    out: list[Block] = []
    for page in doc.primary_slot:
        for block in (page.secondary_slot or []):
            if is_heading_candidate(block, body_size, body_bold):
                out.append(block)
    return out


# --------------------------------------------------------------------------- #
# Clique selection #
# --------------------------------------------------------------------------- #


def _heading_signature(block: Block) -> str:
    """Return the first visible span's font style for a heading block."""
    if not block.primary_slot or not block.primary_slot[0].primary_slot:
        return ""
    return block.primary_slot[0].primary_slot[0].font_style()


def _matches_section_keywords(block: Block) -> bool:
    """Full-match canonical English section names after stripping a leading structural number. For example, "1 Introduction" tokenizes as ["1", "Introduction"], and the numeric prefix must be removed before keyword matching."""
    tokens = tokenize_block(block)
    prefix = extract_structural_number(tokens)
    if prefix is not None:
        tokens = tokens.slice(prefix.length)
    return trie_full_match(SECTION_KEYWORD_TRIE, tokens)


_PSEUDO_CODE_PATTERNS = (
    re.compile(r"^\s*\d+\s*:"),                        # "2:" / "10:" lead -> pseudo-code step
    re.compile(r"[∀-⋿←-⇿≤≥≠∈∉∂∇∑∏√]"),                  # math operators
    re.compile(r"^\s*\d+[a-z]"),                       # "9else", "15return" (no space)
    re.compile(r"^[\d.\s/]+$"),                        # pure numbers / decimals
    re.compile(r"^\s*\d+\s*[+\-*/=]\s*\d"),            # arithmetic
    re.compile(r"^\s*[a-z]+\s*[+\-*/=]\s*"),           # variable assignments
    re.compile(r"\bwhile\b|\bif\b|\belse\b|\bfor\b|\breturn\b|\bdo\b", re.IGNORECASE),  # code keywords
    # Subfigure captions like "(a) RETINA", "(b) IRMA", "(i) plot"
    re.compile(r"^\s*[\(\[]\s*[a-zivx]+\s*[\)\]]\s+\w"),
)

# Author block / affiliation patterns:
# * "Yu Tang†, Leong Hou U‡, ..." -- multiple comma-separated names with
# affiliation markers
# * "Kimi Team" -- short "X Team" / "X Lab" / "X Group" naming
# * "†The University of ..." -- starts with affiliation marker
# * "{user, another}@domain" -- email block
_AUTHOR_PATTERNS = (
    re.compile(r"[†‡§¶∗*]"),                            # affiliation markers
    re.compile(r"@\S+\."),                              # contains email
    re.compile(r"^\s*\S+\s+(?:Team|Group|Lab|Labs|Inc\.|Corp\.|Co\.)\s*$"),
)
# Bullet-list items: "1. " followed by long flowing text (>80 chars total)
_BULLET_LIST_RE = re.compile(r"^\s*\d+\.\s+\w")


def filter_by_clique(headings: list[Block]) -> list[Block]:
    """Keep headings that match the dominant section-keyword style group."""
    if not headings:
        return headings
    anchor_groups: dict[str, list[Block]] = defaultdict(list)
    for state_item in headings:
        if _matches_section_keywords(state_item):
            anchor_groups[_heading_signature(state_item)].append(state_item)
    if not anchor_groups:
        return headings
    winner_sig, winner_group = max(anchor_groups.items(), key=lambda item_pair: len(item_pair[1]))
    if len(winner_group) <= 1:
        return headings
    out: list[Block] = []
    for state_item in headings:
        if _heading_signature(state_item) == winner_sig:
            out.append(state_item)
            continue
        # Different font from heading clique. Only keep if it's a clearly
        # numbered heading that doesn't smell of pseudo-code / math.
        if numbering_kind(state_item.line()) != 1:
            continue
        numbering = numbering_text(state_item.line())
        if not re.match(r"^\d+(?:\.\d+){0,2}$", numbering):
            continue
        text = block_text(state_item)
        if any(pat.search(text) for pat in _PSEUDO_CODE_PATTERNS):
            continue
        # Also require: at least one alphabetic word AFTER the number
        # ("2 Introduction" yes, "9else" no, "1.804 1.737 1.692" no)
        after_num = re.sub(r"^\s*\d+(?:\.\d+){0,2}\s*[.:)]?\s*", "", text)
        if not re.search(r"[A-Za-zÀ-ÿ一-鿿가-힯]{3,}", after_num):
            continue
        out.append(state_item)
    return out
