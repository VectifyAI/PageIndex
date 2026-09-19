"""Trie construction, matching, and token trimming utilities."""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Optional

from ..model import (
    _strip_diacritics,
    avg_char_width2,
    intervals_overlap,
    to_number,
    rect_union,
    EMPTY_RECT,
    avg_char_width,
    Line,
    char_category,
    is_word_category,
    is_punct_category,
    letter_count,
    punct_count,
    info_weight,
    Block,
)

from .token_types import (
    can_extend_token,
    is_trimmable_token,
    TokenView,
    wrap_tokens,
    enumerate_tokens,
    first_token,
    last_token,
)


# --------------------------------------------------------------------------- #
# Token trie matcher and builder.
# --------------------------------------------------------------------------- #


def _de_norm(text: str, case_fold: bool) -> str:
    """Normalize trie keys by optional case folding, NFD decomposition, combining-mark stripping, and NFC recomposition. This strips diacritics without applying compatibility normalization."""
    return _strip_diacritics(text.lower() if case_fold else text)


class TrieConfig:
    """Trie configuration: reverse-match mode and case-fold mode."""

    __slots__ = ("primary_slot", "secondary_slot")

    def __init__(self):
        self.primary_slot: bool = False
        self.secondary_slot: bool = False


class BuiltTrie:
    """Built trie wrapper containing the root node and a reverse-match flag."""

    __slots__ = ("secondary_slot", "primary_slot")

    def __init__(self, primary_item: "TrieNode", candidate_flag: bool):
        self.secondary_slot = primary_item   # root node
        self.primary_slot = candidate_flag   # reverse-match flag


def set_reverse(primary_item: TrieConfig) -> TrieConfig:
    """set reverse flag."""
    primary_item.primary_slot = True
    return primary_item


def set_case_fold(primary_item: TrieConfig, other_flag: bool) -> TrieConfig:
    """set case-fold flag."""
    primary_item.secondary_slot = other_flag
    return primary_item


class TrieNode:
    """- trie node."""

    __slots__ = ("str", "depth", "primary_slot", "children", "dict_suffix_link", "failure_link", "is_terminal", "payload")

    def __init__(self, other_text: str, depth: int, case_fold: bool):
        self.str = other_text
        self.depth = depth
        self.primary_slot = case_fold
        self.children: dict[str, "TrieNode"] = {}
        self.dict_suffix_link = None
        self.failure_link: Optional["TrieNode"] = None
        self.is_terminal = False
        self.payload = None

    def normalize(self, other_text: str) -> str:
        return _de_norm(other_text, self.primary_slot)


def trie_insert_step(node: TrieNode, other_text: str) -> TrieNode:
    """walk one child, creating if absent."""
    key = node.normalize(other_text)
    child = node.children.get(key)
    if child is None:
        child = TrieNode(key, node.depth + 1, node.primary_slot)
        node.children[key] = child
    return child


def trie_walk_step(node: TrieNode, other_text: str) -> TrieNode:
    """Walk one child; if absent, fall back through failure links."""
    key = node.normalize(other_text)
    child = node.children.get(key)
    if child is not None:
        return child
    if node.failure_link is not None:
        return trie_walk_step(node.failure_link, other_text)
    return node


def aho_corasick_match(trie: BuiltTrie, tokens) -> Optional[dict]:
    """Aho-Corasick walk over a trie. Returns the shortest earliest terminal match and its payload. Dictionary-suffix matches use the suffix depth for match length while retaining the current node payload, which is load-bearing for edge cases."""
    if isinstance(tokens, list):
        tokens = wrap_tokens(tokens)
    if trie.primary_slot:
        tokens = tokens.reverse()
    matched_tokens: Optional[TokenView] = None
    matched_reverse = None
    earliest_start = -1
    node: TrieNode = trie.secondary_slot   # root node
    for entry in enumerate_tokens(tokens):
        index = entry["index"]
        token = entry["token"]
        node = trie_walk_step(node, token.str)
        depth = node.depth if node.is_terminal else 0
        if depth > 0 and (earliest_start < 0 or index - depth + 1 <= earliest_start):
            earliest_start = index - depth + 1
            matched_tokens = tokens.slice(earliest_start, index + 1)
            matched_reverse = node.payload
            if trie.primary_slot:
                matched_tokens = matched_tokens.reverse()
        kb_node = node.dict_suffix_link
        kb_depth = kb_node.depth if kb_node is not None else 0
        if kb_depth > 0 and (earliest_start < 0 or index - kb_depth + 1 <= earliest_start):
            earliest_start = index - kb_depth + 1
            matched_tokens = tokens.slice(earliest_start, index + 1)
            matched_reverse = node.payload

            if trie.primary_slot:
                matched_tokens = matched_tokens.reverse()
        # Once a match exists and the current path start has moved past the
        # earliest match start, no later token can produce an earlier match.
        if earliest_start >= 0 and index - node.depth + 1 > earliest_start:
            break
    if matched_tokens is None:
        return None
    return {"tokens": matched_tokens, "payload": matched_reverse}


def aho_corasick_tokens(trie: BuiltTrie, tokens) -> Optional[TokenView]:
    """Return only the matched token view from an Aho-Corasick match."""
    token = aho_corasick_match(trie, tokens)
    return token["tokens"] if token is not None else None


class TrieBuilder:
    """Trie builder context holding the root node and configuration."""

    __slots__ = ("primary_slot", "secondary_slot")

    def __init__(self, query_value: TrieConfig):
        self.primary_slot = TrieNode("", 0, query_value.secondary_slot)   # root node
        self.secondary_slot = query_value                # the config


def _trie_insert_entry(builder: TrieBuilder, entry: str, payload: Optional[Any] = None) -> None:
    """Insert one phrase into the trie after character-by-character tokenization. This keeps punctuation-attached phrases such as ``vol.`` and ``etc.`` aligned with document tokenization. The optional payload is stored only on an empty terminal payload slot."""
    node = builder.primary_slot
    tokens: list[str] = []
    trie = ""
    previous_category = 0
    for char in entry:
        cat = char_category(char)
        if cat == 10 or (trie and not can_extend_token(previous_category, cat, char)):
            if trie:
                tokens.append(trie)
            trie = ""
        if cat != 10:
            trie += char
        previous_category = cat
    if trie:
        tokens.append(trie)
    if builder.secondary_slot.primary_slot:
        tokens.reverse()
    for tok in tokens:
        node = trie_insert_step(node, tok)
    node.is_terminal = True
    # Payload assignment uses truthiness: falsy payloads are skipped, and falsy
    # existing payloads are overwritten. In this package payloads are non-empty
    # dictionary-like objects, so the truthiness contract is stable.
    if payload and not node.payload:
        node.payload = payload


def trie_bulk_insert(builder: TrieBuilder, entries, payload: Optional[Any] = None) -> None:
    """Bulk-insert phrases into ``builder`` with a shared terminal payload."""
    for entry in entries:
        _trie_insert_entry(builder, entry, payload)


def _trie_finalize(builder: TrieBuilder) -> BuiltTrie:
    """Assign Aho-Corasick failure links and dictionary-suffix links with breadth-first traversal, then return a built trie wrapper."""
    from collections import deque

    root = builder.primary_slot
    queue: deque = deque([root])
    while queue:
        node = queue.popleft()
        for child in node.children.values():
            queue.append(child)
            # failure link: longest proper suffix that is a prefix in the trie
            trie = node
            while trie.failure_link is not None:
                child.failure_link = trie.failure_link.children.get(trie.failure_link.normalize(child.str))
                if child.failure_link is not None:
                    break
                trie = trie.failure_link
            if child.failure_link is None:
                child.failure_link = root
            # dictionary-suffix link: nearest failure ancestor that is terminal
            trie = child.failure_link
            while trie is not None:
                if trie.is_terminal:
                    child.dict_suffix_link = trie
                    break
                trie = trie.failure_link

    return BuiltTrie(builder.primary_slot, builder.secondary_slot.primary_slot)


def build_trie(strings: Iterable[str], other_trie: Optional[TrieConfig] = None) -> BuiltTrie:
    """Build a trie from a list of phrase strings."""
    if other_trie is None:
        other_trie = TrieConfig()
    builder = TrieBuilder(other_trie)
    for trie in strings:
        _trie_insert_entry(builder, trie)
    return _trie_finalize(builder)


def trie_prefix_match(trie: BuiltTrie, tokens) -> Optional[TokenView]:
    """Return the longest prefix match against the token trie."""
    # ``tokens`` may be a TokenView or a list; coerce.
    if isinstance(tokens, list):
        tokens = wrap_tokens(tokens)
    if trie.primary_slot:
        tokens = tokens.reverse()

    matched: Optional[TokenView] = None
    node: TrieNode = trie.secondary_slot   # root node
    for entry in enumerate_tokens(tokens):
        if not node.children:
            break
        index = entry["index"]
        token = entry["token"]
        next_node = node.children.get(node.normalize(token.str))
        if next_node is None:
            break
        node = next_node
        if node.is_terminal:
            slice_view = tokens.slice(0, index + 1)
            if trie.primary_slot:
                slice_view = slice_view.reverse()
            matched = slice_view
    return matched


def _trie_full_match(trie: BuiltTrie, tokens) -> bool:
    """full-match check."""
    result = trie_prefix_match(trie, tokens)
    if isinstance(tokens, list):
        tokens_view = wrap_tokens(tokens)
    else:
        tokens_view = tokens
    return result is not None and result.length == tokens_view.length


trie_full_match = _trie_full_match


# --------------------------------------------------------------------------- #
# Token-list strip helpers.
# --------------------------------------------------------------------------- #


def strip_trie_match(tokens: TokenView, other_trie: BuiltTrie) -> TokenView:
    """Strip a matching keyword sequence from a token view."""
    trie = trie_prefix_match(other_trie, tokens)
    if trie is None:
        return tokens
    if other_trie.primary_slot:
        return tokens.slice(0, tokens.length - trie.length)
    return tokens.slice(trie.length)


def strip_leading_if_in(tokens: TokenView, other_items: set) -> TokenView:
    """Strip the leading token if its text is in the provided set."""
    first = first_token(tokens)
    if tokens.length > 0 and first is not None and first.str in other_items:
        return tokens.slice(1)
    return tokens


# Six comma variants only, not general punctuation.
COMMA_CHARS: set[str] = {",", "﹐", "，", "、", "﹑", "､"}


def strip_trailing_comma(tokens: TokenView) -> TokenView:
    """Strip a trailing comma token."""
    last = last_token(tokens)
    if tokens.length > 0 and last is not None and last.str in COMMA_CHARS:
        return tokens.slice(0, tokens.length - 1)
    return tokens


def is_comma_token(token) -> bool:
    """Return True when the token string is one of the supported comma variants."""
    return token is not None and token.str in COMMA_CHARS


def trim_trailing_punct(tokens: TokenView) -> TokenView:
    """Trim trailing punctuation-like tokens."""
    end = tokens.length
    while end > 0:
        tok = tokens.token_at(end - 1)
        if tok is None or not is_trimmable_token(tok):
            break
        end -= 1
    return tokens.slice(0, end)
