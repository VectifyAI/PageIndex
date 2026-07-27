"""Dictionary tables for title detection."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from ..tokens import is_superscript_adjacent, clamp_value, enumerate_tokens, jenkins_hash, trie_prefix_match, set_case_fold, TrieConfig, build_trie, tokenize_block, _de_norm, BuiltTrie, is_word_token


# --------------------------------------------------------------------------- #
# Load title-label and institution dictionaries #
# --------------------------------------------------------------------------- #


_DICT_PATH = Path(__file__).parent.parent / "data" / "dictionaries.json"


def _normalize_text_key(text: str) -> str:
    """NFKC + strip + collapse-whitespace + lowercase."""
    return " ".join(unicodedata.normalize("NFKC", text).strip().split()).lower()


def _load_dicts() -> tuple[BuiltTrie, set[str]]:
    raw = json.loads(_DICT_PATH.read_text(encoding="utf-8"))
    title_label_trie = build_trie(raw.get("title", []), set_case_fold(TrieConfig(), True))
    # institution words use normalized single-token set membership.
    # The title-label dictionary stays a trie because it handles the

    # multi-token "Title:" match; institution words are single-token only.)
    institution_words = set(raw.get("institution_words", []))
    return title_label_trie, institution_words


TITLE_LABEL_TRIE, INSTITUTION_WORDS = _load_dicts()
