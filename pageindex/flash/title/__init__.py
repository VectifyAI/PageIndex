"""Document-title detection. The scoring formula is the heart of title detection: score is a product of layout, recurrence, label, script, width, numbering, punctuation, alignment, and page-position factors. Each factor is in roughly ``[0.1, 3.0]-- the product can grow to a few
thousand for a strong title candidate. The factors are documented in the
scoring body. The multilingual title-keyword and institution-word sets are stored in
``data/dictionaries.json`` as ``title`` and ``institution_words``.
"""

import json
import math
import unicodedata
from pathlib import Path
from typing import Optional

from ..model import (
    _trim_unicode_ws,
    left_aligned,
    right_aligned,
    center_aligned,
    Rect,
    last_span,
    heading_score,
    Line,
    last_line_of,
    first_span_of,
    block_text,
    deaccented_text,
    letter_count,
    dominant_style_of,
    info_weight,
    is_upper_dominant,
    alignment_code,
    Block,
)
from ..stats import DocStats, column_index_of, tally_scripts, dominant_script_family, ScriptHistogram
from ..tokens import is_superscript_adjacent, clamp_value, enumerate_tokens, jenkins_hash, trie_prefix_match, set_case_fold, TrieConfig, build_trie, tokenize_block, _de_norm, BuiltTrie, is_word_token

from .dicts import (
    _DICT_PATH,
    _normalize_text_key,
    _load_dicts,
    INSTITUTION_WORDS,
    TITLE_LABEL_TRIE,
)
from .scoring import (
    TitleCandidate,
    is_cover_like_page,
    is_title_candidate_block,
    score_title_candidate,
)
from .detect import (
    TitleSearchState,
    detect_title,
)

__all__ = ["is_title_candidate_block", "score_title_candidate", "TitleSearchState", "TitleCandidate", "detect_title", "is_cover_like_page", "TITLE_LABEL_TRIE", "INSTITUTION_WORDS"]
