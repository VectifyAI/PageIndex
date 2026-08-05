"""Page-level and document-level layout statistics. The statistics layer computes weighted percentiles, dominant styles, script
families, page spacing measures, and document-wide recurrence signals used by
classification and outline assembly.
"""

import functools
import json
import math
from pathlib import Path
from typing import Optional

from ..model import Span, _format_half_up_one_decimal, Line, info_weight, _max_nan_propagating

from .scripts import (
    _SCRIPT_BUCKET_TABLE_PATH,
    SCRIPT_BUCKET_TABLE,
    char_script_bucket,
    SCRIPT_FAMILY_WEIGHTS,
    ScriptHistogram,
    tally_scripts,
    dominant_script_family,
)
from .aggregates import (
    _percentile_sample_cmp,
    weighted_percentile,
    style_key,
    PageStats,
    compute_page_stats,
    DocStats,
    compute_doc_stats,
    column_index_of,
)

__all__ = [
    "weighted_percentile",
    "style_key",
    "PageStats",
    "compute_page_stats",
    "DocStats",
    "compute_doc_stats",
    "column_index_of",
    "char_script_bucket",
    "tally_scripts",
    "dominant_script_family",
    "ScriptHistogram",
    "SCRIPT_FAMILY_WEIGHTS",
    "SCRIPT_BUCKET_TABLE",
]
