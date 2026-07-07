# pageindex/page_index_md.py
# Deprecation shim. The Markdown indexing pipeline now lives in
# pageindex/index/page_index_md.py (the single source of truth). This module
# re-exports it so legacy imports keep working.
#
# The canonical md_to_tree takes booleans; legacy callers passed 'yes'/'no'
# strings, so the wrapper below coerces them (a bare 'no' is otherwise truthy).
import warnings

warnings.warn(
    "pageindex.page_index_md has moved to pageindex.index.page_index_md; "
    "importing it from the top level is deprecated and will be removed in a "
    "future release.",
    PendingDeprecationWarning,
    stacklevel=2,
)

from .index.page_index_md import *  # noqa: F401,F403,E402
from .index.page_index_md import md_to_tree as _md_to_tree  # noqa: E402

_BOOL_PARAMS = (
    "if_thinning", "if_add_node_summary", "if_add_doc_description",
    "if_add_node_text", "if_add_node_id",
)


def _coerce_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("yes", "true", "1", "y", "on")
    return bool(value)


async def md_to_tree(*args, **kwargs):
    """Legacy wrapper: coerce 'yes'/'no' string flags to bool, then delegate."""
    for key in _BOOL_PARAMS:
        if key in kwargs:
            kwargs[key] = _coerce_bool(kwargs[key])
    return await _md_to_tree(*args, **kwargs)
