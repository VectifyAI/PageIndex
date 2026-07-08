# pageindex/page_index_md.py
# Deprecation shim. The Markdown indexing pipeline now lives in
# pageindex/index/page_index_md.py (the single source of truth). This module
# re-exports it so legacy imports keep working.
#
# The canonical md_to_tree coerces legacy 'yes'/'no' string flags itself (a
# bare 'no' would otherwise be truthy) — this shim used to duplicate that
# coercion in its own wrapper; now it just re-exports the canonical function.
import warnings

warnings.warn(
    "pageindex.page_index_md has moved to pageindex.index.page_index_md; "
    "importing it from the top level is deprecated and will be removed in a "
    "future release.",
    PendingDeprecationWarning,
    stacklevel=2,
)

from .index.page_index_md import *  # noqa: F401,F403,E402
