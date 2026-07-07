# pageindex/page_index.py
# Deprecation shim. The PDF indexing pipeline now lives in
# pageindex/index/page_index.py (the single source of truth). This module
# re-exports it so legacy imports (`from pageindex.page_index import ...`,
# `from pageindex import page_index`) keep working.
import warnings

warnings.warn(
    "pageindex.page_index has moved to pageindex.index.page_index; importing it "
    "from the top level is deprecated and will be removed in a future release.",
    PendingDeprecationWarning,
    stacklevel=2,
)

from .index.page_index import *  # noqa: F401,F403,E402
