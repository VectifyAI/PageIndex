# pageindex/utils.py
# Deprecation shim. The indexing utilities now live in pageindex/index/utils.py,
# which is the single source of truth. This module re-exports them so legacy
# imports (`from pageindex.utils import ...`) keep working.
import warnings

warnings.warn(
    "pageindex.utils has moved to pageindex.index.utils; importing it from the "
    "top level is deprecated and will be removed in a future release.",
    PendingDeprecationWarning,
    stacklevel=2,
)

from .index.utils import *  # noqa: F401,F403,E402
