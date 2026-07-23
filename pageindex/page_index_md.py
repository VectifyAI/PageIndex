# pageindex/page_index_md.py
# Compatibility shim. The Markdown indexing pipeline now lives in
# pageindex/index/page_index_md.py (the single source of truth). This module
# re-exports it so legacy imports keep working.
#
# The canonical md_to_tree coerces legacy 'yes'/'no' string flags itself (a
# bare 'no' would otherwise be truthy) — this shim used to duplicate that
# coercion in its own wrapper; now it just re-exports the canonical function.
from .index.page_index_md import *  # noqa: F401,F403,E402
