# pageindex/utils.py — re-exports from index/utils.py
from .index.utils import *  # noqa: F401,F403,E402
# Legacy 0.2.x alias. index.utils keeps it private (_config) so its star-export
# can't shadow the pageindex.config submodule; re-expose it only in this shim.
from types import SimpleNamespace as config  # noqa: E402,F401
