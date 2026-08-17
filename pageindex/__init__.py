"""PageIndex SDK."""
import os as _os
from typing import TYPE_CHECKING as _TYPE_CHECKING

# LiteLLM's import otherwise fetches its model map over the network — seconds
# of blocking (or a hang offline). setdefault, so an explicit user choice wins.
_os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from .client import PageIndexClient, PageIndexCloudClient, PageIndexLocalClient
from .errors import PageIndexAPIError

if _TYPE_CHECKING:
    from .flash import page_index_flash
    from .page_index_classic import page_index, page_index_main
    from .page_index_md import md_to_tree
    from .tree_optimize import optimize_tree

__all__ = [
    "PageIndexClient", "PageIndexCloudClient", "PageIndexLocalClient",
    "PageIndexAPIError",
    "page_index", "page_index_main", "page_index_flash",
    "optimize_tree", "md_to_tree",
]

_LAZY = {
    "page_index": ".page_index_classic",
    "page_index_main": ".page_index_classic",
    "page_index_flash": ".flash",
    "optimize_tree": ".tree_optimize",
    "md_to_tree": ".page_index_md",
}
_SUBMODULES = {"agent_tools", "client", "cloud_api", "errors", "flash",
               "integrations", "local_api", "local_chat", "local_store",
               "mcp_bridge", "page_index_classic", "page_index_md",
               "tree_optimize", "utils"}

def __getattr__(name):
    if name.startswith("_"):
        # Dunder probes (copy, pickle, inspect) are the frequent unknown
        # names — they must not trigger the classic import below.
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    # Pre-0.2.10 compat: unknown names fall through to the classic module,
    # whose public surface (ConfigLoader, count_tokens, ...) resolved as
    # package attributes. A non-underscore typo pays one classic import
    # before its AttributeError — not worth an allowlist.
    module = importlib.import_module(_LAZY.get(name, ".page_index_classic"),
                                     __name__)
    try:
        value = getattr(module, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}") from None
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)
