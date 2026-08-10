"""Package surface: the SDK eagerly, the indexing stack lazily (PEP 562).

`import pageindex` must stay cheap for cloud-only SDK use -- the indexing
modules cost ~0.7s to import -- so every name beyond client/errors resolves
on first attribute access. Don't add eager imports here.
"""
from typing import TYPE_CHECKING as _TYPE_CHECKING

from .client import PageIndexClient, PageIndexCloudClient, PageIndexLocalClient
from .errors import PageIndexAPIError

if _TYPE_CHECKING:
    # Static-only bindings for the lazily-resolved names below, so type
    # checkers and IDEs see real signatures without the runtime import cost.
    from .flash import page_index_flash
    from .page_index import page_index, page_index_main
    from .page_index_md import md_to_tree
    from .tree_optimize import optimize_tree

__all__ = [
    "PageIndexClient", "PageIndexCloudClient", "PageIndexLocalClient",
    "PageIndexAPIError",
    "page_index", "page_index_main", "page_index_flash",
    "optimize_tree", "md_to_tree",
]

_LAZY = {
    "page_index_flash": ".flash",
    "optimize_tree": ".tree_optimize",
    "md_to_tree": ".page_index_md",
}
# "page_index" is absent: the function of that name shadows the submodule,
# as it always did under `from .page_index import *`.
_SUBMODULES = {"client", "cloud_api", "errors", "flash", "local_api",
               "local_store", "page_index_md", "tree_optimize", "utils"}


def __getattr__(name):
    if name.startswith("_"):
        # Tooling probes dunders (pickle, IPython); never trigger the heavy
        # import for them.
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    module = importlib.import_module(_LAZY.get(name, ".page_index"), __name__)
    try:
        value = getattr(module, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)
