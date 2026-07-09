# pageindex/page_index.py
# Deprecation shim. The PDF indexing pipeline now lives in
# pageindex/index/page_index.py (the single source of truth). This module
# re-exports it so legacy imports (`from pageindex.page_index import ...`,
# `from pageindex import page_index`) keep working.
import sys
import types
import warnings

warnings.warn(
    "pageindex.page_index has moved to pageindex.index.page_index; importing it "
    "from the top level is deprecated and will be removed in a future release.",
    PendingDeprecationWarning,
    stacklevel=2,
)

from .index.page_index import *  # noqa: F401,F403,E402

# pageindex/__init__.py binds the FUNCTION `page_index` as the package
# attribute `pageindex.page_index` (`from .index.page_index import *`). But
# this file is ALSO a real submodule of the same name — the moment anything,
# anywhere in the process, does `import pageindex.page_index` (exactly what
# `from pageindex.page_index import X` triggers), Python's import machinery
# overwrites that package attribute with THIS module object, clobbering the
# function binding. Afterwards `from pageindex import page_index; page_index(x)`
# would raise "TypeError: 'module' object is not callable" — silently, and
# depending entirely on whether this submodule happened to be imported yet.
#
# Fix: make this module itself callable, delegating to the real function, so
# whichever object ends up sitting in the `pageindex.page_index` slot — the
# function or this module — is callable either way. Both `from pageindex.page_index
# import page_index_main` (module attribute access) and
# `from pageindex import page_index; page_index(x)` (call) keep working
# regardless of import order.
class _CallableModule(types.ModuleType):
    def __call__(self, *args, **kwargs):
        return page_index(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
