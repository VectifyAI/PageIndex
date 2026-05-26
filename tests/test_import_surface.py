from __future__ import annotations

import builtins
import importlib
import sys


def test_filesystem_import_works_without_eager_optional_dependencies(monkeypatch):
    blocked_roots = {"litellm", "openai", "PyPDF2", "pymupdf", "sqlite_vec"}
    real_import = builtins.__import__

    def clear_pageindex_modules() -> None:
        for name in list(sys.modules):
            if name == "pageindex" or name.startswith("pageindex."):
                sys.modules.pop(name, None)

    def import_without_optional_deps(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if root in blocked_roots:
            raise ModuleNotFoundError(f"No module named '{root}'", name=root)
        return real_import(name, globals, locals, fromlist, level)

    clear_pageindex_modules()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", import_without_optional_deps)

            filesystem_module = importlib.import_module("pageindex.filesystem")
            from pageindex import PageIndexFileSystem as TopLevelPageIndexFileSystem
            from pageindex.filesystem import PageIndexFileSystem

            assert filesystem_module.PageIndexFileSystem is PageIndexFileSystem
            assert TopLevelPageIndexFileSystem is PageIndexFileSystem
    finally:
        clear_pageindex_modules()
