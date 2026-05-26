from importlib import import_module

from .commands import PIFSCommandExecutor
from .core import PageIndexFileSystem
from .metadata_generation import (
    MetadataGenerationBackend,
    MetadataGenerationError,
    MetadataGenerationInput,
    MetadataGenerationResult,
    MetadataGenerator,
)
from .types import OpenResult, SearchResult

_LAZY_EXPORTS = {
    "HybridProjectionSearchBackend": (".hybrid_projection", "HybridProjectionSearchBackend"),
    "RebuildableSemanticIndex": (".semantic_index", "RebuildableSemanticIndex"),
    "SemanticIndexRecord": (".semantic_index", "SemanticIndexRecord"),
    "SemanticSearchResult": (".semantic_index", "SemanticSearchResult"),
    "SQLiteVecSemanticIndex": (".semantic_index", "SQLiteVecSemanticIndex"),
    "SummaryProjectionIndexer": (".projection_indexing", "SummaryProjectionIndexer"),
}

__all__ = [
    "OpenResult",
    "HybridProjectionSearchBackend",
    "MetadataGenerationBackend",
    "MetadataGenerationError",
    "MetadataGenerationInput",
    "MetadataGenerationResult",
    "MetadataGenerator",
    "PIFSCommandExecutor",
    "PageIndexFileSystem",
    "RebuildableSemanticIndex",
    "SearchResult",
    "SemanticIndexRecord",
    "SemanticSearchResult",
    "SummaryProjectionIndexer",
    "SQLiteVecSemanticIndex",
]


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attribute_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attribute_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
