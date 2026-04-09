from .pageindex_runner import PageIndex, PageIndexConfig

try:
    from .page_index import *
    from .page_index_md import md_to_tree
    from .retrieve import get_document, get_document_structure, get_page_content
    from .client import PageIndexClient
except ModuleNotFoundError:
    # Allows importing lightweight APIs (e.g. PageIndex config/validation)
    # before runtime dependencies are installed.
    pass
