# pageindex/backend/local.py
import hashlib
import os
import re
import sqlite3
import uuid
import shutil
from pathlib import Path

from ..parser.protocol import DocumentParser, ParsedDocument
from ..parser.pdf import PdfParser
from ..parser.markdown import MarkdownParser
from ..storage.protocol import StorageEngine
from ..index.pipeline import build_index
from ..index.utils import parse_pages, get_pdf_page_content, remove_fields
from ..backend.protocol import AgentTools
from ..errors import (FileTypeError, DocumentNotFoundError, CollectionNotFoundError,
                      IndexingError, PageIndexError)

# Matched with .fullmatch() (not .match()): a $-anchored .match() would accept a
# trailing newline ("papers\n") because $ matches just before a final \n.
_COLLECTION_NAME_RE = re.compile(r'[a-zA-Z0-9_-]{1,128}')


class LocalBackend:
    def __init__(self, storage: StorageEngine, files_dir: str, model: str = None,
                 retrieve_model: str = None, index_config=None):
        self._storage = storage
        self._files_dir = Path(files_dir)
        self._model = model
        self._retrieve_model = retrieve_model or model
        self._index_config = index_config
        self._parsers: list[DocumentParser] = [PdfParser(), MarkdownParser()]

    def register_parser(self, parser: DocumentParser) -> None:
        self._parsers.insert(0, parser)  # user parsers checked first

    def get_retrieve_model(self) -> str | None:
        return self._retrieve_model

    def _resolve_parser(self, file_path: str) -> DocumentParser:
        ext = os.path.splitext(file_path)[1].lower()
        for parser in self._parsers:
            if ext in parser.supported_extensions():
                return parser
        raise FileTypeError(f"No parser for extension: {ext}")

    # Collection management
    def _validate_collection_name(self, name: str) -> None:
        if not _COLLECTION_NAME_RE.fullmatch(name):
            raise PageIndexError(f"Invalid collection name: {name!r}. Must be 1-128 chars of [a-zA-Z0-9_-].")

    def create_collection(self, name: str) -> None:
        self._validate_collection_name(name)
        self._storage.create_collection(name)

    def get_or_create_collection(self, name: str) -> None:
        self._validate_collection_name(name)
        self._storage.get_or_create_collection(name)

    def list_collections(self) -> list[str]:
        return self._storage.list_collections()

    def delete_collection(self, name: str) -> None:
        # Validate before touching the filesystem — an unvalidated name like
        # "../.." would make the rmtree below escape files_dir entirely.
        self._validate_collection_name(name)
        self._storage.delete_collection(name)
        col_dir = self._files_dir / name
        if col_dir.exists():
            shutil.rmtree(col_dir)

    @staticmethod
    def _file_hash(file_path: str) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    # Document management
    def add_document(self, collection: str, file_path: str) -> str:
        file_path = os.path.realpath(file_path)
        if not os.path.isfile(file_path):
            # Missing path is a file-not-found error, not an unsupported-type one.
            raise FileNotFoundError(f"No such file: {file_path}")
        # Fail fast before the expensive parse + LLM indexing if the collection
        # doesn't exist — otherwise the FK constraint only trips at save time,
        # after the LLM work (and its cost) is already spent.
        if collection not in self._storage.list_collections():
            raise CollectionNotFoundError(
                f"Collection '{collection}' does not exist; "
                f"create it first (e.g. client.collection('{collection}'))."
            )
        parser = self._resolve_parser(file_path)

        # Dedup is content-only — same file is reused regardless of IndexConfig
        # changes. If you've changed IndexConfig and need a fresh tree, delete
        # the existing doc first or use a new collection.
        file_hash = self._file_hash(file_path)
        existing_id = self._storage.find_document_by_hash(collection, file_hash)
        if existing_id:
            return existing_id

        doc_id = str(uuid.uuid4())

        # Copy file to managed directory
        ext = os.path.splitext(file_path)[1]
        col_dir = self._files_dir / collection
        col_dir.mkdir(parents=True, exist_ok=True)
        managed_path = col_dir / f"{doc_id}{ext}"
        shutil.copy2(file_path, managed_path)

        try:
            # Store images alongside the document: files/{collection}/{doc_id}/images/
            images_dir = str(col_dir / doc_id / "images")
            parsed = parser.parse(file_path, model=self._model, images_dir=images_dir)
            result = build_index(parsed, model=self._model, opt=self._index_config)

            # Cache page text for fast retrieval (avoids re-reading files) and to
            # reconstruct node text on demand (get_document(include_text=True),
            # get_page_content fallback) independent of whether IndexConfig kept
            # text in the stored structure. build_index() already applies
            # if_add_node_text to result["structure"] for every strategy, so no
            # extra stripping is needed here.
            pages = [{"page": n.index, "content": n.content,
                      **({"images": n.images} if n.images else {})}
                     for n in parsed.nodes if n.content]

            self._storage.save_document(collection, doc_id, {
                "doc_name": parsed.doc_name,
                "doc_description": result.get("doc_description", ""),
                "file_path": str(managed_path),
                "file_hash": file_hash,
                "doc_type": ext.lstrip("."),
                "structure": result["structure"],
                "pages": pages,
            })
        except sqlite3.IntegrityError:
            # Lost a concurrent add of the same content (UNIQUE collection+hash).
            # Discard our managed files and return the winner's doc_id.
            managed_path.unlink(missing_ok=True)
            doc_dir = col_dir / doc_id
            if doc_dir.exists():
                shutil.rmtree(doc_dir)
            existing_id = self._storage.find_document_by_hash(collection, file_hash)
            if existing_id:
                return existing_id
            raise
        except Exception as e:
            managed_path.unlink(missing_ok=True)
            doc_dir = col_dir / doc_id
            if doc_dir.exists():
                shutil.rmtree(doc_dir)
            raise IndexingError(f"Failed to index {file_path}: {e}") from e

        return doc_id

    def _require_document(self, collection: str, doc_id: str) -> dict:
        """Return the document's storage row, or raise DocumentNotFoundError.

        Single source of truth for "does this doc exist" — every public method
        and agent tool below goes through this, so a missing doc always
        surfaces the same way instead of each caller re-implementing its own
        (and potentially inconsistent) existence check.
        """
        doc = self._storage.get_document(collection, doc_id)
        if not doc:
            raise DocumentNotFoundError(f"Document {doc_id} not found")
        return doc

    def get_document(self, collection: str, doc_id: str, include_text: bool = False) -> dict:
        """Get document metadata with structure.

        Args:
            include_text: If True, populate each structure node's 'text' field
                from cached page content. WARNING: may be very large — do NOT
                use in agent/LLM contexts as it can exhaust the context window.
        """
        doc = self._require_document(collection, doc_id)
        doc["structure"] = self._storage.get_document_structure(collection, doc_id)
        if include_text:
            pages = self._storage.get_pages(collection, doc_id) or []
            page_map = {p["page"]: p["content"] for p in pages}
            self._fill_node_text(doc["structure"], page_map)
        return doc

    @staticmethod
    def _fill_node_text(nodes: list, page_map: dict) -> None:
        """Recursively fill 'text' on structure nodes from cached page content.

        Two node conventions, one per indexing strategy: content_based (PDF)
        nodes span a start_index..end_index page range; level_based (Markdown)
        nodes map 1:1 to a single page keyed by line_num. Handling only the
        first would silently leave Markdown nodes with no text.
        """
        for node in nodes:
            start = node.get("start_index")
            end = node.get("end_index")
            if start is not None and end is not None:
                node["text"] = "\n".join(
                    page_map.get(p, "") for p in range(start, end + 1)
                )
            elif "line_num" in node:
                node["text"] = page_map.get(node["line_num"], "")
            if "nodes" in node:
                LocalBackend._fill_node_text(node["nodes"], page_map)

    def get_document_structure(self, collection: str, doc_id: str) -> list:
        self._require_document(collection, doc_id)
        return self._storage.get_document_structure(collection, doc_id)

    def get_page_content(self, collection: str, doc_id: str, pages: str) -> list:
        doc = self._require_document(collection, doc_id)
        page_nums = parse_pages(pages)

        # Try cached pages first (fast, no file I/O)
        cached_pages = self._storage.get_pages(collection, doc_id)
        if cached_pages:
            return [p for p in cached_pages if p["page"] in page_nums]

        # Fallback: re-derive from the source file, same as the PDF path below
        # — never from the stored structure, whose 'text' field may have been
        # stripped (if_add_node_text=False, the default). Reachable only for a
        # custom StorageEngine that doesn't cache pages (the built-in
        # SQLiteStorage always does).
        if doc["doc_type"] == "pdf":
            return get_pdf_page_content(doc["file_path"], page_nums)
        else:
            parser = self._resolve_parser(doc["file_path"])
            parsed = parser.parse(doc["file_path"], model=self._model)
            page_map = {n.index: n.content for n in parsed.nodes}
            return [{"page": p, "content": page_map[p]} for p in page_nums if p in page_map]

    def list_documents(self, collection: str) -> list[dict]:
        return self._storage.list_documents(collection)

    def delete_document(self, collection: str, doc_id: str) -> None:
        doc = self._require_document(collection, doc_id)
        if doc.get("file_path"):
            Path(doc["file_path"]).unlink(missing_ok=True)
        # Clean up images directory: files/{collection}/{doc_id}/
        doc_dir = self._files_dir / collection / doc_id
        if doc_dir.exists():
            shutil.rmtree(doc_dir)
        self._storage.delete_document(collection, doc_id)

    def get_agent_tools(self, collection: str, doc_ids: list[str] | None = None) -> AgentTools:
        """Build agent tools.

        - doc_ids=None (open mode): includes ``list_documents``; agent picks docs itself.
        - doc_ids=[...] (scoped mode): no ``list_documents``; the other tools
          hard-enforce the whitelist and reject out-of-scope doc_ids.

        Note ``is not None``: an empty list is a scope of *nothing* (reject every
        doc), NOT open mode. Using truthiness would let ``doc_ids=[]`` collapse to
        ``None`` and silently grant access to the whole collection.
        """
        from agents import function_tool
        import json
        storage = self._storage
        col_name = collection
        backend = self
        scope = set(doc_ids) if doc_ids is not None else None

        def _reject(doc_id: str) -> str | None:
            if scope is not None and doc_id not in scope:
                return json.dumps({
                    "error": f"doc_id '{doc_id}' is not in scope.",
                    "allowed_doc_ids": sorted(scope),
                })
            return None

        @function_tool
        def get_document(doc_id: str) -> str:
            """Get document metadata."""
            rejection = _reject(doc_id)
            if rejection:
                return rejection
            try:
                # _require_document (not backend.get_document) deliberately:
                # the metadata-only row, no 'structure' — keeps this tool's
                # output small for the agent's context window.
                doc = backend._require_document(col_name, doc_id)
            except DocumentNotFoundError:
                return json.dumps({"error": f"doc_id '{doc_id}' not found."})
            return json.dumps(doc)

        @function_tool
        def get_document_structure(doc_id: str) -> str:
            """Get document tree structure (without text)."""
            rejection = _reject(doc_id)
            if rejection:
                return rejection
            try:
                backend._require_document(col_name, doc_id)
            except DocumentNotFoundError:
                return json.dumps({"error": f"doc_id '{doc_id}' not found."})
            structure = storage.get_document_structure(col_name, doc_id)
            return json.dumps(remove_fields(structure, fields=["text"]), ensure_ascii=False)

        @function_tool
        def get_page_content(doc_id: str, pages: str) -> str:
            """Get page content. Use tight ranges: '5-7', '3,8', '12'."""
            rejection = _reject(doc_id)
            if rejection:
                return rejection
            try:
                result = backend.get_page_content(col_name, doc_id, pages)
            except DocumentNotFoundError:
                return json.dumps({"error": f"doc_id '{doc_id}' not found."})
            except (ValueError, AttributeError) as e:
                # A malformed page spec ("all", "5-") is a recoverable bad tool
                # argument: hand the model an actionable error it can correct
                # (mirroring the legacy retrieval tool) rather than letting the
                # ValueError surface as the agent SDK's generic tool-failure text.
                return json.dumps({
                    "error": f"Invalid pages format: {pages!r}. Use '5-7', '3,8', or '12'. Error: {e}"
                })
            return json.dumps(result, ensure_ascii=False)

        tools = [get_document, get_document_structure, get_page_content]

        if scope is None:
            @function_tool
            def list_documents() -> str:
                """List all documents in the collection."""
                return json.dumps(storage.list_documents(col_name))
            tools.insert(0, list_documents)

        return AgentTools(function_tools=tools)

    def _scoped_docs(self, collection: str, doc_ids: list[str]) -> list[dict]:
        """Fetch metadata for the docs in scope; raise if any are missing."""
        by_id = {d["doc_id"]: d for d in self._storage.list_documents(collection)}
        missing = [did for did in doc_ids if did not in by_id]
        if missing:
            raise DocumentNotFoundError(
                f"doc_ids not found in collection '{collection}': {missing}"
            )
        return [by_id[did] for did in doc_ids]

    @staticmethod
    def _normalize_doc_ids(doc_ids: str | list[str] | None) -> list[str] | None:
        if isinstance(doc_ids, str):
            return [doc_ids]
        if doc_ids == []:
            raise ValueError(
                "doc_ids cannot be empty; pass None to query the whole collection"
            )
        return doc_ids

    def query(self, collection: str, question: str,
              doc_ids: str | list[str] | None = None) -> str:
        from ..agent import AgentRunner, SCOPED_SYSTEM_PROMPT, wrap_with_doc_context
        doc_ids = self._normalize_doc_ids(doc_ids)
        tools = self.get_agent_tools(collection, doc_ids)
        instructions = None
        if doc_ids:
            docs = self._scoped_docs(collection, doc_ids)
            question = wrap_with_doc_context(docs, question)
            instructions = SCOPED_SYSTEM_PROMPT
        return AgentRunner(tools=tools, model=self._retrieve_model,
                           instructions=instructions).run(question)

    async def query_stream(self, collection: str, question: str,
                           doc_ids: str | list[str] | None = None):
        from ..agent import QueryStream, SCOPED_SYSTEM_PROMPT, wrap_with_doc_context
        doc_ids = self._normalize_doc_ids(doc_ids)
        tools = self.get_agent_tools(collection, doc_ids)
        instructions = None
        if doc_ids:
            docs = self._scoped_docs(collection, doc_ids)
            question = wrap_with_doc_context(docs, question)
            instructions = SCOPED_SYSTEM_PROMPT
        stream = QueryStream(tools=tools, question=question,
                             model=self._retrieve_model, instructions=instructions)
        async for event in stream:
            yield event
