import os
import uuid
import json
import asyncio
import concurrent.futures
from pathlib import Path

import PyPDF2

from .page_index import page_index
from .page_index_md import (
    md_to_tree,
    extract_nodes_from_markdown,
    extract_node_text_content,
    get_node_summary,
    build_tree_from_nodes,
)
from .retrieve import get_document, get_document_structure, get_page_content
from .utils import (
    ConfigLoader,
    remove_fields,
    hash_text,
    compute_section_hashes,
    find_ancestors,
    structure_to_list,
    write_node_id,
    format_structure,
)

META_INDEX = "_meta.json"


def _normalize_retrieve_model(model: str) -> str:
    """Preserve supported Agents SDK prefixes and route other provider paths via LiteLLM."""
    passthrough_prefixes = ("litellm/", "openai/")
    if not model or "/" not in model:
        return model
    if model.startswith(passthrough_prefixes):
        return model
    return f"litellm/{model}"


class PageIndexClient:
    """
    A client for indexing and retrieving document content.
    Flow: index() -> get_document() / get_document_structure() / get_page_content()

    For agent-based QA, see examples/agentic_vectorless_rag_demo.py.
    """
    def __init__(self, api_key: str = None, model: str = None, retrieve_model: str = None, workspace: str = None):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY") and os.getenv("CHATGPT_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.getenv("CHATGPT_API_KEY")
        self.workspace = Path(workspace).expanduser() if workspace else None
        overrides = {}
        if model:
            overrides["model"] = model
        if retrieve_model:
            overrides["retrieve_model"] = retrieve_model
        opt = ConfigLoader().load(overrides or None)
        self.model = opt.model
        self.retrieve_model = _normalize_retrieve_model(opt.retrieve_model or self.model)
        if self.workspace:
            self.workspace.mkdir(parents=True, exist_ok=True)
        self.documents = {}
        if self.workspace:
            self._load_workspace()

    def index(self, file_path: str, mode: str = "auto") -> str:
        """Index a document. Returns a document_id."""
        # Persist a canonical absolute path so workspace reloads do not
        # reinterpret caller-relative paths against the workspace directory.
        file_path = os.path.abspath(os.path.expanduser(file_path))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = str(uuid.uuid4())
        ext = os.path.splitext(file_path)[1].lower()

        is_pdf = ext == '.pdf'
        is_md = ext in ['.md', '.markdown']

        if mode == "pdf" or (mode == "auto" and is_pdf):
            print(f"Indexing PDF: {file_path}")
            result = page_index(
                doc=file_path,
                model=self.model,
                if_add_node_summary='yes',
                if_add_node_text='yes',
                if_add_node_id='yes',
                if_add_doc_description='yes'
            )
            # Extract per-page text so queries don't need the original PDF
            pages = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(pdf_reader.pages, 1):
                    pages.append({'page': i, 'content': page.extract_text() or ''})

            self.documents[doc_id] = {
                'id': doc_id,
                'type': 'pdf',
                'path': file_path,
                'doc_name': result.get('doc_name', ''),
                'doc_description': result.get('doc_description', ''),
                'page_count': len(pages),
                'structure': result['structure'],
                'pages': pages,
            }

        elif mode == "md" or (mode == "auto" and is_md):
            print(f"Indexing Markdown: {file_path}")
            coro = md_to_tree(
                md_path=file_path,
                if_thinning=False,
                if_add_node_summary='yes',
                summary_token_threshold=200,
                model=self.model,
                if_add_doc_description='yes',
                if_add_node_text='yes',
                if_add_node_id='yes'
            )
            try:
                asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(asyncio.run, coro).result()
            except RuntimeError:
                result = asyncio.run(coro)
            # Compute hashes from the raw file to enable incremental update().
            _md_content = open(file_path, encoding='utf-8').read()
            _node_list, _md_lines = extract_nodes_from_markdown(_md_content)
            _flat_nodes = extract_node_text_content(_node_list, _md_lines)
            self.documents[doc_id] = {
                'id': doc_id,
                'type': 'md',
                'path': file_path,
                'doc_name': result.get('doc_name', ''),
                'doc_description': result.get('doc_description', ''),
                'line_count': result.get('line_count', 0),
                'structure': result['structure'],
                'file_hash': hash_text(_md_content),
                'section_hashes': compute_section_hashes(_flat_nodes),
            }
        else:
            raise ValueError(f"Unsupported file format for: {file_path}")

        print(f"Indexing complete. Document ID: {doc_id}")
        if self.workspace:
            self._save_doc(doc_id)
        return doc_id

    @staticmethod
    def _make_meta_entry(doc: dict) -> dict:
        """Build a lightweight meta entry from a document dict."""
        entry = {
            'type': doc.get('type', ''),
            'doc_name': doc.get('doc_name', ''),
            'doc_description': doc.get('doc_description', ''),
            'path': doc.get('path', ''),
        }
        if doc.get('type') == 'pdf':
            entry['page_count'] = doc.get('page_count')
        elif doc.get('type') == 'md':
            entry['line_count'] = doc.get('line_count')
        return entry

    @staticmethod
    def _read_json(path) -> dict | None:
        """Read a JSON file, returning None on any error."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: corrupt {Path(path).name}: {e}")
            return None

    def _save_doc(self, doc_id: str):
        doc = self.documents[doc_id].copy()
        # Strip text from structure nodes — redundant with pages (PDF only)
        if doc.get('structure') and doc.get('type') == 'pdf':
            doc['structure'] = remove_fields(doc['structure'], fields=['text'])
        path = self.workspace / f"{doc_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        self._save_meta(doc_id, self._make_meta_entry(doc))
        # Drop heavy fields; will lazy-load on demand
        self.documents[doc_id].pop('structure', None)
        self.documents[doc_id].pop('pages', None)

    def _rebuild_meta(self) -> dict:
        """Scan individual doc JSON files and return a meta dict."""
        meta = {}
        for path in self.workspace.glob("*.json"):
            if path.name == META_INDEX:
                continue
            doc = self._read_json(path)
            if doc and isinstance(doc, dict):
                meta[path.stem] = self._make_meta_entry(doc)
        return meta

    def _read_meta(self) -> dict | None:
        """Read and validate _meta.json, returning None on any corruption."""
        meta = self._read_json(self.workspace / META_INDEX)
        if meta is not None and not isinstance(meta, dict):
            print(f"Warning: {META_INDEX} is not a JSON object, ignoring")
            return None
        return meta

    def _save_meta(self, doc_id: str, entry: dict):
        meta = self._read_meta() or self._rebuild_meta()
        meta[doc_id] = entry
        meta_path = self.workspace / META_INDEX
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _load_workspace(self):
        meta = self._read_meta()
        if meta is None:
            meta = self._rebuild_meta()
            if meta:
                print(f"Loaded {len(meta)} document(s) from workspace (legacy mode).")
        for doc_id, entry in meta.items():
            doc = dict(entry, id=doc_id)
            if doc.get('path') and not os.path.isabs(doc['path']):
                doc['path'] = str((self.workspace / doc['path']).resolve())
            self.documents[doc_id] = doc

    def _ensure_doc_loaded(self, doc_id: str):
        """Load full document JSON on demand (structure, pages, etc.)."""
        doc = self.documents.get(doc_id)
        if not doc or doc.get('structure') is not None:
            return
        full = self._read_json(self.workspace / f"{doc_id}.json")
        if not full:
            return
        doc['structure'] = full.get('structure', [])
        if full.get('pages'):
            doc['pages'] = full['pages']
        if full.get('section_hashes'):
            doc['section_hashes'] = full['section_hashes']
        if full.get('file_hash'):
            doc['file_hash'] = full['file_hash']

    def update(self, doc_id: str) -> dict:
        """Incrementally update an indexed MD document.

        Re-summarizes only sections whose own text changed (plus their
        ancestors, whose roll-up may be affected); unchanged sections reuse
        their cached summary. Returns a status dict describing the change set.
        """
        self._ensure_doc_loaded(doc_id)
        doc = self.documents.get(doc_id)
        if not doc:
            raise ValueError(f"Unknown doc_id: {doc_id}")
        if doc.get('type') != 'md':
            raise ValueError("update() only supports MD documents")

        file_path = doc['path']
        content = open(file_path, encoding='utf-8').read()

        # Gate 1: file-level hash — skip entirely if nothing changed.
        new_file_hash = hash_text(content)
        if new_file_hash == doc.get('file_hash'):
            return {"status": "unchanged"}

        # Gate 2: section-level diff.
        node_list, md_lines = extract_nodes_from_markdown(content)
        new_nodes = extract_node_text_content(node_list, md_lines)
        new_hashes = compute_section_hashes(new_nodes)
        old_hashes = doc.get('section_hashes') or {}

        new_keys = set(new_hashes)
        old_keys = set(old_hashes)
        added = new_keys - old_keys
        deleted = old_keys - new_keys
        changed = {p for p in new_keys & old_keys if new_hashes[p] != old_hashes[p]}

        # Dirty sections plus the ancestors of each (roll-up summaries).
        dirty = changed | added
        to_summarize = set(dirty)
        for path in dirty:
            to_summarize.update(find_ancestors(path))

        # Reuse cached summaries for clean sections.
        old_structure_flat = structure_to_list(doc.get('structure', []))
        old_summary_map = {
            n.get('title_path', n.get('title')): n.get('summary') or n.get('prefix_summary', '')
            for n in old_structure_flat
        }

        async def _identity(val):
            return val

        async def _regenerate():
            tasks = {}
            for path, node in {n['title_path']: n for n in new_nodes}.items():
                if path in to_summarize:
                    tasks[path] = get_node_summary(node, summary_token_threshold=200, model=self.model)
                else:
                    tasks[path] = _identity(old_summary_map.get(path, ''))
            return {path: await coro for path, coro in tasks.items()}

        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                summaries = pool.submit(asyncio.run, _regenerate()).result()
        except RuntimeError:
            summaries = asyncio.run(_regenerate())

        for node in new_nodes:
            node['summary'] = summaries.get(node['title_path'], '')

        # Rebuild the tree with fresh node ids.
        new_structure = build_tree_from_nodes(new_nodes)
        write_node_id(new_structure)
        new_structure = format_structure(
            new_structure,
            order=['title', 'node_id', 'line_num', 'summary', 'prefix_summary', 'text', 'nodes'],
        )

        doc['structure'] = new_structure
        doc['file_hash'] = new_file_hash
        doc['section_hashes'] = new_hashes
        doc['line_count'] = content.count('\n') + 1

        if self.workspace:
            tmp = self.workspace / f"{doc_id}.tmp"
            save_doc = dict(doc)
            save_doc['structure'] = new_structure
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(save_doc, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.workspace / f"{doc_id}.json")
            self._save_meta(doc_id, self._make_meta_entry(doc))

        return {
            "status": "updated",
            "updated": sorted(changed),
            "added": sorted(added),
            "deleted": sorted(deleted),
        }

    def get_document(self, doc_id: str) -> str:
        """Return document metadata JSON."""
        return get_document(self.documents, doc_id)

    def get_document_structure(self, doc_id: str) -> str:
        """Return document tree structure JSON (without text fields)."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_document_structure(self.documents, doc_id)

    def get_page_content(self, doc_id: str, pages: str) -> str:
        """Return page content for the given pages string (e.g. '5-7', '3,8', '12')."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_page_content(self.documents, doc_id, pages)
