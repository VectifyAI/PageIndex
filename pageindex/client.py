import os
import uuid
import json
import asyncio
import concurrent.futures
from pathlib import Path

from .page_index import page_index
from .page_index_md import md_to_tree
from .retrieve import get_document, get_document_structure, get_page_content
from .utils import ConfigLoader

class PageIndexClient:
    """
    A client for indexing and retrieving document content.
    Flow: index() -> get_document() / get_document_structure() / get_page_content()

    For agent-based QA, see examples/openai_agents_demo.py.
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
        self.retrieve_model = opt.retrieve_model or self.model
        if self.workspace:
            self.workspace.mkdir(parents=True, exist_ok=True)
        self.documents = {}
        if self.workspace:
            self._load_workspace()

    def index(self, file_path: str, mode: str = "auto") -> str:
        """Index a document. Returns a document_id."""
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
            self.documents[doc_id] = {
                'id': doc_id,
                'path': file_path,
                'type': 'pdf',
                'structure': result['structure'],
                'doc_name': result.get('doc_name', ''),
                'doc_description': result.get('doc_description', '')
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
            self.documents[doc_id] = {
                'id': doc_id,
                'path': file_path,
                'type': 'md',
                'structure': result['structure'],
                'doc_name': result.get('doc_name', ''),
                'doc_description': result.get('doc_description', '')
            }
        else:
            raise ValueError(f"Unsupported file format for: {file_path}")

        print(f"Indexing complete. Document ID: {doc_id}")
        if self.workspace:
            self._save_doc(doc_id)
        return doc_id

    def _save_doc(self, doc_id: str):
        path = self.workspace / f"{doc_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.documents[doc_id], f, ensure_ascii=False, indent=2)

    def _load_workspace(self):
        loaded = 0
        for path in self.workspace.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                self.documents[path.stem] = doc
                loaded += 1
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: skipping corrupt workspace file {path.name}: {e}")
        if loaded:
            print(f"Loaded {loaded} document(s) from workspace.")

    def get_document(self, doc_id: str) -> str:
        """Return document metadata JSON."""
        return get_document(self.documents, doc_id)

    def get_document_structure(self, doc_id: str) -> str:
        """Return document tree structure JSON (without text fields)."""
        return get_document_structure(self.documents, doc_id)

    def get_page_content(self, doc_id: str, pages: str) -> str:
        """Return page content for the given pages string (e.g. '5-7', '3,8', '12')."""
        return get_page_content(self.documents, doc_id, pages)
